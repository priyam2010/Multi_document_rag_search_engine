from groq import Groq
from config import GROQ_API_KEY, LLM_MODEL, TOP_K
from web_search import tavily_search

client = Groq(api_key=GROQ_API_KEY)

MAX_CONTEXT_CHARS = 6000


def safe_str(x):
    return x if isinstance(x, str) else ""


def classify_query(query: str):
    q = query.lower()
    if "latest" in q or "recent" in q:
        return "web"
    if "compare" in q or "vs" in q:
        return "hybrid"
    return "document"


def generate_answer(query, vectorstore, use_web=True):
    route = classify_query(query)
    sources = []
    context_parts = []

    # -------- DOCUMENT SEARCH --------
    if route in ["document", "hybrid"] and vectorstore:
        docs = vectorstore.similarity_search(query, k=TOP_K)
        for d in docs:
            text = safe_str(d.page_content)
            if text.strip():
                context_parts.append(text)
                sources.append(
                    f"[Doc] {d.metadata.get('source_id', 'Unknown')} | Chunk {d.metadata.get('chunk_index', '?')}"
                )

    # -------- WEB SEARCH --------
    if route in ["web", "hybrid"] and use_web:
        for w in tavily_search(query):
            text = safe_str(w.get("content"))
            if text.strip():
                context_parts.append(text)
                sources.append(f"[Web] {w.get('title', 'Unknown')}")

    context = "\n".join(context_parts).strip()

    if not context:
        context = "Answer the question using general knowledge."

    context = context[:MAX_CONTEXT_CHARS]

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Answer clearly and concisely."
        },
        {
            "role": "user",
            "content": f"{query}\n\nContext:\n{context}"
        }
    ]

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            temperature=0.2,
            max_tokens=400,
        )
    except Exception as e:
        # 🚨 DO NOT CRASH STREAMLIT
        return f"Groq API Error: {str(e)}", sources, route

    return response.choices[0].message.content, sources, route






