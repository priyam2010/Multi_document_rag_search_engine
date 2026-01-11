from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from config import GROQ_API_KEY, LLM_MODEL, TOP_K
from web_search import tavily_search

llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name=LLM_MODEL
)

MAX_CONTEXT_CHARS = 10000

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

    # ---------- DOCUMENT SEARCH ----------
    if route in ["document", "hybrid"] and vectorstore:
        docs = vectorstore.similarity_search(query, k=TOP_K)
        for d in docs:
            text = safe_str(d.page_content)
            if text.strip():
                context_parts.append(text)
                sources.append(
                    f"[Doc] {d.metadata.get('source_id', 'Unknown')} | Chunk {d.metadata.get('chunk_index', '?')}"
                )

    # ---------- WEB SEARCH ----------
    if route in ["web", "hybrid"] and use_web:
        for w in tavily_search(query):
            text = safe_str(w.get("content"))
            if text.strip():
                context_parts.append(text)
                sources.append(f"[Web] {w.get('title', 'Unknown')}")

    context = "\n".join(context_parts).strip()

    # 🚨 CRITICAL FIX
    if not context:
        context = "No relevant context found. Answer using general knowledge."

    context = context[:MAX_CONTEXT_CHARS]

    prompt = (
        "You are a helpful assistant.\n"
        "Answer the question clearly and concisely.\n\n"
        f"Question:\n{query}\n\n"
        f"Context:\n{context}"
    )

    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content, sources, route

