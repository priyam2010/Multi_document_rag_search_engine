import os
from langchain_google_genai import ChatGoogleGenerativeAI
from config import GOOGLE_API_KEY, LLM_MODEL, TOP_K
from web_search import tavily_search

def safe_str(x):
    return x if isinstance(x, str) else ""

def classify_query(query: str):
    q = query.lower()
    if any(word in q for word in ["latest", "recent", "current", "today"]):
        return "web"
    if any(word in q for word in ["compare", "vs", "difference"]):
        return "hybrid"
    return "document"

def generate_answer(query, vectorstore, use_web=True):
    # Check if API Key exists to prevent crash
    if not GOOGLE_API_KEY:
        return "Error: GOOGLE_API_KEY is missing in Secrets/Env.", [], "error"

    route = classify_query(query)
    sources = []
    context_parts = []

    # 1. Document Retrieval
    if route in ["document", "hybrid"] and vectorstore:
        docs = vectorstore.similarity_search(query, k=TOP_K)
        for d in docs:
            text = safe_str(d.page_content)
            if text.strip():
                context_parts.append(text)
                sources.append(f"[Doc] {d.metadata.get('title', 'Unknown')} | Chunk {d.metadata.get('chunk_index', '?')}")

    # 2. Web Search Retrieval
    if route in ["web", "hybrid"] and use_web:
        web_results = tavily_search(query)
        for w in web_results:
            text = safe_str(w.get("content"))
            if text.strip():
                context_parts.append(text)
                sources.append(f"[Web] {w.get('title', 'Unknown')}")

    context = "\n\n".join(context_parts).strip()
    if not context:
        context = "No specific context found. Answer using general knowledge."

    # 3. Gemini Generation
    try:
        llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL,
            google_api_key=GOOGLE_API_KEY,
            temperature=0.3
        )

        messages = [
            ("system", "You are a helpful assistant. Use the context to answer clearly."),
            ("human", f"Question: {query}\n\nContext:\n{context}")
        ]

        response = llm.invoke(messages)
        return response.content, list(set(sources)), route

    except Exception as e:
        return f"Gemini Error: {str(e)}", sources, route


