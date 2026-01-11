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
    # Check for API Key
    if not GOOGLE_API_KEY:
        return "Error: GOOGLE_API_KEY is missing. Please set it in your environment or Streamlit secrets.", [], "error"

    route = classify_query(query)
    sources = []
    context_parts = []

    # 1. Document Search
    if route in ["document", "hybrid"] and vectorstore:
        try:
            docs = vectorstore.similarity_search(query, k=TOP_K)
            for d in docs:
                text = safe_str(d.page_content)
                if text.strip():
                    context_parts.append(text)
                    sources.append(
                        f"[Doc] {d.metadata.get('title', 'Unknown')} | Chunk {d.metadata.get('chunk_index', '?')}"
                    )
        except Exception as e:
            print(f"Vectorstore error: {e}")

    # 2. Web Search
    if route in ["web", "hybrid"] and use_web:
        try:
            for w in tavily_search(query):
                text = safe_str(w.get("content"))
                if text.strip():
                    context_parts.append(text)
                    sources.append(f"[Web] {w.get('title', 'Unknown')}")
        except Exception as e:
            print(f"Web search error: {e}")

    context = "\n\n".join(context_parts).strip()
    if not context:
        context = "No specific context found. Answer based on general knowledge."

    # 3. Gemini Generation
    try:
        # Initialize inside the function to prevent top-level crashes
        llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL,
            google_api_key=GOOGLE_API_KEY,
            temperature=0.3,
            max_output_tokens=800
        )

        messages = [
            ("system", "You are a professional assistant. Use the provided context to answer the user question accurately. If the information isn't in the context, use your general knowledge but mention it."),
            ("human", f"Question: {query}\n\nContext:\n{context}")
        ]

        response = llm.invoke(messages)
        return response.content, list(set(sources)), route

    except Exception as e:
        return f"Gemini API Error: {str(e)}", sources, route

