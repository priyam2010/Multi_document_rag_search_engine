from langchain_groq import ChatGroq
from config import TOP_K, LLM_MODEL
from web_search import tavily_search


# -------------------------------
# LLM FACTORY (SAFE FOR STREAMLIT FREE)
# -------------------------------
def get_llm(groq_api_key: str):
    if not groq_api_key:
        raise ValueError("Groq API key is missing")

    return ChatGroq(
        groq_api_key=groq_api_key,
        model_name=LLM_MODEL
    )


# -------------------------------
# QUERY ROUTING
# -------------------------------
def classify_query(query: str):
    query = query.lower()

    if "latest" in query or "current" in query or "recent" in query:
        return "web"
    if "compare" in query or "vs" in query:
        return "hybrid"
    return "document"


# -------------------------------
# ANSWER GENERATION
# -------------------------------
def generate_answer(query, vectorstore, groq_api_key, use_web=True):
    route = classify_query(query)
    sources = []
    context = ""

    # 🔹 DOCUMENT SEARCH
    if route in ["document", "hybrid"] and vectorstore:
        docs = vectorstore.similarity_search(query, k=TOP_K)
        for d in docs:
            context += d.page_content + "\n"
            sources.append(
                f"[Doc] {d.metadata.get('title', 'Unknown')} – "
                f"Chunk {d.metadata.get('chunk_index', 'N/A')}"
            )

    # 🔹 WEB SEARCH
    if route in ["web", "hybrid"] and use_web:
        web_results = tavily_search(query)
        for w in web_results:
            context += w["content"] + "\n"
            sources.append(f"[Web] {w['title']}")

    prompt = f"""
Answer the question using the context below.
Clearly ground the answer in the sources.

Question:
{query}

Context:
{context}
"""

    # ✅ LLM CREATED ONLY HERE
    llm = get_llm(groq_api_key)
    response = llm.invoke(prompt)

    return response.content, sources, route
