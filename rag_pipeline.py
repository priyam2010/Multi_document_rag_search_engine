from langchain_groq import ChatGroq
from config import GROQ_API_KEY, TOP_K
from web_search import tavily_search


import os
from langchain_groq import ChatGroq

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is not set. Check .env loading.")

from config import LLM_MODEL

llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name=LLM_MODEL
)




def classify_query(query: str):
    query = query.lower()

    if "latest" in query or "current" in query or "recent" in query:
        return "web"
    if "compare" in query or "vs" in query:
        return "hybrid"
    return "document"


def generate_answer(query, vectorstore, use_web=True):
    route = classify_query(query)
    sources = []
    context = ""

    if route in ["document", "hybrid"] and vectorstore:
        docs = vectorstore.similarity_search(query, k=TOP_K)
        for d in docs:
            context += d.page_content + "\n"
            sources.append(f"[Doc] {d.metadata['title']} – Chunk {d.metadata['chunk_index']}")

    if route in ["web", "hybrid"] and use_web:
        web_results = tavily_search(query)
        for w in web_results:
            context += w["content"] + "\n"
            sources.append(f"[Web] {w['title']}")

    prompt = f"""
Answer the question using the context below.
Clearly ground the answer in the sources.

Question: {query}

Context:
{context}
"""

    response = llm.invoke(prompt)

    return response.content, sources, route
