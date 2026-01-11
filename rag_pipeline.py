import os
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from config import GROQ_API_KEY, LLM_MODEL, TOP_K
from web_search import tavily_search

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY not set")

llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name=LLM_MODEL
)

MAX_CONTEXT_CHARS = 12000

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
    context = ""

    if route in ["document", "hybrid"] and vectorstore:
        docs = vectorstore.similarity_search(query, k=TOP_K)
        for d in docs:
            context += d.page_content + "\n"
            sources.append(f"[Doc] {d.metadata['source_id']}")

    if route in ["web", "hybrid"] and use_web:
        for w in tavily_search(query):
            context += w["content"] + "\n"
            sources.append(f"[Web] {w['title']}")

    context = context[:MAX_CONTEXT_CHARS]

    prompt = f"""
Answer the question using the context below.
Cite sources when possible.

Question: {query}

Context:
{context}
"""

    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content, sources, route
