from langchain_community.tools.tavily_search import TavilySearchResults
from config import TAVILY_API_KEY

def tavily_search(query):
    if not TAVILY_API_KEY:
        return []

    tool = TavilySearchResults(
        api_key=TAVILY_API_KEY,
        max_results=5
    )

    results = tool.run(query)
    formatted = []

    for r in results:
        formatted.append({
            "title": r.get("title", ""),
            "content": r.get("content", ""),
            "url": r.get("url", "")
        })

    return formatted

