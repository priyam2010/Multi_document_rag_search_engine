from langchain_community.tools.tavily_search import TavilySearchResults
from config import TAVILY_API_KEY


def tavily_search(query):
    tool = TavilySearchResults(
        api_key=TAVILY_API_KEY,
        max_results=5
    )

    results = tool.run(query)

    formatted = []
    for r in results:
        formatted.append({
            "title": r["title"],
            "content": r["content"],
            "url": r["url"]
        })

    return formatted
