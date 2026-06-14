"""External tool calls (web search) for the personal-assistant agent."""

import logging
import os

import httpx

logger = logging.getLogger(__name__)


async def web_search(query: str, max_results: int = 5) -> dict:
    """Run a web search via the Tavily API. Returns
    {answer, results:[{title,url,content}]}. Degrades gracefully to empty
    results if no API key is configured."""
    api_key = os.getenv("TAVILY_API_KEY", "")
    if not api_key:
        logger.info("TAVILY_API_KEY not set; skipping web search")
        return {"answer": "", "results": []}
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": api_key,
                    "query": query,
                    "max_results": max_results,
                    "search_depth": "basic",
                    "include_answer": True,
                },
            )
            if resp.status_code != 200:
                logger.error("Tavily error %s: %s", resp.status_code, resp.text)
                return {"answer": "", "results": []}
            data = resp.json()
            return {
                "answer": data.get("answer", "") or "",
                "results": [
                    {
                        "title": r.get("title", ""),
                        "url": r.get("url", ""),
                        "content": r.get("content", ""),
                    }
                    for r in data.get("results", [])
                ],
            }
    except Exception as e:
        logger.error("web_search error: %s", e)
        return {"answer": "", "results": []}
