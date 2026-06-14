"""Search tooling shared by the learning-tracker research agent and digest job."""

from langchain_tavily import TavilySearch

tavily_search_tool = TavilySearch(max_results=5, topic="general")
