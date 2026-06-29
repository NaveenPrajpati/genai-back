"""Search tooling shared by the learning-tracker research agent and digest job."""

from langchain_tavily import TavilySearch
from langgraph.prebuilt import ToolNode

from app.core.llm import llm

tavily_search_tool = TavilySearch(max_results=5, topic="general")

# ReAct wiring for research_agent: the LLM decides what to search for (and may
# search more than once) before curating resources. The digest job keeps calling
# tavily_search_tool directly with a fixed query.
research_tools = [tavily_search_tool]
research_tool_node = ToolNode(research_tools)
research_llm = llm.bind_tools(research_tools)
