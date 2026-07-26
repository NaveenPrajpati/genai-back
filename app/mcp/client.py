"""MCP client: pulls the meal-planner tools in as LangChain tools.

The supervisor's meal node does not import `agents.meal_planner` at all — it
talks to the MCP server over the wire, so the meal planner is a genuinely
separable service (point `MEAL_MCP_URL` at another host and nothing else
changes) and the same tools an external client sees are the ones the graph uses.

A client is built PER USER rather than cached, because identity rides on the
connection's `X-User-Id` header (see meal_server.IDENTITY). Sharing one client
across users would leak the header of whoever built it first. The cost is a
tools/list round-trip per turn against a local, in-process endpoint.
"""

import logging
from typing import List, Optional

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from app.core.config import MCP_AUTH_TOKEN, MEAL_MCP_TIMEOUT, MEAL_MCP_URL

logger = logging.getLogger(__name__)

MEAL_SERVER = "meal"

# Tools the routing model is allowed to call on its own. `save_meal_plan` is
# excluded deliberately: persisting a week of meals needs human approval, which
# a tool loop cannot pause for, so the supervisor calls it directly after its
# `interrupt()` resolves. See supervisor.workflow.meal_node.
AUTONOMOUS_TOOLS = {
    "list_meal_plans",
    "get_meal_plan",
    "get_grocery_list",
    "get_food_preferences",
    "propose_meal_plan",
    "suggest_meals",
    "log_meal",
    "add_food_dislike",
}

SAVE_PLAN_TOOL = "save_meal_plan"


def _client(user_id: str) -> MultiServerMCPClient:
    headers = {"X-User-Id": user_id}
    if MCP_AUTH_TOKEN:
        headers["Authorization"] = f"Bearer {MCP_AUTH_TOKEN}"
    return MultiServerMCPClient(
        {
            MEAL_SERVER: {
                "transport": "streamable_http",
                "url": MEAL_MCP_URL,
                "headers": headers,
                "timeout": MEAL_MCP_TIMEOUT,
            }
        }
    )


async def load_meal_tools(user_id: str) -> List[BaseTool]:
    """Every meal tool, bound to this user via the connection header. Returns an
    empty list if the server is unreachable, so a dead MCP dependency degrades
    the meal skill instead of failing the whole supervisor turn."""
    try:
        return await _client(user_id).get_tools(server_name=MEAL_SERVER)
    except Exception:
        logger.exception("meal MCP server unreachable at %s", MEAL_MCP_URL)
        return []


def autonomous_tools(tools: List[BaseTool]) -> List[BaseTool]:
    """The subset a tool-calling loop may drive itself (see AUTONOMOUS_TOOLS)."""
    return [t for t in tools if t.name in AUTONOMOUS_TOOLS]


def find_tool(tools: List[BaseTool], name: str) -> Optional[BaseTool]:
    return next((t for t in tools if t.name == name), None)
