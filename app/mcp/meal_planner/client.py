"""Client bindings for the meal-planner MCP server.

Where the server lives and which of its tools a model may drive on its own. The
connection plumbing itself is shared — see app/mcp/client.py.
"""

from typing import List

from langchain_core.tools import BaseTool

from app.core.config import MEAL_MCP_TIMEOUT, MEAL_MCP_URL
from app.mcp.client import find_tool, load_tools, only

MEAL_SERVER = "meal"

# Tools the routing model is allowed to call on its own. `save_meal_plan` is
# excluded deliberately: persisting a week of meals needs human approval, which
# a tool loop cannot pause for, so the supervisor calls it directly after its
# `interrupt()` resolves. See supervisor.workflow.meal_agent.
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


async def load_meal_tools(user_id: str) -> List[BaseTool]:
    """Every meal tool, bound to this user. Empty when the server is down."""
    return await load_tools(MEAL_SERVER, MEAL_MCP_URL, user_id, MEAL_MCP_TIMEOUT)


def autonomous_tools(tools: List[BaseTool]) -> List[BaseTool]:
    """The subset a tool loop may drive itself (see AUTONOMOUS_TOOLS)."""
    return only(tools, AUTONOMOUS_TOOLS)


__all__ = [
    "AUTONOMOUS_TOOLS",
    "MEAL_SERVER",
    "SAVE_PLAN_TOOL",
    "autonomous_tools",
    "find_tool",
    "load_meal_tools",
]
