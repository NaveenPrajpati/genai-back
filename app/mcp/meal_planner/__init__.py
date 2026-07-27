"""The meal planner published over the Model Context Protocol.

`server` is the MCP server itself; `client` is how this app consumes it.
"""

from .client import (
    AUTONOMOUS_TOOLS,
    MEAL_SERVER,
    SAVE_PLAN_TOOL,
    autonomous_tools,
    find_tool,
    load_meal_tools,
)
from .server import mcp

__all__ = [
    "AUTONOMOUS_TOOLS",
    "MEAL_SERVER",
    "SAVE_PLAN_TOOL",
    "autonomous_tools",
    "find_tool",
    "load_meal_tools",
    "mcp",
]
