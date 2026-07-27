"""Shared MCP client plumbing.

Server-agnostic: each MCP package builds its own thin wrapper on top of this
(see meal_planner/client.py) holding that server's URL and tool policy.

A client is built PER USER rather than cached, because identity rides on the
connection's `X-User-Id` header — see the IDENTITY section in any server module.
Sharing one client across users would leak the header of whoever built it first.
The cost is a tools/list round-trip per turn against a local endpoint.
"""

import logging
from typing import List, Optional

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from app.core.config import MCP_AUTH_TOKEN

logger = logging.getLogger(__name__)


def connect(name: str, url: str, user_id: str, timeout: float) -> MultiServerMCPClient:
    """A client for one MCP server, bound to `user_id` via the transport header."""
    headers = {"X-User-Id": user_id}
    if MCP_AUTH_TOKEN:
        headers["Authorization"] = f"Bearer {MCP_AUTH_TOKEN}"
    return MultiServerMCPClient(
        {
            name: {
                "transport": "streamable_http",
                "url": url,
                "headers": headers,
                "timeout": timeout,
            }
        }
    )


async def load_tools(
    name: str, url: str, user_id: str, timeout: float
) -> List[BaseTool]:
    """Every tool the server publishes, as LangChain tools bound to this user.

    Returns an empty list if the server is unreachable, so a dead MCP dependency
    degrades that one skill instead of failing the whole turn.
    """
    try:
        return await connect(name, url, user_id, timeout).get_tools(server_name=name)
    except Exception:
        logger.exception("MCP server %r unreachable at %s", name, url)
        return []


def find_tool(tools: List[BaseTool], name: str) -> Optional[BaseTool]:
    return next((t for t in tools if t.name == name), None)


def only(tools: List[BaseTool], allowed: set) -> List[BaseTool]:
    """The subset a tool-calling loop is allowed to drive itself."""
    return [t for t in tools if t.name in allowed]
