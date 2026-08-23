"""The job helper published over the Model Context Protocol.

`server` is the MCP server itself; `client` is how this app consumes it.
"""

from .client import (
    AUTONOMOUS_TOOLS,
    JOB_SERVER,
    PROPOSE_RESUME_TOOL,
    SAVE_RESUME_TOOL,
    autonomous_tools,
    find_tool,
    load_job_tools,
)
from .server import mcp

__all__ = [
    "AUTONOMOUS_TOOLS",
    "JOB_SERVER",
    "PROPOSE_RESUME_TOOL",
    "SAVE_RESUME_TOOL",
    "autonomous_tools",
    "find_tool",
    "load_job_tools",
    "mcp",
]
