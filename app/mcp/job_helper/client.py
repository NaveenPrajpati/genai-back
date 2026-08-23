"""Client bindings for the job-helper MCP server.

Where the server lives and which of its tools a model may drive on its own. The
connection plumbing itself is shared — see app/mcp/client.py.
"""

from typing import List

from langchain_core.tools import BaseTool

from app.core.config import JOB_MCP_TIMEOUT, JOB_MCP_URL
from app.mcp.client import find_tool, load_tools, only

JOB_SERVER = "jobs"

# Tools the agent's loop is allowed to call on its own.
#
# `save_tailored_resume` is excluded deliberately: a tailored resume is what the
# user sends to an employer under their own name, so it needs human approval,
# which a tool loop cannot pause for. The graph calls it directly once its
# `interrupt()` resolves — see agents/job_helper/workflow.py.
#
# The writes that ARE autonomous are all reversible bookkeeping (tracking a
# posting, a status change, a note, a to-do, recording resume details the user
# just dictated), matching the meal planner's line between logging what happened
# and committing something the user must stand behind. `start_roadmap_for_gap`
# is the one heavy autonomous call; it is gated by requiring a fit assessment to
# have run first.
AUTONOMOUS_TOOLS = {
    "list_applications",
    "get_application",
    "list_resumes",
    "get_resume_details",
    "get_job_search_profile",
    "search_jobs",
    "track_job_posting",
    "update_application_status",
    "add_application_note",
    "assess_resume_fit",
    "propose_tailored_resume",
    "draft_cover_letter",
    "prepare_for_interview",
    "save_resume_details",
    "update_resume_details",
    "add_followup_tasks",
    "start_roadmap_for_gap",
}

SAVE_RESUME_TOOL = "save_tailored_resume"
PROPOSE_RESUME_TOOL = "propose_tailored_resume"


async def load_job_tools(user_id: str) -> List[BaseTool]:
    """Every job-helper tool, bound to this user. Empty when the server is down."""
    return await load_tools(JOB_SERVER, JOB_MCP_URL, user_id, JOB_MCP_TIMEOUT)


def autonomous_tools(tools: List[BaseTool]) -> List[BaseTool]:
    """The subset a tool loop may drive itself (see AUTONOMOUS_TOOLS)."""
    return only(tools, AUTONOMOUS_TOOLS)


__all__ = [
    "AUTONOMOUS_TOOLS",
    "JOB_SERVER",
    "PROPOSE_RESUME_TOOL",
    "SAVE_RESUME_TOOL",
    "autonomous_tools",
    "find_tool",
    "load_job_tools",
]
