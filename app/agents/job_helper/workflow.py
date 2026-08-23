"""LangGraph nodes and graph wiring for the job-helper agent.

    START → load_memory → agent ─┬→ approve_resume → finalize → END
                                 └────────────────→ finalize → END

WHY THIS ONE IS A TOOL LOOP, NOT A CLASSIFIER
─────────────────────────────────────────────
The learning tracker and the personal assistant classify a message into one
intent and then run a node per intent. That fits work with a small, fixed verb
list. A job-search turn does not decompose that way: "should I apply to this?"
is read the posting, score my resume against it, and tell me — three tools whose
order depends on what the first one returns. So the middle of this graph is a
ReAct loop over the MCP tools, and the graph's own structure exists for the one
thing a tool loop cannot express: stopping for a human.

WHY THE SAVE IS NOT IN THE LOOP
───────────────────────────────
A tailored resume goes out under the user's name, so it is not written until
they have seen the changelog and approved it. MCP is stateless request/response
and has nowhere to pause, so the pair is split: `propose_tailored_resume`
(writes nothing) is left to the model inside the loop, while
`save_tailored_resume` is called by `approve_resume` after `interrupt()`
resolves. Same split as the meal planner's propose/save, for the same reason.

This graph is also compiled as a subgraph inside the supervisor. It is compiled
there WITHOUT a checkpointer so it inherits the parent's, which is what lets the
approval below pause the whole supervisor run and resume from one `Command` on
the parent thread.
"""

import logging
from typing import Any, List, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt

from app.agents.approval_store import create_pending, get_pending, resolve
from app.agents.memory_store import get_profile
from app.agents.react import run_tool_loop
from app.core.llm import llm
from app.mcp.client import decode_result
from app.mcp.job_helper.client import (
    PROPOSE_RESUME_TOOL,
    SAVE_RESUME_TOOL,
    autonomous_tools,
    find_tool,
    load_job_tools,
)

from .state import JobHelperState

logger = logging.getLogger(__name__)

RESUME_APPROVAL = "job_save_tailored_resume"

AGENT_SYSTEM = """\
You help the user with their job search: finding openings, tracking the postings \
they are applying to, comparing their resume against one, tailoring it, drafting \
cover letters, preparing for interviews, and keeping each application's status \
current.

Work from their real data. Call list_applications when they mention a job \
without naming an id, and get_job_search_profile before saying anything about \
what they want or what they have. When they paste a job description, track it \
first — everything else needs the application id that returns.

Searching and tracking are different. search_jobs browses and saves nothing; \
when the user picks a listing, track_job_posting is what starts following it. If \
a search comes back with a note about the board's coverage, pass that on rather \
than presenting a partial view as the whole market.

Be honest about fit. If their resume does not evidence something a posting \
requires, say so plainly and offer to start a roadmap on it; never paper over a \
gap, and never imply they have experience they do not.

Their uploaded resume file is never modified, and you should say so when it \
comes up — tailoring produces edits they apply inside their own document, so \
their layout survives. To tailor, call propose_tailored_resume; the proposal is \
shown to them for approval, so do not tell them it has been saved or hand them a \
download link. Say what changed and that it is waiting for their go-ahead.

If they have no resume yet, offer to take the details down as they describe them \
— save_resume_details — rather than insisting on a file.

Finish with a brief plain-language summary of what you did and what it means for \
them. No tool names, no ids.
"""


def _final_text(messages: List[Any]) -> str:
    for message in reversed(messages):
        if isinstance(message, AIMessage) and message.content:
            return message.content
    return ""


def _last_proposal(messages: List[Any]) -> Optional[dict]:
    """The most recent unsaved resume proposal in the tool transcript, if any."""
    for message in reversed(messages):
        if getattr(message, "name", None) != PROPOSE_RESUME_TOOL:
            continue
        result = decode_result(message)
        if isinstance(result, dict) and result.get("content"):
            return result
    return None


# ── nodes ────────────────────────────────────────────────────────────────────
async def load_memory(state: JobHelperState):
    """Load the shared profile and record the user's turn."""
    memory = await get_profile(state["user_id"])
    return {
        "memory": memory,
        "messages": [HumanMessage(content=state.get("query", ""))],
    }


async def agent(state: JobHelperState):
    """The tool loop over the job-helper MCP server.

    Replay-safe: on resume from the approval below, LangGraph re-runs this node
    from the top, so a stored proposal short-circuits it. Without that the user
    would pay for a second tool loop and could be handed a *different* rewrite
    than the one they just approved.

    The stored query is what makes that a replay check rather than a trap. A
    pending approval outlives the turn that raised it — the user can abandon it
    and ask something else on the same thread — and matching on the id alone
    would answer that new question with the old proposal forever.
    """
    user_id = state["user_id"]
    query = state.get("query", "")
    existing = await get_pending(state["thread_id"], [RESUME_APPROVAL])
    payload = (existing or {}).get("payload") or {}
    if existing and payload.get("query") == query:
        return {
            "summary": payload.get("transcript", ""),
            "tailored": payload.get("proposal"),
        }

    tools = await load_job_tools(user_id)
    if not tools:
        return {
            "summary": "The job helper is unavailable right now.",
            "resume_status": "unavailable",
        }

    loop_tools = autonomous_tools(tools)
    context = []
    memory = state.get("memory") or {}
    known = {
        label: memory[key]
        for key, label in (
            ("target_roles", "Target roles"),
            ("current_title", "Current title"),
            ("years_experience", "Years of experience"),
            ("preferred_work_mode", "Work mode"),
        )
        if memory.get(key)
    }
    if known:
        context.append(
            "Known about this user — " + "; ".join(f"{k}: {v}" for k, v in known.items())
        )
    # A turn opened from one application's screen carries its id, so "tailor my
    # resume for this" resolves without a lookup the user cannot see the need for.
    if state.get("application_id"):
        context.append(f"The user is looking at application {state['application_id']}.")
    if state.get("resume_id"):
        context.append(f"Use resume {state['resume_id']} unless they say otherwise.")

    messages = await run_tool_loop(
        llm.bind_tools(loop_tools),
        ToolNode(loop_tools),
        [
            SystemMessage(content="\n\n".join([AGENT_SYSTEM, *context])),
            HumanMessage(content=query),
        ],
    )
    return {"summary": _final_text(messages), "tailored": _last_proposal(messages)}


def route(state: JobHelperState) -> str:
    """A proposal on the state means a human has to see it before it is written."""
    return "approve_resume" if state.get("tailored") else "finalize"


async def approve_resume(state: JobHelperState):
    """Show the rewrite, wait for the user, then persist it or drop it."""
    user_id = state["user_id"]
    thread_id = state["thread_id"]
    proposal = state.get("tailored") or {}

    existing = await get_pending(thread_id, [RESUME_APPROVAL])
    if existing:
        approval_id = str(existing["_id"])
    else:
        approval_id = await create_pending(
            user_id,
            thread_id,
            RESUME_APPROVAL,
            {
                "proposal": proposal,
                "transcript": state.get("summary", ""),
                # What `agent` matches on to tell a replay from a new question.
                "query": state.get("query", ""),
            },
        )

    decision = interrupt(
        {
            "type": RESUME_APPROVAL,
            "approval_id": approval_id,
            "application_id": proposal.get("application_id"),
            "company": proposal.get("company"),
            "role_title": proposal.get("role_title"),
            "changes": proposal.get("changes"),
            "keywords_added": proposal.get("keywords_added"),
            "content": proposal.get("content"),
        }
    )

    if decision != "approved":
        await resolve(approval_id, "rejected")
        return {"resume_status": "rejected by user"}

    tools = await load_job_tools(user_id)
    save_tool = find_tool(tools, SAVE_RESUME_TOOL)
    if not save_tool:
        # The server went away between proposing and approving. Say so rather
        # than reporting a save that did not happen.
        logger.error("job helper: save tool unavailable at approval time")
        return {"resume_status": "could not be saved — the job helper is unavailable"}

    saved = decode_result(
        await save_tool.ainvoke(
            {
                "application_id": proposal.get("application_id"),
                "content": proposal.get("content"),
                "base_resume_id": proposal.get("base_resume_id"),
                "changes": proposal.get("changes") or [],
                "keywords_added": proposal.get("keywords_added") or [],
            }
        )
    )
    await resolve(approval_id, "approved")
    resume_id = saved.get("resume_id") if isinstance(saved, dict) else None
    return {"resume_status": "approved and saved", "resume_version_id": resume_id}


async def finalize(state: JobHelperState):
    """One reply, composed from what the loop said plus what the approval did.

    Deliberately not a second LLM call: the tool loop already wrote the prose,
    and the only thing it could not know is how the approval resolved — which is
    one sentence of known text, not something worth a model round-trip.
    """
    parts = [state.get("summary") or ""]
    status = state.get("resume_status")
    resume_id = state.get("resume_version_id")
    if status == "approved and saved" and resume_id:
        parts.append(
            "Saved as a new version — your original resume file is untouched. "
            "Apply the changes in your own document to keep its layout, or "
            "download this version as "
            f"[.docx](/api/job-helper/resumes/{resume_id}/download?fmt=docx) or "
            f"[.pdf](/api/job-helper/resumes/{resume_id}/download?fmt=pdf)."
        )
    elif status == "rejected by user":
        parts.append("I have not saved that rewrite — your resume is unchanged.")
    elif status:
        parts.append(f"The tailored resume {status}.")

    text = "\n\n".join(p for p in parts if p)
    return {"summary": text, "messages": [AIMessage(content=text)]}


def build_graph() -> StateGraph:
    graph = StateGraph(JobHelperState)
    graph.add_node("load_memory", load_memory)
    graph.add_node("agent", agent)
    graph.add_node("approve_resume", approve_resume)
    graph.add_node("finalize", finalize)

    graph.add_edge(START, "load_memory")
    graph.add_edge("load_memory", "agent")
    graph.add_conditional_edges("agent", route, ["approve_resume", "finalize"])
    graph.add_edge("approve_resume", "finalize")
    graph.add_edge("finalize", END)
    return graph


graph = build_graph()
