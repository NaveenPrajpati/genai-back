"""HTTP routes for the supervisor — one chat surface over all three agents.

The per-agent routers (/learning, /personal-assistant, /meal-planner) are
untouched and still work; this is an additional entry point, not a replacement.

Approvals from any skill surface here identically: the learning tracker's
roadmap review, the personal assistant's delete confirmation and the meal
planner's plan approval all pause the supervisor thread and resume through this
router's POST /approve.
"""

import json
import logging
import uuid
from typing import Annotated, Literal, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from langgraph.types import Command
from pydantic import BaseModel

from app.agents.approval_store import get_pending, list_pending, to_legacy
from app.agents.learning_tracker.repository import write_memory
from app.agents.meal_planner.state import (
    MEAL_MEMORY_INSTRUCTIONS,
    MealMemoryExtract,
)
from app.agents.memory_store import extract_and_save
from app.agents.personal_assistant.state import (
    PA_MEMORY_INSTRUCTIONS,
    PAMemoryExtract,
)
from app.dependencies import get_current_user
from app.mcp.client import MEAL_SERVER, load_meal_tools

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/assistant",
    tags=["assistant"],
    responses={404: {"description": "Not found"}},
)


class QueryRequest(BaseModel):
    text: str
    thread_id: Optional[str] = None
    # Optional deep links, forwarded to whichever skill needs them.
    roadmapId: Optional[str] = None
    plan_id: Optional[str] = None


class ApproveRequest(BaseModel):
    thread_id: str
    decision: Literal["approved", "rejected"]


def _sse(event: dict) -> str:
    """Serialize one event as a Server-Sent Events frame."""
    return f"data: {json.dumps(event, default=str)}\n\n"


def _jsonable(state: dict) -> dict:
    """Drop LangGraph internals and the raw message log before returning."""
    return {
        k: v for k, v in state.items() if not k.startswith("__") and k != "messages"
    }


def _inputs(body: QueryRequest, current_user: dict, thread_id: str) -> dict:
    excluded = {"_id", "expires_at", "password_hash"}
    return {
        "query": body.text,
        "user_id": current_user["uid"],
        "thread_id": thread_id,
        "current_user": {k: v for k, v in current_user.items() if k not in excluded},
        "roadmapId": body.roadmapId,
        "plan_id": body.plan_id,
    }


def _schedule_memory(
    background_tasks: BackgroundTasks,
    user_id: str,
    text: str,
    state: dict,
) -> None:
    """Learn durable facts after the response is sent, using the extractor of
    each skill that actually ran — so a meal turn pays for one extraction, not
    three, and a turn that used no skill pays for none."""
    memory = state.get("memory") or {}
    for skill in state.get("route") or []:
        if skill == "learning":
            background_tasks.add_task(write_memory, user_id, text, memory)
        elif skill == "assistant":
            background_tasks.add_task(
                extract_and_save,
                user_id,
                text,
                PAMemoryExtract,
                PA_MEMORY_INSTRUCTIONS,
                memory,
            )
        elif skill == "meal":
            background_tasks.add_task(
                extract_and_save,
                user_id,
                text,
                MealMemoryExtract,
                MEAL_MEMORY_INSTRUCTIONS,
                memory,
            )


@router.post("/query")
async def ask(
    body: QueryRequest,
    request: Request,
    background_tasks: BackgroundTasks,
    current_user: Annotated[dict, Depends(get_current_user)],
):
    agent = request.app.state.supervisor_agent
    thread_id = body.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    result = await agent.ainvoke(_inputs(body, current_user, thread_id), config=config)

    if "__interrupt__" in result:
        return {
            "status": "needs_approval",
            "thread_id": thread_id,
            "proposal": result["__interrupt__"][0].value,
        }

    _schedule_memory(background_tasks, current_user["uid"], body.text, result)
    return {"status": "done", "thread_id": thread_id, "result": _jsonable(result)}


@router.post("/query/stream")
async def ask_stream(
    body: QueryRequest,
    request: Request,
    background_tasks: BackgroundTasks,
    current_user: Annotated[dict, Depends(get_current_user)],
):
    """Streaming counterpart of /query.

    Two stream modes at once: `updates` drives a `step` event per node so the UI
    can show which skill is working, and `messages` streams the final reply token
    by token. Only finalize's tokens are forwarded — every other node uses
    structured output or tool calls, whose partial content would be JSON noise.
    """
    agent = request.app.state.supervisor_agent
    thread_id = body.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    inputs = _inputs(body, current_user, thread_id)

    async def generate():
        try:
            yield _sse({"type": "thread", "thread_id": thread_id})

            async for mode, chunk in agent.astream(
                inputs, config=config, stream_mode=["updates", "messages"]
            ):
                if mode == "updates":
                    for node in chunk:
                        if not node.startswith("__"):
                            yield _sse({"type": "step", "node": node})
                elif mode == "messages":
                    message, metadata = chunk
                    if (
                        metadata.get("langgraph_node") == "finalize"
                        and message.content
                    ):
                        yield _sse({"type": "token", "token": message.content})

            snapshot = await agent.aget_state(config)
            values = snapshot.values if snapshot else {}
            interrupts = snapshot.interrupts if snapshot else None
            if snapshot and snapshot.next and interrupts:
                yield _sse(
                    {
                        "type": "needs_approval",
                        "thread_id": thread_id,
                        "proposal": interrupts[0].value,
                    }
                )
                return

            _schedule_memory(background_tasks, current_user["uid"], body.text, values)
            yield _sse({"type": "done", "result": _jsonable(values)})
        except Exception as exc:
            logger.exception("supervisor stream failed")
            yield _sse({"type": "error", "message": str(exc)})

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/approve")
async def approve(
    body: ApproveRequest,
    request: Request,
    background_tasks: BackgroundTasks,
    current_user: Annotated[dict, Depends(get_current_user)],
):
    """Resolve whichever skill paused this thread and let the run continue.

    One endpoint covers all three because the pause is a property of the
    supervisor thread, not of the skill that raised it.
    """
    agent = request.app.state.supervisor_agent
    config = {"configurable": {"thread_id": body.thread_id}}

    try:
        approval = await get_pending(body.thread_id)
    except Exception as e:
        logger.error("supervisor approval lookup error: %s", e)
        approval = None

    if not approval:
        raise HTTPException(
            status_code=404, detail="No pending approval for this thread."
        )
    # Ownership check: without it, guessing a thread_id would let one user
    # approve or reject another user's pending action.
    if approval["user_id"] != current_user["uid"]:
        raise HTTPException(
            status_code=403, detail="You do not have access to this approval."
        )

    snapshot = await agent.aget_state(config)
    if not snapshot or not snapshot.next:
        raise HTTPException(
            status_code=404,
            detail="No paused action for this thread. The server may have "
            "restarted — please re-submit your request.",
        )

    result = await agent.ainvoke(Command(resume=body.decision), config=config)

    if "__interrupt__" in result:
        # A multi-skill turn can pause more than once (e.g. approve a roadmap,
        # then approve the meal plan that follows it).
        return {
            "status": "needs_approval",
            "thread_id": body.thread_id,
            "proposal": result["__interrupt__"][0].value,
        }

    query = (snapshot.values or {}).get("query", "")
    _schedule_memory(background_tasks, current_user["uid"], query, result)
    return {"status": "done", "result": _jsonable(result)}


@router.get("/approvals")
async def list_approvals(current_user: Annotated[dict, Depends(get_current_user)]):
    """Everything awaiting this user's decision, whichever skill raised it."""
    try:
        docs = await list_pending(
            current_user["uid"],
            [
                "save_roadmap",
                "update_roadmap",
                "pa_delete_task",
                "supervisor_save_meal_plan",
            ],
        )
        return {"status": "done", "result": [to_legacy(d) for d in docs]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/skills")
async def skills(current_user: Annotated[dict, Depends(get_current_user)]):
    """What the supervisor can currently do — including a live check of the MCP
    server, so a meal planner that is down is visible instead of silent."""
    tools = await load_meal_tools(current_user["uid"])
    return {
        "status": "done",
        "result": {
            "learning": {"transport": "langgraph_subgraph", "available": True},
            "assistant": {"transport": "langgraph_subgraph", "available": True},
            "meal": {
                "transport": "mcp",
                "server": MEAL_SERVER,
                "available": bool(tools),
                "tools": sorted(t.name for t in tools),
            },
        },
    }
