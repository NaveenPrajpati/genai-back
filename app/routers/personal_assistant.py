"""HTTP routes for the personal-assistant agent. Agent logic lives in app.agents.personal_assistant."""

import json
import logging
import uuid
from typing import Optional, Literal, Annotated

from fastapi import APIRouter, Request, HTTPException, Depends, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langgraph.types import Command

from app.database import get_db
from app.dependencies import get_current_user
from app.agents.trigger_store import toggle
from app.agents.approval_store import get_pending, list_pending, to_legacy
from app.agents.memory_store import extract_and_save, get_profile
from app.agents.personal_assistant.state import PAMemoryExtract, PA_MEMORY_INSTRUCTIONS
from app.agents.personal_assistant.repository import (
    TODOS,
    fetch_todos,
    get_todo_by_id,
    update_todo_by_id,
    delete_todo_by_id,
    categorize_agenda,
    fetch_notes,
    add_note,
    fetch_subtasks,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/personal-assistant",
    tags=["personal-assistant"],
    responses={404: {"description": "Not found"}},
)


class QueryRequest(BaseModel):
    text: str
    thread_id: Optional[str] = None


class ApproveRequest(BaseModel):
    thread_id: str
    decision: Literal["approved", "rejected"]


class TaskUpdateRequest(BaseModel):
    title: Optional[str] = None
    details: Optional[str] = None
    priority: Optional[Literal["low", "medium", "high"]] = None
    due_at: Optional[str] = None
    status: Optional[Literal["pending", "done"]] = None


class NoteRequest(BaseModel):
    content: str
    category: Optional[str] = None


def _jsonable(result: dict) -> dict:
    """Drop non-serializable LangGraph internals before returning to clients."""
    return {k: v for k, v in result.items() if not k.startswith("__")}


@router.post("/query")
async def ask(
    body: QueryRequest,
    request: Request,
    background_tasks: BackgroundTasks,
    current_user: Annotated[dict, Depends(get_current_user)],
):
    agent = request.app.state.pa_agent
    thread_id = body.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    _excluded = {"_id", "expires_at", "password_hash"}
    user_data = {k: v for k, v in current_user.items() if k not in _excluded}

    result = await agent.ainvoke(
        {
            "query": body.text,
            "user_id": current_user["uid"],
            "thread_id": thread_id,
            "current_user": user_data,
        },
        config=config,
    )

    # Fire-and-forget: learn durable personal facts after the response is sent.
    background_tasks.add_task(
        extract_and_save,
        current_user["uid"],
        body.text,
        PAMemoryExtract,
        PA_MEMORY_INSTRUCTIONS,
        result.get("memory"),
    )

    if "__interrupt__" in result:
        return {
            "status": "needs_approval",
            "thread_id": thread_id,
            "proposal": result["__interrupt__"][0].value,
        }
    return {"status": "done", "result": _jsonable(result)}


def _sse(event: dict) -> str:
    """Serialize one event as a Server-Sent Events frame."""
    return f"data: {json.dumps(event, default=str)}\n\n"


@router.post("/query/stream")
async def ask_stream(
    body: QueryRequest,
    request: Request,
    background_tasks: BackgroundTasks,
    current_user: Annotated[dict, Depends(get_current_user)],
):
    """Streaming counterpart of /query. Nodes use structured output, so there are
    no text tokens to stream; we emit a `step` event as each graph node completes
    (classify → todo/research/notes → synthesize) for live progress, then the
    final state in `done` (or `needs_approval` for a delete that awaits review)."""
    agent = request.app.state.pa_agent
    thread_id = body.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    # Fire-and-forget: learn durable personal facts after the stream completes.
    background_tasks.add_task(
        extract_and_save,
        current_user["uid"],
        body.text,
        PAMemoryExtract,
        PA_MEMORY_INSTRUCTIONS,
        None,
    )
    _excluded = {"_id", "expires_at", "password_hash"}
    user_data = {k: v for k, v in current_user.items() if k not in _excluded}
    inputs = {
        "query": body.text,
        "user_id": current_user["uid"],
        "thread_id": thread_id,
        "current_user": user_data,
    }

    async def generate():
        try:
            yield _sse({"type": "thread", "thread_id": thread_id})

            async for update in agent.astream(
                inputs, config=config, stream_mode="updates"
            ):
                for node in update:
                    yield _sse({"type": "step", "node": node})

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

            yield _sse({"type": "done", "result": _jsonable(values)})
        except Exception as exc:
            logger.exception("pa stream failed")
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
    current_user: Annotated[dict, Depends(get_current_user)],
):
    agent = request.app.state.pa_agent
    config = {"configurable": {"thread_id": body.thread_id}}

    try:
        approval = await get_pending(body.thread_id)
    except Exception as e:
        logger.error("pa approval ownership lookup error: %s", e)
        approval = None

    # Only a paused delete is resumable here; pa_digest rows are notifications.
    if not approval or approval.get("action_type") != "pa_delete_task":
        raise HTTPException(
            status_code=404, detail="No pending approval for this thread."
        )
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
    return {"status": "done", "result": _jsonable(result)}


@router.get("/approve")
async def list_approvals(current_user: Annotated[dict, Depends(get_current_user)]):
    try:
        docs = await list_pending(current_user["uid"], ["pa_delete_task", "pa_digest"])
        return {"status": "done", "result": [to_legacy(d) for d in docs]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tasks/stats")
async def task_stats(current_user: Annotated[dict, Depends(get_current_user)]):
    pipeline = [
        {"$match": {"user_id": current_user["uid"]}},
        {
            "$group": {
                "_id": {"status": "$status", "priority": "$priority"},
                "count": {"$sum": 1},
            }
        },
    ]
    docs = await get_db()[TODOS].aggregate(pipeline).to_list(length=100)
    stats: dict = {"total": 0, "by_status": {}, "by_priority": {}}
    for doc in docs:
        s = doc["_id"]["status"]
        p = doc["_id"].get("priority") or "none"
        n = doc["count"]
        stats["total"] += n
        stats["by_status"][s] = stats["by_status"].get(s, 0) + n
        stats["by_priority"][p] = stats["by_priority"].get(p, 0) + n
    return {"status": "done", "result": stats}


@router.get("/tasks/agenda")
async def task_agenda(current_user: Annotated[dict, Depends(get_current_user)]):
    """Pending tasks bucketed by due date: overdue / today / upcoming / no_date."""
    todos = await fetch_todos(current_user["uid"], status="pending")
    agenda = categorize_agenda(todos)
    return {
        "status": "done",
        "result": {
            "counts": {k: len(v) for k, v in agenda.items()},
            "buckets": agenda,
        },
    }


@router.get("/tasks")
async def list_tasks(
    current_user: Annotated[dict, Depends(get_current_user)],
    status: Optional[str] = None,
    priority: Optional[str] = None,
):
    todos = await fetch_todos(current_user["uid"], status=status, priority=priority)
    return {"status": "done", "result": todos}


@router.get("/tasks/{task_id}")
async def get_task(
    task_id: str,
    current_user: Annotated[dict, Depends(get_current_user)],
):
    todo = await get_todo_by_id(current_user["uid"], task_id)
    if not todo:
        raise HTTPException(status_code=404, detail="Task not found.")
    return {"status": "done", "result": todo}


@router.get("/tasks/{task_id}/subtasks")
async def list_subtasks(
    task_id: str,
    current_user: Annotated[dict, Depends(get_current_user)],
):
    subtasks = await fetch_subtasks(current_user["uid"], task_id)
    return {"status": "done", "result": subtasks}


@router.put("/tasks/{task_id}")
async def update_task(
    task_id: str,
    body: TaskUpdateRequest,
    current_user: Annotated[dict, Depends(get_current_user)],
):
    updates = body.model_dump(exclude_none=True)
    if not updates:
        raise HTTPException(status_code=400, detail="No fields to update.")
    updated = await update_todo_by_id(current_user["uid"], task_id, updates)
    if not updated:
        raise HTTPException(status_code=404, detail="Task not found.")
    return {"status": "done", "result": updated}


@router.delete("/tasks/{task_id}")
async def delete_task(
    task_id: str,
    current_user: Annotated[dict, Depends(get_current_user)],
):
    deleted = await delete_todo_by_id(current_user["uid"], task_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Task not found.")
    return {"status": "done"}


@router.post("/toggle-trigger")
async def toggle_trigger(current_user: Annotated[dict, Depends(get_current_user)]):
    try:
        # Default cadence: every day at 08:00 in the user's timezone.
        new_enabled = await toggle(
            current_user["uid"],
            "pa_digest",
            defaults={"name": "daily task digest", "schedule_hour": 8},
        )
        return {"status": "done", "enabled": new_enabled}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/memory")
async def get_memory(current_user: Annotated[dict, Depends(get_current_user)]):
    try:
        profile = await get_profile(current_user["uid"])
        result = [{"key": k, "value": v} for k, v in profile.items()]
        return {"status": "done", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/notes")
async def list_notes(current_user: Annotated[dict, Depends(get_current_user)]):
    notes = await fetch_notes(current_user["uid"])
    return {"status": "done", "result": notes}


@router.post("/notes")
async def create_note(
    body: NoteRequest,
    current_user: Annotated[dict, Depends(get_current_user)],
):
    note = await add_note(current_user["uid"], body.content, body.category)
    return {"status": "done", "result": note}
