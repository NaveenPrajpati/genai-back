"""HTTP routes for the personal-assistant agent. Agent logic lives in app.agents.personal_assistant."""

import logging
import uuid
from typing import Optional, Literal, Annotated

from fastapi import APIRouter, Request, HTTPException, Depends
from pydantic import BaseModel
from langgraph.types import Command

from app.core.config import supabase
from app.database import get_db
from app.dependencies import get_current_user
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

    if "__interrupt__" in result:
        return {
            "status": "needs_approval",
            "thread_id": thread_id,
            "proposal": result["__interrupt__"][0].value,
        }
    return {"status": "done", "result": _jsonable(result)}


@router.post("/approve")
async def approve(
    body: ApproveRequest,
    request: Request,
    current_user: Annotated[dict, Depends(get_current_user)],
):
    agent = request.app.state.pa_agent
    config = {"configurable": {"thread_id": body.thread_id}}

    try:
        approval = (
            supabase.table("approvals")
            .select("id, user_id")
            .eq("thread_id", body.thread_id)
            .eq("action_type", "pa_delete_task")
            .eq("status", "pending")
            .maybe_single()
            .execute()
        )
    except Exception as e:
        logger.error("pa approval ownership lookup error: %s", e)
        approval = None

    if not approval or not approval.data:
        raise HTTPException(
            status_code=404, detail="No pending approval for this thread."
        )
    if approval.data["user_id"] != current_user["uid"]:
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
        result = (
            supabase.table("approvals")
            .select("*")
            .eq("user_id", current_user["uid"])
            .in_("action_type", ["pa_delete_task", "pa_digest"])
            .eq("status", "pending")
            .execute()
        )
        return {"status": "done", "result": result.data or []}
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
    user_id = current_user["uid"]
    try:
        existing = (
            supabase.table("triggers")
            .select("*")
            .eq("user_id", user_id)
            .eq("action_type", "pa_digest")
            .execute()
        )
        new_enabled = True
        if existing and existing.data:
            new_enabled = not existing.data[0]["enabled"]
            for t in existing.data:
                supabase.table("triggers").update({"enabled": not t["enabled"]}).eq(
                    "id", t["id"]
                ).execute()
        else:
            supabase.table("triggers").insert(
                {
                    "user_id": user_id,
                    "name": "daily task digest",
                    "schedule": "0 8 * * *",
                    "action_type": "pa_digest",
                    "enabled": True,
                    "last_run_at": None,
                }
            ).execute()
        return {"status": "done", "enabled": new_enabled}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/memory")
async def get_memory(current_user: Annotated[dict, Depends(get_current_user)]):
    try:
        rows = (
            supabase.table("memory")
            .select("key, value")
            .eq("user_id", current_user["uid"])
            .execute()
        )
        return {"status": "done", "result": rows.data or []}
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
