"""HTTP routes for the personal-assistant agent. Agent logic lives in app.agents.personal_assistant."""

import json
import logging
import uuid
from typing import Optional, Literal, Annotated

from fastapi import APIRouter, Request, HTTPException, Depends, BackgroundTasks, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from langgraph.types import Command

from app.database import get_db
from app.dependencies import get_current_user
from app.agents.trigger_store import (
    get_trigger,
    next_run_at,
    schedule_dows,
    toggle,
    update_trigger,
)
from app.agents.approval_store import get_pending, list_pending, to_legacy
from app.agents.memory_store import extract_and_save, forget_profile_key, get_profile
from app.agents.personal_assistant.state import PAMemoryExtract, PA_MEMORY_INSTRUCTIONS
from app.agents.personal_assistant.repository import (
    NOTES_KEY,
    SOURCE_SELF,
    TODOS,
    TODO_SORT_FIELDS,
    bulk_complete,
    delete_completed,
    delete_todos_by_ids,
    fetch_overdue_ids,
    fetch_todos,
    fetch_todos_page,
    get_todo_by_id,
    insert_todo,
    update_todo_by_id,
    delete_todo_by_id,
    categorize_agenda,
    delete_note,
    fetch_notes,
    add_note,
    update_note,
    fetch_subtasks,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/personal-assistant",
    tags=["personal-assistant"],
    responses={404: {"description": "Not found"}},
)

# Read-only intents reveal no durable facts (the user is just checking existing
# data), so we skip the memory-extraction call for them.
READONLY_INTENTS = {"list", "agenda", "recall"}


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


class TaskCreateRequest(BaseModel):
    """A task the user typed in themselves, rather than described to the agent."""

    title: str = Field(min_length=1, max_length=500)
    details: Optional[str] = None
    priority: Literal["low", "medium", "high"] = "medium"
    due_at: Optional[str] = None
    recurrence: Optional[Literal["daily", "weekly", "monthly"]] = None
    # Set when adding a step to an existing goal.
    parent_id: Optional[str] = None


class BulkTaskRequest(BaseModel):
    """Act on many tasks at once.

    Either name the ids explicitly, or name a `scope` the server resolves. The
    scope exists because the useful sweeps — everything overdue, everything
    finished — are defined by a query the client would otherwise have to
    reproduce, and would get subtly wrong: "overdue" is bucketed on the date
    portion only, so a task due later today is not overdue.
    """

    action: Literal["complete", "delete"]
    ids: Optional[list[str]] = None
    scope: Optional[Literal["overdue", "completed"]] = None


class NoteRequest(BaseModel):
    content: str
    category: Optional[str] = None


class NoteUpdateRequest(BaseModel):
    """Both fields optional: a note can be recategorised without retyping it."""

    content: Optional[str] = Field(default=None, min_length=1, max_length=2000)
    category: Optional[str] = None


class TriggerUpdateRequest(BaseModel):
    """Delivery settings for the daily digest."""

    enabled: Optional[bool] = None
    # The scheduler sweeps hourly and fires on the local hour, so this is the
    # full resolution available — there is no minute to set.
    schedule_hour: Optional[int] = Field(default=None, ge=0, le=23)
    timezone: Optional[str] = None
    # Which weekdays the digest may arrive on, Mon=0 .. Sun=6. A list rather
    # than the single day the stored field started as, because the case worth
    # having for a work digest is "weekdays only" — [0,1,2,3,4], five days, not
    # one. An explicit null goes back to every day.
    schedule_dow: Optional[list[Annotated[int, Field(ge=0, le=6)]]] = None


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
    if result.get("intent") not in READONLY_INTENTS:
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

            # Fire-and-forget extraction, now that the resolved intent is known.
            if values.get("intent") not in READONLY_INTENTS:
                background_tasks.add_task(
                    extract_and_save,
                    current_user["uid"],
                    body.text,
                    PAMemoryExtract,
                    PA_MEMORY_INSTRUCTIONS,
                    values.get("memory"),
                )

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
    """Counts by status, priority and originating agent.

    `by_source` is what a client builds the source filter from — it cannot offer
    "Learning" as a chip without knowing there are 31 of them, and the whole
    point of the filter is that those 31 are otherwise indistinguishable from
    the four tasks the user typed. Tasks with no originating agent are counted
    under `self`, matching the value `GET /tasks?source=` takes.
    """
    pipeline = [
        {"$match": {"user_id": current_user["uid"]}},
        {
            "$group": {
                "_id": {
                    "status": "$status",
                    "priority": "$priority",
                    "source": "$source",
                },
                "count": {"$sum": 1},
            }
        },
    ]
    docs = await get_db()[TODOS].aggregate(pipeline).to_list(length=100)
    stats: dict = {"total": 0, "by_status": {}, "by_priority": {}, "by_source": {}}
    for doc in docs:
        s = doc["_id"]["status"]
        p = doc["_id"].get("priority") or "none"
        src = doc["_id"].get("source") or SOURCE_SELF
        n = doc["count"]
        stats["total"] += n
        stats["by_status"][s] = stats["by_status"].get(s, 0) + n
        stats["by_priority"][p] = stats["by_priority"].get(p, 0) + n
        stats["by_source"][src] = stats["by_source"].get(src, 0) + n
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
    q: Optional[str] = Query(
        default=None,
        max_length=200,
        description="Case-insensitive substring match on title and details.",
    ),
    source: Optional[str] = Query(
        default=None,
        max_length=64,
        description=(
            "Originating agent (e.g. 'learning_tracker', 'job_helper'), or "
            "'self' for tasks the user created here. Omit for all of them."
        ),
    ),
    sort: Literal[
        "created_at", "updated_at", "due_at", "priority", "title"
    ] = "created_at",
    order: Optional[Literal["asc", "desc"]] = None,
    limit: int = Query(default=500, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
):
    """One page of the user's tasks, newest first unless told otherwise.

    Searching and sorting run here rather than in the client because the client
    only ever holds the rows one fetch returned. Its search box filtered that
    slice — so at 600 tasks it silently stopped finding the older 100 — and
    "sort by due date" reordered the slice rather than the list. Both look like
    working features right up until the list outgrows a single fetch, which the
    bulk sweeps make more likely rather than less.

    `limit` defaults to the cap this endpoint has always had, so a client that
    passes nothing sees exactly what it saw before. `page.has_more` is how it
    finds out there was more — which, at 500, it previously could not.

    `source` is the other half of that. Every cross-agent task has carried its
    originating agent since `service.create_tasks` shipped, and nothing has ever
    filtered on it — so a roadmap's worth of "Learn:" tasks arrives in the same
    undifferentiated list as the four things the user typed themselves, and
    dominates it. `GET /tasks/stats` returns the counts to build the filter from.
    """
    page = await fetch_todos_page(
        current_user["uid"],
        status=status,
        priority=priority,
        q=q,
        source=source,
        sort=sort,
        order=order,
        limit=limit,
        offset=offset,
    )
    return {
        "status": "done",
        # `result` stays the plain list it has always been and the pagination
        # sits beside it rather than wrapping it, so the deployed client keeps
        # rendering while it catches up.
        "result": page["items"],
        "page": {
            "total": page["total"],
            "limit": limit,
            "offset": offset,
            "has_more": offset + len(page["items"]) < page["total"],
            "sort": sort,
            "order": order or TODO_SORT_FIELDS[sort],
        },
    }


@router.post("/tasks", status_code=201)
async def create_task(
    body: TaskCreateRequest,
    current_user: Annotated[dict, Depends(get_current_user)],
):
    """Create a task directly, with no agent turn.

    Every task previously had to go through `/query`: the user described what
    they wanted, an LLM classified the intent, a second LLM call extracted the
    fields, and a task came out the other side. That is the right path for "remind
    me to call the dentist on Friday" and an absurd one for someone who already
    knows the four fields and just wants to type them — it costs two model calls,
    a second or two, and a non-zero chance of the extractor mishearing a title
    the user had already written down correctly.
    """
    task = await insert_todo(current_user["uid"], body.model_dump(exclude_none=True))
    return {"status": "done", "result": task}


@router.post("/tasks/bulk")
async def bulk_tasks(
    body: BulkTaskRequest,
    current_user: Annotated[dict, Depends(get_current_user)],
):
    """Complete or delete many tasks in one request.

    Clearing out a finished week previously meant one HTTP round trip per task,
    fired off a list screen with no way to report which of them failed.

    Note the asymmetry with `/query`: deleting through the agent is gated behind
    a human-in-the-loop interrupt because *the agent* chose the targets from a
    fuzzy title match and could have chosen wrong. Here the caller names them,
    so the confirmation belongs in the UI that built the list rather than in a
    second server round trip.
    """
    user_id = current_user["uid"]

    if body.scope == "overdue":
        if body.action != "complete":
            raise HTTPException(
                status_code=400,
                detail="The 'overdue' scope only supports completing tasks.",
            )
        ids = await fetch_overdue_ids(user_id)
    elif body.scope == "completed":
        if body.action != "delete":
            raise HTTPException(
                status_code=400,
                detail="The 'completed' scope only supports deleting tasks.",
            )
        # Handled as a single query rather than by resolving ids first: nothing
        # here needs per-task treatment, and the set could be large.
        deleted = await delete_completed(user_id)
        return {"status": "done", "result": {"action": "delete", "deleted": deleted}}
    else:
        ids = body.ids or []

    if not ids:
        return {
            "status": "done",
            "result": {"action": body.action, "completed": 0, "deleted": 0, "tasks": []},
        }

    if body.action == "complete":
        outcome = await bulk_complete(user_id, ids)
        return {"status": "done", "result": {"action": "complete", **outcome}}

    deleted = await delete_todos_by_ids(user_id, ids)
    return {"status": "done", "result": {"action": "delete", "deleted": deleted}}


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


@router.get("/trigger")
async def get_trigger_state(current_user: Annotated[dict, Depends(get_current_user)]):
    """The current state of this user's daily digest.

    `POST /toggle-trigger` was write-only, so a client had nothing to render the
    switch from and had to assume "off" on every launch — which silently
    inverted the control for anyone who had already turned the digest on: their
    next tap disabled the thing they were trying to keep.

    A user who never opted in has no row, which is reported as disabled at the
    default hour rather than as an error — that is what they will get if they
    switch it on.
    """
    try:
        trig = await get_trigger(current_user["uid"], "pa_digest")
        return {
            "status": "done",
            "result": {
                "enabled": bool(trig.get("enabled")) if trig else False,
                "schedule_hour": (trig or {}).get("schedule_hour", 8),
                "timezone": (trig or {}).get("timezone", "UTC"),
                # Always a list here even for a row that stored a bare int, so
                # a client has one shape to render. None means every day.
                "schedule_dow": schedule_dows(trig) if trig else None,
                # None when off — there is no next run to count down to.
                "next_run_at": next_run_at(trig) if trig else None,
                "last_run_at": (trig or {}).get("last_run_at"),
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/trigger")
async def set_trigger_state(
    body: TriggerUpdateRequest,
    current_user: Annotated[dict, Depends(get_current_user)],
):
    """Set when — and whether — the daily digest arrives.

    The hour was hard-coded to 08:00 at the one place a trigger row could be
    created, even though the scheduler has always read `schedule_hour` and
    `timezone` per user and fired at each user's own local time. The capability
    was there; nothing exposed it.

    Setting the timezone matters as much as the hour: a row defaults to UTC, so
    a user who never set one has been getting their "8am" digest at 8am UTC.

    `schedule_dow` is the same story one level down. `is_due` and `next_run_at`
    have always honoured it and nothing has ever set it, so every digest has
    been daily — including Sundays, for what is mostly a work digest.
    """
    updates = body.model_dump(exclude_none=True)
    # An explicit `"schedule_dow": null` clears the day restriction. Every other
    # field here relies on omission meaning "leave alone", so exclude_none can't
    # tell "back to every day" from "not mentioned"; what the request actually
    # carried can.
    clearing_dow = "schedule_dow" in body.model_fields_set and body.schedule_dow is None
    if not updates and not clearing_dow:
        raise HTTPException(status_code=400, detail="No fields to update.")
    if body.schedule_dow == []:
        # An empty list would store "fires on no day at all". Silently reading
        # that as "every day" is the kind of guess that turns a digest off
        # without telling anyone.
        raise HTTPException(
            status_code=400,
            detail="schedule_dow must name at least one day (Mon=0 .. Sun=6), "
            "or be null for every day.",
        )
    if "schedule_dow" in updates:
        # Deduped and ordered, so two requests naming the same days store the
        # same thing.
        updates["schedule_dow"] = sorted(set(updates["schedule_dow"]))
    try:
        trig = await update_trigger(
            current_user["uid"],
            "pa_digest",
            updates,
            defaults={"name": "daily task digest", "schedule_hour": 8},
            unset=["schedule_dow"] if clearing_dow else None,
        )
        return {
            "status": "done",
            "result": {
                "enabled": bool(trig.get("enabled")),
                "schedule_hour": trig.get("schedule_hour", 8),
                "timezone": trig.get("timezone", "UTC"),
                "schedule_dow": schedule_dows(trig),
                "next_run_at": next_run_at(trig),
                "last_run_at": trig.get("last_run_at"),
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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


@router.delete("/memory/{key}")
async def forget_memory(
    key: str,
    current_user: Annotated[dict, Depends(get_current_user)],
):
    """Forget one learned fact.

    "What I know" was read-only, and the facts on it are a model's reading of
    prose — so they are sometimes wrong, and being wrong is not harmless. The
    profile is merged into the state behind every reply, which means a mis-heard
    `work_hours` or a priority the user dropped months ago keeps steering
    answers with no way to stop it. Exactly the gap notes had before they became
    editable, on the screen next to it.
    """
    if key == NOTES_KEY:
        # `data.pa_notes` is the notes list sharing the same document, not a
        # learned fact. Clearing it here would forget every note at once, from a
        # screen whose other rows delete a single value.
        raise HTTPException(
            status_code=400,
            detail="Notes are deleted one at a time via DELETE /notes/{note_id}.",
        )
    if "." in key or key.startswith("$"):
        # The key becomes a field path in the $unset. A dotted one would reach
        # into a nested field this endpoint has no business touching.
        raise HTTPException(status_code=400, detail="Invalid memory key.")
    try:
        forgotten = await forget_profile_key(current_user["uid"], key)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    if not forgotten:
        raise HTTPException(status_code=404, detail="No such memory.")
    return {"status": "done"}


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


@router.put("/notes/{note_id}")
async def edit_note(
    note_id: str,
    body: NoteUpdateRequest,
    current_user: Annotated[dict, Depends(get_current_user)],
):
    """Correct a remembered fact.

    Notes feed straight into the prompt that writes every reply, so a fact the
    extractor misheard keeps being applied. Until now they were append-only:
    the only available fix was to add a contradicting note beside the wrong one
    and hope the model preferred the newer.
    """
    if body.content is None and body.category is None:
        raise HTTPException(status_code=400, detail="No fields to update.")
    note = await update_note(
        current_user["uid"], note_id, content=body.content, category=body.category
    )
    if not note:
        raise HTTPException(status_code=404, detail="Note not found.")
    return {"status": "done", "result": note}


@router.delete("/notes/{note_id}")
async def remove_note(
    note_id: str,
    current_user: Annotated[dict, Depends(get_current_user)],
):
    """Forget a fact. No approval gate: the user named this one themselves."""
    if not await delete_note(current_user["uid"], note_id):
        raise HTTPException(status_code=404, detail="Note not found.")
    return {"status": "done"}
