"""HTTP routes for the learning-tracker agent. Agent logic lives in app.agents.learning_tracker."""

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Optional, Literal, Annotated
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from fastapi import APIRouter, Request, HTTPException, Depends, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from bson import ObjectId
from langgraph.types import Command

from app.dependencies import get_current_user
from app.database import get_db
from app.agents.approval_store import get_pending
from app.agents.learning_tracker.repository import set_topic_covered, write_memory

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/learning",
    tags=["learning"],
    responses={404: {"description": "Not found"}},
)

MEMORY_INTENTS = {
    "create_roadmap",
    "modify_roadmap",
    "explain",
    "query_roadmap",
    "find_resources",
}


class QueryRequest(BaseModel):
    text: str
    roadmapId: Optional[str] = None
    thread_id: Optional[str] = None


def _sse(event: dict) -> str:
    """Serialize one event as a Server-Sent Events frame."""
    return f"data: {json.dumps(event)}\n\n"


@router.post("/query")
async def ask(
    body: QueryRequest,
    request: Request,
    background_tasks: BackgroundTasks,
    current_user: Annotated[dict, Depends(get_current_user)],
):
    agent = request.app.state.agent

    thread_id = body.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    _excluded = {"_id", "expires_at", "password_hash"}
    user_data = {k: v for k, v in current_user.items() if k not in _excluded}
    result = await agent.ainvoke(
        {
            "query": body.text,
            "user_id": current_user["uid"],
            "thread_id": thread_id,
            "roadmapId": body.roadmapId,
            "current_user": user_data,
        },
        config=config,
    )
    logger.info("final -- %s", result)

    # Fire-and-forget memory extraction after the response is sent — no added latency.
    if result.get("intent") in MEMORY_INTENTS:
        background_tasks.add_task(
            write_memory,
            current_user["uid"],
            body.text,
            result.get("memory", {}),
        )

    if "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        return {
            "status": "needs_approval",
            "thread_id": thread_id,  # app sends this back to /approve
            "proposal": payload,
        }

    return {"status": "done", "result": result}


@router.post("/query/stream")
async def ask_stream(
    body: QueryRequest,
    request: Request,
    background_tasks: BackgroundTasks,
    current_user: Annotated[dict, Depends(get_current_user)],
):
    """Streaming counterpart of /query. Streams the tutor agent's explanation
    token-by-token over SSE; for other intents (quiz, roadmap, …) no tokens are
    emitted and the final state arrives in the `done` event."""
    agent = request.app.state.agent

    thread_id = body.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    _excluded = {"_id", "expires_at", "password_hash"}
    user_data = {k: v for k, v in current_user.items() if k not in _excluded}
    inputs = {
        "query": body.text,
        "user_id": current_user["uid"],
        "thread_id": thread_id,
        "roadmapId": body.roadmapId,
        "current_user": user_data,
    }

    async def generate():
        try:
            yield _sse({"type": "thread", "thread_id": thread_id})

            # stream_mode="messages" yields (chunk, metadata) for every LLM token
            # across all nodes. We only forward text tokens from tutor_agent —
            # other nodes use structured output (empty .content) and would leak
            # tool-call JSON otherwise.
            async for chunk, metadata in agent.astream(
                inputs, config=config, stream_mode="messages"
            ):
                if metadata.get("langgraph_node") == "tutor_agent" and chunk.content:
                    yield _sse({"type": "token", "token": chunk.content})

            snapshot = await agent.aget_state(config)
            values = snapshot.values if snapshot else {}

            # A node hit an interrupt (e.g. roadmap approval) — surface it instead
            # of a normal result, mirroring /query.
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

            intent = values.get("intent")
            if intent in MEMORY_INTENTS:
                background_tasks.add_task(
                    write_memory,
                    current_user["uid"],
                    body.text,
                    values.get("memory", {}),
                )

            yield _sse(
                {
                    "type": "done",
                    "result": {
                        "intent": intent,
                        "topic_explaination": values.get("topic_explaination"),
                        "quiz": values.get("quiz"),
                        "quizId": values.get("quizId"),
                        "suggestions": values.get("suggestions"),
                        "next_topic": values.get("next_topic"),
                        "progress": values.get("progress"),
                    },
                }
            )
        except Exception as exc:
            logger.exception("learning stream failed")
            yield _sse({"type": "error", "message": str(exc)})

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


class ApproveRequest(BaseModel):
    thread_id: str
    decision: Literal["approved", "rejected"]


@router.post("/approvals")
async def approve(
    body: ApproveRequest,
    request: Request,
    current_user: Annotated[dict, Depends(get_current_user)],
):
    agent = request.app.state.agent
    config = {"configurable": {"thread_id": body.thread_id}}

    # The thread/approval must belong to the caller (prevents IDOR where a user
    # approves or rejects someone else's pending plan by guessing the thread_id).
    approval = None
    try:
        approval = await get_pending(body.thread_id)
        logger.info("approval found: %s", approval)
    except Exception as e:
        logger.error("approval ownership lookup error: %s", e)

    if not approval:
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
            detail="No pending approval for this thread. The server may have restarted — please re-submit your plan request.",
        )

    result = await agent.ainvoke(Command(resume=body.decision), config=config)
    return {"status": "done", "result": result}


class Answer(BaseModel):
    question: int
    answer: int


class SubmitQuiz(BaseModel):
    quizId: str
    answers: list[Answer]


@router.post("/submit-quiz")
async def submit_quiz(
    body: SubmitQuiz, current_user: Annotated[dict, Depends(get_current_user)]
):
    user_id = current_user["uid"]
    logger.info("--- %s", user_id)
    try:
        quiz = await get_db()["quizzes"].find_one(
            {"_id": ObjectId(body.quizId), "user_id": user_id}
        )
        if not quiz:
            raise HTTPException(status_code=404, detail="Quiz not found.")

        questions = quiz.get("questions", [])
        selected = {a.question: a.answer for a in body.answers}

        correct = 0
        review = []  # only the questions the user got wrong
        for idx, q in enumerate(questions):
            chosen = selected.get(idx)
            if chosen == q.get("answer"):
                correct += 1
            else:
                review.append(
                    {
                        "question": idx,
                        "selected": chosen,
                        "correctAnswer": q.get("answer"),
                        "correctOption": (
                            q.get("options", [])[q.get("answer")]
                            if q.get("answer") is not None
                            else None
                        ),
                    }
                )

        return {
            "status": "done",
            "result": {
                "total": len(questions),
                "correct": correct,
                "review": review,
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/roadmaps")
async def getPlans(current_user: Annotated[dict, Depends(get_current_user)]):
    user_id = current_user["uid"]
    logger.info("--- %s", user_id)
    try:
        cursor = get_db()["roadmaps"].find({"user_id": user_id})
        docs = await cursor.to_list(None)
        for doc in docs:
            doc["_id"] = str(doc["_id"])
        logger.info("approvals found: %s", len(docs))

        if not docs:
            return {"status": "done", "message": "roadmaps not found", "result": []}

        return {"status": "done", "message": "roadmaps fetched", "result": docs}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/memory")
async def get_memory(current_user: Annotated[dict, Depends(get_current_user)]):
    """Let the UI show the learner what the system remembers about them."""
    try:
        doc = await get_db()["memories"].find_one({"user_id": current_user["uid"]})
        return {"status": "done", "result": doc.get("data", {}) if doc else {}}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class MemoryUpdate(BaseModel):
    data: dict


@router.put("/memory")
async def put_memory(
    body: MemoryUpdate, current_user: Annotated[dict, Depends(get_current_user)]
):
    """Explicit user-managed edits (e.g. a settings screen). Merges the given keys
    into the stored profile; keys not sent are left untouched."""
    try:
        set_doc = {f"data.{k}": v for k, v in body.data.items()}
        set_doc["updatedAt"] = datetime.now(timezone.utc).isoformat()
        await get_db()["memories"].update_one(
            {"user_id": current_user["uid"]},
            {
                "$set": set_doc,
                "$setOnInsert": {"createdAt": datetime.now(timezone.utc).isoformat()},
            },
            upsert=True,
        )
        return {"status": "done"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/memory")
async def delete_memory(current_user: Annotated[dict, Depends(get_current_user)]):
    """Clear everything we remember about the learner (privacy / reset)."""
    try:
        await get_db()["memories"].delete_one({"user_id": current_user["uid"]})
        return {"status": "done"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/digests")
async def get_digests(
    current_user: Annotated[dict, Depends(get_current_user)],
    limit: int = 20,
):
    """Return the caller's daily learning digests, most recent first."""
    try:
        cursor = (
            get_db()["learning_digests"]
            .find({"user_id": current_user["uid"]})
            .sort("createdAt", -1)
            .limit(limit)
        )
        docs = await cursor.to_list(None)
        for doc in docs:
            doc["_id"] = str(doc["_id"])
        return {"status": "done", "result": docs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class ProgressUpdate(BaseModel):
    roadmapId: str
    topicId: str
    covered: bool = True


@router.post("/progress")
async def update_progress(
    body: ProgressUpdate, current_user: Annotated[dict, Depends(get_current_user)]
):
    """Directly set a topic's covered flag — the primary progress path (e.g. a
    checkbox in the UI, which already knows the topic id). No LLM involved."""
    try:
        updated = await set_topic_covered(
            body.roadmapId, body.topicId, body.covered, user_id=current_user["uid"]
        )
        if not updated:
            raise HTTPException(status_code=404, detail="Roadmap or topic not found.")
        return {
            "status": "done",
            "topicId": body.topicId,
            "covered": body.covered,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class Trigger(BaseModel):
    user_id: str
    action_type: str = "learning_digest"
    enabled: bool = True
    # Local hour-of-day (0-23) the digest should fire, interpreted in `timezone`.
    schedule_hour: int = 9
    # IANA timezone name (e.g. "Asia/Kolkata"). The hourly sweep converts to this
    # to decide whether it's the user's chosen hour right now.
    timezone: str = "UTC"
    createdAt: Optional[str] = None
    updatedAt: Optional[str] = None
    last_run_at: Optional[str] = None


class TriggerSettings(BaseModel):
    """Partial update for a user's trigger. Only provided fields are changed."""

    action_type: str = "learning_digest"
    enabled: Optional[bool] = None
    schedule_hour: Optional[int] = None
    timezone: Optional[str] = None


@router.get("/triggers")
async def get_triggers(current_user: Annotated[dict, Depends(get_current_user)]):
    """Return the caller's trigger settings for rendering toggles in settings. A
    user who has never opted in has no row, so the list may be empty — the UI
    should treat a missing trigger as disabled."""
    try:
        cursor = get_db()["triggers"].find({"user_id": current_user["uid"]})
        docs = await cursor.to_list(None)
        for doc in docs:
            doc["_id"] = str(doc["_id"])
        return {"status": "done", "result": docs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/toggle-trigger")
async def toggle_trigger(current_user: Annotated[dict, Depends(get_current_user)]):
    """Opt in/out of the daily learning digest. The first call creates an enabled
    entry; each subsequent call flips it on/off. run_triggers only generates a digest
    for users whose entry is enabled."""
    try:
        user_id = current_user["uid"]
        col = get_db()["triggers"]
        existing = await col.find_one(
            {"user_id": user_id, "action_type": "learning_digest"}
        )
        if existing:
            enabled = not existing.get("enabled", True)
            await col.update_one(
                {"_id": existing["_id"]},
                {
                    "$set": {
                        "enabled": enabled,
                        "updatedAt": datetime.now(timezone.utc).isoformat(),
                    }
                },
            )
        else:
            enabled = True
            await col.insert_one(
                {
                    "user_id": user_id,
                    "action_type": "learning_digest",
                    "enabled": True,
                    "schedule_hour": 9,
                    "timezone": "UTC",
                    "createdAt": datetime.now(timezone.utc).isoformat(),
                }
            )
        return {"status": "done", "enabled": enabled}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/trigger-settings")
async def update_trigger_settings(
    body: TriggerSettings,
    current_user: Annotated[dict, Depends(get_current_user)],
):
    """Update a trigger's delivery settings (enabled / schedule_hour / timezone).
    Creates the row if the user has never opted in, so the settings screen can
    save without a prior toggle."""
    try:
        update: dict = {}
        if body.enabled is not None:
            update["enabled"] = body.enabled
        if body.schedule_hour is not None:
            if not 0 <= body.schedule_hour <= 23:
                raise HTTPException(
                    status_code=422, detail="schedule_hour must be 0-23."
                )
            update["schedule_hour"] = body.schedule_hour
        if body.timezone is not None:
            try:
                ZoneInfo(body.timezone)
            except ZoneInfoNotFoundError:
                raise HTTPException(
                    status_code=422, detail=f"Unknown timezone: {body.timezone}"
                )
            update["timezone"] = body.timezone
        if not update:
            raise HTTPException(status_code=422, detail="No settings provided.")

        update["updatedAt"] = datetime.now(timezone.utc).isoformat()
        result = await get_db()["triggers"].update_one(
            {"user_id": current_user["uid"], "action_type": body.action_type},
            {
                "$set": update,
                "$setOnInsert": {
                    "user_id": current_user["uid"],
                    "action_type": body.action_type,
                    "createdAt": datetime.now(timezone.utc).isoformat(),
                },
            },
            upsert=True,
        )
        return {"status": "done", "matched": result.matched_count, **update}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
