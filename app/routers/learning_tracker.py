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
from langgraph.types import Command

from app.dependencies import get_current_user
from app.database import get_db
from app.agents.approval_store import get_pending
from app.agents.memory_store import MEMORIES, get_profile, save_profile
from app.agents.learning_tracker.repository import (
    LEARNING_NS,
    fetch_quiz,
    fetch_roadmap,
    grade_quiz,
    learning_stats,
    list_roadmaps,
    record_quiz_attempt,
    resolve_roadmap_id,
    roadmap_progress,
    set_roadmap_status,
    set_topic_progress,
    write_memory,
)
from app.agents.learning_tracker.state import ProgressStatus, RoadmapStatus

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/learning",
    tags=["learning"],
    responses={404: {"description": "Not found"}},
)

# Intents worth extracting durable facts from. Read-only intents like
# query_roadmap are excluded — checking your roadmap reveals no new facts.
MEMORY_INTENTS = {
    "create_roadmap",
    "modify_roadmap",
    "explain",
    "find_resources",
}


class QueryRequest(BaseModel):
    text: str
    roadmapId: Optional[str] = None
    thread_id: Optional[str] = None


def _sse(event: dict) -> str:
    """Serialize one event as a Server-Sent Events frame."""
    return f"data: {json.dumps(event)}\n\n"


def _pause_status(payload: dict) -> str:
    """Which kind of pause the graph hit. Onboarding asks the learner a question
    and resumes via /onboarding; everything else is a proposal awaiting approval
    and resumes via /approvals."""
    return "needs_input" if payload.get("type") == "onboarding" else "needs_approval"


# Graph-state keys the client is allowed to see. A whitelist, not a blacklist:
# the state also carries `current_user` (uid, token_version,
# email_verify_code_hash) and the learner's whole memory profile, none of which
# belongs in a chat response. Returning the raw state leaked all of it.
_CLIENT_FIELDS = (
    "intent",
    "topic_explaination",
    "quiz",
    "quizId",
    "quiz_result",
    "suggestions",
    "next_topic",
    "progress",
    "log_status",
    "roadmap",
    "roadmapId",
    "roadmap_status",
    "pa_tasks_created",
)


def _result(values: dict) -> dict:
    """The turn's outcome, projected to what the client actually needs."""
    return {k: values.get(k) for k in _CLIENT_FIELDS}


def _turn(result: dict, thread_id: str) -> dict:
    """One response shape for every route that advances a turn — /query,
    /approvals, /onboarding.

    The pause check comes first because resuming can pause again immediately:
    answering onboarding runs straight on into the roadmap approval, and that
    second interrupt has to surface as a pause rather than be buried in the
    state dump the client would otherwise try to render.
    """
    interrupts = result.get("__interrupt__")
    if interrupts:
        payload = interrupts[0].value
        return {
            "status": _pause_status(payload),
            "thread_id": thread_id,
            "proposal": payload,
        }
    return {"status": "done", "thread_id": thread_id, "result": _result(result)}


@router.post("/query")
async def ask(
    body: QueryRequest,
    request: Request,
    background_tasks: BackgroundTasks,
    current_user: Annotated[dict, Depends(get_current_user)],
):
    agent = request.app.state.learning_agent

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
    logger.info("final intent=%s", result.get("intent"))

    # Fire-and-forget memory extraction after the response is sent — no added latency.
    if result.get("intent") in MEMORY_INTENTS:
        background_tasks.add_task(
            write_memory,
            current_user["uid"],
            body.text,
            result.get("memory", {}),
        )

    return _turn(result, thread_id)


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
    agent = request.app.state.learning_agent

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

            # A node hit an interrupt (roadmap approval, onboarding) — surface it
            # instead of a normal result, mirroring /query.
            interrupts = snapshot.interrupts if snapshot else None
            if snapshot and snapshot.next and interrupts:
                payload = interrupts[0].value
                yield _sse(
                    {
                        "type": _pause_status(payload),
                        "thread_id": thread_id,
                        "proposal": payload,
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

            yield _sse({"type": "done", "result": _result(values)})
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
    agent = request.app.state.learning_agent
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
    return _turn(result, body.thread_id)


class OnboardingAnswers(BaseModel):
    thread_id: str
    # Null / empty means the learner skipped. Either way onboarding is marked
    # done so the questions don't come back on their next message.
    answers: Optional[dict] = None


@router.post("/onboarding")
async def submit_onboarding(
    body: OnboardingAnswers,
    request: Request,
    current_user: Annotated[dict, Depends(get_current_user)],
):
    """Resume a run paused on the onboarding questions.

    Ownership is checked against the thread's own state rather than an approvals
    row — onboarding creates no approval, and the checkpointed `user_id` is the
    identity the graph actually ran under.
    """
    agent = request.app.state.learning_agent
    config = {"configurable": {"thread_id": body.thread_id}}

    snapshot = await agent.aget_state(config)
    if not snapshot or not snapshot.next:
        raise HTTPException(
            status_code=404, detail="Nothing is waiting for input on this thread."
        )
    if (snapshot.values or {}).get("user_id") != current_user["uid"]:
        raise HTTPException(
            status_code=403, detail="You do not have access to this thread."
        )

    # Resuming finishes the turn the learner originally sent, which on a first
    # run means going straight into roadmap generation — so this very often
    # comes back as another pause rather than a finished result.
    result = await agent.ainvoke(Command(resume=body.answers or {}), config=config)
    return _turn(result, body.thread_id)


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
    """Grade a quiz submitted from the UI. Shares `grade_quiz` with the chat path
    (quiz_grader_agent), so the same answers can't score differently."""
    user_id = current_user["uid"]
    quiz = await fetch_quiz(user_id, body.quizId)
    if not quiz:
        raise HTTPException(status_code=404, detail="Quiz not found.")

    questions = quiz.get("questions", [])
    result = grade_quiz(questions, {a.question: a.answer for a in body.answers})
    await record_quiz_attempt(
        user_id, body.quizId, quiz.get("roadmapId"), quiz.get("topicId"), result
    )
    return {"status": "done", "result": result}


@router.get("/current-state")
async def getCurrentState(current_user: Annotated[dict, Depends(get_current_user)]):
    """What the learner is working on right now: their active roadmap and how far
    along it is. The home screen's one call — no roadmapId needed."""
    user_id = current_user["uid"]
    roadmapId = await resolve_roadmap_id(user_id)
    roadmap = await fetch_roadmap(roadmapId, user_id)
    if not roadmap:
        return {
            "status": "done",
            "message": "What do you want to learn today?",
            "result": None,
        }

    return {
        "status": "done",
        "message": "roadmap fetched",
        "result": {"roadmap": roadmap, "progress": roadmap_progress(roadmap)},
    }


@router.get("/roadmaps")
async def getPlans(
    current_user: Annotated[dict, Depends(get_current_user)],
    status: Optional[RoadmapStatus] = None,
    limit: int = 20,
    skip: int = 0,
):
    """The learner's roadmaps, newest first. Paginated and status-filterable so a
    long-running account doesn't ship every roadmap in full on every load."""
    docs = await list_roadmaps(
        current_user["uid"], status=status, limit=limit, skip=skip
    )
    return {
        "status": "done",
        "message": "roadmaps fetched" if docs else "roadmaps not found",
        "result": docs,
    }


@router.get("/stats")
async def get_stats(current_user: Annotated[dict, Depends(get_current_user)]):
    """Progress across all of the learner's roadmaps — the landing screen's
    summary strip. Fetched in parallel with GET /roadmaps."""
    return {"status": "done", "result": await learning_stats(current_user["uid"])}


@router.get("/roadmaps/{roadmapId}")
async def getPlan(
    roadmapId: str, current_user: Annotated[dict, Depends(get_current_user)]
):
    roadmap = await fetch_roadmap(roadmapId, current_user["uid"])
    if not roadmap:
        raise HTTPException(status_code=404, detail="Roadmap not found.")
    return {
        "status": "done",
        "result": {"roadmap": roadmap, "progress": roadmap_progress(roadmap)},
    }


class RoadmapStatusUpdate(BaseModel):
    status: RoadmapStatus


@router.patch("/roadmaps/{roadmapId}")
async def updateRoadmapStatus(
    roadmapId: str,
    body: RoadmapStatusUpdate,
    current_user: Annotated[dict, Depends(get_current_user)],
):
    """Park, resume, or archive a roadmap. Which one is `active` decides what a
    bare "what should I study next?" resolves to."""
    updated = await set_roadmap_status(roadmapId, current_user["uid"], body.status)
    if not updated:
        raise HTTPException(status_code=404, detail="Roadmap not found.")
    return {"status": "done", "roadmapId": roadmapId, "roadmap_status": body.status}


@router.get("/memory")
async def get_memory(current_user: Annotated[dict, Depends(get_current_user)]):
    """Let the UI show the learner what the system remembers about them. Reads the
    same `learning` namespace the agent reads, so the screen can't disagree with
    what actually personalizes the roadmaps."""
    return {
        "status": "done",
        "result": await get_profile(current_user["uid"], LEARNING_NS),
    }


@router.get("/state", deprecated=True)
async def get_state(current_user: Annotated[dict, Depends(get_current_user)]):
    """Deprecated alias of GET /memory — kept for shipped clients. Use /memory."""
    return await get_memory(current_user)


class MemoryUpdate(BaseModel):
    data: dict


@router.put("/memory")
async def put_memory(
    body: MemoryUpdate, current_user: Annotated[dict, Depends(get_current_user)]
):
    """Explicit user-managed edits (e.g. a settings screen). Merges the given keys
    into the stored profile; keys not sent are left untouched."""
    await save_profile(current_user["uid"], body.data, LEARNING_NS)
    return {"status": "done"}


@router.delete("/memory")
async def delete_memory(current_user: Annotated[dict, Depends(get_current_user)]):
    """Clear what we remember about the learner (privacy / reset). Drops only the
    learning profile — the PA's and meal planner's namespaces are untouched."""
    try:
        await get_db()[MEMORIES].update_one(
            {"user_id": current_user["uid"]}, {"$unset": {LEARNING_NS: ""}}
        )
        return {"status": "done"}
    except Exception as e:
        logger.error("delete_memory error: %s", e)
        raise HTTPException(status_code=500, detail="Could not clear memory.")


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
    # `status` is the full vocabulary (in_progress, needs_review, skipped, …);
    # `covered` is the original boolean the shipped client sends. When both are
    # absent this marks the topic completed, as it always did.
    status: Optional[ProgressStatus] = None
    covered: bool = True
    mastery_score: Optional[int] = None


@router.post("/progress")
async def update_progress(
    body: ProgressUpdate, current_user: Annotated[dict, Depends(get_current_user)]
):
    """Directly set a topic's progress — the primary path (e.g. a checkbox in the
    UI, which already knows the topic id). No LLM involved."""
    status = body.status or ("completed" if body.covered else "not_started")
    updated = await set_topic_progress(
        body.roadmapId,
        body.topicId,
        status,
        current_user["uid"],
        mastery_score=body.mastery_score,
    )
    if not updated:
        raise HTTPException(status_code=404, detail="Roadmap or topic not found.")
    return {
        "status": "done",
        "topicId": body.topicId,
        "progress_status": status,
        "covered": status == "completed",
    }


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
