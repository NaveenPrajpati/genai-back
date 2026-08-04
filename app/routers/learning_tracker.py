"""HTTP routes for the learning-tracker agent. Agent logic lives in app.agents.learning_tracker."""

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Optional, Literal, Annotated
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from fastapi import APIRouter, Request, HTTPException, Depends, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from langgraph.types import Command

from app.dependencies import get_current_user
from app.database import get_db
from app.agents.approval_store import get_pending
from app.agents.memory_store import MEMORIES, get_profile, save_profile
from app.core.config import CHECKPOINT_PASS_SCORE, DIGEST_QUIZ_PASS_SCORE
from app.agents.learning_tracker.service import build_checkpoint
from app.agents.learning_tracker.triggers import build_digest
from app.agents.learning_tracker.repository import (
    LEARNING_NS,
    ActiveRoadmapLimit,
    apply_checkpoint,
    completion_forecast,
    create_note,
    delete_note,
    digest_quiz_gate,
    due_reviews,
    fetch_digest,
    fetch_quiz,
    fetch_roadmap,
    find_topic,
    grade_quiz,
    in_progress_topic,
    learning_focus,
    learning_stats,
    list_digests,
    list_notes,
    list_roadmaps,
    mark_digest,
    note_counts,
    profile_drift,
    profile_snapshot,
    record_quiz_attempt,
    update_note,
    resolve_roadmap_id,
    roadmap_progress,
    set_roadmap_status,
    set_topic_progress,
    topic_digest_states,
    write_memory,
)
from app.agents.learning_tracker.state import (
    CheckpointOutcome,
    ProgressStatus,
    RoadmapStatus,
)

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


async def _roadmap_view(roadmap: dict, user_id: str) -> dict:
    """A roadmap with everything derived from the learner's profile attached.

    The profile already shapes generation, but silently — this is what lets the
    UI show the pace it implies, what was personalized, and whether the profile
    has moved on since. One assembler so the detail and home screens can't
    disagree about any of it.
    """
    memory = await get_profile(user_id, LEARNING_NS)
    snapshot = roadmap.get("personalization")
    return {
        "roadmap": roadmap,
        "progress": roadmap_progress(roadmap),
        "forecast": completion_forecast(roadmap, memory),
        "personalization": snapshot,
        # Which inputs have changed since this roadmap was built. Empty for
        # roadmaps generated before snapshots existed — see profile_drift.
        "profile_changes": profile_drift(snapshot, memory),
        # The profile as it stands, so an "update this roadmap" prompt can name
        # the new values instead of vaguely saying something changed.
        "current_personalization": profile_snapshot(memory),
        # {topicId: n} — lets the roadmap mark which topics have been written
        # about without shipping the notes themselves.
        "note_counts": await note_counts(user_id, str(roadmap["_id"])),
    }


@router.get("/current-state")
async def getCurrentState(current_user: Annotated[dict, Depends(get_current_user)]):
    """What the learner is working on right now: their active roadmap, how far
    along it is, and when their stated pace gets them to the end. The home
    screen's one call — no roadmapId needed."""
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
        "result": await _roadmap_view(roadmap, user_id),
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
        "result": await _roadmap_view(roadmap, current_user["uid"]),
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
    bare "what should I study next?" resolves to, and what the daily digest
    sweep runs on.

    Resuming past MAX_ACTIVE_ROADMAPS is a 409 naming the roadmaps holding the
    slots — the learner has to park one, and it's not the server's call which.
    """
    try:
        updated = await set_roadmap_status(roadmapId, current_user["uid"], body.status)
    except ActiveRoadmapLimit as e:
        raise HTTPException(
            status_code=409,
            detail={
                "message": (
                    f"You can run {e.limit} roadmaps at a time. "
                    "Pause one to make room for this."
                ),
                "limit": e.limit,
                "active": e.active,
            },
        )
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
    status: Optional[Literal["unread", "marked"]] = None,
    active_only: bool = False,
    limit: int = 20,
    roadmapId: Optional[str] = None,
    topicId: Optional[str] = None,
):
    """The caller's digests, most recent first.

    `status=unread&active_only=true` is the catch-up view: everything still
    outstanding across roadmaps they're actually working on.

    `roadmapId` and then `topicId` narrow the archive to one roadmap and one of
    its topics. Both are scoped by the caller's uid in the query, so an id
    belonging to someone else matches nothing rather than reading across.
    """
    return {
        "status": "done",
        "result": await list_digests(
            current_user["uid"],
            status=status,
            active_only=active_only,
            limit=limit,
            roadmapId=roadmapId,
            topicId=topicId,
        ),
    }


class DigestMark(BaseModel):
    # Required when the digest carries a recall check (every digest after the
    # first on a topic).
    answers: list[Answer] = Field(default_factory=list)
    # Generate the next digest for this topic straight after marking.
    generate_next: bool = False


@router.post("/digests/{digestId}/mark")
async def acknowledge_digest(
    digestId: str,
    body: DigestMark,
    current_user: Annotated[dict, Depends(get_current_user)],
):
    """Acknowledge a digest, optionally pulling the next one in the same step.

    A digest carrying a recall check can only be marked by passing it — that's
    what makes acknowledgement mean "I read this" rather than "I dismissed this".
    The check covers earlier digests only, so it's always answerable.
    """
    user_id = current_user["uid"]
    digest = await fetch_digest(digestId, user_id)
    if not digest:
        raise HTTPException(status_code=404, detail="Digest not found.")

    quiz_result = None
    quiz_id = digest.get("quizId")
    if quiz_id and digest.get("status") != "marked":
        quiz = await fetch_quiz(user_id, quiz_id)
        if quiz:
            graded = grade_quiz(
                quiz.get("questions", []), {a.question: a.answer for a in body.answers}
            )
            if graded["score"] < DIGEST_QUIZ_PASS_SCORE:

                raise HTTPException(
                    status_code=422,
                    detail={
                        "message": "Not quite — check the earlier tips and try again.",
                        "quiz_result": graded,
                        "pass_score": DIGEST_QUIZ_PASS_SCORE,
                    },
                )
            quiz_result = graded
            await record_quiz_attempt(
                user_id, quiz_id, digest.get("roadmapId"), digest.get("topicId"), graded
            )

    if not await mark_digest(digestId, user_id):
        raise HTTPException(status_code=404, detail="Digest not found.")

    generated = None
    if body.generate_next:
        roadmap = await fetch_roadmap(digest.get("roadmapId"), user_id)
        if roadmap:
            try:
                generated = await build_digest(user_id, roadmap, None, notify=False)
            except Exception as e:
                # Marking already succeeded; a generation failure must not undo it.
                logger.error("digest generate-after-mark failed: %s", e)

    remaining = await list_digests(user_id, status="unread", active_only=True, limit=50)
    return {
        "status": "done",
        "result": {
            "digestId": digestId,
            "quiz_result": quiz_result,
            "generated": generated,
            "remaining": len(remaining),
            "next": remaining[0] if remaining else None,
            # Once the tips have covered the topic, the checkpoint is what comes
            # next — not another digest.
            "coverage_complete": bool(digest.get("coverage_complete")),
            "topicId": digest.get("topicId"),
            "roadmapId": digest.get("roadmapId"),
        },
    }


@router.get("/focus")
async def get_focus(current_user: Annotated[dict, Depends(get_current_user)]):
    """What the learner is working on and when the next digest lands — what the
    home screen shows once the queue is clear.

    `result.roadmaps` carries one entry per active roadmap, newest first, each
    with its own topic, progress and `blocked_reason`; `next_at` and `cap` sit at
    the top because they're account-wide.
    """
    return {"status": "done", "result": await learning_focus(current_user["uid"])}


@router.post("/digests/generate")
async def generate_digest(
    current_user: Annotated[dict, Depends(get_current_user)],
    roadmapId: Optional[str] = None,
    topicId: Optional[str] = None,
):
    """Pull the next digest now rather than waiting for tomorrow's sweep.

    Deliberately not folded into marking: a digest costs a web search and an LLM
    call, so it happens when the learner asks for it, not as a side effect of
    acknowledging one. `build_digest` applies the same guards either way — the
    unread cap and the outstanding recall check — so this can't be used to stack
    them up or to skip a check.
    """
    user_id = current_user["uid"]
    roadmap = await fetch_roadmap(await resolve_roadmap_id(user_id, roadmapId), user_id)
    if not roadmap:
        raise HTTPException(status_code=404, detail="No active roadmap to digest.")

    # Checked here as well as inside `build_digest` only to say which check is
    # outstanding. `build_digest` returns a bare None, and "nothing new to send"
    # would send the learner looking for a digest that isn't the problem.
    topic = in_progress_topic(roadmap)
    if topic:
        gated_on = digest_quiz_gate(
            await topic_digest_states(user_id, str(roadmap["_id"]), topic.get("id"))
        )
        if gated_on:
            raise HTTPException(
                status_code=409,
                detail={
                    "message": (
                        f"Pass the recall check on digest #{gated_on} first — "
                        "it's on the digest itself."
                    ),
                    "blocked_reason": "awaiting_quiz",
                    "sequence": gated_on,
                },
            )

    digest = await build_digest(user_id, roadmap, topicId, notify=False)
    if not digest:
        raise HTTPException(
            status_code=409,
            detail="Nothing new to send — finish the digest you already have, or the roadmap is complete.",
        )
    return {"status": "done", "result": digest}


NoteKind = Literal["note", "snippet", "link", "question"]


class NoteCreate(BaseModel):
    roadmapId: str
    topicId: str
    # One field rather than four features: a jotting, a code snippet, a saved
    # link, and a question to come back to differ in how they render and filter,
    # not in what they are.
    kind: NoteKind = "note"
    body: str = Field(min_length=1, max_length=10_000)
    url: Optional[str] = None


@router.post("/notes")
async def add_note(
    body: NoteCreate, current_user: Annotated[dict, Depends(get_current_user)]
):
    note = await create_note(
        current_user["uid"],
        body.roadmapId,
        body.topicId,
        body.kind,
        body.body.strip(),
        body.url,
    )
    if not note:
        raise HTTPException(status_code=500, detail="Could not save that note.")
    return {"status": "done", "result": note}


@router.get("/notes")
async def get_notes(
    current_user: Annotated[dict, Depends(get_current_user)],
    roadmapId: Optional[str] = None,
    topicId: Optional[str] = None,
    kind: Optional[NoteKind] = None,
    limit: int = 100,
    skip: int = 0,
):
    """Everything the learner has written down, newest first. Unfiltered this is
    the consolidated "my notes" view; scoped to a topic it backs the notes
    section on the roadmap screen."""
    return {
        "status": "done",
        "result": await list_notes(
            current_user["uid"],
            roadmapId=roadmapId,
            topicId=topicId,
            kind=kind,
            limit=limit,
            skip=skip,
        ),
    }


class NoteUpdate(BaseModel):
    body: Optional[str] = Field(default=None, min_length=1, max_length=10_000)
    url: Optional[str] = None
    # Questions get ticked off once answered.
    resolved: Optional[bool] = None


@router.patch("/notes/{noteId}")
async def edit_note(
    noteId: str,
    body: NoteUpdate,
    current_user: Annotated[dict, Depends(get_current_user)],
):
    updates = body.model_dump(exclude_none=True)
    if "body" in updates:
        updates["body"] = updates["body"].strip()
    if not updates:
        raise HTTPException(status_code=422, detail="Nothing to update.")
    if not await update_note(noteId, current_user["uid"], updates):
        raise HTTPException(status_code=404, detail="Note not found.")
    return {"status": "done", **updates}


@router.delete("/notes/{noteId}")
async def remove_note(
    noteId: str, current_user: Annotated[dict, Depends(get_current_user)]
):
    if not await delete_note(noteId, current_user["uid"]):
        raise HTTPException(status_code=404, detail="Note not found.")
    return {"status": "done"}


class CheckpointRequest(BaseModel):
    roadmapId: str
    # Force a new question set instead of resuming the one already issued.
    regenerate: bool = False


@router.post("/topics/{topicId}/checkpoint")
async def start_checkpoint(
    topicId: str,
    body: CheckpointRequest,
    current_user: Annotated[dict, Depends(get_current_user)],
):
    """Issue the active-recall checkpoint for a topic.

    Reuses the outstanding question set on a retry so a learner who failed sees
    the same material again rather than a fresh roll of the dice; a scheduled
    review always gets new questions, since recognising an old question isn't
    recall. Answers are stripped — grading happens server-side.
    """
    user_id = current_user["uid"]
    roadmap = await fetch_roadmap(body.roadmapId, user_id)
    topic = find_topic(roadmap, topicId)
    if not topic:
        raise HTTPException(status_code=404, detail="Topic not found.")

    is_review = topic.get("progress_status") == "completed"
    existing = None
    if not body.regenerate and not is_review:
        existing = await get_db()["quizzes"].find_one(
            {"user_id": user_id, "roadmapId": body.roadmapId, "topicId": topicId},
            sort=[("_id", -1)],
        )

    if existing:
        quizId, questions = str(existing["_id"]), existing.get("questions", [])
    else:
        memory = await get_profile(user_id, LEARNING_NS)
        generated = await build_checkpoint(
            topic, (roadmap or {}).get("title", ""), memory, is_review
        )
        questions = [q.model_dump() for q in generated.quiz]
        res = await get_db()["quizzes"].insert_one(
            {
                "user_id": user_id,
                "roadmapId": body.roadmapId,
                "topicId": topicId,
                "kind": "review" if is_review else "checkpoint",
                "questions": questions,
                "createdAt": datetime.now(timezone.utc).isoformat(),
            }
        )
        quizId = str(res.inserted_id)

    return {
        "status": "done",
        "result": {
            "quizId": quizId,
            "topicId": topicId,
            "title": topic.get("title"),
            "is_review": is_review,
            "pass_score": CHECKPOINT_PASS_SCORE,
            "questions": [
                {"question": q.get("question"), "options": q.get("options")}
                for q in questions
            ],
        },
    }


class CheckpointSubmission(BaseModel):
    quizId: str
    answers: list[Answer]


@router.post("/checkpoint/submit")
async def submit_checkpoint(
    body: CheckpointSubmission,
    current_user: Annotated[dict, Depends(get_current_user)],
):
    """Grade a checkpoint and let the result drive the topic's state: passing is
    what completes it, and either way the next review gets scheduled."""
    user_id = current_user["uid"]
    quiz = await fetch_quiz(user_id, body.quizId)
    # A quiz raised in chat without a roadmap has nothing to attach a result to.
    if not quiz or not quiz.get("topicId") or not quiz.get("roadmapId"):
        raise HTTPException(status_code=404, detail="Checkpoint not found.")

    graded = grade_quiz(
        quiz.get("questions", []), {a.question: a.answer for a in body.answers}
    )
    await record_quiz_attempt(
        user_id, body.quizId, quiz.get("roadmapId"), quiz.get("topicId"), graded
    )

    outcome = await apply_checkpoint(
        quiz["roadmapId"], quiz["topicId"], user_id, graded["score"]
    )
    if outcome is None:
        raise HTTPException(status_code=404, detail="Roadmap or topic not found.")

    return {
        "status": "done",
        "result": CheckpointOutcome(
            **graded, **outcome, pass_score=CHECKPOINT_PASS_SCORE
        ).model_dump(),
    }


@router.get("/reviews")
async def get_reviews(
    current_user: Annotated[dict, Depends(get_current_user)],
    limit: int = 20,
):
    """Topics whose spaced-repetition review has come due, soonest first."""
    return {
        "status": "done",
        "result": await due_reviews(current_user["uid"], limit=limit),
    }


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
    """Set a topic's progress directly — used for every transition the learner
    controls outright: starting, skipping, or reopening a topic.

    Completion is the exception. A topic is completed by passing its checkpoint
    (POST /checkpoint/submit), not by asserting it, so that a finished roadmap
    means something was recalled rather than something was tapped. Reopening a
    topic stays free: the gate is on claiming knowledge, not on retracting it.
    """
    status = body.status or ("completed" if body.covered else "not_started")
    if status == "completed":
        raise HTTPException(
            status_code=409,
            detail="Pass this topic's checkpoint to complete it.",
        )
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
