"""HTTP routes for the learning-tracker agent. Agent logic lives in app.agents.learning_tracker."""

import json
import logging
import uuid
from datetime import datetime, timedelta, timezone
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
from app.core.config import (
    CHECKPOINT_MAX_ATTEMPTS_PER_DAY,
    CHECKPOINT_PASS_SCORE,
    CHECKPOINT_QUESTIONS,
    DIGEST_QUIZ_PASS_SCORE,
    DIGEST_RETEACH_AFTER,
    FEYNMAN_LADDER_BONUS,
    FEYNMAN_MIN_WORDS,
    FEYNMAN_PASS_SCORE,
    LEARNING_CHECKPOINT_RATE_LIMIT,
    LEARNING_CHECKPOINT_RATE_WINDOW,
    LEARNING_EXPLAIN_RATE_LIMIT,
    LEARNING_EXPLAIN_RATE_WINDOW,
    LEARNING_QUERY_RATE_LIMIT,
    LEARNING_QUERY_RATE_WINDOW,
    MISCONCEPTION_REPROBE_DAYS,
    redis_client,
)
from app.services import rate_limit
from app.agents.learning_tracker.service import (
    build_briefing,
    build_checkpoint,
    grade_oneliner,
    judge_explanation,
    refresh_misconceptions,
)
from app.agents.learning_tracker.context import learner_context, situation_key
from app.agents.learning_tracker.practice import build_practice_deck
from app.agents.learning_tracker.triggers import (
    build_digest,
    build_revision_digest,
    escalate_to_reteach,
    pull_next_digest,
)
from app.agents.learning_tracker.repository import (
    LEARNING_NS,
    ActiveRoadmapLimit,
    apply_checkpoint,
    asked_questions,
    checkpoint_pass_mark,
    completion_forecast,
    create_note,
    delete_note,
    delete_roadmap,
    digest_quiz_gate,
    due_misconceptions,
    due_reviews,
    failed_attempts,
    fetch_digest,
    fetch_quiz,
    fetch_roadmap,
    find_topic,
    get_misconceptions,
    grade_quiz,
    in_progress_topic,
    learning_focus,
    list_misconceptions,
    learning_stats,
    list_digests,
    list_notes,
    list_roadmaps,
    mark_digest,
    mark_misconceptions_addressed,
    note_counts,
    open_questions,
    public_question,
    profile_drift,
    profile_snapshot,
    quiz_graded,
    recent_attempts,
    record_explanation,
    record_quiz_attempt,
    record_revision,
    redact_review,
    retry_block,
    revision_outstanding,
    roadmap_mastery,
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
    # Every turn is at least an intent classification and an agent call, and a
    # "build me a roadmap" turn is a good deal more. Capped before the graph runs.
    await rate_limit.limit_user(
        "learning_query",
        current_user["uid"],
        LEARNING_QUERY_RATE_LIMIT,
        LEARNING_QUERY_RATE_WINDOW,
    )
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
    # Same bucket as /query — it is the same turn, and splitting them would let a
    # client double its allowance by alternating.
    await rate_limit.limit_user(
        "learning_query",
        current_user["uid"],
        LEARNING_QUERY_RATE_LIMIT,
        LEARNING_QUERY_RATE_WINDOW,
    )
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
        user_id,
        body.quizId,
        quiz.get("roadmapId"),
        quiz.get("topicId"),
        result,
        questions=questions,
        kind=quiz.get("kind") or "chat",
    )
    # A quiz raised in chat gates nothing, so there's nothing to protect by
    # withholding the answers — seeing them is the whole point of asking.
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
        # {topicId: mastery} from the same computation the home screen's numbers
        # come from. Without it the topic screen had nothing to show but
        # `mastery_score` — the last checkpoint's score — under the word mastery,
        # and the two screens disagreed about the same topic.
        "topic_mastery": await roadmap_mastery(user_id, roadmap),
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


@router.delete("/roadmaps/{roadmapId}")
async def deleteRoadmap(
    roadmapId: str, current_user: Annotated[dict, Depends(get_current_user)]
):
    """Delete a roadmap and everything stored against it — digests, notes,
    quizzes, graded attempts, misconception analysis.

    Irreversible, and distinct from `PATCH … {"status": "archived"}`, which keeps
    the record and the history. Archiving is the reversible way to put a roadmap
    down; this is for one the learner wants gone.

    `result.linked_tasks` counts to-dos in the personal assistant that came from
    this roadmap. They are deliberately left alone — the learner may have
    rescheduled or annotated them — so the client can mention they're still there.
    """
    removed = await delete_roadmap(roadmapId, current_user["uid"])
    if removed is None:
        raise HTTPException(status_code=404, detail="Roadmap not found.")
    return {"status": "done", "roadmapId": roadmapId, "result": removed}


@router.get("/memory")
async def get_memory(current_user: Annotated[dict, Depends(get_current_user)]):
    """Let the UI show the learner what the system remembers about them. Reads the
    same `learning` namespace the agent reads, so the screen can't disagree with
    what actually personalizes the roadmaps."""
    return {
        "status": "done",
        "result": await get_profile(current_user["uid"], LEARNING_NS),
    }


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


async def _record_written_answer(
    user_id: str,
    roadmapId: str,
    topicId: str,
    topicTitle: str,
    question: dict,
    answer: str,
) -> None:
    """Grade one typed sentence and file what it reveals.

    Runs off the request path: the learner has already been let through on the
    tap questions, so nothing they are waiting for depends on this. Failing here
    costs a piece of evidence, never an acknowledgement.
    """
    try:
        judged = await grade_oneliner(
            question.get("question", ""), question.get("expected"), answer
        )
    except Exception as e:
        logger.error("one-liner grading failed topic=%s: %s", topicId, e)
        return

    correct = judged.score >= 50
    await record_quiz_attempt(
        user_id,
        None,
        roadmapId,
        topicId,
        {"total": 1, "correct": 1 if correct else 0, "score": judged.score, "review": []},
        kind="written",
        # Their sentence next to what the question wanted — the shape of evidence
        # the tracker reads, and richer than any distractor index.
        misses=(
            []
            if correct
            else [
                {
                    "question": question.get("question", ""),
                    "chosen": answer,
                    "correct": question.get("expected"),
                }
            ]
        ),
    )
    if not correct:
        await refresh_misconceptions(user_id, roadmapId, topicId, topicTitle)


class DigestMark(BaseModel):
    # Required when the digest carries a recall check (every digest after the
    # first on a topic).
    answers: list[Answer] = Field(default_factory=list)
    # Typed answers to the open question, keyed by its position in the set. From
    # the fourth digest on, one question is answered in a sentence rather than
    # tapped.
    written: dict[int, str] = Field(default_factory=dict)
    # Generate the next digest for this topic straight after marking.
    generate_next: bool = False


@router.post("/digests/{digestId}/mark")
async def acknowledge_digest(
    digestId: str,
    body: DigestMark,
    background_tasks: BackgroundTasks,
    current_user: Annotated[dict, Depends(get_current_user)],
):
    """Acknowledge a digest, optionally pulling the next one in the same step.

    A digest carrying a recall check can only be marked by passing it — that's
    what makes acknowledgement mean "I read this" rather than "I dismissed this".
    The check covers earlier digests only, so it's always answerable.

    From the fourth digest the check also asks for one typed sentence. It has to
    be attempted, but being *wrong* doesn't block the mark: the tap questions are
    the gate, and putting an LLM's opinion of a sentence in front of a nudge
    would make the cheapest interaction in the system the most brittle. What it's
    really for is the diagnosis — it goes to the misconception tracker either way.
    """
    user_id = current_user["uid"]
    digest = await fetch_digest(digestId, user_id)
    if not digest:
        raise HTTPException(status_code=404, detail="Digest not found.")

    quiz_result = None
    quiz_id = digest.get("quizId")
    if quiz_id and digest.get("status") != "marked":
        quiz = await fetch_quiz(user_id, quiz_id)
        # Gated on the CLOSED questions, not on there being questions at all. A
        # set with nothing to tap scores 0 against a pass mark of 100, so it would
        # make the digest permanently unmarkable — and an unmarkable digest holds
        # a slot under DIGEST_MAX_UNREAD forever, stopping the topic getting any
        # further digests. A written answer is evidence, never the gate.
        all_questions = (quiz or {}).get("questions") or []
        gating = [q for q in all_questions if q.get("kind") != "open"]
        if quiz and gating:
            graded = grade_quiz(
                quiz.get("questions", []), {a.question: a.answer for a in body.answers}
            )
            # Recorded before the refusal, not after. A failed recall check is
            # the single most informative thing a digest produces, and raising
            # first meant the only attempt ever written was the one that got
            # everything right — which by definition carries no misses, so the
            # digest check contributed nothing to the misconception tracker at
            # all.
            await record_quiz_attempt(
                user_id,
                quiz_id,
                digest.get("roadmapId"),
                digest.get("topicId"),
                graded,
                questions=quiz.get("questions", []),
                kind="digest",
            )

            if graded["score"] < DIGEST_QUIZ_PASS_SCORE:
                # Twice on one set of tips is a fact about the tips. The learner
                # is still not let through — they read the new explanation and
                # answer the rebuilt check — but sending them back to re-read the
                # same words a third time is the one thing certain not to work.
                failures = await failed_attempts(
                    user_id,
                    digest.get("roadmapId"),
                    digest.get("topicId"),
                    quiz_id,
                    DIGEST_QUIZ_PASS_SCORE,
                )
                escalating = failures >= DIGEST_RETEACH_AFTER
                if escalating:
                    background_tasks.add_task(
                        escalate_to_reteach,
                        user_id,
                        digest.get("roadmapId"),
                        digest.get("topicId"),
                        quiz_id,
                        [
                            quiz["questions"][r["question"]].get("question", "")
                            for r in graded["review"]
                            if 0 <= r["question"] < len(quiz.get("questions") or [])
                        ],
                    )
                raise HTTPException(
                    status_code=422,
                    detail={
                        "message": (
                            "That's on us, not you — we'll explain this a "
                            "different way. Check back in a moment."
                            if escalating
                            else "Not quite — check the earlier tips and try again."
                        ),
                        # Redacted, exactly as a failed checkpoint is. This
                        # carried the raw grading, so the correct option for every
                        # missed question was on the wire — on a gate the learner
                        # retries until they pass, which makes it the one place
                        # the answer must not be. What replaces it is more useful
                        # anyway: what each missed question was testing, and
                        # which tip to go back to.
                        "quiz_result": redact_review(
                            graded, False, quiz.get("questions", [])
                        ),
                        "pass_score": DIGEST_QUIZ_PASS_SCORE,
                        # The client can stop saying "try again" and start saying
                        # "a new explanation is coming".
                        "reteaching": escalating,
                        "attempts": failures,
                    },
                )
            quiz_result = graded

        # The typed sentence, handled outside the gate above: it is evidence, not
        # a pass mark, and it still has to be collected on a set that happens to
        # have nothing to tap. It has to be attempted — a blank is noise in the
        # tracker rather than a reading of how someone thinks — but whether it is
        # *right* never blocks the acknowledgement.
        for idx, q in open_questions(all_questions):
            answer = (body.written.get(idx) or "").strip()
            if not answer:
                raise HTTPException(
                    status_code=422,
                    detail={
                        "message": "Answer the written question in a sentence.",
                        "question": idx,
                        "blocked_reason": "written_answer_required",
                    },
                )
            background_tasks.add_task(
                _record_written_answer,
                user_id,
                digest.get("roadmapId"),
                digest.get("topicId"),
                digest.get("topicTitle", ""),
                q,
                answer,
            )

    if not await mark_digest(digestId, user_id):
        raise HTTPException(status_code=404, detail="Digest not found.")

    # Acknowledging the revision digest is what settles the debt a failed
    # checkpoint opened, and so what re-opens the retry.
    revision_cleared = False
    if digest.get("kind") == "revision":
        revision_cleared = await record_revision(
            digest.get("roadmapId"), digest.get("topicId"), user_id
        )

    generated = None
    if body.generate_next and not revision_cleared:
        # Not after revision: what comes next there is the retry, not a digest.
        roadmap = await fetch_roadmap(digest.get("roadmapId"), user_id)
        if roadmap:
            try:
                # Declines on its own if the tips already cover the topic — the
                # coverage check runs before any generation, so asking for the
                # next digest on a finished topic costs nothing and returns None.
                generated = await build_digest(user_id, roadmap, None, notify=False)
            except Exception as e:
                # Marking already succeeded; a generation failure must not undo it.
                logger.error("digest generate-after-mark failed: %s", e)

    # Re-read after generating: `build_digest` moves the topic to `needs_review`
    # the moment coverage settles, and that transition is the answer to "why is
    # there no next digest?". Taking it off the digest that was just marked would
    # report the state from before this request.
    topic_status = None
    if digest.get("roadmapId"):
        fresh = await fetch_roadmap(digest.get("roadmapId"), user_id)
        topic_status = (find_topic(fresh, digest.get("topicId")) or {}).get(
            "progress_status"
        )

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
            # Where the topic stands now. `needs_review` with no `generated` is
            # the drip-feed having ended, not a failure to produce one — it's what
            # lets the client offer the checkpoint instead of "nothing new to send".
            "topic_status": topic_status,
            # The retry is now unblocked: send them to the checkpoint.
            "revision_cleared": revision_cleared,
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


# A briefing holds while the learner's position holds, so it is keyed on a
# fingerprint of that position rather than on time — see `situation_key`. The TTL
# is a backstop against a stale key surviving a deploy that changes the prompt,
# not the thing that expires it.
BRIEFING_CACHE_TTL = 6 * 60 * 60


@router.get("/briefing")
async def get_briefing(current_user: Annotated[dict, Depends(get_current_user)]):
    """What the assistant says on arrival, before being asked anything.

    The screen already lists what is outstanding. This is the judgement over that
    list: which one thing to do first, why, and the actions to do it — which is
    the part that previously lived in the client as a table of fixed copy keyed
    on `blocked_reason`, and so could never weigh two situations against each
    other or explain itself.

    Generated once per *situation*, not per request. The learner opening the app
    four times in an evening gets one LLM call; the fifth, after they read a
    digest, gets a new one because the situation actually moved. A model failure
    degrades to `fallback_briefing`, which is deterministic and follows the same
    priority order — so this endpoint always answers.
    """
    user_id = current_user["uid"]
    ctx = await learner_context(user_id, with_stats=True)
    key = f"learning:briefing:{user_id}:{situation_key(ctx)}"

    try:
        cached = await redis_client.get(key)
        if cached:
            return {"status": "done", "result": json.loads(cached)}
    except Exception:
        # A cache that cannot be read is a cache miss, never a failed request.
        logger.warning("briefing cache read failed user=%s", user_id)

    briefing = await build_briefing(ctx)

    try:
        await redis_client.set(key, json.dumps(briefing), ex=BRIEFING_CACHE_TTL)
    except Exception:
        logger.warning("briefing cache write failed user=%s", user_id)

    return {"status": "done", "result": briefing}


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
    # Every guard lives in `pull_next_digest`, which the tutor's own
    # `pull_next_lesson` tool calls as well: one copy of the rules, two ways in.
    # This route's job is now only to turn a refusal into the status and detail
    # shape shipped clients already read.
    #
    # That now includes the per-user cap on the most expensive thing the learner
    # can ask for — a web search and two or three model calls. It used to be
    # applied here, which capped this door and left the chat one open.
    outcome = await pull_next_digest(user_id, roadmapId, topicId)
    if "digest" in outcome:
        return {"status": "done", "result": outcome["digest"]}

    reason = outcome.get("reason")
    if reason == "no_roadmap":
        raise HTTPException(status_code=404, detail="No active roadmap to digest.")

    # Over the hourly cap. Same 429 and `Retry-After` the limiter raised when it
    # sat on this route; the detail is now the refusal sentence, which is a
    # string like every other detail here and says what the learner ran out of.
    if reason == "rate_limited":
        raise HTTPException(
            status_code=429,
            detail=outcome["refused"],
            headers={"Retry-After": str(outcome["retry_after"])},
        )

    # A bare string rather than an object, as before: the client renders a string
    # detail directly, and an object would reach a <Text> child as "[object Object]".
    if reason == "nothing_to_send":
        raise HTTPException(status_code=409, detail=outcome["refused"])

    detail = {"message": outcome["refused"], "blocked_reason": reason}
    if reason == "needs_revision":
        detail["topicId"] = outcome["topicId"]
    if reason == "awaiting_quiz":
        detail["sequence"] = outcome["sequence"]
    raise HTTPException(status_code=409, detail=detail)


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

    Every attempt gets a fresh question set, generated against what this learner
    has already been asked and what their past mistakes point to. Only an
    issued-but-ungraded quiz is handed back, which is a learner resuming mid-
    attempt rather than retrying. Answers are stripped — grading is server-side.
    """
    user_id = current_user["uid"]
    # CHECKPOINT_MAX_ATTEMPTS_PER_DAY is per topic; a roadmap with twenty of them
    # is bounded by nothing without this.
    await rate_limit.limit_user(
        "learning_checkpoint",
        user_id,
        LEARNING_CHECKPOINT_RATE_LIMIT,
        LEARNING_CHECKPOINT_RATE_WINDOW,
    )
    roadmap = await fetch_roadmap(body.roadmapId, user_id)
    topic = find_topic(roadmap, topicId)
    if not topic:
        raise HTTPException(status_code=404, detail="Topic not found.")

    # A failed attempt owes one round of revision before the next one. Without
    # this, "retry" is re-rolling the same questions until they land by accident
    # — and since a retry deliberately reuses the same question set, that's a
    # short walk.
    if revision_outstanding(topic):
        raise HTTPException(
            status_code=409,
            detail={
                "message": (
                    "Go over the revision tips for this topic first — "
                    "they're written from the questions you missed."
                ),
                "blocked_reason": "needs_revision",
                "topicId": topicId,
                "weak_points": topic.get("weak_points") or [],
            },
        )

    is_review = topic.get("progress_status") == "completed"

    # Retry limits. Checked before anything is generated: refusing after paying
    # for an LLM call would burn the call and still refuse.
    attempts = await recent_attempts(
        user_id, body.roadmapId, topicId, kinds=("checkpoint", "review")
    )
    blocked = retry_block(attempts)
    if blocked:
        raise HTTPException(
            status_code=429,
            detail={
                "message": (
                    "That's today's attempts on this topic — come back tomorrow "
                    "with fresh eyes."
                    if blocked["reason"] == "daily_limit"
                    else "Take a few minutes with the material before trying again."
                ),
                "blocked_reason": blocked["reason"],
                **blocked,
            },
        )

    # An issued-but-unsubmitted quiz is resumed — that's the learner reopening
    # the app mid-attempt, not a retry. Once it has been graded it is never
    # handed out again: re-issuing a set they've already seen tests whether they
    # remember being caught on it, and combined with any answer disclosure it
    # would make a retry pure transcription.
    #
    # Restricted to checkpoint sets. The newest quiz on a topic is just as often
    # a digest recall check or one raised in chat — both are stamped with the
    # same topicId — and handing one of those back means the topic is completed
    # by answering two questions about last week's tips.
    existing = None
    if not body.regenerate and not is_review:
        candidate = await get_db()["quizzes"].find_one(
            {
                "user_id": user_id,
                "roadmapId": body.roadmapId,
                "topicId": topicId,
                "kind": {"$in": ["checkpoint", "review"]},
            },
            sort=[("_id", -1)],
        )
        if candidate and not any(
            a.get("quizId") == str(candidate["_id"]) for a in attempts
        ):
            existing = candidate

    if existing:
        quizId, questions = str(existing["_id"]), existing.get("questions", [])
        # Resumed mid-attempt: keep the bar it was issued under. Falls back to
        # deriving it for sets stored before the mark was recorded.
        pass_score = existing.get("pass_score") or checkpoint_pass_mark(len(questions))
    else:
        memory = await get_profile(user_id, LEARNING_NS)
        # Both cached reads — the misconception analysis runs in the background
        # after each graded attempt, so issuing a checkpoint stays one LLM call.
        analysis = await get_misconceptions(user_id, body.roadmapId, topicId)
        # Filtered by the re-probe clock. Probing the same belief in consecutive
        # checkpoints measures whether they remember last week's question rather
        # than whether they now understand; leaving it a while and coming back is
        # what tells a correction that stuck from one that was papered over.
        confusions = due_misconceptions(
            (analysis or {}).get("patterns"),
            "last_probed_at",
            MISCONCEPTION_REPROBE_DAYS,
        )
        already_asked = await asked_questions(user_id, body.roadmapId, topicId)
        generated = await build_checkpoint(
            topic,
            (roadmap or {}).get("title", ""),
            memory,
            is_review,
            misconceptions=confusions,
            asked=already_asked,
        )
        if confusions:
            await mark_misconceptions_addressed(
                user_id,
                body.roadmapId,
                topicId,
                [c.get("label") for c in confusions if c.get("label")],
                "last_probed_at",
            )
        questions = [q.model_dump() for q in generated.quiz]
        if len(questions) != CHECKPOINT_QUESTIONS:
            # Not fatal — the bar below adapts to what actually arrived — but it
            # means the generator is under-delivering, which is invisible from
            # the outside and quietly changes how hard the gate is.
            logger.warning(
                "checkpoint generated %d questions, expected %d (topic=%s)",
                len(questions),
                CHECKPOINT_QUESTIONS,
                topicId,
            )
        pass_score = checkpoint_pass_mark(len(questions))
        res = await get_db()["quizzes"].insert_one(
            {
                "user_id": user_id,
                "roadmapId": body.roadmapId,
                "topicId": topicId,
                "kind": "review" if is_review else "checkpoint",
                "questions": questions,
                # Stored with the set, so grading marks it against the bar it was
                # advertised under rather than whatever the config says later.
                "pass_score": pass_score,
                "createdAt": datetime.now(timezone.utc).isoformat(),
            }
        )
        quizId = str(res.inserted_id)

    today = [a for a in attempts if _within_a_day(a.get("createdAt"))]
    return {
        "status": "done",
        "result": {
            "quizId": quizId,
            "topicId": topicId,
            "title": topic.get("title"),
            "is_review": is_review,
            "pass_score": pass_score,
            # So the learner knows what a failure costs before they answer,
            # rather than discovering the limit by hitting it.
            "attempts_today": len(today),
            "attempt_limit": CHECKPOINT_MAX_ATTEMPTS_PER_DAY,
            "questions": [
                {"question": q.get("question"), "options": q.get("options")}
                for q in questions
            ],
        },
    }


def _within_a_day(stamp: Optional[str]) -> bool:
    try:
        return datetime.fromisoformat(stamp) >= datetime.now(timezone.utc) - timedelta(
            days=1
        )
    except (TypeError, ValueError):
        return False


class CheckpointSubmission(BaseModel):
    quizId: str
    answers: list[Answer]


@router.post("/checkpoint/submit")
async def submit_checkpoint(
    body: CheckpointSubmission,
    background_tasks: BackgroundTasks,
    current_user: Annotated[dict, Depends(get_current_user)],
):
    """Grade a checkpoint and let the result drive the topic's state: passing is
    what completes it, and either way the next review gets scheduled.

    A failure comes back with the hint and the outcome behind each missed
    question, never the answer. The learner has to know what to go back over;
    they must not be able to write the right box down and re-enter it.
    """
    user_id = current_user["uid"]
    quiz = await fetch_quiz(user_id, body.quizId)
    # A quiz raised in chat without a roadmap has nothing to attach a result to.
    if not quiz or not quiz.get("topicId") or not quiz.get("roadmapId"):
        raise HTTPException(status_code=404, detail="Checkpoint not found.")

    # One grading per set. The cooldown and the daily cap are enforced where a
    # set is issued, so a second POST with the same quizId would walk straight
    # past both — and past the revision a failure owes — using the misses the
    # first attempt just reported back.
    if await quiz_graded(user_id, body.quizId):
        raise HTTPException(
            status_code=409,
            detail={
                "message": (
                    "That checkpoint has already been graded. Start a new one "
                    "when you're ready to try again."
                ),
                "blocked_reason": "already_graded",
            },
        )

    questions = quiz.get("questions", [])
    graded = grade_quiz(questions, {a.question: a.answer for a in body.answers})
    await record_quiz_attempt(
        user_id,
        body.quizId,
        quiz.get("roadmapId"),
        quiz.get("topicId"),
        graded,
        questions=questions,
        kind=quiz.get("kind") or "checkpoint",
    )

    # The text of what they got wrong, not just the indices `graded["review"]`
    # carries — the revision digest is written against these, and question
    # numbers from a quiz it never sees would tell it nothing.
    missed = [
        questions[r["question"]].get("question", "")
        for r in graded["review"]
        if 0 <= r["question"] < len(questions)
    ]

    # The bar this set was issued under. Derived for sets stored before it was
    # recorded, so an attempt in flight across the change is graded on its own
    # terms rather than on today's config.
    pass_score = quiz.get("pass_score") or checkpoint_pass_mark(len(questions))

    outcome = await apply_checkpoint(
        quiz["roadmapId"],
        quiz["topicId"],
        user_id,
        graded["score"],
        missed=missed,
        pass_score=pass_score,
    )
    if outcome is None:
        raise HTTPException(status_code=404, detail="Roadmap or topic not found.")

    # Re-read the history off the request path: the analysis is an LLM call, and
    # nothing on this response depends on it. It lands before the next checkpoint
    # is issued, which is the only thing that reads it.
    roadmap = await fetch_roadmap(quiz["roadmapId"], user_id)
    topic = find_topic(roadmap, quiz["topicId"])
    background_tasks.add_task(
        refresh_misconceptions,
        user_id,
        quiz["roadmapId"],
        quiz["topicId"],
        (topic or {}).get("title", ""),
    )

    return {
        "status": "done",
        "result": CheckpointOutcome(
            **redact_review(graded, outcome["passed"], questions),
            **outcome,
            pass_score=pass_score,
        ).model_dump(),
    }


class ExplanationSubmission(BaseModel):
    roadmapId: str
    text: str = Field(min_length=1, max_length=8_000)
    # Where the words came from. Dictated answers ramble, restart and lose their
    # punctuation; the judge is told to ignore all of that, and knowing which
    # mode produced the text is what lets that instruction be honest rather than
    # a guess about everyone.
    source: Literal["text", "voice"] = "text"


@router.post("/topics/{topicId}/explain")
async def explain_topic(
    topicId: str,
    body: ExplanationSubmission,
    background_tasks: BackgroundTasks,
    current_user: Annotated[dict, Depends(get_current_user)],
):
    """The Feynman checkpoint: explain this topic in your own words.

    Optional, and never a gate — a poor explanation costs nothing and can be
    retried, because an exercise that can lose you progress is one nobody
    volunteers for. It pays out through the review ladder instead: a passing
    explanation puts the topic an extra rung along at its next checkpoint, so
    someone who can say it plainly sees it again later than someone who only
    recognised the right option.

    Its real value is the diagnosis. A wrong multiple-choice answer says which
    of four options someone tapped; a flawed explanation shows how they are
    thinking, and that goes into the misconception tracker through the same path
    every other attempt does.
    """
    user_id = current_user["uid"]
    # A model call per submission, and — being optional and never a gate — the
    # one cost-bearing route with no domain limit of its own.
    await rate_limit.limit_user(
        "learning_explain",
        user_id,
        LEARNING_EXPLAIN_RATE_LIMIT,
        LEARNING_EXPLAIN_RATE_WINDOW,
    )
    roadmap = await fetch_roadmap(body.roadmapId, user_id)
    topic = find_topic(roadmap, topicId)
    if not topic:
        raise HTTPException(status_code=404, detail="Topic not found.")

    words = len(body.text.split())
    if words < FEYNMAN_MIN_WORDS:
        # Graded, this would produce a confident verdict about nothing, and store
        # it as evidence — so it's refused rather than scored badly.
        raise HTTPException(
            status_code=422,
            detail={
                "message": (
                    f"Give it a few more words — around {FEYNMAN_MIN_WORDS} is "
                    "enough to work with. Say it as you would to a friend."
                ),
                "words": words,
                "min_words": FEYNMAN_MIN_WORDS,
            },
        )

    try:
        judged = await judge_explanation(topic, body.text)
    except Exception as e:
        logger.error("explanation judging failed topic=%s: %s", topicId, e)
        raise HTTPException(
            status_code=503, detail="Couldn't read that just now — try again shortly."
        )

    passed = judged.score >= FEYNMAN_PASS_SCORE
    verdict = judged.model_dump()
    await record_explanation(
        body.roadmapId, topicId, user_id, verdict, body.text, passed
    )

    # Stored whole, returned without `probe` — it describes how a future question
    # will come back at this belief, and a learner who reads it knows what's
    # coming. GET /misconceptions strips it for the same reason.
    verdict["misconceptions"] = [
        {k: v for k, v in m.items() if k != "probe"}
        for m in verdict.get("misconceptions") or []
    ]

    # Recorded as an attempt so the existing analysis picks it up. The misses are
    # the outcomes the explanation got wrong, carrying the learner's own words —
    # far better evidence than a distractor index, and it reaches the tracker
    # through the same path as everything else rather than a second pipeline that
    # would have to be kept in step with the first.
    #
    # An outcome merely left out is NOT a miss: silence is a gap, and treating it
    # as a wrong belief would have the tracker chase something never said.
    misses = [
        {
            "question": o.outcome,
            "chosen": o.evidence or "(said, but incorrectly)",
            "correct": None,
        }
        for o in judged.outcomes
        if o.verdict == "wrong"
    ] + [
        {"question": m.label, "chosen": m.detail, "correct": None}
        for m in judged.misconceptions
    ]
    await record_quiz_attempt(
        user_id,
        None,
        body.roadmapId,
        topicId,
        {
            "total": len(judged.outcomes) or 1,
            "correct": sum(1 for o in judged.outcomes if o.verdict == "solid"),
            "score": judged.score,
            "review": [],
        },
        kind="explain",
        misses=misses,
    )

    background_tasks.add_task(
        refresh_misconceptions, user_id, body.roadmapId, topicId, topic.get("title", "")
    )

    return {
        "status": "done",
        "result": {
            "passed": passed,
            "pass_score": FEYNMAN_PASS_SCORE,
            **verdict,
            # What passing bought, stated plainly — an incentive nobody is told
            # about doesn't incentivise anything.
            "ladder_bonus": FEYNMAN_LADDER_BONUS if passed else 0,
            "topicId": topicId,
            "roadmapId": body.roadmapId,
        },
    }


@router.post("/practice")
async def start_practice(current_user: Annotated[dict, Depends(get_current_user)]):
    """A short mixed-topic practice deck, for a day with nothing waiting.

    The only retrieval in the app that interleaves: every other check sits inside
    one topic, which lets the surrounding context carry half the answer. It gates
    nothing, completes nothing, and — unlike a checkpoint — comes back with the
    answers, because there is no retry here to protect.

    Refusals are answers rather than errors: a learner who has not finished a
    topic yet has nothing to retrieve, and the thing to do is finish one.
    """
    user_id = current_user["uid"]
    # A deck is a single model call over several topics, so it sits in the same
    # bucket as a checkpoint rather than getting a cheaper one of its own.
    await rate_limit.limit_user(
        "learning_checkpoint",
        user_id,
        LEARNING_CHECKPOINT_RATE_LIMIT,
        LEARNING_CHECKPOINT_RATE_WINDOW,
    )

    deck = await build_practice_deck(user_id)
    if "refused" in deck:
        raise HTTPException(status_code=409, detail=deck["refused"])

    questions = deck["questions"]
    # `topicId` is null on the document and set per question instead: a deck spans
    # topics, and the attempt it produces has to be attributed to each of them
    # separately or the misconception tracker learns the wrong thing.
    res = await get_db()["quizzes"].insert_one(
        {
            "user_id": user_id,
            "roadmapId": None,
            "topicId": None,
            "kind": "practice",
            "questions": questions,
            "createdAt": datetime.now(timezone.utc).isoformat(),
        }
    )

    return {
        "status": "done",
        "result": {
            "quizId": str(res.inserted_id),
            "topics": deck["topics"],
            # Answer-free, like every other set that leaves this server.
            "questions": [
                {**public_question(q), "topicTitle": q.get("topicTitle"), "topicId": q.get("topicId")}
                for q in questions
            ],
        },
    }


class PracticeSubmission(BaseModel):
    quizId: str
    answers: list[Answer]


@router.post("/practice/submit")
async def submit_practice(
    body: PracticeSubmission,
    background_tasks: BackgroundTasks,
    current_user: Annotated[dict, Depends(get_current_user)],
):
    """Grade a practice deck.

    Two things separate this from every other grading route. The answers come
    back in full — practice unlocks nothing, so there is no transcription to
    prevent and immediate feedback is simply better teaching. And the attempt is
    recorded per *topic*, tagged `practice`: it feeds the misconception tracker,
    which is the diagnostic point of the exercise, and is kept out of mastery, so
    a voluntary drill can never cost the learner their headline number.
    """
    user_id = current_user["uid"]
    quiz = await fetch_quiz(user_id, body.quizId)
    if not quiz or quiz.get("kind") != "practice":
        raise HTTPException(status_code=404, detail="Practice set not found.")
    if await quiz_graded(user_id, body.quizId):
        raise HTTPException(status_code=409, detail="That practice set is already graded.")

    questions = quiz.get("questions", [])
    graded = grade_quiz(questions, {a.question: a.answer for a in body.answers})

    # One attempt per topic the deck touched, holding only that topic's questions.
    # A single attempt spanning five topics would attribute every miss to all of
    # them, and the misconception analysis reads exactly this.
    by_topic: dict[tuple, list[int]] = {}
    for i, q in enumerate(questions):
        key = (q.get("roadmapId"), q.get("topicId"))
        if all(key):
            by_topic.setdefault(key, []).append(i)

    chosen = {a.question: a.answer for a in body.answers}
    breakdown = []
    for (roadmapId, topicId), positions in by_topic.items():
        subset = [questions[i] for i in positions]
        # Re-graded against the subset so `review` indices line up with it — the
        # attempt is stored against this topic alone and has to be readable on
        # its own terms months later.
        selected = {new: chosen[old] for new, old in enumerate(positions) if old in chosen}
        result = grade_quiz(subset, selected)
        await record_quiz_attempt(
            user_id,
            body.quizId,
            roadmapId,
            topicId,
            result,
            questions=subset,
            kind="practice",
        )
        title = next((q.get("topicTitle") for q in subset if q.get("topicTitle")), "")
        background_tasks.add_task(
            refresh_misconceptions, user_id, roadmapId, topicId, title or ""
        )
        breakdown.append(
            {
                "roadmapId": roadmapId,
                "topicId": topicId,
                "title": title,
                "correct": result["correct"],
                "total": result["total"],
            }
        )

    # Weakest first: with the deck mixed, the useful reading is which topic let
    # them down, and a list in deck order buries it.
    breakdown.sort(key=lambda b: (b["correct"] / b["total"]) if b["total"] else 1)

    # Everything a checkpoint shows on a pass AND everything it shows on a
    # failure. `grade_quiz` reports the answer but not the hint, and `reveal=True`
    # passes its result through untouched — so the hint the generator wrote would
    # be stored and never seen. Here it costs nothing to give: there is no retry
    # for it to leak into, and "what to go back over" is the point of the round.
    review = [
        {
            **r,
            "outcome": (questions[r["question"]] or {}).get("outcome"),
            "hint": (questions[r["question"]] or {}).get("hint"),
        }
        if isinstance(r.get("question"), int) and 0 <= r["question"] < len(questions)
        else r
        for r in graded.get("review") or []
    ]

    return {
        "status": "done",
        "result": {**graded, "review": review, "answers_revealed": True, "topics": breakdown},
    }


@router.get("/misconceptions")
async def get_misconception_report(
    current_user: Annotated[dict, Depends(get_current_user)],
    roadmapId: Optional[str] = None,
):
    """What the learner keeps getting wrong, and why it looks that way.

    Derived from their wrong answers across digest checks, checkpoints and
    reviews. Read-only and cached — the analysis runs in the background after
    each graded attempt, never on this request.

    `probe` is omitted: it's the instruction for writing a question that catches
    this misunderstanding, and a learner who reads it knows what's coming.
    """
    report = await list_misconceptions(current_user["uid"], roadmapId)
    for entry in report:
        entry["patterns"] = [
            {k: v for k, v in p.items() if k != "probe"} for p in entry["patterns"]
        ]
    return {"status": "done", "result": report}


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
