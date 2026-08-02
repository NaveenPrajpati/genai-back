"""Learning digests: generating one, and the daily job that schedules them."""

import logging
from datetime import datetime, timezone
from typing import Optional

from langchain_core.prompts import ChatPromptTemplate

from app.core.config import DIGEST_MAX_UNREAD
from app.core.llm import llm
from app.database import get_db
from app.agents.trigger_store import due_triggers, mark_ran
from app.services.push_service import send_push_notification
from .service import build_digest_quiz, check_coverage
from .state import CoverageOutput, TopicTipsOutput
from .repository import (
    DIGESTS,
    in_progress_topic,
    set_topic_progress,
    topic_digests,
    unread_digest_count,
)
from .tools import tavily_search_tool

logger = logging.getLogger(__name__)

_TIPS_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are teaching ONE topic to a learner, a few points at a time.\n"
            "Write 3-5 bullets that TEACH. Every bullet must carry something the "
            "learner can use straight away — a definition, a rule, a gotcha, a "
            "rule of thumb, or a tiny worked example. Where a short snippet makes "
            "the idea land, put it inline.\n"
            "NEVER write a bullet that merely points at a resource. "
            "\"W3Schools has a tutorial\", \"check out Khan Academy\", \"use the "
            "official docs\" — these teach nothing, and links are already listed "
            "separately below the bullets. Treat the search results as reference "
            "material for getting the facts right, not as the subject.\n"
            "Aim at these outcomes, in order, and go deep enough that a learner "
            "could actually do them:\n{outstanding}\n"
            "Do not repeat what has already been sent:\n{covered}\n"
            "Keep each bullet to one or two sentences.",
        ),
        (
            "human",
            "Topic: {topic}\n"
            "What it covers: {description}\n"
            "Roadmap context: {summary}\n"
            "Reference material:\n{results}",
        ),
    ]
)


async def build_digest(
    user_id: str, roadmap: dict, notify: bool = True
) -> Optional[dict]:
    """Generate and store one digest for the topic a learner is working on.

    Shared by the daily sweep and the on-demand "next digest" action, so a digest
    pulled early is the same artefact as one that arrived on schedule.

    Returns None when there's nothing to send:
      * no topic is `in_progress` — nothing has been picked up yet;
      * DIGEST_MAX_UNREAD digests for it are already waiting.

    Each digest past the first also carries a short recall check over the earlier
    ones, and every digest re-asks whether the topic has now been covered end to
    end — at which point the drip-feed stops and the checkpoint takes over.
    """
    topic = in_progress_topic(roadmap)
    if not topic:
        return None

    roadmap_id = str(roadmap["_id"])
    topic_id = topic.get("id")
    unread = await unread_digest_count(user_id, roadmap_id, topic_id)
    if unread >= DIGEST_MAX_UNREAD:
        logger.info(
            "digest skipped, %s unread on topic=%s (cap %s)",
            unread,
            topic_id,
            DIGEST_MAX_UNREAD,
        )
        return None

    prior = await topic_digests(user_id, roadmap_id, topic_id)
    prior_bullets = [b for d in prior for b in d.get("bullets") or []]
    sequence = len(prior) + 1

    topic_title = topic.get("title", "")
    results = []
    try:
        search = await tavily_search_tool.ainvoke(
            {"query": f"best free learning resources for {topic_title}"}
        )
        results = search.get("results", []) if isinstance(search, dict) else search
    except Exception as e:
        logger.error("tavily digest search error: %s", e)

    chain = _TIPS_PROMPT | llm.with_structured_output(TopicTipsOutput)
    tips: TopicTipsOutput = await chain.ainvoke(
        {
            "topic": topic_title,
            "summary": roadmap.get("summary", ""),
            "results": results,
            # Steer each digest at what the previous ones didn't reach, so the
            # drip-feed advances instead of restating the same introduction.
            "covered": "\n".join(f"- {b}" for b in prior_bullets) or "- (nothing yet)",
        }
    )

    # From the second digest on, acknowledging requires recalling the earlier
    # ones. Built from `prior_bullets` only — quizzing someone on the digest they
    # haven't read yet would make marking impossible.
    quiz_id = None
    if sequence >= 2 and prior_bullets:
        try:
            quiz = await build_digest_quiz(topic_title, prior_bullets)
            res_quiz = await get_db()["quizzes"].insert_one(
                {
                    "user_id": user_id,
                    "roadmapId": roadmap_id,
                    "topicId": topic_id,
                    "kind": "digest",
                    "questions": [q.model_dump() for q in quiz.quiz],
                    "createdAt": datetime.now(timezone.utc).isoformat(),
                }
            )
            quiz_id = str(res_quiz.inserted_id)
        except Exception as e:
            # A digest the learner can't acknowledge is worse than one without a
            # recall check, so a generation failure degrades to no quiz.
            logger.error("digest quiz generation failed topic=%s: %s", topic_id, e)

    coverage = CoverageOutput(covered=False, missing=[])
    try:
        coverage = await check_coverage(topic, prior_bullets + list(tips.bullets))
    except Exception as e:
        logger.error("coverage check failed topic=%s: %s", topic_id, e)

    if coverage.covered:
        # Everything worth drip-feeding has been sent, so the topic moves to
        # `needs_review`: the final checkpoint is what completes it now, and no
        # further digests are generated for it (in_progress_topic won't match).
        await set_topic_progress(roadmap_id, topic_id, "needs_review", user_id)
        logger.info("topic %s fully covered — awaiting checkpoint", topic_id)

    now = datetime.now(timezone.utc).isoformat()
    doc = {
        "user_id": user_id,
        "roadmapId": roadmap_id,
        "topicId": topic_id,
        "topicTitle": topic_title,
        "sequence": sequence,
        "bullets": tips.bullets,
        "resources": [
            {"title": r.get("title"), "url": r.get("url")}
            for r in results
            if isinstance(r, dict)
        ],
        # Must be passed before this digest can be marked. Null on the first.
        "quizId": quiz_id,
        # Set once the tips have taught the whole topic: the client then offers
        # the checkpoint instead of another digest.
        "coverage_complete": coverage.covered,
        "missing_outcomes": coverage.missing,
        # Unread until the learner acknowledges it — that acknowledgement is the
        # only signal we have that a digest actually landed.
        "status": "unread",
        "createdAt": now,
        "updatedAt": now,
    }
    res = await get_db()[DIGESTS].insert_one(doc)
    logger.info(
        "learning digest #%s created user=%s topic=%s covered=%s",
        sequence,
        user_id,
        topic_title,
        coverage.covered,
    )

    if notify:
        await send_push_notification(
            user_id,
            title=f"Today's tips: {topic_title}",
            body=(
                tips.bullets[0]
                if tips.bullets
                else "Your daily learning digest is ready."
            ),
            data={"type": "learning_digest", "topicId": topic.get("id")},
        )

    return {**doc, "_id": str(res.inserted_id)}


async def run_triggers(agent=None):
    """Hourly sweep: for every user who opted in via /toggle-trigger, fire only
    when the current hour matches their chosen schedule_hour in their timezone,
    then generate a digest for each active roadmap's current topic."""
    logger.info("learning digest job running")
    now = datetime.now(timezone.utc)

    try:
        triggers = await due_triggers("learning_digest", now)
    except Exception as e:
        logger.error("run_triggers trigger fetch error: %s", e)
        return

    for trig in triggers:
        user_id = trig.get("user_id")
        try:
            # Active only: an archived or finished roadmap has nothing to nudge
            # the learner about, and every extra roadmap is a search + an LLM call.
            cursor = get_db()["roadmaps"].find({"user_id": user_id, "status": "active"})
            roadmaps = await cursor.to_list(None)
        except Exception as e:
            logger.error("run_triggers roadmap fetch error user=%s: %s", user_id, e)
            continue

        for roadmap in roadmaps:
            try:
                await build_digest(user_id, roadmap)
            except Exception as e:
                logger.error(
                    "learning digest error roadmap=%s: %s", roadmap.get("_id"), e
                )

        # Record when this user's digest last ran.
        try:
            await mark_ran(trig, now)
        except Exception as e:
            logger.error("trigger last_run update error user=%s: %s", user_id, e)
