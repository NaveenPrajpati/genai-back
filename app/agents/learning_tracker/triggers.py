"""Learning digests: generating one, and the daily job that schedules them."""

import logging
from datetime import datetime, timezone
from typing import Optional

from langchain_core.prompts import ChatPromptTemplate

from app.core.config import (
    DIGEST_MAX_UNREAD,
    DIGEST_MIN_BEFORE_CHECKPOINT,
    DIGEST_ONELINER_FROM,
    LEARNING_DIGEST_RATE_LIMIT,
    LEARNING_DIGEST_RATE_WINDOW,
    MISCONCEPTION_RETEACH_DAYS,
)
from app.core.llm import llm
from app.database import get_db
from app.agents.trigger_store import due_triggers, mark_ran
from app.services import rate_limit
from app.services.push_service import send_push_notification
from app.agents.memory_store import get_profile
from .service import (
    build_digest_quiz,
    build_oneliner,
    build_reteach,
    check_coverage,
)
from .state import CoverageOutput, TopicTipsOutput
from .repository import (
    DIGESTS,
    LEARNING_NS,
    coverage_may_end,
    fetch_roadmap,
    replace_quiz_questions,
    digest_carries_quiz,
    digest_quiz_gate,
    digest_quiz_window,
    due_misconceptions,
    find_topic,
    get_misconceptions,
    mark_misconceptions_addressed,
    in_progress_topic,
    public_question,
    resolve_roadmap_id,
    revision_outstanding,
    set_topic_progress,
    teaching_digests,
    topic_digest_states,
    topic_digests,
)
from .tools import tavily_search_tool

logger = logging.getLogger(__name__)

# Injected into the tips prompt when this learner's own wrong answers point at a
# specific confusion. Teaching that ignores what someone already believes is
# teaching into a headwind: the misconception is what the new bullet has to
# displace, and naming it is what turns a general explanation into one aimed at
# this learner.
_MISCONCEPTION_BLOCK = (
    "This learner's own answers show they are getting these wrong:\n{patterns}\n"
    "Write at least one bullet that takes the confusion head-on — say plainly "
    "what is actually the case and why the other reading is tempting but wrong. "
    "Do NOT mention their mistakes, quiz scores or history; write it as teaching, "
    "not as a correction notice. A bullet that says 'you may have thought…' "
    "reads as an accusation and teaches nothing extra.\n"
)

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
            '"W3Schools has a tutorial", "check out Khan Academy", "use the '
            'official docs" — these teach nothing, and links are already listed '
            "separately below the bullets. Treat the search results as reference "
            "material for getting the facts right, not as the subject.\n"
            "Never name a course, book, author, website, channel or platform in a "
            "bullet. If the reference material turns out to be mostly lists of "
            "places to learn this — course roundups, 'top 10' posts, forum threads "
            "recommending tutorials — then it contains nothing to teach: ignore it "
            "entirely and write the bullets from your own knowledge of the topic.\n"
            "Aim at these outcomes, in order, and go deep enough that a learner "
            # "could actually do them:\n{outstanding}\n"
            "Do not repeat what has already been sent:\n{covered}\n"
            "{misconceptions}"
            "Keep each bullet to one or two sentences.",
        ),
        (
            "human",
            "Topic: {topic}\n"
            # "What it covers: {description}\n"
            "Roadmap context: {summary}\n" "Reference material:\n{results}",
        ),
    ]
)


async def build_digest(
    user_id: str, roadmap: dict, topicId: Optional[str] = None, notify: bool = True
) -> Optional[dict]:
    """Generate and store one digest for the topic a learner is working on.

    Shared by the daily sweep and the on-demand "next digest" action, so a digest
    pulled early is the same artefact as one that arrived on schedule.

    Returns None when there's nothing to send:
      * no topic is `in_progress` — nothing has been picked up yet;
      * DIGEST_MAX_UNREAD digests for it are already waiting;
      * the previous recall check hasn't been passed (see `digest_quiz_gate`);
      * the tips already sent cover the topic — the checkpoint comes next.

    Every other digest — #2, #4, #6 — carries a short recall check over the ones
    since the last check.
    """
    topic = in_progress_topic(roadmap)
    if not topic:
        return None

    roadmap_id = str(roadmap["_id"])
    topic_id = topic.get("id")

    prior = await topic_digests(user_id, roadmap_id, topic_id)
    # Counted over the drip-feed alone. A revision or re-teach digest covers
    # ground already sent and stores `sequence: None` to say so — letting one
    # take the next number would move where the recall checks fall for
    # everything after it.
    sequence = len(teaching_digests(prior)) + 1

    # Both guards run before the web search and the LLM call — a digest that
    # won't be stored shouldn't cost anything to decline.
    unread = sum(1 for d in prior if d.get("status") != "marked")
    if unread >= DIGEST_MAX_UNREAD:
        logger.info(
            "digest skipped, %s unread on topic=%s (cap %s)",
            unread,
            topic_id,
            DIGEST_MAX_UNREAD,
        )
        return None

    gated_on = digest_quiz_gate(prior)
    if gated_on:
        logger.info(
            "digest #%s held on topic=%s: recall check for #%s not passed",
            sequence,
            topic_id,
            gated_on,
        )
        return None

    prior_bullets = [b for d in prior for b in d.get("bullets") or []]
    topic_title = topic.get("title", "")

    # Coverage is decided BEFORE anything is generated, not after. Asking
    # afterwards means the answer arrives once a search and a tips call have
    # already been spent on a digest the topic didn't need — and the learner
    # receives it, so the drip-feed always overshoots by one.
    #
    # The last digest carries the verdict already, and it was computed over
    # exactly these bullets — `prior(1..N-1) + new(N)` is `prior(1..N)` — so it is
    # reused rather than re-derived, either way it fell. That keeps the steady
    # state at one coverage call per digest while still deciding before the spend,
    # and it catches a topic left `in_progress` after coverage completed, which is
    # the one way a covered topic keeps receiving digests. Only a digest written
    # before the field existed needs the check run here.
    if prior_bullets:
        stored = prior[-1].get("coverage_complete")
        if stored is None:
            try:
                stored = (await check_coverage(topic, prior_bullets)).covered
            except Exception as e:
                # Unknown coverage must not stop the drip-feed: a topic that goes
                # quiet because a check failed is worse than one extra digest.
                logger.error(
                    "pre-generation coverage check failed topic=%s: %s", topic_id, e
                )
                stored = False
        # The floor is re-applied rather than trusted from the stored flag, which
        # on a digest written before the floor existed is the raw verdict.
        if stored and coverage_may_end(sequence - 1):
            await set_topic_progress(roadmap_id, topic_id, "needs_review", user_id)
            logger.info(
                "digest #%s not generated, topic=%s already covered — checkpoint next",
                sequence,
                topic_id,
            )
            return None

    results = []
    try:
        # Search the SUBJECT, not the marketplace around it. Asking for "best
        # free learning resources for X" returns course roundups and "top 10"
        # listicles, whose content is about where to learn X rather than about X
        # — so the bullets written from them drift into naming courses and
        # authors, and the recall check built from those bullets ends up quizzing
        # the learner on who wrote which tutorial.
        search = await tavily_search_tool.ainvoke(
            {
                "query": (
                    f"{topic_title} explained: key concepts, worked examples, "
                    "common mistakes"
                )
            }
        )
        results = search.get("results", []) if isinstance(search, dict) else search
    except Exception as e:
        logger.error("tavily digest search error: %s", e)

    # What this learner is getting wrong, so the tips can displace it rather than
    # explain past it. A cached read — the analysis runs in the background after
    # each graded attempt — and filtered by the re-teach clock, so a correction
    # already made recently isn't repeated in the next digest.
    analysis = await get_misconceptions(user_id, roadmap_id, topic_id)
    confusions = due_misconceptions(
        (analysis or {}).get("patterns"), "last_taught_at", MISCONCEPTION_RETEACH_DAYS
    )

    chain = _TIPS_PROMPT | llm.with_structured_output(TopicTipsOutput)
    tips: TopicTipsOutput = await chain.ainvoke(
        {
            "topic": topic_title,
            "summary": roadmap.get("summary", ""),
            "results": results,
            # Steer each digest at what the previous ones didn't reach, so the
            # drip-feed advances instead of restating the same introduction.
            "covered": "\n".join(f"- {b}" for b in prior_bullets) or "- (nothing yet)",
            "misconceptions": (
                _MISCONCEPTION_BLOCK.format(
                    patterns="\n".join(
                        f"- {c.get('label')}: {c.get('detail')}" for c in confusions
                    )
                )
                if confusions
                else ""
            ),
        }
    )

    if confusions:
        await mark_misconceptions_addressed(
            user_id,
            roadmap_id,
            topic_id,
            [c.get("label") for c in confusions if c.get("label")],
            "last_taught_at",
        )
        logger.info(
            "digest #%s on topic=%s aimed at %s misconception(s)",
            sequence,
            topic_id,
            len(confusions),
        )

    # Every other digest, acknowledging requires recalling what came since the
    # last check. Built from earlier digests only — quizzing someone on the one
    # they haven't read yet would make marking impossible.
    quiz_id = None
    quiz = None
    checked = [b for d in digest_quiz_window(prior) for b in d.get("bullets") or []]
    if digest_carries_quiz(sequence) and checked:
        try:
            built = await build_digest_quiz(topic_title, checked)
            # From DIGEST_ONELINER_FROM on, one of the questions is answered in
            # prose instead of tapped. Late rather than from the start: the habit
            # forms on tap-only checks, and by the fourth digest there is enough
            # material that a sentence is worth the keystroke. It's appended, so
            # the tap questions keep their positions and their grading.
            if sequence >= DIGEST_ONELINER_FROM:
                try:
                    built.quiz.append(await build_oneliner(topic_title, checked))
                except Exception as e:
                    # The tap questions are the gate; losing the sentence costs
                    # signal, not the learner's ability to acknowledge the digest.
                    logger.error("one-liner generation failed topic=%s: %s", topic_id, e)
            # An empty set is a legitimate outcome — the generator is told to
            # write nothing rather than ask about the provenance of a tip that
            # only points at a resource. It must not be attached: a digest whose
            # quiz has no questions scores 0 against a pass mark of 100, so it can
            # never be marked, and being unmarkable it holds a slot under
            # DIGEST_MAX_UNREAD forever and jams the whole topic.
            if built.quiz:
                quiz = built
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
            else:
                logger.info(
                    "digest quiz empty topic=%s — nothing checkable in the tips",
                    topic_id,
                )
        except Exception as e:
            # A digest the learner can't acknowledge is worse than one without a
            # recall check, so a generation failure degrades to no quiz.
            logger.error("digest quiz generation failed topic=%s: %s", topic_id, e)

    # Re-asked now that this digest's bullets exist. This is what the gate above
    # reads on the next call, and it's stored on the digest so the client can
    # offer the checkpoint on the one that completed the topic rather than a call
    # later. The topic still moves now — this digest is worth reading, but no
    # further one is worth generating.
    coverage = CoverageOutput(covered=False, missing=[])
    try:
        coverage = await check_coverage(topic, prior_bullets + list(tips.bullets))
    except Exception as e:
        logger.error("coverage check failed topic=%s: %s", topic_id, e)

    # A narrow topic is genuinely covered by its first digest — and ending there
    # would mean it never carries a recall check, since those ride the second.
    # The verdict stands; acting on it waits for the floor.
    ends_here = coverage.covered and coverage_may_end(sequence)
    if coverage.covered and not ends_here:
        logger.info(
            "topic %s covered at digest #%s — held for the recall check on #%s",
            topic_id,
            sequence,
            DIGEST_MIN_BEFORE_CHECKPOINT,
        )

    if ends_here:
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
        #
        # The effective decision, not the raw verdict — this is what the card
        # reads, and a digest that says "take the checkpoint" while the server
        # intends to send one more is the two of them disagreeing on screen. It
        # is also what the next call's pre-generation guard consults, which is
        # the same question: should the feed stop here?
        "coverage_complete": ends_here,
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
            # Enough to open the digest itself, not just the topic it belongs
            # to: every learning read is scoped by roadmap, so a payload without
            # roadmapId can't be navigated from.
            data={
                "type": "learning_digest",
                "roadmapId": roadmap_id,
                "topicId": topic_id,
                "digestId": str(res.inserted_id),
            },
        )

    # The same answer-free shape GET /digests hands back. Returning the models
    # whole sent `answer`, `expected` and `hint` to the client on the one path
    # that skips the read projection — the check is issued and given away in a
    # single response.
    doc["quiz"] = (
        [public_question(q.model_dump()) for q in quiz.quiz] if quiz else None
    )
    return {**doc, "_id": str(res.inserted_id)}


_REVISION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "A learner has just failed the checkpoint on a topic they have "
            "already been taught. Write 3-5 bullets that close the specific gaps "
            "the failed questions expose.\n"
            "This is NOT a re-introduction. They have read everything below and "
            "still got these questions wrong, so restating it the same way will "
            "fail the same way. Come at each miss from a different angle: a "
            "worked example where the earlier bullet gave a rule, the "
            "distinction being confused, the case that breaks the wrong mental "
            "model.\n"
            "Address every question they missed. Do not reveal or restate the "
            "answer options — they will face these questions again, and a bullet "
            "that hands over the answer teaches nothing.\n"
            "Questions they got wrong:\n{missed}\n"
            "What they have already been told (do not simply repeat it):\n{covered}",
        ),
        ("human", "Topic: {topic}\nRoadmap context: {summary}"),
    ]
)


async def build_revision_digest(
    user_id: str, roadmap: dict, topicId: str, notify: bool = True
) -> Optional[dict]:
    """Generate the revision digest a failed checkpoint owes the learner.

    Separate from `build_digest` because it answers a different question. A
    teaching digest asks "what hasn't been covered yet?" and stops when coverage
    is complete; a revision digest starts from coverage being complete and asks
    "what did they get wrong?". Running the teaching path here would generate one
    more digest about a topic already declared fully taught, which is the loop
    this exists to break.

    No web search: the material is the learner's own missed questions and the
    bullets they've already been sent. Nothing on the open web knows either.

    Returns None when nothing is owed, or when one is already waiting — a stack
    of revision digests is a worse answer to a failed checkpoint than one.
    """
    topic = find_topic(roadmap, topicId)
    if not topic or not revision_outstanding(topic):
        return None

    roadmap_id = str(roadmap["_id"])
    prior = await topic_digests(user_id, roadmap_id, topicId)
    if any(
        d.get("kind") == "revision" and d.get("status") != "marked" for d in prior
    ):
        logger.info("revision digest already waiting on topic=%s", topicId)
        return None

    missed = topic.get("weak_points") or []
    covered = [b for d in prior for b in d.get("bullets") or []]
    topic_title = topic.get("title", "")

    chain = _REVISION_PROMPT | llm.with_structured_output(TopicTipsOutput)
    tips: TopicTipsOutput = await chain.ainvoke(
        {
            "topic": topic_title,
            "summary": roadmap.get("summary", ""),
            "missed": "\n".join(f"- {m}" for m in missed) or "- (not recorded)",
            "covered": "\n".join(f"- {b}" for b in covered) or "- (nothing yet)",
        }
    )

    now = datetime.now(timezone.utc).isoformat()
    doc = {
        "user_id": user_id,
        "roadmapId": roadmap_id,
        "topicId": topicId,
        "topicTitle": topic_title,
        # Deliberately outside the teaching sequence: revision doesn't advance
        # the drip-feed, so it must not shift where the recall-check cadence
        # falls for the topic's remaining digests.
        "sequence": None,
        "kind": "revision",
        "bullets": tips.bullets,
        "resources": [],
        # No recall check. The checkpoint waiting on the other side of this IS
        # the assessment; gating revision behind its own quiz would put two
        # quizzes back to back between the learner and their retry.
        "quizId": None,
        "coverage_complete": True,
        "missing_outcomes": [],
        "weak_points": missed,
        "status": "unread",
        "createdAt": now,
        "updatedAt": now,
    }
    res = await get_db()[DIGESTS].insert_one(doc)
    logger.info(
        "revision digest created user=%s topic=%s misses=%s",
        user_id,
        topic_title,
        len(missed),
    )

    if notify:
        await send_push_notification(
            user_id,
            title=f"Revision: {topic_title}",
            body=(
                tips.bullets[0]
                if tips.bullets
                else "A few things to go over before you retry."
            ),
            data={
                "type": "learning_revision",
                "roadmapId": roadmap_id,
                "topicId": topicId,
                "digestId": str(res.inserted_id),
            },
        )

    doc["quiz"] = None
    return {**doc, "_id": str(res.inserted_id)}


async def escalate_to_reteach(
    user_id: str,
    roadmapId: str,
    topicId: Optional[str],
    quizId: str,
    missed: Optional[list] = None,
) -> Optional[dict]:
    """Re-explain the material behind a recall check the learner keeps failing.

    Triggered when the same check has been failed DIGEST_RETEACH_AFTER times.
    Two failures on one set of tips is a fact about the tips: the learner has
    read them twice and still can't answer, so sending them back to re-read the
    same words a third time is the one thing certain not to work.

    Adaptive on this side, fail-closed on theirs. The learner is not let through
    — the check still has to be passed. What changes is that its questions are
    rewritten against a fresh explanation, so passing is possible on the strength
    of teaching that might actually land, rather than on persistence.

    Returns the re-teach digest, or None if there was nothing to re-explain.
    """
    digests = await topic_digests(user_id, roadmapId, topicId)
    # The material the failed check covers: the window it was built from.
    blocked = next((d for d in digests if d.get("quizId") == quizId), None)
    window = digest_quiz_window(
        [d for d in digests if (d.get("sequence") or 0) < (blocked or {}).get("sequence", 0)]
        if blocked
        else digests
    )
    bullets = [b for d in window for b in d.get("bullets") or []]
    if not bullets:
        logger.info("no material to re-teach for quiz=%s", quizId)
        return None

    topic = find_topic(await fetch_roadmap(roadmapId, user_id), topicId) or {}
    topic_title = topic.get("title") or (blocked or {}).get("topicTitle") or ""

    # Their stated preference decides the angle — this is the one moment where
    # "how do you like things explained" has something concrete to change.
    style = None
    try:
        profile = await get_profile(user_id, LEARNING_NS)
        style = (profile or {}).get("preferred_explanation_style")
    except Exception as e:
        logger.error("reteach profile fetch failed user=%s: %s", user_id, e)

    try:
        retaught = await build_reteach(topic_title, bullets, style, missed)
    except Exception as e:
        logger.error("reteach generation failed quiz=%s: %s", quizId, e)
        return None
    if not retaught.bullets:
        return None

    # The check is rewritten against old material AND new, so it stays a test of
    # the same ideas while becoming answerable from the new explanation.
    try:
        rebuilt = await build_digest_quiz(topic_title, bullets + list(retaught.bullets))
        if rebuilt.quiz:
            await replace_quiz_questions(
                quizId, user_id, [q.model_dump() for q in rebuilt.quiz]
            )
    except Exception as e:
        # The re-explanation is the valuable half; keeping the old questions is
        # survivable, where losing the new teaching is not.
        logger.error("reteach quiz rebuild failed quiz=%s: %s", quizId, e)

    now = datetime.now(timezone.utc).isoformat()
    doc = {
        "user_id": user_id,
        "roadmapId": roadmapId,
        "topicId": topicId,
        "topicTitle": topic_title,
        # Outside the drip-feed: this covers ground already sent, so counting it
        # would shift where the recall-check cadence falls for what comes after.
        "sequence": None,
        "kind": "reteach",
        "style": style,
        "bullets": retaught.bullets,
        "resources": [],
        # The gate is the check on the digest this re-explains, not a second one.
        "quizId": None,
        "coverage_complete": False,
        "missing_outcomes": [],
        "status": "unread",
        "createdAt": now,
        "updatedAt": now,
    }
    res = await get_db()[DIGESTS].insert_one(doc)
    logger.info(
        "re-teach digest created user=%s topic=%s style=%s",
        user_id,
        topic_title,
        style or "analogy",
    )

    await send_push_notification(
        user_id,
        title=f"Another way to look at {topic_title}",
        body=(
            retaught.bullets[0]
            if retaught.bullets
            else "Let's come at this from a different angle."
        ),
        data={
            "type": "learning_reteach",
            "roadmapId": roadmapId,
            "topicId": topicId,
            "digestId": str(res.inserted_id),
        },
    )

    doc["quiz"] = None
    return {**doc, "_id": str(res.inserted_id)}


async def pull_next_digest(
    user_id: str, roadmapId: Optional[str] = None, topicId: Optional[str] = None
) -> dict:
    """Everything that decides what "give me the next lesson" produces.

    Extracted so the HTTP route and the tutor's `pull_next_lesson` tool cannot
    drift: the guards here are the ones that keep the drip-feed honest, and a
    second copy of them behind the chat path would be a second chance to get one
    of them wrong. The route turns a refusal into an HTTP status; the tool turns
    the same refusal into a sentence.

    Returns `{"digest": …}` on success, or `{"refused": <why>, "reason": <code>}`
    — never raises for an ordinary refusal, because "you can't have one yet" is
    an answer rather than a failure.
    """
    # The account-level spend cap, and the reason it lives here rather than on
    # the route: the route is not the only door. `pull_next_lesson` reaches this
    # function straight from a chat turn, which is capped only by
    # `learning_query` at 30/minute — so a cap sitting on the route left the most
    # expensive operation in the product running on the loosest budget in it.
    #
    # Counted before the roadmap is read. Every branch below can reach a web
    # search and two or three model calls, and a cap that starts counting after
    # the cheap refusals is one a caller can walk past by asking for a topic that
    # refuses.
    retry_after = await rate_limit.check_user(
        "learning_digest",
        user_id,
        LEARNING_DIGEST_RATE_LIMIT,
        LEARNING_DIGEST_RATE_WINDOW,
    )
    if retry_after is not None:
        return {
            "refused": (
                "That's a lot of lessons in one stretch — the next one can be "
                f"written in {retry_after}s."
            ),
            "reason": "rate_limited",
            "retry_after": retry_after,
        }

    roadmap = await fetch_roadmap(await resolve_roadmap_id(user_id, roadmapId), user_id)
    if not roadmap:
        return {"refused": "There's no active roadmap to pull a lesson from.",
                "reason": "no_roadmap"}

    # A topic owed revision after a failed checkpoint gets that instead. It's no
    # longer `in_progress`, so the teaching path below would find nothing to send
    # and report "nothing new" to a learner who is in fact blocked.
    owed = next(
        (t for t in roadmap.get("topics") or [] if revision_outstanding(t)), None
    )
    if owed and topicId in (None, owed["id"]):
        revision = await build_revision_digest(
            user_id, roadmap, owed["id"], notify=False
        )
        if revision:
            return {"digest": revision}
        return {
            "refused": "Your revision tips are already waiting — go over those first.",
            "reason": "needs_revision",
            "topicId": owed["id"],
        }

    # Checked here as well as inside `build_digest` only to say which check is
    # outstanding. `build_digest` returns a bare None, and "nothing new to send"
    # would send the learner looking for a digest that isn't the problem.
    topic = in_progress_topic(roadmap)
    if topic:
        gated_on = digest_quiz_gate(
            await topic_digest_states(user_id, str(roadmap["_id"]), topic.get("id"))
        )
        if gated_on:
            return {
                "refused": (
                    f"Pass the recall check on digest #{gated_on} first — "
                    "it's on the digest itself."
                ),
                "reason": "awaiting_quiz",
                "topicId": topic.get("id"),
                "sequence": gated_on,
            }

    digest = await build_digest(user_id, roadmap, topicId, notify=False)
    if not digest:
        return {
            "refused": (
                "Nothing new to send — finish the digest you already have, or the "
                "roadmap is complete."
            ),
            "reason": "nothing_to_send",
        }
    return {"digest": digest}


async def run_triggers(agent=None):
    """Hourly sweep: for every user who opted in via /toggle-trigger, fire only
    when the current hour matches their chosen schedule_hour in their timezone,
    then generate a digest for each active roadmap's current topic.

    A roadmap whose topic is owed revision after a failed checkpoint gets that
    instead — the topic is no longer being taught, so the teaching path would
    find nothing to send and the learner would sit blocked until they thought to
    ask for revision by hand.
    """
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
                owed = next(
                    (
                        t
                        for t in roadmap.get("topics") or []
                        if revision_outstanding(t)
                    ),
                    None,
                )
                if owed:
                    await build_revision_digest(user_id, roadmap, owed["id"])
                else:
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
