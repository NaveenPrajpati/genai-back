"""Practice: retrieval across topics, on a day with nothing waiting.

Everything else in this tracker is scheduled. A digest arrives, its check gates
the next one, a checkpoint gates the topic, and a review comes back on a ladder.
That is the right shape for teaching, and it leaves a learner who is caught up
with nothing to do — which is the state most days are in, and the one where the
habit is actually won or lost.

It is also the only practice in the app that **interleaves**. Every other check
sits inside one topic, and studying one thing at a time is the most comfortable
and least effective way to revise: it lets you recognise the answer from the
context rather than retrieve it. A deck that mixes five topics forces the
retrieval to start from the question.

Three rules hold here, and they follow from the rest of the product.

**Practice gates nothing.** It cannot complete a topic, cannot mark a digest
read, cannot advance a review. Passing is worth nothing except knowing you knew
it.

**Practice cannot cost anything either.** Attempts are recorded and feed the
misconception tracker — that is the whole diagnostic value — but they are kept
out of mastery. The same argument as the Feynman card: an exercise that can lower
your headline number is one nobody volunteers for, and a voluntary exercise
nobody volunteers for is not a feature.

**Answers are shown.** Withholding them exists so a failed checkpoint cannot be
transcribed into a passing retry. There is no retry to protect here.
"""

import logging
from typing import Optional

from langchain_core.prompts import ChatPromptTemplate

from app.core.llm import llm

from .repository import (
    asked_questions,
    get_misconceptions,
    list_roadmaps,
    roadmap_mastery,
)
from .state import PracticeDeckOutput

logger = logging.getLogger(__name__)

#: How many questions a deck holds. Small on purpose: this is the thing you do
#: when you have five spare minutes, and a deck that takes twenty is one that
#: gets postponed.
PRACTICE_QUESTIONS = 5

#: At most one question per topic until the deck runs out of topics, which is
#: what makes it interleaved rather than a quiz with a wide net.
MAX_TOPICS = PRACTICE_QUESTIONS

#: Roadmaps worth practising from. `archived` and `paused` are both deliberate
#: "not now" signals from the learner and are left alone; `completed` very much
#: is not — a finished roadmap is the richest source of things worth keeping.
PRACTICE_STATUSES = ("active", "completed")

#: A topic has to have been taught before there is anything to retrieve.
PRACTICE_TOPIC_STATUSES = ("completed", "needs_review")


async def practice_candidates(user_id: str) -> list[dict]:
    """Topics worth practising, most worth it first.

    Ranked by what the learner's own history says is shaky: a topic with named
    misconceptions leads, then one whose mastery has slipped, then one whose
    review is overdue, then everything else by age. Ties break toward the topic
    practised least recently, so a deck doesn't circle the same three.

    Returns the topic dicts enriched with `roadmapId`, `roadmapTitle` and the
    reason they were picked — the reason is not shown to the learner, but it is
    what the deck generator aims each question at.
    """
    try:
        roadmaps = await list_roadmaps(user_id, limit=100)
    except Exception as e:
        logger.error("practice_candidates roadmap fetch failed user=%s: %s", user_id, e)
        return []

    candidates: list[dict] = []
    for roadmap in roadmaps:
        if roadmap.get("status") not in PRACTICE_STATUSES:
            continue
        roadmap_id = str(roadmap["_id"])
        # One read for the whole roadmap rather than one per topic — the same
        # numbers the topic screen and the home screen show, from the same place.
        mastery_by_topic = await roadmap_mastery(user_id, roadmap)

        for topic in roadmap.get("topics") or []:
            if topic.get("progress_status") not in PRACTICE_TOPIC_STATUSES:
                continue

            mastery = mastery_by_topic.get(topic.get("id"))
            report = await get_misconceptions(user_id, roadmap_id, topic.get("id"))
            patterns = (report or {}).get("patterns") or []

            candidates.append(
                {
                    "roadmapId": roadmap_id,
                    "roadmapTitle": roadmap.get("title"),
                    "topic": topic,
                    "misconceptions": patterns,
                    "mastery": (mastery or {}).get("mastery"),
                    "overdue_days": (mastery or {}).get("overdue_days") or 0,
                    "attempts": (mastery or {}).get("attempts") or 0,
                }
            )

    def rank(c: dict) -> tuple:
        # Sorted ascending, so every term is negated where "more" means "sooner".
        # A topic with no mastery yet sorts as 100: never having been graded is
        # not evidence of weakness, and letting None read as 0 would put every
        # untested topic at the front of every deck.
        return (
            -len(c["misconceptions"]),
            c["mastery"] if c["mastery"] is not None else 100,
            -c["overdue_days"],
            c["attempts"],
        )

    candidates.sort(key=rank)
    return candidates


def spread(count: int, buckets: int) -> list[int]:
    """How many questions each topic gets. Even, remainder to the front — and the
    front is where the shakiest topics are."""
    if buckets <= 0:
        return []
    base, extra = divmod(count, buckets)
    return [base + (1 if i < extra else 0) for i in range(buckets)]


_PRACTICE_SYSTEM = """You are writing a short mixed practice set for a learner \
revising several topics at once.

Write exactly {count} multiple-choice questions, distributed across the topics \
below as specified. Each question must carry the `topic_index` of the topic it \
belongs to.

Topics:
{topics}

Rules:
- Ground every question in its own topic's outcomes. A question that needs \
material from another topic is unanswerable and reads as a trick.
- Four options, exactly one correct, and `answer` is its 0-based index.
- Wrong options must be plausible to someone who half-knows the topic. An \
obviously silly option turns a four-way question into a two-way one.
- Where a topic lists misunderstandings, aim its question at that belief. Target \
the belief, not the wording of any question they have seen.
- Say nothing about the learner's history. A question that announces itself as \
remedial tells them where to concentrate and reads as an accusation.
- `outcome` names what the question tests, in a few words. `hint` points at what \
to re-read WITHOUT stating, paraphrasing or positionally naming the answer.
- Do not repeat, or lightly reword, any question listed as already asked.

This is practice: it completes nothing and unlocks nothing. Write questions worth \
getting wrong — the point is to find out, not to award a pass."""


def _topic_block(candidates: list[dict], shares: list[int], asked: list[list[str]]) -> str:
    """The topics as the prompt sees them, indexed so questions can point back."""
    blocks = []
    for i, (c, share) in enumerate(zip(candidates, shares)):
        topic = c["topic"]
        outcomes = "\n".join(f"    - {o}" for o in topic.get("learning_outcomes") or [])
        lines = [
            f'[{i}] "{topic.get("title")}" (from {c["roadmapTitle"]}) — write {share} question(s)',
            f'    What it covers: {topic.get("description") or "—"}',
            f"    Outcomes:\n{outcomes}" if outcomes else "    Outcomes: (none given)",
        ]
        if c["misconceptions"]:
            patterns = "\n".join(
                f'    - {m.get("label")}: {m.get("detail")} (probe: {m.get("probe")})'.rstrip()
                for m in c["misconceptions"]
            )
            lines.append(f"    Known misunderstandings to aim at:\n{patterns}")
        if asked[i]:
            already = "\n".join(f"    - {q}" for q in asked[i][:12])
            lines.append(f"    Already asked (do not repeat):\n{already}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


async def build_practice_deck(user_id: str, count: int = PRACTICE_QUESTIONS) -> dict:
    """Assemble and write one interleaved practice deck.

    Returns `{"questions": [...], "topics": [...]}` where each question carries
    the `topicId` it was written for, or `{"refused": <why>}` when the learner has
    nothing to practise yet — which is an answer, not a failure: it means they
    have not finished a topic, and the thing to do is finish one.
    """
    candidates = await practice_candidates(user_id)
    if not candidates:
        return {
            "refused": (
                "Nothing to practise yet — this draws on topics you've already been "
                "taught, so finish one first."
            )
        }

    chosen = candidates[:MAX_TOPICS]
    shares = spread(count, len(chosen))
    asked = [
        await asked_questions(
            user_id, c["roadmapId"], c["topic"].get("id"), kinds=("checkpoint", "review", "practice")
        )
        for c in chosen
    ]

    chain = ChatPromptTemplate.from_messages(
        [("system", _PRACTICE_SYSTEM), ("human", "Write the practice set.")]
    ) | llm.with_structured_output(PracticeDeckOutput)

    try:
        result: PracticeDeckOutput = await chain.ainvoke(
            {"count": count, "topics": _topic_block(chosen, shares, asked)}
        )
    except Exception as e:
        logger.error("build_practice_deck generation failed user=%s: %s", user_id, e)
        return {"refused": "Couldn't put a practice set together just now. Try again shortly."}

    # `topic_index` is validated rather than trusted, exactly as `stage_order` is
    # on a roadmap draft: an index the model invented would file a question — and
    # the wrong answer that follows it — against a topic it isn't about, which is
    # how a misconception tracker learns something untrue.
    questions = []
    for q in result.questions:
        if not 0 <= q.topic_index < len(chosen):
            logger.warning("practice deck dropped a question with topic_index=%s", q.topic_index)
            continue
        if not q.options or not 0 <= q.answer < len(q.options):
            logger.warning("practice deck dropped a question with an unanswerable answer index")
            continue
        c = chosen[q.topic_index]
        questions.append(
            {
                "question": q.question,
                "options": q.options,
                "answer": q.answer,
                "kind": "choice",
                "outcome": q.outcome,
                "hint": q.hint,
                "roadmapId": c["roadmapId"],
                "topicId": c["topic"].get("id"),
                "topicTitle": c["topic"].get("title"),
            }
        )

    if not questions:
        return {"refused": "Couldn't put a practice set together just now. Try again shortly."}

    return {
        "questions": questions,
        "topics": [
            {
                "roadmapId": c["roadmapId"],
                "topicId": c["topic"].get("id"),
                "title": c["topic"].get("title"),
            }
            for c in chosen
        ],
    }
