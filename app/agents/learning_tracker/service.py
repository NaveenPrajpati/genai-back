"""Cross-agent capability layer for the learning-tracker agent.

Mirrors personal_assistant.service: a thin, intent-free domain API other agents
(and the LT graph itself) can call. The PA hands off here when a user asks it to
"learn X" — but the PA must NOT invoke the LT *graph*, whose roadmap_agent pauses
on a human-approval `interrupt()`. That interrupt would propagate into the PA's
run. So roadmap *generation* lives here as a direct, interrupt-free operation;
the graph still wraps it with the approval step for the interactive path.

Both roadmap prompts live here as the single source of truth — the graph imports
them rather than keeping the second copy that used to drift.
"""

import logging
from typing import Optional, List

from langchain_core.prompts import ChatPromptTemplate

from app.core.config import (
    CHECKPOINT_QUESTIONS,
    DIGEST_QUIZ_QUESTIONS,
    MISCONCEPTION_MAX_PATTERNS,
    MISCONCEPTION_MIN_EVIDENCE,
)
from app.core.llm import llm, fast_llm, get_llm
from app.agents.personal_assistant.service import (
    TaskSpec,
    create_tasks as create_pa_tasks,
)
from .state import (
    CoverageOutput,
    ExplanationJudgement,
    MisconceptionOutput,
    Question,
    QuizOutput,
    RoadmapDraft,
    RoadmapOutput,
)
from .repository import (
    get_misconceptions,
    insertRoadmapToDb,
    materialize_roadmap,
    profile_snapshot,
    save_misconceptions,
    topic_misses,
)

logger = logging.getLogger(__name__)

_NEW_ROADMAP_SYSTEM = (
    "You are an expert curriculum designer and learning path architect.\n"
    "Given a topic the user wants to learn, produce a complete, sequenced roadmap:\n"
    "1. Break the subject into ordered topics (order field starts at 1).\n"
    "2. Group topics into broad stages (e.g. Foundations, Intermediate, Advanced), "
    "each with its own order starting at 1. Set every topic's stage_order to the "
    "order of the stage it belongs to.\n"
    "3. For each topic list its prerequisites by title — only topics that appear "
    "earlier in the list.\n"
    "4. Estimate realistic study minutes per topic, and total hours for the roadmap.\n"
    "5. Suggest 1-2 free learning resources per topic, each with a title, a URL if "
    "you know one, and its resource_type.\n"
    "Personalize based on the exact subject in the user query. Be specific and practical.\n"
    "Learner profile (use to tailor depth, pacing, and resources):\n{memory}"
)

_MODIFY_ROADMAP_SYSTEM = (
    "You are an expert curriculum designer. The user wants to modify an existing "
    "learning roadmap.\n"
    "Apply the requested change (add topic, remove topic, reorder, adjust hours, "
    "update resources, etc.).\n"
    "Return the FULL updated roadmap — keep all unchanged topics intact.\n"
    "For every topic that already exists, copy its id into existing_id verbatim. "
    "That is how the learner's progress on that topic survives the edit: a topic "
    "returned without an existing_id is treated as brand new and starts from zero. "
    "Leave existing_id null only for topics you are genuinely adding.\n"
    "Maintain correct order values, stage_order links, and prerequisites after any "
    "structural change.\n"
    "Existing roadmap:\n{existing_roadmap}\n"
    "Learner profile (use to tailor depth, pacing, and resources):\n{memory}"
)


async def build_roadmap(topic: str, memory: Optional[dict] = None) -> RoadmapDraft:
    """Generate a fresh roadmap draft for `topic` (no ids, persistence, or approval)."""
    chain = ChatPromptTemplate.from_messages(
        [("system", _NEW_ROADMAP_SYSTEM), ("human", "{text}")]
    ) | llm.with_structured_output(RoadmapDraft)

    return await chain.ainvoke({"text": topic, "memory": memory or "none"})


async def revise_roadmap(
    request: str, existing_roadmap: dict, memory: Optional[dict] = None
) -> RoadmapDraft:
    """Apply the user's requested change to an existing roadmap, as a draft. The
    caller merges it onto the stored document — see repository.merge_roadmap."""
    chain = ChatPromptTemplate.from_messages(
        [("system", _MODIFY_ROADMAP_SYSTEM), ("human", "{text}")]
    ) | llm.with_structured_output(RoadmapDraft)
    return await chain.ainvoke(
        {
            "text": request,
            "existing_roadmap": existing_roadmap,
            "memory": memory or "none",
        }
    )


_CHECKPOINT_SYSTEM = (
    "You are writing a short active-recall checkpoint for ONE topic in a learning "
    "roadmap. The learner must pass it to mark the topic complete, so:\n"
    "- Test understanding and application, not trivia or wording recall.\n"
    "- Stay strictly within this topic. Do not ask about later topics.\n"
    "- Every question must be answerable from the topic's description and "
    "learning outcomes.\n"
    "- Exactly one option is correct, and `answer` is its 0-based index.\n"
    "- Make the wrong options plausible; an obviously silly distractor tests nothing.\n"
    "- Set `outcome` to the learning outcome or sub-skill the question checks.\n"
    "- Set `hint` to one sentence pointing at what to go back over. The learner "
    "sees the hint WITHOUT the answer when they get the question wrong, so it "
    "must not state or paraphrase the correct option, identify it by position, "
    "or rule the wrong ones out. 'Re-read how ownership transfers on assignment' "
    "is a hint; 'it's the one about borrowing' is not.\n"
    "Write exactly {count} questions.\n"
    "Topic: {title}\n"
    "Description: {description}\n"
    "Learning outcomes:\n{outcomes}\n"
    "Roadmap context: {roadmap_title}\n"
    "Pitch the difficulty to the learner's profile:\n{memory}"
)

_REVIEW_NOTE = (
    "\nThis is a spaced-repetition REVIEW of a topic the learner completed "
    "earlier. Ask about it from a different angle than a first-pass quiz would — "
    "the goal is durable recall, not recognising a question they've already seen."
)

_AVOID_NOTE = (
    "\nThis learner has already been asked the following on this topic. Do not "
    "ask any of them again, and do not ask a reworded version of one — the same "
    "question in new clothes tests whether they remember the answer, which is "
    "exactly what a review must not measure:\n{asked}\n"
    "Come at the same outcomes from angles these did not cover: apply it to a "
    "scenario, compare it with a neighbouring idea, ask what breaks if it is done "
    "wrong, or work from a concrete case back to the rule. If the topic is small "
    "enough that you have genuinely run out of distinct questions, change the "
    "setting and the specifics rather than repeating one."
)

_MISCONCEPTION_NOTE = (
    "\nThis learner has a history on this topic. These are the misunderstandings "
    "their past wrong answers point to:\n{patterns}\n"
    "Aim at least one question squarely at each, up to half the set — a checkpoint "
    "that only re-probes old mistakes stops covering the topic. Target the belief, "
    "not the wording: reusing a question they've already seen tests whether they "
    "remember being caught, not whether they now understand.\n"
    "Say nothing about their history in the question text. A question that "
    "announces itself as remedial tells them where to concentrate and reads as an "
    "accusation."
)


async def build_checkpoint(
    topic: dict,
    roadmap_title: str,
    memory: Optional[dict] = None,
    is_review: bool = False,
    misconceptions: Optional[List[dict]] = None,
    asked: Optional[List[str]] = None,
) -> QuizOutput:
    """Generate a topic-scoped checkpoint quiz.

    Grounded in the topic's own description and learning outcomes so the
    questions can't wander into material the learner hasn't reached yet — the
    failure mode that would make a completion gate feel arbitrary.

    `misconceptions` are the patterns inferred from this learner's previous wrong
    answers across every quiz kind. Passing them makes a retry a genuinely
    different test: same outcomes, aimed at what they actually got wrong, rather
    than a fresh roll of the dice or a repeat of the set they just failed.

    `asked` is what this learner has already been shown on this topic. Without
    it, regenerating produces near-identical questions — the same description and
    outcomes lead the model to the same obvious few every time, which over a
    spaced-repetition ladder measures memory of the answer, not the material.
    """
    outcomes = "\n".join(f"- {o}" for o in topic.get("learning_outcomes") or [])
    system = _CHECKPOINT_SYSTEM + (_REVIEW_NOTE if is_review else "")
    if asked:
        system += _AVOID_NOTE.format(asked="\n".join(f"- {a}" for a in asked))
    if misconceptions:
        system += _MISCONCEPTION_NOTE.format(
            patterns="\n".join(
                f"- {m.get('label')}: {m.get('detail')} "
                f"(probe: {m.get('probe')})".strip()
                for m in misconceptions
            )
        )
    chain = ChatPromptTemplate.from_messages(
        [("system", system), ("human", "Write the checkpoint.")]
    ) | llm.with_structured_output(QuizOutput)
    return await chain.ainvoke(
        {
            "count": CHECKPOINT_QUESTIONS,
            "title": topic.get("title", ""),
            "description": topic.get("description", ""),
            "outcomes": outcomes or "- (none given)",
            "roadmap_title": roadmap_title or "general",
            "memory": memory or "none",
        }
    )


_FEYNMAN_SYSTEM = (
    "You are judging whether a learner understands a topic, from how they "
    "explain it in their own words. This is the Feynman test: someone who has "
    "understood something can say it plainly.\n"
    "Grade against the learning outcomes below and nothing else. For each "
    "outcome say whether their explanation is `solid`, `partial`, `missing` "
    "(never addressed) or `wrong` (addressed incorrectly), quoting the words that "
    "led you there.\n"
    "Judge the understanding, not the performance. Rambling, informal phrasing, "
    "false starts, filler and bad grammar are irrelevant — this may have been "
    "spoken aloud. An analogy that carries the idea is worth more than textbook "
    "vocabulary used without meaning. Do not reward someone for reciting the "
    "outcome back at you: parroting a phrase they were given is the thing this "
    "test exists to see through.\n"
    "`score` is the proportion of the outcomes genuinely conveyed, weighted by "
    "how central each is to the topic.\n"
    "Fill `misconceptions` with beliefs the explanation shows to be wrong — the "
    "reason this is worth grading at all. Only what they actually said supports; "
    "an outcome they simply skipped is a gap, not a misconception, and inventing "
    "one from silence would put a future checkpoint on the trail of a problem "
    "they may not have.\n"
    "Write `feedback` to the learner in the second person, two sentences at most: "
    "the strongest thing they showed, then the one thing to shore up.\n"
    "Topic: {topic}\n"
    "Description: {description}\n"
    "Learning outcomes:\n{outcomes}\n"
    "Their explanation:\n{explanation}"
)


async def judge_explanation(
    topic: dict, explanation: str
) -> ExplanationJudgement:
    """Grade a learner's own-words explanation of a topic against its outcomes.

    The counterpart to the multiple-choice checkpoint, and the only place the
    system sees a learner's actual reasoning. Everything else it knows about
    someone's understanding is inferred from which of four options they tapped.
    """
    outcomes = "\n".join(f"- {o}" for o in topic.get("learning_outcomes") or [])
    chain = ChatPromptTemplate.from_messages(
        [("system", _FEYNMAN_SYSTEM), ("human", "Judge the explanation.")]
    ) | llm.with_structured_output(ExplanationJudgement)
    return await chain.ainvoke(
        {
            "topic": topic.get("title", ""),
            "description": topic.get("description", ""),
            "outcomes": outcomes or "- (none given)",
            "explanation": explanation,
        }
    )


_ONELINER_SYSTEM = (
    "Write ONE open question about the study tips below. The learner answers it "
    "in a single typed sentence, so it must be answerable in one — ask for a "
    "definition in their own words, the reason something is done, or when they "
    "would reach for it.\n"
    "It must be answerable from the tips alone. Do not ask them to recall a name, "
    "a number, or exact wording: this exists to show how they are thinking, and a "
    "fact they either remember or don't shows nothing.\n"
    "Set `expected` to what a correct one-sentence answer has to convey — the "
    "idea, not a form of words. The learner never sees it.\n"
    "Set `kind` to \"open\", leave `options` empty, and set `outcome` to what the "
    "question checks.\n"
    "Topic: {topic}\n"
    "Tips already sent:\n{bullets}"
)


async def build_oneliner(topic_title: str, bullets: List[str]) -> Question:
    """The single free-text question that rides along on later digest checks."""
    chain = ChatPromptTemplate.from_messages(
        [("system", _ONELINER_SYSTEM), ("human", "Write the question.")]
    ) | fast_llm.with_structured_output(Question)
    q = await chain.ainvoke({"topic": topic_title, "bullets": "\n".join(f"- {b}" for b in bullets)})
    # The model is asked for these but structured output will happily return a
    # `choice` with options; force the shape the grader depends on.
    q.kind = "open"
    q.options = []
    return q


_ONELINER_GRADE_SYSTEM = (
    "A learner answered one open question about study tips they were sent, in a "
    "single sentence. Decide whether their answer conveys what it needed to.\n"
    "Be generous about form and strict about meaning. Spelling, grammar, brevity "
    "and informality do not matter; whether they have the idea does. Accept a "
    "correct answer phrased differently from the expected one. Reject an answer "
    "that restates the question, says nothing specific, or is obviously a "
    "placeholder.\n"
    "If it is wrong, `misconceptions` should say what they appear to believe — "
    "that is the point of asking in prose rather than offering four options.\n"
    "Question: {question}\n"
    "What a correct answer must convey: {expected}\n"
    "Their answer: {answer}"
)


async def grade_oneliner(
    question: str, expected: Optional[str], answer: str
) -> ExplanationJudgement:
    """Grade one free-text sentence. Same shape as the Feynman judgement so both
    feed the misconception tracker through one path."""
    chain = ChatPromptTemplate.from_messages(
        [("system", _ONELINER_GRADE_SYSTEM), ("human", "Grade it.")]
    ) | fast_llm.with_structured_output(ExplanationJudgement)
    return await chain.ainvoke(
        {
            "question": question,
            "expected": expected or "(not recorded — judge against the question)",
            "answer": answer,
        }
    )


_MISCONCEPTION_SYSTEM = (
    "You are diagnosing a learner from their wrong answers on ONE topic.\n"
    "Below is every question they got wrong, what they picked, the right answer, "
    "and which kind of check it came from — a `digest` recall check, a "
    "`checkpoint` they must pass to complete the topic, or a `review` of material "
    "they finished earlier and are revisiting.\n"
    "Find at most {count} RECURRING misunderstandings. A misunderstanding is a "
    "belief that explains several wrong answers at once — not a restatement of "
    "one. If the mistakes have nothing in common, return fewer patterns, or none: "
    "a learner having one bad day is not a misconception, and inventing one sends "
    "every future checkpoint chasing something that was never there.\n"
    "Weight the evidence. A mistake repeated across different kinds of check, or "
    "one that persists into a `review`, is a real gap; a single early `digest` "
    "slip probably isn't. Prefer what they got wrong most recently — a belief "
    "they have since corrected must not keep steering their questions.\n"
    "For each pattern give: `label`, a short name; `detail`, what they appear to "
    "believe and what is actually true; `probe`, how a future question could "
    "re-test that belief from a different angle; and `evidence`, the 0-based "
    "indices of the misses that support it.\n"
    "Topic: {topic}\n"
    "Their wrong answers:\n{misses}"
)


async def extract_misconceptions(
    topic_title: str, misses: List[dict]
) -> MisconceptionOutput:
    """Read a topic's wrong answers for the belief underneath them.

    Deliberately allowed to return nothing. The useful output here is a pattern
    that survives contact with several mistakes; a model pressed to always
    produce one will paraphrase the most recent miss and call it a diagnosis,
    which then steers checkpoints at a problem the learner doesn't have.
    """
    rendered = "\n".join(
        f"{i}. [{m.get('kind') or 'quiz'}] {m.get('question')}\n"
        f"   picked: {m.get('chosen') or '(left blank)'}\n"
        f"   correct: {m.get('correct') or '(unknown)'}"
        for i, m in enumerate(misses)
    )
    chain = ChatPromptTemplate.from_messages(
        [("system", _MISCONCEPTION_SYSTEM), ("human", "Diagnose.")]
    ) | llm.with_structured_output(MisconceptionOutput)
    return await chain.ainvoke(
        {
            "count": MISCONCEPTION_MAX_PATTERNS,
            "topic": topic_title,
            "misses": rendered,
        }
    )


async def refresh_misconceptions(
    user_id: str, roadmapId: str, topicId: str, topic_title: str
) -> Optional[list[dict]]:
    """Re-read a topic's wrong answers if there's new evidence worth reading.

    Runs off the request path (a background task after grading), so the checkpoint
    that consumes it only ever does a cached read. Skips when the evidence is too
    thin to generalise from, and when nothing has been added since last time —
    re-deriving the same patterns from the same misses is a wasted LLM call.
    """
    misses = await topic_misses(user_id, roadmapId, topicId)
    if len(misses) < MISCONCEPTION_MIN_EVIDENCE:
        return None

    stored = await get_misconceptions(user_id, roadmapId, topicId)
    if stored and stored.get("misses_analyzed") == len(misses):
        return stored.get("patterns")

    try:
        out = await extract_misconceptions(topic_title, misses)
    except Exception as e:
        logger.error("misconception analysis failed topic=%s: %s", topicId, e)
        return stored.get("patterns") if stored else None

    patterns = [p.model_dump() for p in out.patterns]
    await save_misconceptions(user_id, roadmapId, topicId, patterns, len(misses))
    logger.info(
        "misconceptions for topic=%s: %s pattern(s) from %s miss(es)",
        topicId,
        len(patterns),
        len(misses),
    )
    return patterns


_DIGEST_QUIZ_SYSTEM = (
    "You are checking whether a learner actually read the study tips they were "
    "sent about ONE topic.\n"
    "Write up to {count} multiple-choice questions answerable purely from the "
    "tips below — nothing from outside them, and nothing about the newest tips, "
    "which they haven't acknowledged yet.\n"
    "Question the IDEA in a tip, never its provenance. Do not ask which book, "
    "course, site, channel or author something came from, where to learn the "
    "topic, or what a named tutorial covers — knowing that Wes McKinney wrote a "
    "book is not knowing pandas, and a check made of those questions measures "
    "nothing about the material.\n"
    "Some tips may only point at a resource rather than teach anything. Skip "
    "those: there is nothing in them to have understood. If that leaves too "
    "little to ask about, write fewer questions, or none at all — a short check "
    "beats a check about trivia.\n"
    "Keep them short and concrete. Exactly one option is correct, and `answer` is "
    "its 0-based index. Make the wrong options plausible.\n"
    "Topic: {topic}\n"
    "Tips already sent:\n{bullets}"
)

_COVERAGE_SYSTEM = (
    "You are deciding whether a drip-feed of study tips has finished teaching a "
    "topic.\n"
    "Given the topic's description and its learning outcomes, judge whether the "
    "tips sent so far substantively cover ALL of the outcomes.\n"
    "Set covered=true only if a learner who absorbed these tips could meet every "
    "outcome. List any outcomes still untouched in `missing` — those are what the "
    "next tips should be about.\n"
    "Be strict: saying a topic is covered ends the drip-feed and sends the learner "
    "to a graded checkpoint.\n"
    "Topic: {topic}\n"
    "Description: {description}\n"
    "Learning outcomes:\n{outcomes}\n"
    "Tips sent so far:\n{bullets}"
)


async def build_digest_quiz(
    topic_title: str,
    bullets: List[str],
    questioncount: Optional[int] = DIGEST_QUIZ_QUESTIONS,
) -> QuizOutput:
    """A short recall check over tips the learner has already been sent."""
    chain = ChatPromptTemplate.from_messages(
        [("system", _DIGEST_QUIZ_SYSTEM), ("human", "Write the check.")]
    ) | fast_llm.with_structured_output(QuizOutput)
    return await chain.ainvoke(
        {
            "count": 4 if questioncount > 4 else questioncount,
            "topic": topic_title,
            "bullets": "\n".join(f"- {b}" for b in bullets),
        }
    )


async def check_coverage(topic: dict, bullets: List[str]) -> CoverageOutput:
    """Have the digests so far taught the whole topic?

    Judged against the topic's own learning outcomes rather than a digest count,
    so a topic with two outcomes isn't dragged out to the same length as one with
    eight.
    """
    outcomes = topic.get("learning_outcomes") or []
    if not outcomes:
        # Nothing concrete to measure against — don't declare victory on a guess.
        return CoverageOutput(covered=False, missing=[])

    chain = ChatPromptTemplate.from_messages(
        [("system", _COVERAGE_SYSTEM), ("human", "Judge the coverage.")]
    ) | llm.with_structured_output(CoverageOutput)
    return await chain.ainvoke(
        {
            "topic": topic.get("title", ""),
            "description": topic.get("description", ""),
            "outcomes": "\n".join(f"- {o}" for o in outcomes),
            "bullets": "\n".join(f"- {b}" for b in bullets) or "- (none yet)",
        }
    )


def roadmap_task_specs(roadmap: RoadmapOutput, roadmap_id: str) -> List[TaskSpec]:
    """Map a roadmap's topics to PA to-do specs. `source_ref` keys each task to
    its topic so re-runs (modify, resume-after-interrupt) don't duplicate — which
    only holds because topic ids are server-minted and survive a modify."""
    return [
        TaskSpec(
            title=f"Learn: {topic.title}",
            details=topic.description,
            source_ref=f"{roadmap_id}:{topic.id}",
        )
        for topic in roadmap.topics
    ]


async def sync_roadmap_to_pa(
    user_id: Optional[str], roadmap: RoadmapOutput, roadmap_id: Optional[str]
) -> int:
    """Push a roadmap's topics into the PA as tracked to-dos. Returns the count
    of newly created tasks (deduped by source_ref)."""
    if not roadmap_id:
        return 0
    created = await create_pa_tasks(
        user_id, roadmap_task_specs(roadmap, roadmap_id), source="learning_tracker"
    )
    return len(created)


async def generate_roadmap(
    user_id: Optional[str], topic: str, memory: Optional[dict] = None
) -> dict:
    """Full cross-agent entry point: build a roadmap, persist it, and sync its
    topics to the PA's to-do list. Skips the interactive HITL approval (the
    caller is another agent, not the LT chat flow). Returns a compact summary."""
    roadmap = materialize_roadmap(
        await build_roadmap(topic, memory), personalization=profile_snapshot(memory)
    )
    roadmap_id = await insertRoadmapToDb(roadmap, user_id)
    tasks_created = await sync_roadmap_to_pa(user_id, roadmap, roadmap_id)
    logger.info(
        "generate_roadmap: topic=%r roadmapId=%s topics=%d pa_tasks=%d",
        topic,
        roadmap_id,
        len(roadmap.topics),
        tasks_created,
    )
    return {
        "roadmapId": roadmap_id,
        "title": roadmap.title,
        "summary": roadmap.summary,
        "topic_count": len(roadmap.topics),
        "pa_tasks_created": tasks_created,
    }
