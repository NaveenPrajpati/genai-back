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

from app.core.config import CHECKPOINT_QUESTIONS, DIGEST_QUIZ_QUESTIONS
from app.core.llm import llm, fast_llm
from app.agents.personal_assistant.service import (
    TaskSpec,
    create_tasks as create_pa_tasks,
)
from .state import CoverageOutput, QuizOutput, RoadmapDraft, RoadmapOutput
from .repository import insertRoadmapToDb, materialize_roadmap, profile_snapshot

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


async def build_checkpoint(
    topic: dict,
    roadmap_title: str,
    memory: Optional[dict] = None,
    is_review: bool = False,
) -> QuizOutput:
    """Generate a topic-scoped checkpoint quiz.

    Grounded in the topic's own description and learning outcomes so the
    questions can't wander into material the learner hasn't reached yet — the
    failure mode that would make a completion gate feel arbitrary.
    """
    outcomes = "\n".join(f"- {o}" for o in topic.get("learning_outcomes") or [])
    system = _CHECKPOINT_SYSTEM + (_REVIEW_NOTE if is_review else "")
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


_DIGEST_QUIZ_SYSTEM = (
    "You are checking whether a learner actually read the study tips they were "
    "sent about ONE topic.\n"
    "Write exactly {count} multiple-choice questions answerable purely from the "
    "tips below — nothing from outside them, and nothing about the newest tips, "
    "which they haven't acknowledged yet.\n"
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
