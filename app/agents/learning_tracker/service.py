"""Cross-agent capability layer for the learning-tracker agent.

Mirrors personal_assistant.service: a thin, intent-free domain API other agents
(and the LT graph itself) can call. The PA hands off here when a user asks it to
"learn X" — but the PA must NOT invoke the LT *graph*, whose roadmap_agent pauses
on a human-approval `interrupt()`. That interrupt would propagate into the PA's
run. So roadmap *generation* lives here as a direct, interrupt-free operation;
the graph still wraps it with the approval step for the interactive path.
"""

import logging
from typing import Optional, List

from langchain_core.prompts import ChatPromptTemplate

from app.core.llm import llm
from app.agents.personal_assistant.service import TaskSpec, create_tasks as create_pa_tasks
from .state import RoadmapOutput
from .repository import insertRoadmapToDb

logger = logging.getLogger(__name__)

_NEW_ROADMAP_SYSTEM = (
    "You are an expert curriculum designer and learning path architect.\n"
    "Given a topic the user wants to learn, produce a complete, sequenced roadmap:\n"
    "1. Break the subject into ordered topics (order field starts at 1).\n"
    "2. For each topic list its prerequisites by title — only topics that appear earlier in the list.\n"
    "3. Group topics into broad stages (e.g. Foundations, Intermediate, Advanced).\n"
    "4. Estimate realistic study hours per topic and a total.\n"
    "5. Suggest 1-2 free learning resources (course names, docs, book titles) per topic.\n"
    "Personalize based on the exact subject in the user query. Be specific and practical.\n"
    "Learner profile (use to tailor depth, pacing, and resources):\n{memory}"
)


async def build_roadmap(topic: str, memory: Optional[dict] = None) -> RoadmapOutput:
    """Generate a fresh roadmap for `topic` (no persistence, no approval)."""
    chain = ChatPromptTemplate.from_messages(
        [("system", _NEW_ROADMAP_SYSTEM), ("human", "{text}")]
    ) | llm.with_structured_output(RoadmapOutput)
    return await chain.ainvoke({"text": topic, "memory": memory or "none"})


def roadmap_task_specs(roadmap: RoadmapOutput, roadmap_id: str) -> List[TaskSpec]:
    """Map a roadmap's topics to PA to-do specs. `source_ref` keys each task to
    its topic so re-runs (modify, resume-after-interrupt) don't duplicate."""
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
    roadmap = await build_roadmap(topic, memory)
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
