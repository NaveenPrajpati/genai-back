"""LangChain tools that let one agent's ReAct loop hand off to another.

These wrap the agents' cross-agent service APIs (personal_assistant.service,
learning_tracker.service) — never their graphs — so handoffs stay synchronous,
cheap, and free of the callee's HITL interrupts.

Each tool is built by a factory that binds `user_id` from graph state. The LLM
must not supply the user id (it would hallucinate one), so it is a closure
variable, not a tool argument.
"""

import logging
from typing import List, Optional

from langchain_core.tools import tool

from app.agents.personal_assistant.service import TaskSpec, create_tasks
from app.agents.learning_tracker.service import generate_roadmap

logger = logging.getLogger(__name__)


def make_push_tasks_to_pa_tool(user_id: str):
    """Tool: let the calling agent create to-dos in the personal assistant."""

    @tool
    async def push_tasks_to_personal_assistant(
        titles: List[str], details: Optional[List[str]] = None
    ) -> dict:
        """Add one or more to-dos to the user's personal-assistant task list so
        they're tracked and surfaced in their agenda. Pass a list of short task
        titles; optionally a parallel list of detail strings. Use this when work
        you produce should become trackable tasks for the user."""
        specs = [
            TaskSpec(
                title=t,
                details=(details[i] if details and i < len(details) else None),
            )
            for i, t in enumerate(titles)
        ]
        created = await create_tasks(user_id, specs, source="agent_handoff")
        return {"created": len(created), "titles": [c["title"] for c in created]}

    return push_tasks_to_personal_assistant


def make_start_learning_roadmap_tool(user_id: str):
    """Tool: let the calling agent (e.g. PA) hand off to the learning tracker."""

    @tool
    async def start_learning_roadmap(topic: str) -> dict:
        """Create a structured, sequenced learning roadmap for a topic the user
        wants to learn, and add its steps to their task list. Call this when the
        user expresses a goal to *learn* or *study* a subject (not just look up a
        fact). Returns the roadmap title, summary, and how many tasks were added."""
        result = await generate_roadmap(user_id, topic)
        logger.info("start_learning_roadmap handoff: topic=%r -> %s", topic, result.get("roadmapId"))
        return result

    return start_learning_roadmap
