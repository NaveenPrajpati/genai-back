"""Cross-agent capability layer for the personal-assistant agent.

Other agents (learning_tracker, meal_planner) collaborate with the PA by
*producing* tasks the PA owns and tracks. They must NOT invoke the PA graph for
this: that path runs intent classification + synthesis (extra LLM cost) and
brings the PA's own thread/checkpointer and HITL interrupts into the caller's
run. Instead they call this thin, intent-free domain API directly.

Every task created here carries provenance (`source`, `source_ref`) so the
origin is traceable and re-runs are idempotent — important because the agents
that call this (e.g. roadmap_agent) are replayed from the top when a graph
resumes from an interrupt.
"""

import logging
from typing import Optional, List, Literal

from pydantic import BaseModel

from app.database import get_db
from .repository import TODOS, insert_todo, _serialize

logger = logging.getLogger(__name__)


class TaskSpec(BaseModel):
    """A single to-do another agent wants the PA to track."""

    title: str
    details: Optional[str] = None
    priority: Literal["low", "medium", "high"] = "medium"
    due_at: Optional[str] = None  # ISO date
    parent_id: Optional[str] = None
    # Stable identifier of the originating item (e.g. a roadmap topic id), used
    # to skip re-creating the same task when the calling graph replays.
    source_ref: Optional[str] = None


async def create_tasks(
    user_id: str,
    tasks: List[TaskSpec],
    source: str,
) -> List[dict]:
    """Create PA to-dos on behalf of another agent.

    `source` is the originating agent (e.g. "learning_tracker"). Tasks whose
    `source_ref` already exists for this user+source are skipped, so calling
    this twice for the same roadmap is a no-op for already-created topics.
    Returns the tasks that were newly created (serialized).
    """
    refs = [t.source_ref for t in tasks if t.source_ref]
    existing_refs: set[str] = set()
    if refs:
        cursor = get_db()[TODOS].find(
            {"user_id": user_id, "source": source, "source_ref": {"$in": refs}},
            {"source_ref": 1},
        )
        existing_refs = {d["source_ref"] async for d in cursor}

    created: List[dict] = []
    for t in tasks:
        if t.source_ref and t.source_ref in existing_refs:
            continue
        doc = await insert_todo(
            user_id,
            {
                **t.model_dump(exclude_none=True),
                "source": source,
            },
        )
        created.append(doc)

    logger.info(
        "create_tasks: source=%s requested=%d created=%d (user=%s)",
        source,
        len(tasks),
        len(created),
        user_id,
    )
    return created
