"""Daily task-digest job for the personal-assistant agent."""

import logging
import uuid
from datetime import datetime, timezone

from app.agents.trigger_store import due_triggers, mark_ran
from app.agents.approval_store import create_pending
from app.services.push_service import send_push_notification
from .repository import fetch_todos, categorize_agenda

logger = logging.getLogger(__name__)


def _compose_digest(agenda: dict) -> str:
    """Build a friendly, prioritized digest line from a categorized agenda."""
    overdue = agenda.get("overdue", [])
    today = agenda.get("today", [])
    upcoming = agenda.get("upcoming", [])
    if not (overdue or today or upcoming or agenda.get("no_date")):
        return "You're all caught up — no pending tasks. 🎉"
    segments = []
    if overdue:
        names = ", ".join(t.get("title", "") for t in overdue[:5])
        segments.append(f"⚠️ {len(overdue)} overdue: {names}")
    if today:
        names = ", ".join(t.get("title", "") for t in today[:5])
        segments.append(f"📅 {len(today)} due today: {names}")
    if upcoming:
        names = ", ".join(t.get("title", "") for t in upcoming[:5])
        segments.append(f"⏳ {len(upcoming)} upcoming: {names}")
    return " | ".join(segments)


async def run_pa_triggers(agent=None):
    """For each enabled pa_digest trigger, snapshot the user's pending tasks into
    an approvals-table notification the user can review via GET /approve."""
    logger.info("pa digest job running")
    now = datetime.now(timezone.utc)
    try:
        triggers = await due_triggers("pa_digest", now)
    except Exception as e:
        logger.error("run_pa_triggers fetch error: %s", e)
        return

    for t in triggers:
        user_id = t.get("user_id")
        try:
            pending = await fetch_todos(user_id, status="pending")
            agenda = categorize_agenda(pending)
            digest = _compose_digest(agenda)
            await create_pending(
                user_id,
                str(uuid.uuid4()),
                "pa_digest",
                {
                    "generated_at": now.isoformat(),
                    "pending_count": len(pending),
                    "digest": digest,
                    "counts": {k: len(v) for k, v in agenda.items()},
                    "tasks": pending,
                },
            )
            await mark_ran(t, now)
            await send_push_notification(
                user_id,
                title="Your daily agenda",
                body=digest,
                data={"type": "pa_digest", "pending_count": len(pending)},
            )
            logger.info(
                "pa digest created for user=%s (%d pending)",
                user_id,
                len(pending),
            )
        except Exception as e:
            logger.error("pa digest error for user=%s: %s", user_id, e)
