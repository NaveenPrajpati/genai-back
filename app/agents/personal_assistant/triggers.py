"""Daily task-digest job for the personal-assistant agent."""

import logging
import uuid
from datetime import datetime

from app.core.config import supabase
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
    now = datetime.now()
    try:
        triggers = (
            supabase.table("triggers")
            .select("*")
            .eq("enabled", True)
            .eq("action_type", "pa_digest")
            .execute()
        )
    except Exception as e:
        logger.error("run_pa_triggers fetch error: %s", e)
        return

    for t in triggers.data or []:
        try:
            pending = await fetch_todos(t["user_id"], status="pending")
            agenda = categorize_agenda(pending)
            digest = _compose_digest(agenda)
            supabase.table("approvals").insert(
                {
                    "user_id": t["user_id"],
                    "thread_id": str(uuid.uuid4()),
                    "action_type": "pa_digest",
                    "payload": {
                        "generated_at": now.isoformat(),
                        "pending_count": len(pending),
                        "digest": digest,
                        "counts": {k: len(v) for k, v in agenda.items()},
                        "tasks": pending,
                    },
                    "status": "pending",
                }
            ).execute()
            supabase.table("triggers").update({"last_run_at": now.isoformat()}).eq(
                "id", t["id"]
            ).execute()
            await send_push_notification(
                t["user_id"],
                title="Your daily agenda",
                body=digest,
                data={"type": "pa_digest", "pending_count": len(pending)},
            )
            logger.info(
                "pa digest created for user=%s (%d pending)",
                t["user_id"],
                len(pending),
            )
        except Exception as e:
            logger.error("pa digest error for user=%s: %s", t.get("user_id"), e)
