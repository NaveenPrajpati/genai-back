"""Scheduled follow-up sweep for the job helper.

An application sent and then forgotten is the failure mode this exists for: the
user applied, heard nothing, and never nudged. The sweep finds applications that
have sat at `applied` past the follow-up window and turns each into a to-do on
their existing task list, plus one push notification.

Opt-in state lives in the shared Mongo `triggers` collection (action_type
"job_followup"). The scheduler sweeps hourly and trigger_store.is_due fires each
user only at their chosen local schedule_hour in their timezone (default 09:00
daily).

NO MODEL CALLS
──────────────
Everything here is a date comparison and a task insert. A nudge that costs an
LLM call per application per day would be the most expensive trigger in the app
and would say the same sentence every time.
"""

import logging
from datetime import datetime, timedelta, timezone

from app.agents.trigger_store import due_triggers, mark_ran
from app.services.push_service import send_push_notification

from . import repository as repo
from .service import push_followups

logger = logging.getLogger(__name__)

TRIGGER_TYPE = "job_followup"

# Days at `applied` with no movement before a nudge. Two weeks is the point past
# which silence usually is the answer, and a follow-up still reads as interest
# rather than impatience.
FOLLOWUP_AFTER_DAYS = 14

# How many nudges one sweep may raise for a user. Someone who applied to thirty
# roles in one week does not need thirty to-dos on the same morning.
MAX_NUDGES_PER_RUN = 5


async def run_job_triggers(agent=None) -> None:
    """Hourly sweep. `agent` is accepted and unused: the scheduler passes every
    trigger the compiled graph, and this one needs no inference to do its job."""
    logger.info("job follow-up sweep running")
    now = datetime.now(timezone.utc)
    cutoff = (now - timedelta(days=FOLLOWUP_AFTER_DAYS)).isoformat()

    try:
        triggers = await due_triggers(TRIGGER_TYPE, now)
    except Exception as e:
        logger.error("job trigger fetch error: %s", e)
        return

    for trigger in triggers:
        user_id = trigger.get("user_id")
        # Per-user isolation: one user's failure must not abort the whole sweep.
        try:
            stale = await repo.stale_applications(user_id, cutoff)
            # Already nudged applications stay in the query's results — their
            # status has not changed — so the marker, not the query, is what
            # stops a daily repeat of the same to-do.
            pending = [a for a in stale if not a.get("followup_nudged_at")][
                :MAX_NUDGES_PER_RUN
            ]
            for application in pending:
                label = " at ".join(
                    b
                    for b in (application.get("role_title"), application.get("company"))
                    if b
                ) or "a job you applied to"
                await push_followups(
                    user_id,
                    application["id"],
                    [f"Follow up on {label}"],
                )
                await repo.update_application(
                    user_id, application["id"], {"followup_nudged_at": now.isoformat()}
                )

            if pending:
                first = pending[0]
                body = (
                    f"{first.get('company') or 'One application'} has been quiet for "
                    f"{FOLLOWUP_AFTER_DAYS} days."
                    if len(pending) == 1
                    else f"{len(pending)} applications have been quiet for "
                    f"{FOLLOWUP_AFTER_DAYS} days."
                )
                await send_push_notification(
                    user_id,
                    title="Time to follow up",
                    body=body,
                    data={"type": TRIGGER_TYPE, "count": len(pending)},
                )
                logger.info(
                    "[job trigger] %d follow-up(s) raised for user=%s",
                    len(pending),
                    user_id,
                )

            await mark_ran(trigger, now)
        except Exception as e:
            logger.error("[job trigger] error for user=%s: %s", user_id, e)
