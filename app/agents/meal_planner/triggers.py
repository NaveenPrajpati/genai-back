"""Scheduled meal-plan generation sweep for the meal-planner agent.

Opt-in state lives in the shared Mongo `triggers` collection (action_type
"schedule"); meal plans and approvals stay in Supabase. The scheduler sweeps
hourly and trigger_store.is_due fires each user only at their chosen local
schedule_hour/schedule_dow in their timezone (default Sunday 18:00 UTC)."""

import logging
import uuid
from datetime import datetime, timezone

from app.core.config import supabase
from app.agents.trigger_store import due_triggers, mark_ran
from app.agents.approval_store import create_pending
from app.services.push_service import send_push_notification
from .repository import get_monday

logger = logging.getLogger(__name__)


async def run_triggers(agent):
    logger.info("meal-plan trigger sweep running")
    now = datetime.now(timezone.utc)
    week_start = get_monday()
    try:
        triggers = await due_triggers("schedule", now)
    except Exception as e:
        logger.error("run_triggers fetch error: %s", e)
        return

    for t in triggers:
        user_id = t.get("user_id")
        # Per-user isolation: one user's failure must not abort the whole sweep.
        try:
            thread_id = str(uuid.uuid4())

            # Check if user already has an approved plan to re-use
            latest = (
                supabase.table("meal_plans")
                .select("id, meal_slots(*)")
                .eq("user", user_id)
                .eq("status", "approved")
                .order("created_at", desc=True)
                .limit(1)
                .maybe_single()
                .execute()
            )

            if latest and latest.data:
                # Re-use existing plan: create approval directly without LLM
                slots = latest.data.get("meal_slots", [])
                proposed = [
                    {
                        "plan_id": latest.data["id"],
                        "day_of_week": s["day_of_week"],
                        "meal_type": s["meal_type"],
                        "recipe_name": s["recipe_name"],
                        "protein_g": s["protein_g"],
                    }
                    for s in slots
                ]
                await create_pending(
                    user_id,
                    thread_id,
                    "save_plan",
                    {"week_start": week_start, "plan": proposed},
                )
                logger.info(
                    f"[trigger] Approval created for existing plan, user={user_id}"
                )
                await send_push_notification(
                    user_id,
                    title="Next week's meal plan is ready",
                    body=f"Review your plan for the week of {week_start}.",
                    data={"type": "save_plan", "week_start": week_start},
                )
            else:
                # No existing plan: invoke agent to generate one
                config = {"configurable": {"thread_id": thread_id}}
                agent_result = await agent.ainvoke(
                    {
                        "query": "Plan my meals for next week",
                        "user_id": user_id,
                        "thread_id": thread_id,
                    },
                    config=config,
                )
                if "__interrupt__" in agent_result:
                    logger.info(f"[trigger] New plan approval created, user={user_id}")
                    await send_push_notification(
                        user_id,
                        title="Next week's meal plan is ready",
                        body=f"Review your plan for the week of {week_start}.",
                        data={"type": "save_plan", "week_start": week_start},
                    )

            await mark_ran(t, now)
        except Exception as e:
            logger.error(f"[trigger] error for user={user_id}: {e}")
