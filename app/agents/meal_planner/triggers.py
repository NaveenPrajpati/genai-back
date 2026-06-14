"""Scheduled meal-plan generation sweep for the meal-planner agent."""

import logging
import uuid
from datetime import datetime

from app.core.config import supabase
from .repository import get_monday

logger = logging.getLogger(__name__)


async def run_triggers(agent):
    logger.info("This job runs every sunday on 6:30 pm")
    now = datetime.now()
    week_start = get_monday()
    try:
        # Only meal-plan schedules — other features (e.g. personal_assistant)
        # share this table with their own action_type.
        triggers = (
            supabase.table("triggers")
            .select("*")
            .eq("enabled", True)
            .eq("action_type", "schedule")
            .execute()
        )
    except Exception as e:
        logger.error("run_triggers fetch error: %s", e)
        return

    for t in triggers.data or []:
        # Per-user isolation: one user's failure must not abort the whole sweep.
        try:
            thread_id = str(uuid.uuid4())

            # Check if user already has an approved plan to re-use
            latest = (
                supabase.table("meal_plans")
                .select("id, meal_slots(*)")
                .eq("user", t["user_id"])
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
                supabase.table("approvals").insert(
                    {
                        "user_id": t["user_id"],
                        "thread_id": thread_id,
                        "action_type": "save_plan",
                        "payload": {"week_start": week_start, "plan": proposed},
                        "status": "pending",
                    }
                ).execute()
                logger.info(
                    f"[trigger] Approval created for existing plan, user={t['user_id']}"
                )
            else:
                # No existing plan: invoke agent to generate one
                config = {"configurable": {"thread_id": thread_id}}
                agent_result = await agent.ainvoke(
                    {
                        "query": "Plan my meals for next week",
                        "user_id": t["user_id"],
                        "thread_id": thread_id,
                    },
                    config=config,
                )
                if "__interrupt__" in agent_result:
                    logger.info(
                        f"[trigger] New plan approval created, user={t['user_id']}"
                    )

            supabase.table("triggers").update({"last_run_at": now.isoformat()}).eq(
                "id", t["id"]
            ).execute()
        except Exception as e:
            logger.error(f"[trigger] error for user={t.get('user_id')}: {e}")
