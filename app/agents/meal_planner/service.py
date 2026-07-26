"""Cross-agent capability layer for the meal-planner agent.

Mirrors personal_assistant.service and learning_tracker.service: a thin,
intent-free domain API. This one has a second consumer beyond other agents —
app/mcp/meal_server.py publishes these functions as MCP tools, so the meal
planner is reachable by any MCP client (the supervisor graph, Claude Desktop,
the MCP Inspector).

Plan *generation* and plan *persistence* are deliberately two calls
(`build_plan` then `save_plan`). MCP is stateless request/response with nowhere
to pause, so a human-approval step cannot live inside a tool call: the caller
proposes, gets the proposal approved however it likes (the supervisor graph uses
a LangGraph `interrupt()`), and only then saves. The graph's own plan_agent
follows the same split, so both paths share one implementation.
"""

import logging
from typing import Optional, List

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

from app.core.config import supabase
from app.core.llm import llm
from app.agents.react import run_tool_loop
from .state import PlanOutput, ResearchOutput
from .repository import remember

logger = logging.getLogger(__name__)

_PLAN_SYSTEM = (
    "You are an expert at planning diet plan , so plan diet for week my week start from monday means monday is day of week 0 ,recipes for dinner, lunch, breakfast for all days of week along with protien in grams in each meal "
    "User diet: {diet}. Disliked: {disliked}.\n"
    "Learned preferences (honor these strictly) — diet restrictions: "
    "{diet_restrictions}; allergies: {allergies}; preferred cuisines: "
    "{preferred_cuisines}; household size: {household_size}; cooking "
    "skill: {cooking_skill}; nutrition goals: {nutrition_goals}.\n"
)


async def build_plan(
    request: str,
    current_user: Optional[dict] = None,
    memory: Optional[dict] = None,
) -> List[dict]:
    """Generate a weekly plan proposal. Pure generation — nothing is persisted
    and no approval is sought. Returns a list of meal-slot dicts."""
    current_user = current_user or {}
    memory = memory or {}
    chain = ChatPromptTemplate.from_messages(
        [("system", _PLAN_SYSTEM), ("human", "{text}")]
    ) | llm.with_structured_output(PlanOutput)
    result: PlanOutput = await chain.ainvoke(
        {
            "text": request,
            "diet": current_user.get("diet", "vegetarian"),
            "disliked": memory.get("disliked_dishes", []),
            "diet_restrictions": memory.get("diet_restrictions") or "none",
            "allergies": memory.get("allergies") or "none",
            "preferred_cuisines": memory.get("preferred_cuisines") or "none",
            "household_size": memory.get("household_size") or "unknown",
            "cooking_skill": memory.get("cooking_skill") or "unknown",
            "nutrition_goals": memory.get("nutrition_goals") or "none",
        }
    )
    return [slot.model_dump(mode="json") for slot in result.plan]


async def save_plan(
    user_id: str,
    week_start: str,
    slots: List[dict],
    plan_id: Optional[str] = None,
    existing_liked: Optional[List[str]] = None,
) -> Optional[str]:
    """Persist an approved plan and return its plan_id.

    With `plan_id` the existing plan is regenerated in place (its slots are
    cleared first); without one a new plan row is created. Slot writes upsert on
    (plan_id, day_of_week, meal_type), so a replay overwrites rather than
    duplicating.
    """
    if not plan_id:
        try:
            plan_row = (
                supabase.table("meal_plans")
                .insert(
                    {"user": user_id, "week_start": week_start, "status": "approved"}
                )
                .execute()
            )
            plan_id = plan_row.data[0]["id"] if plan_row.data else None
        except Exception as e:
            logger.error("meal_plan insert error: %s", e)
    else:
        try:
            supabase.table("meal_slots").delete().eq("plan_id", plan_id).execute()
        except Exception as e:
            logger.error("meal_slots clear error: %s", e)

    merged = list(
        dict.fromkeys(
            (existing_liked or []) + [s["recipe_name"] for s in (slots or [])]
        )
    )
    await remember(user_id, "liked_dishes", merged)

    for slot in slots or []:
        try:
            supabase.table("meal_slots").upsert(
                {
                    "plan_id": plan_id,
                    "day_of_week": slot["day_of_week"],
                    "meal_type": slot["meal_type"].lower(),
                    "recipe_name": slot["recipe_name"],
                    "protein_g": slot["protein_g"],
                },
                on_conflict="plan_id,day_of_week,meal_type",
            ).execute()
        except Exception as e:
            logger.error("slot insert error: %s", e)

    return plan_id


async def suggest_meals(
    request: str,
    current_user: Optional[dict] = None,
    memory: Optional[dict] = None,
) -> List[dict]:
    """Nutrition-aware meal suggestions via the research ReAct loop: the model
    calls get_nutrition for each dish it proposes, then we distil the enriched
    conversation into structured suggestions."""
    # Imported here (not at module import time) because tools.py builds a
    # bound LLM at import, and service.py is imported by the MCP server.
    from .tools import research_tool_node, research_llm

    current_user = current_user or {}
    memory = memory or {}
    messages = [
        SystemMessage(
            content=(
                "You are a nutrition expert. Suggest meals matching the user's request.\n"
                "For EVERY meal you suggest, call get_nutrition with its ingredient list "
                "(with quantities e.g. '200g chicken breast') to get accurate nutrition data.\n"
                f"User diet: {current_user.get('diet', 'vegetarian')}. "
                f"Disliked: {memory.get('disliked_dishes', [])}."
            )
        ),
        HumanMessage(content=request),
    ]
    messages = await run_tool_loop(research_llm, research_tool_node, messages)
    structured: ResearchOutput = await llm.with_structured_output(
        ResearchOutput
    ).ainvoke(
        messages
        + [
            HumanMessage(
                content="Return all meal suggestions with their nutrition data in structured format."
            )
        ]
    )
    return [m.model_dump() for m in structured.suggestions]


async def list_plans(user_id: str) -> List[dict]:
    """Every meal plan belonging to the user, newest week first."""
    try:
        res = (
            supabase.table("meal_plans")
            .select("id, week_start, status")
            .eq("user", user_id)
            .order("week_start", desc=True)
            .execute()
        )
        return res.data or []
    except Exception as e:
        logger.error("list_plans error: %s", e)
        return []


async def get_plan_slots(
    plan_id: str, meal_types: Optional[List[str]] = None
) -> List[dict]:
    """The filled meal slots of one plan, optionally narrowed to certain meal
    types (breakfast/lunch/dinner)."""
    try:
        query = (
            supabase.table("meal_slots")
            .select("day_of_week, meal_type, recipe_name, protein_g")
            .eq("plan_id", plan_id)
        )
        if meal_types:
            query = query.in_("meal_type", meal_types)
        res = query.order("day_of_week").execute()
        return res.data or []
    except Exception as e:
        logger.error("get_plan_slots error: %s", e)
        return []
