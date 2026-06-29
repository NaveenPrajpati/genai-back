"""Supabase persistence and domain helpers for the meal-planner agent."""

import logging
from datetime import date, datetime, timedelta, timezone
from typing import Optional, List

from langchain_core.prompts import ChatPromptTemplate

from app.core.config import supabase
from app.core.llm import llm
from app.database import get_db
from .state import RecipeOutput, QueryOutput

logger = logging.getLogger(__name__)

MEMORIES = "memories"


# Shared prompt: given a recipe name, ask the LLM to fill in nutrients,
# cooking time and ingredients. Used by both log_agent and conflict resolution.
searchRecipePrompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert at finding nutrients and cooking time , ingredients of given recipe",
        ),
        ("human", "{text}"),
    ]
)


def get_monday(today: Optional[date] = None) -> str:
    # weekday(): Monday=0, Sunday=6
    if today is None:
        today = date.today()
    monday = today - timedelta(days=today.weekday())
    return monday.isoformat()


async def findRecipeInDb(
    recipe: Optional[str] = None, filters: Optional[QueryOutput] = None
):
    try:
        res = (
            supabase.table("recipes")
            .select("id, name, protein_g")
            .ilike("name", recipe)
            .maybe_single()
            .execute()
        )
        logger.info("findRecipeInDb result: %s", res)
        return res.data if res else None
    except Exception as e:
        logger.error("findRecipeInDb error: %s", e)
        return None


async def insertRecipeInDb(recipe: RecipeOutput):
    try:
        res = (
            supabase.table("recipes")
            .insert(recipe.model_dump(mode="json", exclude_none=True))
            .execute()
        )
        logger.info("insertRecipeInDb result: %s", res.data)
        return res.data
    except Exception as e:
        logger.error("insertRecipeInDb error: %s", e)
        return None


async def insertRecipeInMealSlot(data: dict):
    try:
        # Upsert (not insert) so re-logging the same plan/day/meal_type replaces
        # the slot instead of creating a duplicate row. Requires a unique
        # constraint on (plan_id, day_of_week, meal_type) — see migrations/.
        res = (
            supabase.table("meal_slots")
            .upsert(
                {
                    "plan_id": data["plan_id"],
                    "day_of_week": data["day_of_week"],
                    "meal_type": data["meal_type"],
                    "recipe_id": data["recipe_id"],
                    "recipe_name": data["recipe_name"],
                    "protein_g": data["protein_g"],
                },
                on_conflict="plan_id,day_of_week,meal_type",
            )
            .execute()
        )
        logger.info("insertRecipeInMealSlot result: %s", res.data)
        return res.data
    except Exception as e:
        logger.error("insertRecipeInMealSlot error: %s", e)
        return None


async def log_recipe_to_slot(
    plan_id: str, recipe_name: str, day_of_week: int, meal_type: str
):
    """Find-or-create a recipe by name, then attach it to a meal slot.

    Shared by log_agent and the conflict-resolution endpoint so the
    find/enrich/insert logic lives in exactly one place.
    """
    recipe_present = await findRecipeInDb(recipe_name)
    if recipe_present:
        recipe_id = recipe_present["id"]
        name = recipe_present["name"]
        protein = recipe_present["protein_g"]
    else:
        logger.info("recipe not present: %s", recipe_name)
        chain = searchRecipePrompt | llm.with_structured_output(RecipeOutput)
        details: RecipeOutput = await chain.ainvoke({"text": recipe_name})
        logger.info("recipe data %s", details)
        inserted = await insertRecipeInDb(details)
        recipe_id = inserted[0]["id"] if inserted else None
        name = inserted[0]["name"] if inserted else recipe_name
        protein = inserted[0]["protein_g"] if inserted else None

    return await insertRecipeInMealSlot(
        {
            "plan_id": plan_id,
            "day_of_week": day_of_week,
            "meal_type": meal_type.lower(),
            "recipe_id": recipe_id,
            "recipe_name": name,
            "protein_g": protein,
        }
    )


async def findMealSlotsInDb(plan_id: str, meal_types: List[str]):
    try:
        res = (
            supabase.table("meal_slots")
            .select("day_of_week, meal_type, recipe_name, protein_g")
            .eq("plan_id", plan_id)
            .in_("meal_type", meal_types)
            .execute()
        )
        logger.info("findMealSlotsInDb result: %s", res)
        return res.data if res else None
    except Exception as e:
        logger.error("findMealSlotsInDb error: %s", e)
        return None


async def remember(user_id: str, key: str, value):
    """Set a single memory field in the user's Mongo `memories` doc."""
    now = datetime.now(timezone.utc).isoformat()
    try:
        await get_db()[MEMORIES].update_one(
            {"user_id": user_id},
            {
                "$set": {f"data.{key}": value, "updatedAt": now},
                "$setOnInsert": {"createdAt": now},
            },
            upsert=True,
        )
    except Exception as e:
        logger.error("remember error: %s", e)


async def get_disliked_dishes(user_id: str) -> list:
    """Return the user's current disliked_dishes list (empty on miss/error)."""
    try:
        doc = await get_db()[MEMORIES].find_one({"user_id": user_id})
        if doc:
            return list((doc.get("data") or {}).get("disliked_dishes", []) or [])
    except Exception as e:
        logger.error("get_disliked_dishes error: %s", e)
    return []


async def add_disliked_dish(user_id: str, dish: str) -> list:
    """Append a dish to the user's disliked_dishes memory (de-duplicated)."""
    existing = await get_disliked_dishes(user_id)
    merged = list(dict.fromkeys(existing + [dish]))
    await remember(user_id, "disliked_dishes", merged)
    return merged


async def remove_disliked_dish(user_id: str, dish: str) -> list:
    """Remove a dish from the user's disliked_dishes memory."""
    existing = await get_disliked_dishes(user_id)
    merged = [d for d in existing if d != dish]
    await remember(user_id, "disliked_dishes", merged)
    return merged


async def build_grocery_list(plan_id: str) -> list:
    """Aggregate ingredients across every meal slot in a plan into a shopping
    list. Quantities accumulate per (ingredient name, unit), counting each slot
    separately so a dish eaten N times contributes N times."""
    try:
        slots_res = (
            supabase.table("meal_slots")
            .select("recipe_id, recipe_name")
            .eq("plan_id", plan_id)
            .execute()
        )
        slots = slots_res.data or []
    except Exception as e:
        logger.error("build_grocery_list slots error: %s", e)
        return []

    ids = list({s["recipe_id"] for s in slots if s.get("recipe_id")})
    names = list({s["recipe_name"] for s in slots if s.get("recipe_name")})

    # Slots from the log path carry a recipe_id; plan-generated slots only have a
    # recipe_name. Look up ingredients by both so either kind resolves.
    by_id: dict = {}
    by_name: dict = {}
    try:
        if ids:
            r = (
                supabase.table("recipes")
                .select("id, name, ingredients")
                .in_("id", ids)
                .execute()
            )
            for rec in r.data or []:
                by_id[rec["id"]] = rec.get("ingredients") or []
        if names:
            r = (
                supabase.table("recipes")
                .select("id, name, ingredients")
                .in_("name", names)
                .execute()
            )
            for rec in r.data or []:
                by_name[rec["name"]] = rec.get("ingredients") or []
    except Exception as e:
        logger.error("build_grocery_list recipes error: %s", e)

    agg: dict = {}
    for s in slots:
        ingredients = by_id.get(s.get("recipe_id")) if s.get("recipe_id") else None
        if ingredients is None:
            ingredients = by_name.get(s.get("recipe_name"), [])
        for ing in ingredients:
            name = (ing.get("name") or "").strip()
            if not name:
                continue
            unit = ing.get("unit")
            qty = ing.get("qty")
            key = (name.lower(), unit)
            entry = agg.setdefault(
                key, {"name": name, "qty": None, "unit": unit, "checked": False}
            )
            if qty is not None:
                entry["qty"] = (entry["qty"] or 0) + qty

    return sorted(agg.values(), key=lambda x: x["name"].lower())


async def verify_plan_ownership(plan_id: str, user_id: str) -> bool:
    """Return True if the plan exists and belongs to the user."""
    try:
        res = (
            supabase.table("meal_plans")
            .select("id")
            .eq("id", plan_id)
            .eq("user", user_id)
            .maybe_single()
            .execute()
        )
        return bool(res and res.data)
    except Exception as e:
        logger.error("verify_plan_ownership error: %s", e)
        return False
