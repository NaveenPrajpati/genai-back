from typing import TypedDict, Optional, List, Literal, Annotated
from fastapi import APIRouter, Request, HTTPException, Depends
from pydantic import BaseModel
from app.dependencies import get_current_user
from app.core.llm import llm
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langgraph.graph import START, StateGraph, END
from app.core.config import supabase
from datetime import date, timedelta, datetime
from langgraph.types import interrupt, Command
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage
import logging
import uuid
import os
import httpx

load_dotenv()

logger = logging.getLogger(__name__)


mealRouter = APIRouter(
    prefix="/learning-tracker",
    tags=["learning-tracker"],
    responses={404: {"description": "Not found"}},
)


class QueryRequest(BaseModel):
    text: str
    plan_id: Optional[str] = None
    thread_id: Optional[str] = None


class PlannerState(TypedDict, total=False):
    query: str
    intent: str
    current_user: dict
    user_id: str
    thread_id: str
    memory: dict
    plan_status: Optional[str]
    log_status: Optional[str]
    conflict: Optional[dict]
    plan_id: Optional[str]
    suggestions: Optional[list]
    meal_slots: Optional[list]


graph = StateGraph(PlannerState)


class LogOutput(BaseModel):
    recipe: str
    day_of_week: int
    meal_type: str
    conflict: bool
    suggestion: Optional[str]


class GroceryItem(BaseModel):
    plan_id: Optional[str] = None
    name: str
    qty: Optional[float] = None
    unit: Optional[str] = None
    checked: bool = False


class RecipeOutput(BaseModel):
    name: str
    ingredients: list[GroceryItem] = []
    protein_g: Optional[int] = None
    prep_minutes: Optional[int] = None
    source_url: Optional[str] = None
    summary: Optional[str] = None


class IntentOutput(BaseModel):
    intent: str


def get_monday(today: Optional[date] = None) -> str:
    # weekday(): Monday=0, Sunday=6
    if today is None:
        today = date.today()
    monday = today - timedelta(days=today.weekday())
    return monday.isoformat()


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


async def classify_intent(state: PlannerState):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Classify the user message into one intent:\n"
                "- log: recording a specific meal eaten or to be eaten (mentions a dish + day/meal slot)\n"
                "- plan: generate a full weekly meal plan from scratch (no existing plan mentioned)\n"
                "- update: regenerate or change an existing meal plan (words like update, change, redo, modify, regenerate)\n"
                "- research: ask for food/nutrition info or suggestions (no specific slot)\n"
                "- query: view or check what is already in the meal plan\n"
                "Reply with one word only: log, plan, update, research, or query.",
            ),
            ("human", "{text}"),
        ]
    )
    chain = prompt | llm.with_structured_output(IntentOutput)
    result: IntentOutput = await chain.ainvoke({"text": state.get("query", "")})
    logger.info("%s", result)
    return {
        "intent": result.intent,
    }


class QueryOutput(BaseModel):
    meal_type: List[str]
    time: str


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


async def log_agent(state: PlannerState):
    findPrompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Extract the recipe, day_of_week (Monday=0), and meal_type from the text.\n"
                "User diet: {diet}. Disliked: {disliked}.\n"
                "If the requested dish conflicts with their diet, set conflict=true "
                "and suggest an alternative instead of extracting it.",
            ),
            ("human", "{text}"),
        ]
    )

    current_user = state.get("current_user") or {}
    memory = state.get("memory") or {}

    chain = findPrompt | llm.with_structured_output(LogOutput)
    result: LogOutput = await chain.ainvoke(
        {
            "text": state["query"],
            "diet": current_user.get("diet", "vegetarian"),
            "disliked": memory.get("disliked_dishes", []),
        }
    )
    logger.info("log data %s", result)
    if result.conflict:
        # Surface the full context so the client can call /resolve-conflict
        # to accept the suggestion (log it) or reject it (record a dislike).
        return {
            "intent": "log",
            "log_status": "conflict",
            "suggestions": [result.suggestion] if result.suggestion else [],
            "conflict": {
                "original": result.recipe,
                "suggestion": result.suggestion,
                "day_of_week": result.day_of_week,
                "meal_type": result.meal_type,
            },
        }

    plan_id = state.get("plan_id")
    if not plan_id:
        return {"intent": "log", "log_status": "no_plan"}

    await log_recipe_to_slot(
        plan_id, result.recipe, result.day_of_week, result.meal_type
    )
    return {
        "intent": "log",
        "log_status": "logged",
    }


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


async def query_agent(state: PlannerState):
    findPrompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert at extracting information about having food , so tell me what types of meal (dinner,lunch,breakfast ) user want to know and for when (today,week)",
            ),
            ("human", "{text}"),
        ]
    )

    chain = findPrompt | llm.with_structured_output(QueryOutput)
    result: QueryOutput = await chain.ainvoke({"text": state["query"]})
    logger.info("query data %s", result)
    plan_id = state.get("plan_id")
    if not plan_id:
        return {"intent": "query", "meal_slots": []}
    slots = await findMealSlotsInDb(plan_id, result.meal_type)
    return {
        "intent": "query",
        "meal_slots": slots or [],
    }


class NutritionData(BaseModel):
    calories: float = 0
    protein_g: float = 0
    carbs_g: float = 0
    fat_g: float = 0


class ResearchMeal(BaseModel):
    meal_type: str
    recipe_name: str
    ingredients: list[str]
    prep_minutes: int
    nutrition: Optional[NutritionData] = None


class ResearchOutput(BaseModel):
    suggestions: List[ResearchMeal]


@tool
async def get_nutrition(ingredients: list[str]) -> dict:
    """Fetch accurate nutrition data for a recipe from the Edamam API.
    Call this for every meal you suggest.
    Pass ingredients with quantities e.g. ['200g chicken breast', '1 cup rice'].
    Returns calories, protein_g, carbs_g, fat_g for the full recipe."""
    app_id = os.getenv("EDAMAM_APP_ID", "")
    app_key = os.getenv("EDAMAM_APP_KEY", "")
    if not app_id or not app_key:
        return {"calories": 0, "protein_g": 0, "carbs_g": 0, "fat_g": 0}
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                "https://api.edamam.com/api/nutrition-details",
                params={"app_id": app_id, "app_key": app_key},
                json={"ingr": ingredients},
            )
            if resp.status_code != 200:
                logger.error(f"Edamam error {resp.status_code}: {resp.text}")
                return {"calories": 0, "protein_g": 0, "carbs_g": 0, "fat_g": 0}
            data = resp.json()
            n = data.get("totalNutrients", {})
            return {
                "calories": round(data.get("calories", 0), 1),
                "protein_g": round(n.get("PROCNT", {}).get("quantity", 0), 1),
                "carbs_g": round(n.get("CHOCDF", {}).get("quantity", 0), 1),
                "fat_g": round(n.get("FAT", {}).get("quantity", 0), 1),
            }
    except Exception as e:
        logger.error(f"nutrition API error: {e}")
        return {"calories": 0, "protein_g": 0, "carbs_g": 0, "fat_g": 0}


_research_tools = [get_nutrition]
_research_tool_node = ToolNode(_research_tools)
_research_llm = llm.bind_tools(_research_tools)


async def research_agent(state: PlannerState):
    current_user = state.get("current_user") or {}
    memory = state.get("memory") or {}

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
        HumanMessage(content=state["query"]),
    ]

    # Tool-calling loop: LLM calls get_nutrition for each meal it suggests
    while True:
        response = await _research_llm.ainvoke(messages)
        messages.append(response)
        if not response.tool_calls:
            break
        tool_results = await _research_tool_node.ainvoke({"messages": messages})
        messages.extend(tool_results["messages"])

    # Final pass: extract structured output from the enriched conversation
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
    logger.info("research data %s", structured)
    return {
        "intent": "research",
        "suggestions": [m.model_dump() for m in structured.suggestions],
    }


class MealSlots(BaseModel):
    plan_id: Optional[str] = None
    day_of_week: int = 0
    meal_type: Literal["dinner", "lunch", "breakfast"]
    recipe_id: Optional[str] = None
    recipe_name: Optional[str] = None
    protein_g: Optional[int] = None


class PlanOutput(BaseModel):
    plan: list[MealSlots] = []


async def remember(user_id: str, key: str, value):
    try:
        supabase.table("memory").upsert(
            {"user_id": user_id, "key": key, "value": value},
            on_conflict="user_id,key",
        ).execute()
    except Exception as e:
        logger.error("remember error: %s", e)


async def get_disliked_dishes(user_id: str) -> list:
    """Return the user's current disliked_dishes list (empty on miss/error)."""
    try:
        row = (
            supabase.table("memory")
            .select("value")
            .eq("user_id", user_id)
            .eq("key", "disliked_dishes")
            .maybe_single()
            .execute()
        )
        return list(row.data["value"]) if row and row.data else []
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


async def plan_agent(state: PlannerState):
    # LangGraph re-runs this node from the top on resume. Check Supabase first
    # so we reuse the original proposal instead of creating a duplicate.
    week_start = get_monday()
    approval_id = None
    proposed = None
    try:
        existing_row = (
            supabase.table("approvals")
            .select("id, payload")
            .eq("thread_id", state["thread_id"])
            .eq("status", "pending")
            .maybe_single()
            .execute()
        )
        if existing_row and existing_row.data:
            approval_id = existing_row.data["id"]
            proposed = existing_row.data["payload"]["plan"]
    except Exception as e:
        logger.error("approval lookup error: %s", e)

    if not approval_id:
        # First run: generate plan via LLM and insert approval.
        suggestionPrompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an expert at planning diet plan , so plan diet for week my week start from monday means monday is day of week 0 ,recipes for dinner, lunch, breakfast for all days of week along with protien in grams in each meal "
                    "User diet: {diet}. Disliked: {disliked}.\n",
                ),
                ("human", "{text}"),
            ]
        )
        current_user = state.get("current_user") or {}
        memory = state.get("memory") or {}
        chain = suggestionPrompt | llm.with_structured_output(PlanOutput)
        result: PlanOutput = await chain.ainvoke(
            {
                "text": state["query"],
                "diet": current_user.get("diet", "vegetarian"),
                "disliked": memory.get("disliked_dishes", []),
            }
        )
        logger.info("plan data %s", result)
        proposed = [slot.model_dump(mode="json") for slot in result.plan]

        try:
            res = (
                supabase.table("approvals")
                .insert(
                    {
                        "user_id": state["user_id"],
                        "thread_id": state["thread_id"],
                        "action_type": "save_plan",
                        "payload": {"week_start": week_start, "plan": proposed},
                        "status": "pending",
                    }
                )
                .execute()
            )
            approval_id = res.data[0]["id"] if res.data else None
        except Exception as e:
            logger.error("approval insert error: %s", e)

    is_update = state.get("intent") == "update" or bool(state.get("plan_id"))
    action_type = "update_plan" if is_update else "save_plan"

    decision = interrupt(
        {
            "type": action_type,
            "approval_id": approval_id,
            "week_start": week_start,
            "plan": proposed,
        }
    )

    if decision != "approved":
        if approval_id:
            supabase.table("approvals").update(
                {"status": "rejected", "resolved_at": datetime.now().isoformat()}
            ).eq("id", approval_id).execute()
        return {"intent": state.get("intent", "plan"), "plan_status": "rejected"}

    # Approved: for update reuse the existing plan row; for new plan create one.
    plan_id = state.get("plan_id") if is_update else None
    if not plan_id:
        try:
            plan_row = (
                supabase.table("meal_plans")
                .insert(
                    {
                        "user": state["user_id"],
                        "week_start": week_start,
                        "status": "approved",
                    }
                )
                .execute()
            )
            plan_id = plan_row.data[0]["id"] if plan_row.data else None
        except Exception as e:
            logger.error("meal_plan insert error: %s", e)
    else:
        # Clear existing slots so we start fresh with the new proposal.
        try:
            supabase.table("meal_slots").delete().eq("plan_id", plan_id).execute()
        except Exception as e:
            logger.error("meal_slots clear error: %s", e)

    existing = (state.get("memory") or {}).get("liked_dishes", [])
    merged = list(
        dict.fromkeys(existing + [s["recipe_name"] for s in (proposed or [])])
    )
    await remember(state["user_id"], "liked_dishes", merged)

    for slot in proposed or []:
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

    if approval_id:
        supabase.table("approvals").update(
            {"status": "approved", "resolved_at": datetime.now().isoformat()}
        ).eq("id", approval_id).execute()

    return {
        "intent": state.get("intent", "plan"),
        "plan_status": "approved",
        "plan_id": plan_id,
    }


async def load_memory(state: PlannerState):
    user_id = state["user_id"]
    memory = {}

    try:
        rows = (
            supabase.table("memory")
            .select("key, value")
            .eq("user_id", user_id)
            .execute()
        )
        if rows and rows.data:
            memory = {r["key"]: r["value"] for r in rows.data}
    except Exception as e:
        logger.error("load memory error: %s", e)

    return {"memory": memory}


def decide_agent(state: PlannerState):
    intent = state.get("intent")
    if intent == "log":
        return "log_agent"
    elif intent == "query":
        return "query_agent"
    elif intent == "research":
        return "research_agent"
    elif intent in ("plan", "update"):
        return "plan_agent"
    return END


graph.add_node("load_memory", load_memory)
graph.add_node("classify_intent", classify_intent)
graph.add_node("log_agent", log_agent)
graph.add_node("query_agent", query_agent)
graph.add_node("research_agent", research_agent)
graph.add_node("plan_agent", plan_agent)
graph.add_edge(START, "load_memory")
graph.add_edge("load_memory", "classify_intent")
graph.add_conditional_edges(
    "classify_intent",
    decide_agent,
    ["log_agent", "query_agent", "research_agent", "plan_agent", END],
)


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


@mealRouter.post("/query")
async def ask(
    body: QueryRequest,
    request: Request,
    current_user: Annotated[dict, Depends(get_current_user)],
):
    agent = request.app.state.agent

    if body.plan_id and not await verify_plan_ownership(
        body.plan_id, current_user["uid"]
    ):
        raise HTTPException(
            status_code=403, detail="You do not have access to this plan."
        )

    # "update" intent requires a plan_id to know which plan to regenerate.
    # Do a lightweight pre-check so we fail fast with a readable error.
    text_lower = body.text.lower()
    update_keywords = ("update", "change", "redo", "modify", "regenerate")
    if any(kw in text_lower for kw in update_keywords) and not body.plan_id:
        raise HTTPException(
            status_code=400,
            detail="Provide plan_id to update an existing plan.",
        )

    thread_id = body.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    _excluded = {"_id", "expires_at", "password_hash"}
    user_data = {k: v for k, v in current_user.items() if k not in _excluded}
    result = await agent.ainvoke(
        {
            "query": body.text,
            "user_id": current_user["uid"],
            "thread_id": thread_id,
            "plan_id": body.plan_id,
            "current_user": user_data,
        },
        config=config,
    )
    logger.info("final -- %s", result)

    if "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        return {
            "status": "needs_approval",
            "thread_id": thread_id,  # app sends this back to /approve
            "proposal": payload,
        }

    return {"status": "done", "result": result}


class ApproveRequest(BaseModel):
    thread_id: str
    decision: Literal["approved", "rejected"]


@mealRouter.post("/approve")
async def approve(
    body: ApproveRequest,
    request: Request,
    current_user: Annotated[dict, Depends(get_current_user)],
):
    agent = request.app.state.agent
    config = {"configurable": {"thread_id": body.thread_id}}

    # The thread/approval must belong to the caller (prevents IDOR where a user
    # approves or rejects someone else's pending plan by guessing the thread_id).
    try:
        approval = (
            supabase.table("approvals")
            .select("id, user_id")
            .eq("thread_id", body.thread_id)
            .eq("status", "pending")
            .maybe_single()
            .execute()
        )
    except Exception as e:
        logger.error("approval ownership lookup error: %s", e)
        approval = None

    if not approval or not approval.data:
        raise HTTPException(
            status_code=404, detail="No pending approval for this thread."
        )
    if approval.data["user_id"] != current_user["uid"]:
        raise HTTPException(
            status_code=403, detail="You do not have access to this approval."
        )

    snapshot = await agent.aget_state(config)
    if not snapshot or not snapshot.next:
        raise HTTPException(
            status_code=404,
            detail="No pending approval for this thread. The server may have restarted — please re-submit your plan request.",
        )

    result = await agent.ainvoke(Command(resume=body.decision), config=config)
    return {"status": "done", "result": result}


class ResolveConflictRequest(BaseModel):
    plan_id: str
    recipe: str  # the suggested (or chosen) dish to act on
    day_of_week: int
    meal_type: Literal["dinner", "lunch", "breakfast"]
    decision: Literal["accept", "reject"]


@mealRouter.post("/resolve-conflict")
async def resolve_conflict(
    body: ResolveConflictRequest,
    current_user: Annotated[dict, Depends(get_current_user)],
):
    """Finish a 'conflict' from log_agent.

    accept → log the suggested recipe into the slot.
    reject → record the suggestion as a disliked dish so it isn't offered again.
    """
    if not await verify_plan_ownership(body.plan_id, current_user["uid"]):
        raise HTTPException(
            status_code=403, detail="You do not have access to this plan."
        )

    if body.decision == "reject":
        await add_disliked_dish(current_user["uid"], body.recipe)
        return {"status": "done", "log_status": "rejected"}

    slot = await log_recipe_to_slot(
        body.plan_id, body.recipe, body.day_of_week, body.meal_type
    )
    return {"status": "done", "log_status": "logged", "slot": slot}


class DislikeRequest(BaseModel):
    dish: str


@mealRouter.get("/disliked")
async def list_disliked(current_user: Annotated[dict, Depends(get_current_user)]):
    return {"status": "done", "result": await get_disliked_dishes(current_user["uid"])}


@mealRouter.post("/disliked")
async def add_disliked(
    body: DislikeRequest,
    current_user: Annotated[dict, Depends(get_current_user)],
):
    merged = await add_disliked_dish(current_user["uid"], body.dish)
    return {"status": "done", "result": merged}


@mealRouter.delete("/disliked")
async def delete_disliked(
    body: DislikeRequest,
    current_user: Annotated[dict, Depends(get_current_user)],
):
    merged = await remove_disliked_dish(current_user["uid"], body.dish)
    return {"status": "done", "result": merged}


@mealRouter.get("/meal-slots/{plan_id}")
async def get_meal_slots(
    plan_id: str,
    current_user: Annotated[dict, Depends(get_current_user)],
):
    if not await verify_plan_ownership(plan_id, current_user["uid"]):
        raise HTTPException(
            status_code=403, detail="You do not have access to this plan."
        )
    try:
        res = (
            supabase.table("meal_slots")
            .select("id, day_of_week, meal_type, recipe_id, recipe_name, protein_g")
            .eq("plan_id", plan_id)
            .order("day_of_week")
            .order("meal_type")
            .execute()
        )
        return {"status": "done", "plan_id": plan_id, "slots": res.data or []}
    except Exception as e:
        logger.error("get_meal_slots error: %s", e)
        raise HTTPException(status_code=500, detail="Failed to fetch meal slots.")


@mealRouter.get("/grocery-list/{plan_id}")
async def grocery_list(
    plan_id: str,
    current_user: Annotated[dict, Depends(get_current_user)],
):
    if not await verify_plan_ownership(plan_id, current_user["uid"]):
        raise HTTPException(
            status_code=403, detail="You do not have access to this plan."
        )
    items = await build_grocery_list(plan_id)
    return {"status": "done", "plan_id": plan_id, "result": items}


@mealRouter.get("/approve")
async def list_approvals(current_user: Annotated[dict, Depends(get_current_user)]):
    user_id = current_user["uid"]
    logger.info("--- %s", user_id)
    try:
        result = (
            supabase.table("approvals")
            .select("*")
            .eq("user_id", user_id)
            .eq("status", "pending")
            .execute()
        )
        logger.info("%s", result)

        if not result.data:
            return {"status": "done", "message": "no approval found", "result": []}

        return {"status": "done", "result": result.data}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@mealRouter.get("/plans")
async def getPlans(current_user: Annotated[dict, Depends(get_current_user)]):
    user_id = current_user["uid"]
    logger.info("--- %s", user_id)
    try:
        result = supabase.table("meal_plans").select("*").eq("user", user_id).execute()
        logger.info("%s", result)

        if not result.data:
            return {"status": "done", "message": "plans not found", "result": []}

        return {"status": "done", "message": "plans fetched", "result": result.data}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class Trigger(BaseModel):
    id: str
    name: str
    schedule: str
    action_type: str
    enabled: bool = True
    last_run_at: Optional[datetime] = None


@mealRouter.post("/toggle-trigger")
async def toggle_trigger(current_user: Annotated[dict, Depends(get_current_user)]):
    user_id = current_user["uid"]

    try:
        result = supabase.table("triggers").select("*").eq("user_id", user_id).execute()
        if result and result.data:
            for t in result.data or []:
                supabase.table("triggers").update({"enabled": not t["enabled"]}).eq(
                    "id", t["id"]
                ).execute()
        else:
            supabase.table("triggers").insert(
                {
                    "user_id": user_id,
                    "name": "plan my schedule",
                    "schedule": "30 18 * * 0",
                    "action_type": "schedule",
                    "enabled": True,
                    "last_run_at": None,
                }
            ).execute()

        return {"status": "done"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
