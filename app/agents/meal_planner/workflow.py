"""LangGraph nodes and graph wiring for the meal-planner agent."""

import logging

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import START, StateGraph, END
from langgraph.types import interrupt

from app.core.config import supabase
from app.core.llm import llm, fast_llm
from app.services.cache import cached_value
from app.core.config import CACHE_CLASSIFY_THRESHOLD
from app.agents.approval_store import get_pending, create_pending, resolve
from app.agents.react import run_tool_loop
from app.agents.memory_store import get_profile
from .state import (
    PlannerState,
    IntentOutput,
    LogOutput,
    QueryOutput,
    ResearchOutput,
    PlanOutput,
)
from .repository import (
    get_monday,
    findMealSlotsInDb,
    log_recipe_to_slot,
    remember,
)
from .tools import research_tool_node, research_llm

logger = logging.getLogger(__name__)


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
    chain = prompt | fast_llm.with_structured_output(IntentOutput)
    query = state.get("query", "")

    async def produce():
        result: IntentOutput = await chain.ainvoke({"text": query})
        logger.info("%s", result)
        return result.model_dump()

    # Intent depends only on the message and is user-independent → global scope,
    # shared across users, with the loose classification threshold.
    data = await cached_value(
        query, "agent:meal:classify_intent", CACHE_CLASSIFY_THRESHOLD, produce
    )
    return {
        "intent": data["intent"],
    }


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

    chain = findPrompt | fast_llm.with_structured_output(LogOutput)
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

    chain = findPrompt | fast_llm.with_structured_output(QueryOutput)
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

    # ReAct loop: LLM calls get_nutrition for each meal it suggests, reads the
    # nutrition data, and may suggest/look up more before finishing.
    messages = await run_tool_loop(research_llm, research_tool_node, messages)

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


async def plan_agent(state: PlannerState):
    # LangGraph re-runs this node from the top on resume. Check Supabase first
    # so we reuse the original proposal instead of creating a duplicate.
    week_start = get_monday()
    approval_id = None
    proposed = None
    existing_row = await get_pending(state["thread_id"])
    if existing_row:
        approval_id = str(existing_row["_id"])
        proposed = existing_row["payload"]["plan"]

    if not approval_id:
        # First run: generate plan via LLM and insert approval.
        suggestionPrompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an expert at planning diet plan , so plan diet for week my week start from monday means monday is day of week 0 ,recipes for dinner, lunch, breakfast for all days of week along with protien in grams in each meal "
                    "User diet: {diet}. Disliked: {disliked}.\n"
                    "Learned preferences (honor these strictly) — diet restrictions: "
                    "{diet_restrictions}; allergies: {allergies}; preferred cuisines: "
                    "{preferred_cuisines}; household size: {household_size}; cooking "
                    "skill: {cooking_skill}; nutrition goals: {nutrition_goals}.\n",
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
                "diet_restrictions": memory.get("diet_restrictions") or "none",
                "allergies": memory.get("allergies") or "none",
                "preferred_cuisines": memory.get("preferred_cuisines") or "none",
                "household_size": memory.get("household_size") or "unknown",
                "cooking_skill": memory.get("cooking_skill") or "unknown",
                "nutrition_goals": memory.get("nutrition_goals") or "none",
            }
        )
        logger.info("plan data %s", result)
        proposed = [slot.model_dump(mode="json") for slot in result.plan]

        approval_id = await create_pending(
            state["user_id"],
            state["thread_id"],
            "save_plan",
            {"week_start": week_start, "plan": proposed},
        )

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
        await resolve(approval_id, "rejected")
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

    await resolve(approval_id, "approved")

    return {
        "intent": state.get("intent", "plan"),
        "plan_status": "approved",
        "plan_id": plan_id,
    }


async def load_memory(state: PlannerState):
    user_id = state["user_id"]
    # Learned long-term profile + app-managed prefs (diet, allergies, cuisines,
    # disliked dishes …) all live in the shared Mongo `memories` doc.
    memory = await get_profile(user_id)

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


def build_graph() -> StateGraph:
    graph = StateGraph(PlannerState)
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
    return graph


graph = build_graph()
