"""HTTP routes for the meal-planner agent. Agent logic lives in app.agents.meal_planner."""

import logging
import uuid
from datetime import datetime
from typing import Optional, Literal, Annotated

from fastapi import APIRouter, Request, HTTPException, Depends
from pydantic import BaseModel
from langgraph.types import Command

from app.core.config import supabase
from app.dependencies import get_current_user
from app.agents.meal_planner.repository import (
    verify_plan_ownership,
    get_disliked_dishes,
    add_disliked_dish,
    remove_disliked_dish,
    build_grocery_list,
    log_recipe_to_slot,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/meal-planner",
    tags=["meal-planner"],
    responses={404: {"description": "Not found"}},
)


class QueryRequest(BaseModel):
    text: str
    plan_id: Optional[str] = None
    thread_id: Optional[str] = None


@router.post("/query")
async def ask(
    body: QueryRequest,
    request: Request,
    current_user: Annotated[dict, Depends(get_current_user)],
):
    agent = request.app.state.meal_agent

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


@router.post("/approve")
async def approve(
    body: ApproveRequest,
    request: Request,
    current_user: Annotated[dict, Depends(get_current_user)],
):
    agent = request.app.state.meal_agent
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


@router.post("/resolve-conflict")
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


@router.get("/disliked")
async def list_disliked(current_user: Annotated[dict, Depends(get_current_user)]):
    return {"status": "done", "result": await get_disliked_dishes(current_user["uid"])}


@router.post("/disliked")
async def add_disliked(
    body: DislikeRequest,
    current_user: Annotated[dict, Depends(get_current_user)],
):
    merged = await add_disliked_dish(current_user["uid"], body.dish)
    return {"status": "done", "result": merged}


@router.delete("/disliked")
async def delete_disliked(
    body: DislikeRequest,
    current_user: Annotated[dict, Depends(get_current_user)],
):
    merged = await remove_disliked_dish(current_user["uid"], body.dish)
    return {"status": "done", "result": merged}


@router.get("/meal-slots/{plan_id}")
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


@router.get("/grocery-list/{plan_id}")
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


@router.get("/approve")
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


@router.get("/plans")
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


@router.post("/toggle-trigger")
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
