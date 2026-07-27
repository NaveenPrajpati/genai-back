"""The meal planner, published as a Model Context Protocol server.

Everything here is a thin wrapper over `agents.meal_planner.service` — the MCP
layer owns protocol concerns (tool schemas, identity, ownership checks) and
nothing else, so the graph and the protocol can never drift apart.

──────────────────────────────────────────────────────────────────────────────
IDENTITY
──────────────────────────────────────────────────────────────────────────────
Every tool acts on one user's data, but `user_id` is deliberately NOT a tool
argument: the calling model would have to supply it, and a model that can name
the tenant it reads is a model that can hallucinate (or be talked into) someone
else's. Instead identity travels out-of-band, the same way a bearer token does:

  • streamable-HTTP  → `X-User-Id` request header, set by the client per user
  • stdio            → `MCP_USER_ID` env var (single-user: Claude Desktop, the
                       MCP Inspector)

Tools that name a `plan_id` still verify that plan belongs to the caller, so a
guessed id from any client is rejected rather than trusted.

──────────────────────────────────────────────────────────────────────────────
PROPOSE / SAVE SPLIT
──────────────────────────────────────────────────────────────────────────────
`propose_meal_plan` generates without persisting; `save_meal_plan` persists an
already-approved plan. MCP is stateless request/response — a tool call has
nowhere to pause for a human — so the approval step lives in the caller (the
supervisor graph raises a LangGraph `interrupt()` between the two calls).

──────────────────────────────────────────────────────────────────────────────
RUNNING IT
──────────────────────────────────────────────────────────────────────────────
Mounted at /mcp by app.main for the supervisor. Standalone, for an MCP client
such as Claude Desktop or `npx @modelcontextprotocol/inspector`:

    MCP_USER_ID=<uid> python -m app.mcp.meal_planner          # stdio
    python -m app.mcp.meal_planner --http                     # :8100/mcp
"""

import logging
import os
from typing import Annotated, Literal, Optional

from mcp.server.fastmcp import Context, FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from pydantic import BaseModel, Field

from app.agents.meal_planner import service
from app.agents.meal_planner.repository import (
    add_disliked_dish,
    build_grocery_list,
    get_disliked_dishes,
    get_monday,
    log_recipe_to_slot,
    verify_plan_ownership,
)
from app.agents.memory_store import get_profile
from app.database import get_db

logger = logging.getLogger(__name__)

INSTRUCTIONS = """\
Tools for a user's weekly meal planning: browse and read their plans, propose a
new week of meals, log what they actually ate, build a grocery list, and record
food preferences.

Creating a plan is two steps on purpose: call propose_meal_plan to generate one,
show it to the user, and only call save_meal_plan once they have approved it.
Never call save_meal_plan with a plan the user has not seen.
"""

# stateless_http: each request is self-contained, so no session state has to
# survive between calls. That keeps the mounted server safe to restart and lets
# a client open a fresh per-user connection (with its own X-User-Id) cheaply.
mcp = FastMCP("meal-planner", instructions=INSTRUCTIONS, stateless_http=True)


class ProposedSlot(BaseModel):
    """One meal in a weekly plan."""

    day_of_week: int = Field(description="Monday=0 … Sunday=6", ge=0, le=6)
    meal_type: Literal["breakfast", "lunch", "dinner"]
    recipe_name: str
    protein_g: Optional[int] = Field(default=None, description="Protein in grams")


# ── identity ─────────────────────────────────────────────────────────────────
def _caller_id(ctx: Context) -> str:
    """The user this call acts for. See IDENTITY in the module docstring."""
    request = getattr(ctx.request_context, "request", None)
    if request is not None:
        header = request.headers.get("x-user-id")
        if header:
            return header
    env_uid = os.getenv("MCP_USER_ID")
    if env_uid:
        return env_uid
    raise ToolError(
        "No user identity on this MCP connection: send an X-User-Id header "
        "(HTTP) or set MCP_USER_ID (stdio)."
    )


async def _profile(user_id: str) -> tuple[dict, dict]:
    """(user document, learned memory profile) — the personalisation the
    planning and suggestion prompts expect."""
    user = {}
    try:
        user = await get_db()["users"].find_one({"uid": user_id}) or {}
    except Exception as e:
        logger.error("mcp user lookup error: %s", e)
    return user, await get_profile(user_id)


async def _require_own_plan(user_id: str, plan_id: str) -> None:
    """Reject a plan_id the caller does not own, whoever sent it."""
    if not await verify_plan_ownership(plan_id, user_id):
        raise ToolError(f"Plan {plan_id} does not exist or does not belong to you.")


# ── read ─────────────────────────────────────────────────────────────────────
@mcp.tool()
async def list_meal_plans(ctx: Context) -> list[dict]:
    """List the user's meal plans, newest week first. Each entry has an `id`
    (pass it as plan_id to other tools), the `week_start` date, and a status.
    Call this first when the user refers to "my plan" without naming one."""
    return await service.list_plans(_caller_id(ctx))


@mcp.tool()
async def get_meal_plan(
    ctx: Context,
    plan_id: Annotated[str, Field(description="Plan id from list_meal_plans")],
    meal_types: Annotated[
        Optional[list[Literal["breakfast", "lunch", "dinner"]]],
        Field(description="Only these meal types; omit for the whole week"),
    ] = None,
) -> list[dict]:
    """Read the meals scheduled in one plan, as day/meal/recipe/protein rows."""
    user_id = _caller_id(ctx)
    await _require_own_plan(user_id, plan_id)
    return await service.get_plan_slots(plan_id, meal_types)


@mcp.tool()
async def get_grocery_list(ctx: Context, plan_id: str) -> list[dict]:
    """Build a shopping list for a plan: every recipe's ingredients, aggregated
    by name and unit across the week."""
    user_id = _caller_id(ctx)
    await _require_own_plan(user_id, plan_id)
    return await build_grocery_list(plan_id)


@mcp.tool()
async def get_food_preferences(ctx: Context) -> dict:
    """What is known about how this user eats — diet, allergies, preferred
    cuisines, nutrition goals, and dishes they have rejected before. Consult
    this before suggesting food."""
    user_id = _caller_id(ctx)
    user, memory = await _profile(user_id)
    return {
        "diet": user.get("diet", "vegetarian"),
        "diet_restrictions": memory.get("diet_restrictions"),
        "allergies": memory.get("allergies"),
        "preferred_cuisines": memory.get("preferred_cuisines"),
        "nutrition_goals": memory.get("nutrition_goals"),
        "household_size": memory.get("household_size"),
        "cooking_skill": memory.get("cooking_skill"),
        "disliked_dishes": await get_disliked_dishes(user_id),
    }


# ── generate ─────────────────────────────────────────────────────────────────
@mcp.tool()
async def propose_meal_plan(
    ctx: Context,
    request: Annotated[
        str, Field(description="What the user wants, in their own words")
    ],
) -> dict:
    """Generate a full week of meals honouring the user's diet, allergies and
    dislikes. NOTHING IS SAVED — this returns a proposal to show the user.
    Returns `week_start` and `slots`; pass both to save_meal_plan once approved."""
    user_id = _caller_id(ctx)
    user, memory = await _profile(user_id)
    slots = await service.build_plan(request, user, memory)
    return {"week_start": get_monday(), "slots": slots, "saved": False}


@mcp.tool()
async def suggest_meals(
    ctx: Context,
    request: Annotated[str, Field(description="What kind of meals to suggest")],
) -> list[dict]:
    """Suggest individual dishes (not a full week) with real nutrition data
    looked up per recipe. Use for "what should I eat for…" style questions."""
    user_id = _caller_id(ctx)
    user, memory = await _profile(user_id)
    return await service.suggest_meals(request, user, memory)


# ── write ────────────────────────────────────────────────────────────────────
@mcp.tool()
async def save_meal_plan(
    ctx: Context,
    week_start: Annotated[str, Field(description="ISO date of the Monday")],
    slots: Annotated[list[ProposedSlot], Field(description="The approved meals")],
    plan_id: Annotated[
        Optional[str],
        Field(description="Replace this existing plan instead of creating one"),
    ] = None,
) -> dict:
    """Persist a meal plan THE USER HAS ALREADY APPROVED. With plan_id the
    existing plan is regenerated in place; without one a new plan is created."""
    user_id = _caller_id(ctx)
    if plan_id:
        await _require_own_plan(user_id, plan_id)
    _, memory = await _profile(user_id)
    saved_id = await service.save_plan(
        user_id,
        week_start,
        [s.model_dump() for s in slots],
        plan_id=plan_id,
        existing_liked=memory.get("liked_dishes", []),
    )
    return {"plan_id": saved_id, "slots_saved": len(slots), "saved": True}


@mcp.tool()
async def log_meal(
    ctx: Context,
    plan_id: str,
    recipe: Annotated[str, Field(description="Dish that was eaten")],
    day_of_week: Annotated[int, Field(description="Monday=0 … Sunday=6", ge=0, le=6)],
    meal_type: Literal["breakfast", "lunch", "dinner"],
) -> dict:
    """Record what the user actually ate in one slot of their plan, replacing
    whatever was planned there."""
    user_id = _caller_id(ctx)
    await _require_own_plan(user_id, plan_id)
    await log_recipe_to_slot(plan_id, recipe, day_of_week, meal_type)
    return {"logged": recipe, "day_of_week": day_of_week, "meal_type": meal_type}


@mcp.tool()
async def add_food_dislike(
    ctx: Context, dish: Annotated[str, Field(description="Dish to stop suggesting")]
) -> dict:
    """Remember that the user does not want a dish, so future plans and
    suggestions skip it."""
    disliked = await add_disliked_dish(_caller_id(ctx), dish)
    return {"disliked_dishes": disliked}


def main() -> None:
    """Standalone entry point. Invoked by __main__.py — see RUNNING IT above."""
    import sys

    logging.basicConfig(level=logging.INFO)
    if "--http" in sys.argv:
        mcp.settings.host = os.getenv("MCP_HOST", "127.0.0.1")
        mcp.settings.port = int(os.getenv("MCP_PORT", "8100"))
        mcp.run(transport="streamable-http")
    else:
        # stdio speaks the protocol over the pipe — nothing may be printed to
        # stdout, so logging goes to stderr (basicConfig's default).
        mcp.run(transport="stdio")
