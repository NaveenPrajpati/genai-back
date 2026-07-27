"""Tests for the meal planner's MCP surface.

The protocol plumbing is the SDK's job; what matters here is what we put on top
of it — identity that cannot be supplied by the calling model, ownership checks
on every plan id, and the propose/save split that keeps a human in the loop.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from mcp.server.fastmcp.exceptions import ToolError

import app.mcp.meal_planner.server as server
from app.mcp.meal_planner.client import (
    AUTONOMOUS_TOOLS,
    SAVE_PLAN_TOOL,
    autonomous_tools,
    find_tool,
)


def _ctx(headers=None):
    """A Context as the HTTP transport supplies it (stdio passes request=None)."""
    request = SimpleNamespace(headers=headers) if headers is not None else None
    return SimpleNamespace(request_context=SimpleNamespace(request=request))


# --------------------------------------------------------------------------- #
# identity
# --------------------------------------------------------------------------- #
def test_identity_comes_from_the_request_header():
    assert server._caller_id(_ctx({"x-user-id": "u1"})) == "u1"


def test_identity_falls_back_to_the_env_var_over_stdio(monkeypatch):
    monkeypatch.setenv("MCP_USER_ID", "stdio-user")
    assert server._caller_id(_ctx()) == "stdio-user"


def test_an_unidentified_call_is_refused_not_guessed(monkeypatch):
    monkeypatch.delenv("MCP_USER_ID", raising=False)
    with pytest.raises(ToolError, match="No user identity"):
        server._caller_id(_ctx({}))


def test_the_header_wins_over_the_env_var(monkeypatch):
    # A multi-user HTTP deployment must never fall back to a process-wide id.
    monkeypatch.setenv("MCP_USER_ID", "stdio-user")
    assert server._caller_id(_ctx({"x-user-id": "u1"})) == "u1"


def test_no_tool_takes_a_user_id_argument():
    """Identity travels out-of-band on purpose: a user_id the model fills in is
    a user_id the model can get wrong or be talked into changing."""
    for tool in server.mcp._tool_manager.list_tools():
        assert "user_id" not in (tool.parameters.get("properties") or {}), tool.name


# --------------------------------------------------------------------------- #
# ownership
# --------------------------------------------------------------------------- #
async def test_a_foreign_plan_id_is_rejected(monkeypatch):
    monkeypatch.setattr(server, "verify_plan_ownership", AsyncMock(return_value=False))
    with pytest.raises(ToolError, match="does not belong to you"):
        await server._require_own_plan("u1", "someone-elses-plan")


async def test_an_owned_plan_id_passes(monkeypatch):
    monkeypatch.setattr(server, "verify_plan_ownership", AsyncMock(return_value=True))
    await server._require_own_plan("u1", "my-plan")  # must not raise


@pytest.mark.parametrize(
    "tool, extra_args, backend_attr, backend_owner",
    [
        (server.get_meal_plan, {}, "get_plan_slots", "service"),
        (server.get_grocery_list, {}, "build_grocery_list", "server"),
        (
            server.log_meal,
            {"recipe": "dal", "day_of_week": 0, "meal_type": "dinner"},
            "log_recipe_to_slot",
            "server",
        ),
    ],
)
async def test_plan_scoped_tools_check_ownership_before_touching_data(
    monkeypatch, tool, extra_args, backend_attr, backend_owner
):
    monkeypatch.setattr(server, "verify_plan_ownership", AsyncMock(return_value=False))
    target = server.service if backend_owner == "service" else server
    backend = AsyncMock()
    monkeypatch.setattr(target, backend_attr, backend)

    with pytest.raises(ToolError, match="does not belong to you"):
        await tool(_ctx({"x-user-id": "u1"}), "not-mine", **extra_args)

    backend.assert_not_awaited()


# --------------------------------------------------------------------------- #
# propose / save split
# --------------------------------------------------------------------------- #
async def test_propose_generates_without_persisting(monkeypatch):
    build = AsyncMock(return_value=[{"recipe_name": "dal"}])
    save = AsyncMock()
    monkeypatch.setattr(server.service, "build_plan", build)
    monkeypatch.setattr(server.service, "save_plan", save)
    monkeypatch.setattr(server, "_profile", AsyncMock(return_value=({}, {})))

    result = await server.propose_meal_plan(_ctx({"x-user-id": "u1"}), "high protein")

    assert result["saved"] is False
    assert result["slots"] == [{"recipe_name": "dal"}]
    save.assert_not_awaited()


async def test_save_persists_the_approved_plan(monkeypatch):
    save = AsyncMock(return_value="plan-9")
    monkeypatch.setattr(server.service, "save_plan", save)
    monkeypatch.setattr(server, "_profile", AsyncMock(return_value=({}, {})))

    result = await server.save_meal_plan(
        _ctx({"x-user-id": "u1"}),
        "2026-01-05",
        [
            server.ProposedSlot(
                day_of_week=0, meal_type="dinner", recipe_name="dal", protein_g=20
            )
        ],
    )

    assert result == {"plan_id": "plan-9", "slots_saved": 1, "saved": True}
    assert save.await_args.args[0] == "u1"


async def test_save_checks_ownership_before_overwriting_an_existing_plan(monkeypatch):
    monkeypatch.setattr(server, "verify_plan_ownership", AsyncMock(return_value=False))
    save = AsyncMock()
    monkeypatch.setattr(server.service, "save_plan", save)

    with pytest.raises(ToolError, match="does not belong to you"):
        await server.save_meal_plan(
            _ctx({"x-user-id": "u1"}), "2026-01-05", [], plan_id="not-mine"
        )
    save.assert_not_awaited()


# --------------------------------------------------------------------------- #
# what the client lets a tool loop drive by itself
# --------------------------------------------------------------------------- #
def test_the_save_tool_is_withheld_from_the_autonomous_loop():
    """Saving needs human approval, which a tool loop has no way to pause for,
    so the supervisor calls it directly once its interrupt resolves."""
    assert SAVE_PLAN_TOOL not in AUTONOMOUS_TOOLS

    tools = []
    for name in [*AUTONOMOUS_TOOLS, SAVE_PLAN_TOOL]:
        tool = MagicMock()
        tool.name = name
        tools.append(tool)

    allowed = {t.name for t in autonomous_tools(tools)}
    assert allowed == AUTONOMOUS_TOOLS
    assert find_tool(tools, SAVE_PLAN_TOOL) is not None


def test_every_autonomous_tool_actually_exists_on_the_server():
    """Guards against a rename on the server silently shrinking what the
    supervisor is allowed to call."""
    published = {t.name for t in server.mcp._tool_manager.list_tools()}
    assert AUTONOMOUS_TOOLS | {SAVE_PLAN_TOOL} == published
