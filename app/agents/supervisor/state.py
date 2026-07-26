"""State and structured-output schemas for the supervisor graph."""

import operator
from typing import Annotated, List, Literal, Optional, TypedDict

from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

# The three skills the supervisor can dispatch to. "learning" and "assistant"
# are LangGraph subgraphs; "meal" is reached over MCP.
Skill = Literal["learning", "assistant", "meal"]


def merge_results(left: Optional[dict], right: Optional[dict]) -> dict:
    """Reducer for `results`: each skill writes under its own key, so later
    writes extend the dict rather than replacing what earlier skills produced."""
    return {**(left or {}), **(right or {})}


class SupervisorState(TypedDict, total=False):
    # --- input ---
    query: str
    user_id: str
    thread_id: str
    current_user: dict
    # Optional deep links, forwarded to whichever subgraph needs them.
    roadmapId: Optional[str]
    plan_id: Optional[str]

    # --- routing ---
    memory: dict
    route: List[str]  # ordered skills to run this turn
    route_reason: str
    # Appended by each skill node as it finishes; the dispatcher runs whatever
    # is in `route` but not yet here, so a resumed run never repeats a skill.
    completed: Annotated[List[str], operator.add]
    results: Annotated[dict, merge_results]

    # --- output ---
    response: str
    messages: Annotated[list, add_messages]


class RouteDecision(BaseModel):
    """Which skills this message needs, in the order they should run."""

    skills: List[Skill] = Field(
        default_factory=list,
        description=(
            "Skills to invoke, in execution order. Empty when the supervisor can "
            "answer directly without any skill."
        ),
    )
    reason: str = Field(description="One short sentence on why these skills")
