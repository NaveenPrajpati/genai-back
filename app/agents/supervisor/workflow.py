"""The supervisor graph: one conversation over all three agents.

    START → load_context → route ─┬→ learning_agent ─┐
                                  ├→ assistant_agent ─┼→ (dispatch again) → finalize → END
                                  └→ meal_agent ──────┘

WHY THE SKILLS ARE REACHED THREE DIFFERENT WAYS
───────────────────────────────────────────────
  • learning / assistant — compiled LangGraph SUBGRAPHS, invoked inside a node.
    They keep their own state schemas, so each node maps supervisor state in and
    a compact summary out. They are compiled WITHOUT a checkpointer: a subgraph
    inherits the parent's, which is what makes an `interrupt()` deep inside the
    learning tracker's roadmap approval pause the whole supervisor run and
    resume from a single `Command(resume=...)` on the parent thread.
  • meal — reached over MCP (app/mcp/meal_planner/). No import of the meal agent
    here at all; it is a network dependency that happens to live in-process.

ONE SKILL PER NODE, NOT A LOOP INSIDE ONE NODE
──────────────────────────────────────────────
`route` returns an ordered list of skills and the dispatcher walks it, but each
skill is its own graph node. LangGraph checkpoints per node, so when a later
skill interrupts for approval, resuming replays only that node — the skills that
already ran are not re-executed and not re-billed. Running the queue inside a
single node would redo all of them on every resume.
"""

import json
import logging
from typing import Any, List, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt

from app.agents.approval_store import create_pending, get_pending, resolve
from app.agents.learning_tracker import graph as learning_graph
from app.agents.memory_store import get_profile
from app.agents.personal_assistant import graph as pa_graph
from app.agents.react import run_tool_loop
from app.core.config import CACHE_CLASSIFY_THRESHOLD
from app.core.llm import fast_llm, llm
from app.mcp.meal_planner.client import (
    SAVE_PLAN_TOOL,
    autonomous_tools,
    find_tool,
    load_meal_tools,
)
from app.services.cache import cached_value
from .state import RouteDecision, SupervisorState

logger = logging.getLogger(__name__)

# Compiled once at import. No checkpointer: they inherit the supervisor's, which
# is what lets their interrupts surface on the parent thread.
learning_subgraph = learning_graph.compile()
assistant_subgraph = pa_graph.compile()

MEAL_APPROVAL = "supervisor_save_meal_plan"
PROPOSE_TOOL = "propose_meal_plan"

SKILL_NODES = {
    "learning": "learning_agent",
    "assistant": "assistant_agent",
    "meal": "meal_agent",
}

ROUTER_SYSTEM = """\
You route a user's message to the skills that can serve it. The skills:

- learning: study roadmaps, study progress, quizzes, learning resources, and \
EXPLAINING OR TEACHING any concept or topic — including a bare "explain X", \
"how does X work", "what is X". Teaching is always this skill, never a direct answer.
- assistant: the user's own to-do list and tasks, reminders, agenda and due \
dates, personal notes and facts to remember, breaking a goal into subtasks, and \
general web research. Only tasks the user tracks — a shopping or grocery list \
that comes from a meal plan belongs to meal.
- meal: weekly meal plans, recipes, nutrition, GROCERY AND SHOPPING LISTS, and \
food likes, dislikes, diets and allergies.

Return the skills needed, in the order they should run — usually exactly one. \
Return two when the message genuinely asks for both ("plan my meals and add the \
shopping to my tasks" → meal, assistant).

When two skills could plausibly claim a message, these win: a grocery or \
shopping list is always meal, never assistant. Explaining or teaching is always \
learning, never a direct answer.

Anything that reads or changes the user's own data — their tasks, agenda, \
reminders, notes, roadmap, progress, meal plans — needs a skill. Route it even \
when the message is short or sounds casual. Returning no skill means answering \
with nothing but the conversation so far, so reserve it for greetings, thanks, \
and questions about this conversation itself.

Examples:
  "what's due tomorrow?"                  → assistant
  "what do I need to buy this week?"      → meal
  "how does TCP work?"                    → learning
  "am I on track with my studies?"        → learning
  "swap Tuesday dinner for something else" → meal
  "break my thesis into steps"            → assistant
  "make me a week of dinners and put the shopping on my list" → meal, assistant
  "good morning"                          → (none)
"""

FINALIZE_SYSTEM = """\
You are one assistant with three skills: learning coach, personal assistant, and \
meal planner. The skill results below are your own work — never mention skills, \
agents, routing, or tools, and never apologise for what you did not do.

Write a short, natural reply telling the user what happened and what matters \
from it. Use plain prose or a tight list; do not dump raw data structures. If a \
result reports an error, say plainly what did not work. Honor the user's \
preferred communication style if their profile gives one.
"""


# ── routing ──────────────────────────────────────────────────────────────────
async def load_context(state: SupervisorState):
    """Load the shared profile once and record the user's turn.

    The subgraphs re-read the profile in their own `load_memory` — one extra
    indexed Mongo read each, kept deliberately so they stay runnable standalone
    through their existing per-agent routes.
    """
    memory = await get_profile(state["user_id"])
    return {
        "memory": memory,
        "messages": [HumanMessage(content=state.get("query", ""))],
    }


async def route(state: SupervisorState):
    """Pick the skills for this turn."""
    query = state.get("query", "")
    chain = ChatPromptTemplate.from_messages(
        [("system", ROUTER_SYSTEM), ("human", "{text}")]
    ) | fast_llm.with_structured_output(RouteDecision)

    async def produce():
        decision: RouteDecision = await chain.ainvoke({"text": query})
        logger.info("supervisor route: %s (%s)", decision.skills, decision.reason)
        return decision.model_dump()

    # Routing depends only on the message and is user-independent → the same
    # global semantic cache the per-agent classifiers use.
    data = await cached_value(
        query, "agent:supervisor:route", CACHE_CLASSIFY_THRESHOLD, produce
    )
    # Dedupe, preserving order: a repeated skill would be skipped by the
    # dispatcher anyway (it runs what is in `route` but not in `completed`).
    skills = list(dict.fromkeys(data.get("skills") or []))
    return {"route": skills, "route_reason": data.get("reason", "")}


def dispatch(state: SupervisorState) -> str:
    """Next un-run skill in the queue, or finalize when the queue is done."""
    completed = set(state.get("completed") or [])
    for skill in state.get("route") or []:
        if skill not in completed:
            return SKILL_NODES[skill]
    return "finalize"


# ── skills ───────────────────────────────────────────────────────────────────
def _subgraph_input(state: SupervisorState) -> dict:
    return {
        "query": state.get("query", ""),
        "user_id": state["user_id"],
        "thread_id": state["thread_id"],
        "current_user": state.get("current_user") or {},
    }


async def learning_agent(state: SupervisorState):
    """Learning tracker as a subgraph. A roadmap approval interrupts in here and
    pauses the supervisor run."""
    out = await learning_subgraph.ainvoke(
        {**_subgraph_input(state), "roadmapId": state.get("roadmapId")}
    )
    roadmap = out.get("roadmap") if isinstance(out.get("roadmap"), dict) else {}
    topics = roadmap.get("topics") or []
    summary = {
        "intent": out.get("intent"),
        "roadmap_status": out.get("roadmap_status"),
        "roadmapId": out.get("roadmapId"),
        "roadmap_title": roadmap.get("title"),
        # Only when a roadmap survived: on rejection `roadmap` is absent, and a
        # literal 0 here reads to the finalizer as "produced an empty roadmap".
        "topic_count": len(topics) or None,
        "tasks_added_to_todo_list": out.get("pa_tasks_created"),
        "explanation": out.get("topic_explaination"),
        "quiz_questions": len(out.get("quiz") or []) or None,
        "quizId": out.get("quizId"),
        "quiz_result": out.get("quiz_result"),
        "next_topic": out.get("next_topic"),
        "progress": out.get("progress"),
        "resources": out.get("suggestions"),
    }
    return {
        "completed": ["learning"],
        "results": {"learning": {k: v for k, v in summary.items() if v is not None}},
        "roadmapId": out.get("roadmapId") or state.get("roadmapId"),
    }


async def assistant_agent(state: SupervisorState):
    """Personal assistant as a subgraph. Its delete-task approval interrupts in
    here the same way."""
    out = await assistant_subgraph.ainvoke(_subgraph_input(state))
    summary = {
        "intent": out.get("intent"),
        "status": out.get("task_status"),
        # The PA graph already synthesises prose; hand that to the finalizer as
        # the skill's own account of what it did rather than re-deriving it.
        "summary": out.get("response"),
        "tasks": [
            {
                "title": t.get("title"),
                "priority": t.get("priority"),
                "due": t.get("due_at"),
            }
            for t in (out.get("todos") or [])[:20]
        ]
        or None,
        "subtasks": [s.get("title") for s in (out.get("subtasks") or [])] or None,
        "agenda_counts": (
            {k: len(v) for k, v in out["agenda"].items()} if out.get("agenda") else None
        ),
        "notes": [n.get("content") for n in (out.get("notes") or [])[-20:]] or None,
        "research": out.get("research"),
    }
    return {
        "completed": ["assistant"],
        "results": {"assistant": {k: v for k, v in summary.items() if v is not None}},
    }


def _tool_result(payload: Any) -> Any:
    """Decode one MCP tool result, from either a ToolMessage or a direct
    `.ainvoke` return.

    MCP does not hand back a plain string: a result is a list of typed content
    blocks, and the value we want is the JSON carried in the first text block.
    Anything that is not JSON is passed through unchanged.
    """
    content = getattr(payload, "content", payload)
    if isinstance(content, list):
        texts = [
            block.get("text")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        content = texts[0] if texts else None
    if isinstance(content, str):
        try:
            return json.loads(content)
        except (ValueError, TypeError):
            return content
    return content


def _last_proposal(messages: List[Any]) -> Optional[dict]:
    """The most recent unsaved plan proposal in the tool transcript, if any."""
    for message in reversed(messages):
        if getattr(message, "name", None) != PROPOSE_TOOL:
            continue
        result = _tool_result(message)
        if isinstance(result, dict) and result.get("slots"):
            return result
    return None


def _final_text(messages: List[Any]) -> str:
    for message in reversed(messages):
        if isinstance(message, AIMessage) and message.content:
            return message.content
    return ""


async def meal_agent(state: SupervisorState):
    """Meal planner over MCP: a tool loop against the MCP server, then a human
    approval gate before anything is persisted.

    The gate cannot live in the tool loop — MCP has no way to pause a call — so
    `propose_meal_plan` (safe, writes nothing) is left to the model while
    `save_meal_plan` is called by this node after `interrupt()` resolves.
    """
    user_id = state["user_id"]
    thread_id = state["thread_id"]

    tools = await load_meal_tools(user_id)
    if not tools:
        return {
            "completed": ["meal"],
            "results": {
                "meal": {"error": "The meal planner is unavailable right now."}
            },
        }

    # Replay-safe, like the meal graph's own plan_agent: on resume this node runs
    # again from the top, so reuse the stored proposal instead of paying for a
    # second tool loop and offering the user a different plan than they approved.
    approval_id = None
    proposal = None
    transcript = ""
    existing = await get_pending(thread_id)
    if existing and existing.get("action_type") == MEAL_APPROVAL:
        approval_id = str(existing["_id"])
        payload = existing.get("payload") or {}
        proposal = payload.get("proposal")
        transcript = payload.get("transcript", "")
    else:
        loop_tools = autonomous_tools(tools)
        messages = await run_tool_loop(
            llm.bind_tools(loop_tools),
            ToolNode(loop_tools),
            [
                SystemMessage(
                    content=(
                        "You are a meal planning assistant with tools over the "
                        "user's plans. Check their preferences before suggesting "
                        "food, and look up the plan id when they refer to their "
                        "plan without naming one. To create a week of meals call "
                        "propose_meal_plan — the proposal is shown to the user "
                        "for approval, so do not claim it has been saved. "
                        "Finish with a brief plain-language summary."
                    )
                ),
                HumanMessage(content=state.get("query", "")),
            ],
        )
        transcript = _final_text(messages)
        proposal = _last_proposal(messages)
        if proposal:
            approval_id = await create_pending(
                user_id,
                thread_id,
                MEAL_APPROVAL,
                {"proposal": proposal, "transcript": transcript},
            )

    if not approval_id:
        return {
            "completed": ["meal"],
            "results": {"meal": {"summary": transcript}},
        }

    decision = interrupt(
        {
            "type": MEAL_APPROVAL,
            "approval_id": approval_id,
            "week_start": proposal.get("week_start"),
            "plan": proposal.get("slots"),
        }
    )

    if decision != "approved":
        await resolve(approval_id, "rejected")
        return {
            "completed": ["meal"],
            "results": {
                "meal": {"summary": transcript, "plan_status": "rejected by user"}
            },
        }

    save_tool = find_tool(tools, SAVE_PLAN_TOOL)
    saved = _tool_result(
        await save_tool.ainvoke(
            {
                "week_start": proposal.get("week_start"),
                "slots": proposal.get("slots"),
                "plan_id": state.get("plan_id"),
            }
        )
    )
    await resolve(approval_id, "approved")
    plan_id = saved.get("plan_id") if isinstance(saved, dict) else None
    return {
        "completed": ["meal"],
        "results": {
            "meal": {
                "summary": transcript,
                "plan_status": "approved and saved",
                "plan_id": plan_id,
                "meals_saved": len(proposal.get("slots") or []),
            }
        },
        "plan_id": plan_id or state.get("plan_id"),
    }


# ── response ─────────────────────────────────────────────────────────────────
def _render(state: SupervisorState) -> str:
    """Flatten the turn into the context the finalizer writes from."""
    parts = []
    memory = state.get("memory") or {}
    profile = {
        label: memory[key]
        for key, label in (
            ("communication_style", "Preferred style"),
            ("priorities", "Standing priorities"),
            ("skill_level", "Skill level"),
            ("diet_restrictions", "Diet"),
        )
        if memory.get(key)
    }
    if profile:
        parts.append(
            "User profile — " + "; ".join(f"{k}: {v}" for k, v in profile.items())
        )
    results = state.get("results") or {}
    if not results:
        parts.append("No skill was needed — answer the user directly.")
    for skill, payload in results.items():
        parts.append(f"[{skill}]\n{json.dumps(payload, indent=2, default=str)}")
    return "\n\n".join(parts)


async def finalize(state: SupervisorState):
    """One voice over whatever ran. Plain text (no structured output) so the
    route can stream these tokens to the client."""
    response = await llm.ainvoke(
        [
            SystemMessage(content=FINALIZE_SYSTEM),
            HumanMessage(
                content=f'User said: "{state.get("query", "")}"\n\n{_render(state)}'
            ),
        ]
    )
    text = (
        response.content if isinstance(response.content, str) else str(response.content)
    )
    return {"response": text, "messages": [AIMessage(content=text)]}


def build_graph() -> StateGraph:
    graph = StateGraph(SupervisorState)
    graph.add_node("load_context", load_context)
    graph.add_node("route", route)
    graph.add_node("learning_agent", learning_agent)
    graph.add_node("assistant_agent", assistant_agent)
    graph.add_node("meal_agent", meal_agent)
    graph.add_node("finalize", finalize)

    graph.add_edge(START, "load_context")
    graph.add_edge("load_context", "route")

    targets = [*SKILL_NODES.values(), "finalize"]
    graph.add_conditional_edges("route", dispatch, targets)
    # Back to the dispatcher after each skill, so a multi-skill turn walks the
    # queue and a single-skill turn falls straight through to finalize. A skill
    # is excluded from its own target list: `dispatch` skips anything already in
    # `completed`, so it can never route back into the node it just left.
    for node in SKILL_NODES.values():
        graph.add_conditional_edges(node, dispatch, [t for t in targets if t != node])

    graph.add_edge("finalize", END)
    return graph


graph = build_graph()
