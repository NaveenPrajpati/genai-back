"""
personal_assistant.py
======================
A Personal Intelligent Assistant built as a LangGraph multi-agent system.

Pieces (mirrors the architecture of meal_planner.py):
  * Multi-agent graph
      - todo_agent     → manages a To-Do list stored in MongoDB
      - research_agent → researches a topic via a web Search API (Tavily)
  * Human-in-the-Loop  → destructive actions (deleting tasks) pause for approval
                         via LangGraph `interrupt()` + the Supabase `approvals` table
  * Persistent memory  → per-user facts in the Supabase `memory` table, loaded at
                         the start of every run so context survives across sessions
  * Automation engine  → `run_pa_triggers` produces a scheduled task digest, driven
                         by rows in the Supabase `triggers` table (action_type=pa_digest)

State stores (shared with meal_planner, namespaced by action_type / memory key):
  * MongoDB  `todos`        collection — the To-Do list
  * Supabase `approvals`    table      — HITL proposals (action_type="pa_delete_task")
  * Supabase `memory`       table      — persistent memory (keys prefixed "pa_")
  * Supabase `triggers`     table      — automation schedules (action_type="pa_digest")
"""

from typing import TypedDict, Optional, List, Literal, Annotated
from datetime import datetime
import logging
import os
import uuid

import httpx
from bson import ObjectId
from dotenv import load_dotenv
from fastapi import APIRouter, Request, HTTPException, Depends
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import START, StateGraph, END
from langgraph.types import interrupt, Command

from app.core.llm import llm
from app.core.config import supabase
from app.database import get_db
from app.dependencies import get_current_user

load_dotenv()

logger = logging.getLogger(__name__)

paRouter = APIRouter(
    prefix="/personal-assistant",
    tags=["personal-assistant"],
    responses={404: {"description": "Not found"}},
)

TODOS = "todos"  # MongoDB collection name
MEMORY_RESEARCH_KEY = "pa_research_history"
MEMORY_COMPLETED_KEY = "pa_completed_history"


# --------------------------------------------------------------------------- #
# State + schemas
# --------------------------------------------------------------------------- #
class PAState(TypedDict, total=False):
    query: str
    intent: str
    current_user: dict
    user_id: str
    thread_id: str
    memory: dict
    todos: Optional[list]
    research: Optional[dict]
    suggestions: Optional[list]
    task_status: Optional[str]
    response: Optional[str]


class IntentOutput(BaseModel):
    intent: Literal["add", "list", "complete", "delete", "update", "research"]


class TaskInput(BaseModel):
    title: str
    details: Optional[str] = None
    priority: Optional[Literal["low", "medium", "high"]] = "medium"
    due_at: Optional[str] = None  # ISO date/datetime if the user gave one


class TaskUpdateInput(BaseModel):
    """Fields the LLM extracts when the user wants to modify an existing task."""

    title: Optional[str] = None  # existing task to match
    new_title: Optional[str] = None
    new_priority: Optional[Literal["low", "medium", "high"]] = None
    new_due_at: Optional[str] = None
    new_details: Optional[str] = None


class TaskUpdateRequest(BaseModel):
    title: Optional[str] = None
    details: Optional[str] = None
    priority: Optional[Literal["low", "medium", "high"]] = None
    due_at: Optional[str] = None
    status: Optional[Literal["pending", "done"]] = None


class TaskSelector(BaseModel):
    """Which existing task(s) a complete/delete request refers to."""

    title: Optional[str] = None
    match_all: bool = False


class ResearchOutput(BaseModel):
    summary: str
    key_points: List[str] = []
    sources: List[str] = []


class QueryRequest(BaseModel):
    text: str
    thread_id: Optional[str] = None


class ApproveRequest(BaseModel):
    thread_id: str
    decision: Literal["approved", "rejected"]


# --------------------------------------------------------------------------- #
# MongoDB To-Do helpers (kept small so they're trivially mockable in tests)
# --------------------------------------------------------------------------- #
def _serialize(doc: dict) -> dict:
    """Make a Mongo doc JSON-safe (ObjectId -> str)."""
    out = dict(doc)
    if "_id" in out:
        out["id"] = str(out.pop("_id"))
    return out


async def insert_todo(user_id: str, data: dict) -> dict:
    now = datetime.now().isoformat()
    doc = {
        "user_id": user_id,
        "title": data["title"],
        "details": data.get("details"),
        "priority": data.get("priority", "medium"),
        "due_at": data.get("due_at"),
        "status": "pending",
        "created_at": now,
        "updated_at": now,
    }
    res = await get_db()[TODOS].insert_one(doc)
    doc["_id"] = res.inserted_id
    return _serialize(doc)


async def fetch_todos(
    user_id: str,
    status: Optional[str] = None,
    priority: Optional[str] = None,
) -> list:
    query: dict = {"user_id": user_id}
    if status:
        query["status"] = status
    if priority:
        query["priority"] = priority
    cursor = get_db()[TODOS].find(query).sort("created_at", -1)
    docs = await cursor.to_list(length=500)
    return [_serialize(d) for d in docs]


async def find_pending_todos(
    user_id: str, title: Optional[str], match_all: bool
) -> list:
    """Resolve a TaskSelector to the matching pending todos (serialized)."""
    docs = (
        await get_db()[TODOS]
        .find({"user_id": user_id, "status": "pending"})
        .to_list(length=500)
    )
    if match_all:
        matched = docs
    elif title:
        needle = title.lower()
        matched = [d for d in docs if needle in (d.get("title", "").lower())]
    else:
        matched = []
    return [_serialize(d) for d in matched]


async def complete_todo(user_id: str, title: str) -> Optional[dict]:
    matches = await find_pending_todos(user_id, title, match_all=False)
    if not matches:
        return None
    target_id = matches[0]["id"]
    await get_db()[TODOS].update_one(
        {"_id": ObjectId(target_id), "user_id": user_id},
        {"$set": {"status": "done", "updated_at": datetime.now().isoformat()}},
    )
    return matches[0]


async def delete_todos_by_ids(user_id: str, ids: List[str]) -> int:
    if not ids:
        return 0
    res = await get_db()[TODOS].delete_many(
        {"user_id": user_id, "_id": {"$in": [ObjectId(i) for i in ids]}}
    )
    return res.deleted_count


async def get_todo_by_id(user_id: str, task_id: str) -> Optional[dict]:
    try:
        doc = await get_db()[TODOS].find_one(
            {"_id": ObjectId(task_id), "user_id": user_id}
        )
        return _serialize(doc) if doc else None
    except Exception:
        return None


async def update_todo(user_id: str, title: str, updates: dict) -> Optional[dict]:
    """Find a pending task by title and apply field updates (agent path)."""
    matches = await find_pending_todos(user_id, title, match_all=False)
    if not matches:
        return None
    target_id = matches[0]["id"]
    updates["updated_at"] = datetime.now().isoformat()
    await get_db()[TODOS].update_one(
        {"_id": ObjectId(target_id), "user_id": user_id},
        {"$set": updates},
    )
    doc = await get_db()[TODOS].find_one({"_id": ObjectId(target_id)})
    return _serialize(doc) if doc else None


async def update_todo_by_id(user_id: str, task_id: str, updates: dict) -> Optional[dict]:
    """Update a task by its MongoDB ID (direct API path)."""
    try:
        updates["updated_at"] = datetime.now().isoformat()
        result = await get_db()[TODOS].update_one(
            {"_id": ObjectId(task_id), "user_id": user_id},
            {"$set": updates},
        )
        if result.matched_count == 0:
            return None
        doc = await get_db()[TODOS].find_one({"_id": ObjectId(task_id)})
        return _serialize(doc) if doc else None
    except Exception:
        return None


async def delete_todo_by_id(user_id: str, task_id: str) -> bool:
    """Delete a single task by its MongoDB ID (direct API path, no HITL)."""
    try:
        res = await get_db()[TODOS].delete_one(
            {"_id": ObjectId(task_id), "user_id": user_id}
        )
        return res.deleted_count > 0
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# Persistent memory helpers (Supabase `memory` table)
# --------------------------------------------------------------------------- #
async def load_memory(state: PAState):
    user_id = state["user_id"]
    memory: dict = {}
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
        logger.error("pa load memory error: %s", e)
    return {"memory": memory}


async def remember(user_id: str, key: str, value):
    try:
        supabase.table("memory").upsert(
            {"user_id": user_id, "key": key, "value": value},
            on_conflict="user_id,key",
        ).execute()
    except Exception as e:
        logger.error("pa remember error: %s", e)


async def append_memory_list(user_id: str, key: str, item: str, cap: int = 50):
    """Append an item to a list-valued memory entry, de-duplicated and capped."""
    try:
        row = (
            supabase.table("memory")
            .select("value")
            .eq("user_id", user_id)
            .eq("key", key)
            .maybe_single()
            .execute()
        )
        existing = list(row.data["value"]) if row and row.data else []
    except Exception as e:
        logger.error("pa append_memory_list lookup error: %s", e)
        existing = []
    merged = list(dict.fromkeys(existing + [item]))[-cap:]
    await remember(user_id, key, merged)
    return merged


# --------------------------------------------------------------------------- #
# Search API (Tavily) — used by research_agent
# --------------------------------------------------------------------------- #
async def web_search(query: str, max_results: int = 5) -> dict:
    """Run a web search via the Tavily API. Returns
    {answer, results:[{title,url,content}]}. Degrades gracefully to empty
    results if no API key is configured."""
    api_key = os.getenv("TAVILY_API_KEY", "")
    if not api_key:
        logger.info("TAVILY_API_KEY not set; skipping web search")
        return {"answer": "", "results": []}
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": api_key,
                    "query": query,
                    "max_results": max_results,
                    "search_depth": "basic",
                    "include_answer": True,
                },
            )
            if resp.status_code != 200:
                logger.error("Tavily error %s: %s", resp.status_code, resp.text)
                return {"answer": "", "results": []}
            data = resp.json()
            return {
                "answer": data.get("answer", "") or "",
                "results": [
                    {
                        "title": r.get("title", ""),
                        "url": r.get("url", ""),
                        "content": r.get("content", ""),
                    }
                    for r in data.get("results", [])
                ],
            }
    except Exception as e:
        logger.error("web_search error: %s", e)
        return {"answer": "", "results": []}


# --------------------------------------------------------------------------- #
# Graph nodes
# --------------------------------------------------------------------------- #
async def classify_intent(state: PAState):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Classify the user's message into exactly one intent:\n"
                "- add: create a new to-do / task\n"
                "- list: view or check existing tasks\n"
                "- complete: mark an existing task as done\n"
                "- delete: remove/cancel an existing task (destructive)\n"
                "- update: change an existing task's title, priority, due date, or details\n"
                "- research: look up information about a topic or task\n"
                "Reply with one word only: add, list, complete, delete, update, or research.",
            ),
            ("human", "{text}"),
        ]
    )
    chain = prompt | llm.with_structured_output(IntentOutput)
    result: IntentOutput = await chain.ainvoke({"text": state.get("query", "")})
    logger.info("pa intent: %s", result)
    return {"intent": result.intent}


async def _extract_selector(text: str) -> TaskSelector:
    chain = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "From the user's message, identify which existing task they mean. "
                "Return the task title to match on, or set match_all=true if they "
                "refer to all/every task.",
            ),
            ("human", "{text}"),
        ]
    ) | llm.with_structured_output(TaskSelector)
    return await chain.ainvoke({"text": text})


async def todo_agent(state: PAState):
    """Manages the MongoDB to-do list. Deleting tasks is gated behind HITL."""
    intent = state.get("intent")
    user_id = state["user_id"]

    if intent == "add":
        chain = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Extract a single to-do task from the user's message: a short "
                    "title, optional details, priority (low/medium/high), and an "
                    "ISO due date if one is mentioned.",
                ),
                ("human", "{text}"),
            ]
        ) | llm.with_structured_output(TaskInput)
        task: TaskInput = await chain.ainvoke({"text": state["query"]})
        created = await insert_todo(user_id, task.model_dump(exclude_none=True))
        return {"intent": "add", "task_status": "added", "todos": [created]}

    if intent == "list":
        todos = await fetch_todos(user_id, status="pending")
        return {"intent": "list", "task_status": "listed", "todos": todos}

    if intent == "complete":
        selector = await _extract_selector(state["query"])
        done = await complete_todo(user_id, selector.title or state["query"])
        if not done:
            return {"intent": "complete", "task_status": "not_found"}
        await append_memory_list(user_id, MEMORY_COMPLETED_KEY, done["title"])
        return {
            "intent": "complete",
            "task_status": "completed",
            "todos": await fetch_todos(user_id, status="pending"),
        }

    if intent == "delete":
        return await _delete_with_approval(state)

    if intent == "update":
        chain = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "From the user's message, extract: which existing task to update "
                    "(title field to match) and what changes to apply "
                    "(new_title, new_priority, new_due_at, new_details). "
                    "Only populate fields that the user explicitly wants to change.",
                ),
                ("human", "{text}"),
            ]
        ) | llm.with_structured_output(TaskUpdateInput)
        update_input: TaskUpdateInput = await chain.ainvoke({"text": state["query"]})
        updates = {
            k: v
            for k, v in {
                "title": update_input.new_title,
                "priority": update_input.new_priority,
                "due_at": update_input.new_due_at,
                "details": update_input.new_details,
            }.items()
            if v is not None
        }
        updated = await update_todo(
            user_id, update_input.title or state["query"], updates
        )
        if not updated:
            return {"intent": "update", "task_status": "not_found"}
        return {"intent": "update", "task_status": "updated", "todos": [updated]}

    return {"intent": intent, "task_status": "unknown"}


async def _delete_with_approval(state: PAState):
    """Destructive: build a deletion proposal, pause for human approval, then act.

    Re-run safe: on resume LangGraph replays this node from the top, so we look
    up an existing pending approval for this thread before creating a new one.
    """
    user_id = state["user_id"]
    thread_id = state["thread_id"]

    approval_id = None
    proposed = None
    try:
        existing = (
            supabase.table("approvals")
            .select("id, payload")
            .eq("thread_id", thread_id)
            .eq("status", "pending")
            .maybe_single()
            .execute()
        )
        if existing and existing.data:
            approval_id = existing.data["id"]
            proposed = existing.data["payload"]["tasks"]
    except Exception as e:
        logger.error("pa approval lookup error: %s", e)

    if not approval_id:
        selector = await _extract_selector(state["query"])
        matches = await find_pending_todos(
            user_id, selector.title, selector.match_all
        )
        if not matches:
            return {"intent": "delete", "task_status": "not_found"}
        proposed = [{"id": m["id"], "title": m["title"]} for m in matches]
        try:
            res = (
                supabase.table("approvals")
                .insert(
                    {
                        "user_id": user_id,
                        "thread_id": thread_id,
                        "action_type": "pa_delete_task",
                        "payload": {"tasks": proposed},
                        "status": "pending",
                    }
                )
                .execute()
            )
            approval_id = res.data[0]["id"] if res.data else None
        except Exception as e:
            logger.error("pa approval insert error: %s", e)

    decision = interrupt(
        {"type": "pa_delete_task", "approval_id": approval_id, "tasks": proposed}
    )

    if decision != "approved":
        if approval_id:
            supabase.table("approvals").update(
                {"status": "rejected", "resolved_at": datetime.now().isoformat()}
            ).eq("id", approval_id).execute()
        return {"intent": "delete", "task_status": "delete_rejected"}

    deleted = await delete_todos_by_ids(
        user_id, [t["id"] for t in (proposed or [])]
    )
    if approval_id:
        supabase.table("approvals").update(
            {"status": "approved", "resolved_at": datetime.now().isoformat()}
        ).eq("id", approval_id).execute()
    return {
        "intent": "delete",
        "task_status": f"deleted:{deleted}",
        "todos": await fetch_todos(user_id, status="pending"),
    }


async def research_agent(state: PAState):
    """Researches the user's topic via the Search API and summarizes it."""
    topic = state["query"]
    search = await web_search(topic)

    context_lines = []
    if search.get("answer"):
        context_lines.append(f"Search answer: {search['answer']}")
    for r in search.get("results", []):
        context_lines.append(f"- {r['title']} ({r['url']})\n  {r['content']}")
    context = "\n".join(context_lines) or "No external results available."

    messages = [
        SystemMessage(
            content=(
                "You are a research assistant. Using the provided search results, "
                "write a concise summary, a few key points, and list the source "
                "URLs. If the results are empty, answer from general knowledge and "
                "leave sources empty."
            )
        ),
        HumanMessage(content=f"Topic: {topic}\n\nSearch results:\n{context}"),
    ]
    structured: ResearchOutput = await llm.with_structured_output(
        ResearchOutput
    ).ainvoke(messages)

    await append_memory_list(state["user_id"], MEMORY_RESEARCH_KEY, topic)

    return {
        "intent": "research",
        "research": structured.model_dump(),
        "suggestions": structured.key_points,
    }


def decide_agent(state: PAState):
    intent = state.get("intent")
    if intent in ("add", "list", "complete", "delete", "update"):
        return "todo_agent"
    if intent == "research":
        return "research_agent"
    return END


class SynthesisOutput(BaseModel):
    response: str


async def synthesize_response(state: PAState):
    """Turn the structured operation result into a short, friendly plain-English reply."""
    intent = state.get("intent", "")
    task_status = state.get("task_status", "")
    todos = state.get("todos") or []
    research = state.get("research")

    parts = [f"Intent: {intent}", f"Status: {task_status}"]
    if todos:
        task_lines = "\n".join(
            f"  - [{t.get('priority', '?')}] {t.get('title', '')} "
            f"(due: {t.get('due_at') or 'none'}, status: {t.get('status', '')})"
            for t in todos[:20]
        )
        parts.append(f"Tasks:\n{task_lines}")
    if research:
        parts.append(f"Research summary: {research.get('summary', '')}")
        if research.get("key_points"):
            parts.append("Key points: " + "; ".join(research["key_points"]))

    context = "\n".join(parts)
    messages = [
        SystemMessage(
            content=(
                "You are a concise personal assistant. Based on the operation result "
                "below, write a short, friendly reply to the user. Describe what "
                "happened in plain language — do not dump raw data."
            )
        ),
        HumanMessage(
            content=f'User said: "{state.get("query", "")}"\n\nResult:\n{context}'
        ),
    ]
    result: SynthesisOutput = await llm.with_structured_output(SynthesisOutput).ainvoke(
        messages
    )
    return {"response": result.response}


# --------------------------------------------------------------------------- #
# Graph wiring
# --------------------------------------------------------------------------- #
graph = StateGraph(PAState)
graph.add_node("load_memory", load_memory)
graph.add_node("classify_intent", classify_intent)
graph.add_node("todo_agent", todo_agent)
graph.add_node("research_agent", research_agent)
graph.add_node("synthesize", synthesize_response)
graph.add_edge(START, "load_memory")
graph.add_edge("load_memory", "classify_intent")
graph.add_conditional_edges(
    "classify_intent",
    decide_agent,
    ["todo_agent", "research_agent", END],
)
graph.add_edge("todo_agent", "synthesize")
graph.add_edge("research_agent", "synthesize")
graph.add_edge("synthesize", END)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _jsonable(result: dict) -> dict:
    """Drop non-serializable LangGraph internals before returning to clients."""
    return {k: v for k, v in result.items() if not k.startswith("__")}


# --------------------------------------------------------------------------- #
# HTTP endpoints
# --------------------------------------------------------------------------- #
@paRouter.post("/query")
async def ask(
    body: QueryRequest,
    request: Request,
    current_user: Annotated[dict, Depends(get_current_user)],
):
    agent = request.app.state.pa_agent
    thread_id = body.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    _excluded = {"_id", "expires_at", "password_hash"}
    user_data = {k: v for k, v in current_user.items() if k not in _excluded}

    result = await agent.ainvoke(
        {
            "query": body.text,
            "user_id": current_user["uid"],
            "thread_id": thread_id,
            "current_user": user_data,
        },
        config=config,
    )

    if "__interrupt__" in result:
        return {
            "status": "needs_approval",
            "thread_id": thread_id,
            "proposal": result["__interrupt__"][0].value,
        }
    return {"status": "done", "result": _jsonable(result)}


@paRouter.post("/approve")
async def approve(
    body: ApproveRequest,
    request: Request,
    current_user: Annotated[dict, Depends(get_current_user)],
):
    agent = request.app.state.pa_agent
    config = {"configurable": {"thread_id": body.thread_id}}

    # The pending approval must belong to the caller (prevents IDOR via guessed
    # thread_id) and must be a personal-assistant action.
    try:
        approval = (
            supabase.table("approvals")
            .select("id, user_id")
            .eq("thread_id", body.thread_id)
            .eq("action_type", "pa_delete_task")
            .eq("status", "pending")
            .maybe_single()
            .execute()
        )
    except Exception as e:
        logger.error("pa approval ownership lookup error: %s", e)
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
            detail="No paused action for this thread. The server may have "
            "restarted — please re-submit your request.",
        )

    result = await agent.ainvoke(Command(resume=body.decision), config=config)
    return {"status": "done", "result": _jsonable(result)}


@paRouter.get("/approve")
async def list_approvals(current_user: Annotated[dict, Depends(get_current_user)]):
    try:
        result = (
            supabase.table("approvals")
            .select("*")
            .eq("user_id", current_user["uid"])
            .in_("action_type", ["pa_delete_task", "pa_digest"])
            .eq("status", "pending")
            .execute()
        )
        return {"status": "done", "result": result.data or []}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@paRouter.get("/tasks/stats")
async def task_stats(current_user: Annotated[dict, Depends(get_current_user)]):
    pipeline = [
        {"$match": {"user_id": current_user["uid"]}},
        {
            "$group": {
                "_id": {"status": "$status", "priority": "$priority"},
                "count": {"$sum": 1},
            }
        },
    ]
    docs = await get_db()[TODOS].aggregate(pipeline).to_list(length=100)
    stats: dict = {"total": 0, "by_status": {}, "by_priority": {}}
    for doc in docs:
        s = doc["_id"]["status"]
        p = doc["_id"].get("priority") or "none"
        n = doc["count"]
        stats["total"] += n
        stats["by_status"][s] = stats["by_status"].get(s, 0) + n
        stats["by_priority"][p] = stats["by_priority"].get(p, 0) + n
    return {"status": "done", "result": stats}


@paRouter.get("/tasks")
async def list_tasks(
    current_user: Annotated[dict, Depends(get_current_user)],
    status: Optional[str] = None,
    priority: Optional[str] = None,
):
    todos = await fetch_todos(current_user["uid"], status=status, priority=priority)
    return {"status": "done", "result": todos}


@paRouter.get("/tasks/{task_id}")
async def get_task(
    task_id: str,
    current_user: Annotated[dict, Depends(get_current_user)],
):
    todo = await get_todo_by_id(current_user["uid"], task_id)
    if not todo:
        raise HTTPException(status_code=404, detail="Task not found.")
    return {"status": "done", "result": todo}


@paRouter.put("/tasks/{task_id}")
async def update_task(
    task_id: str,
    body: TaskUpdateRequest,
    current_user: Annotated[dict, Depends(get_current_user)],
):
    updates = body.model_dump(exclude_none=True)
    if not updates:
        raise HTTPException(status_code=400, detail="No fields to update.")
    updated = await update_todo_by_id(current_user["uid"], task_id, updates)
    if not updated:
        raise HTTPException(status_code=404, detail="Task not found.")
    return {"status": "done", "result": updated}


@paRouter.delete("/tasks/{task_id}")
async def delete_task(
    task_id: str,
    current_user: Annotated[dict, Depends(get_current_user)],
):
    deleted = await delete_todo_by_id(current_user["uid"], task_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Task not found.")
    return {"status": "done"}


@paRouter.post("/toggle-trigger")
async def toggle_trigger(current_user: Annotated[dict, Depends(get_current_user)]):
    user_id = current_user["uid"]
    try:
        existing = (
            supabase.table("triggers")
            .select("*")
            .eq("user_id", user_id)
            .eq("action_type", "pa_digest")
            .execute()
        )
        new_enabled = True
        if existing and existing.data:
            new_enabled = not existing.data[0]["enabled"]
            for t in existing.data:
                supabase.table("triggers").update({"enabled": not t["enabled"]}).eq(
                    "id", t["id"]
                ).execute()
        else:
            supabase.table("triggers").insert(
                {
                    "user_id": user_id,
                    "name": "daily task digest",
                    "schedule": "0 8 * * *",
                    "action_type": "pa_digest",
                    "enabled": True,
                    "last_run_at": None,
                }
            ).execute()
        return {"status": "done", "enabled": new_enabled}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@paRouter.get("/memory")
async def get_memory(current_user: Annotated[dict, Depends(get_current_user)]):
    try:
        rows = (
            supabase.table("memory")
            .select("key, value")
            .eq("user_id", current_user["uid"])
            .execute()
        )
        return {"status": "done", "result": rows.data or []}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --------------------------------------------------------------------------- #
# Automation engine — daily task digest
# --------------------------------------------------------------------------- #
async def run_pa_triggers(agent=None):
    """For each enabled pa_digest trigger, snapshot the user's pending tasks into
    an approvals-table notification the user can review via GET /approve."""
    logger.info("pa digest job running")
    now = datetime.now()
    try:
        triggers = (
            supabase.table("triggers")
            .select("*")
            .eq("enabled", True)
            .eq("action_type", "pa_digest")
            .execute()
        )
    except Exception as e:
        logger.error("run_pa_triggers fetch error: %s", e)
        return

    for t in triggers.data or []:
        try:
            pending = await fetch_todos(t["user_id"], status="pending")
            supabase.table("approvals").insert(
                {
                    "user_id": t["user_id"],
                    "thread_id": str(uuid.uuid4()),
                    "action_type": "pa_digest",
                    "payload": {
                        "generated_at": now.isoformat(),
                        "pending_count": len(pending),
                        "tasks": pending,
                    },
                    "status": "pending",
                }
            ).execute()
            supabase.table("triggers").update(
                {"last_run_at": now.isoformat()}
            ).eq("id", t["id"]).execute()
            logger.info(
                "pa digest created for user=%s (%d pending)",
                t["user_id"],
                len(pending),
            )
        except Exception as e:
            logger.error("pa digest error for user=%s: %s", t.get("user_id"), e)
