"""MongoDB to-do persistence and Supabase memory helpers for the personal-assistant agent."""

import logging
from datetime import datetime, date, timedelta
from typing import Optional, List

from bson import ObjectId
from dateutil import parser as date_parser
from dateutil.relativedelta import relativedelta

from app.core.config import supabase
from app.database import get_db

logger = logging.getLogger(__name__)

TODOS = "todos"
NOTES_KEY = "pa_notes"


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
        "recurrence": data.get("recurrence"),
        # Set when this is a subtask spawned from a larger goal (breakdown).
        "parent_id": data.get("parent_id"),
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


def _next_due(due_at: Optional[str], recurrence: str) -> str:
    """Compute the next due date for a recurring task. Bases off the existing
    due date when present, otherwise off now."""
    try:
        base = date_parser.parse(due_at) if due_at else datetime.now()
    except (ValueError, TypeError):
        base = datetime.now()
    if recurrence == "daily":
        base = base + timedelta(days=1)
    elif recurrence == "weekly":
        base = base + timedelta(weeks=1)
    elif recurrence == "monthly":
        base = base + relativedelta(months=1)
    return base.isoformat()


async def complete_todo(user_id: str, title: str) -> Optional[dict]:
    matches = await find_pending_todos(user_id, title, match_all=False)
    if not matches:
        return None
    target = matches[0]
    target_id = target["id"]
    await get_db()[TODOS].update_one(
        {"_id": ObjectId(target_id), "user_id": user_id},
        {"$set": {"status": "done", "updated_at": datetime.now().isoformat()}},
    )
    # Recurring task: queue up the next occurrence so the user never loses it.
    recurrence = target.get("recurrence")
    if recurrence:
        follow_up = await insert_todo(
            user_id,
            {
                "title": target["title"],
                "details": target.get("details"),
                "priority": target.get("priority", "medium"),
                "due_at": _next_due(target.get("due_at"), recurrence),
                "recurrence": recurrence,
            },
        )
        target = {**target, "next_occurrence": follow_up}
    return target


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


async def update_todo_by_id(
    user_id: str, task_id: str, updates: dict
) -> Optional[dict]:
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
# Notes / personal facts (stored as a list under the pa_notes memory key)
# --------------------------------------------------------------------------- #
async def fetch_notes(user_id: str) -> list:
    try:
        row = (
            supabase.table("memory")
            .select("value")
            .eq("user_id", user_id)
            .eq("key", NOTES_KEY)
            .maybe_single()
            .execute()
        )
        return list(row.data["value"]) if row and row.data else []
    except Exception as e:
        logger.error("pa fetch_notes error: %s", e)
        return []


async def add_note(
    user_id: str, content: str, category: Optional[str] = None, cap: int = 100
) -> dict:
    """Append a timestamped personal fact, newest-capped."""
    existing = await fetch_notes(user_id)
    note = {
        "content": content,
        "category": category,
        "created_at": datetime.now().isoformat(),
    }
    merged = (existing + [note])[-cap:]
    await remember(user_id, NOTES_KEY, merged)
    return note


# --------------------------------------------------------------------------- #
# Due-date awareness
# --------------------------------------------------------------------------- #
def categorize_agenda(todos: list) -> dict:
    """Bucket todos by due date into overdue / today / upcoming / no_date.

    Comparison is on the date portion only, so a task due any time today counts
    as today rather than overdue."""
    today = date.today().isoformat()
    buckets: dict = {"overdue": [], "today": [], "upcoming": [], "no_date": []}
    for t in todos:
        due = t.get("due_at")
        if not due:
            buckets["no_date"].append(t)
            continue
        day = str(due)[:10]
        if day < today:
            buckets["overdue"].append(t)
        elif day == today:
            buckets["today"].append(t)
        else:
            buckets["upcoming"].append(t)
    return buckets


# --------------------------------------------------------------------------- #
# Subtasks (child todos linked by parent_id)
# --------------------------------------------------------------------------- #
async def insert_subtasks(user_id: str, parent_id: str, titles: List[str]) -> list:
    created = []
    for title in titles:
        if not title.strip():
            continue
        created.append(
            await insert_todo(user_id, {"title": title, "parent_id": parent_id})
        )
    return created


async def fetch_subtasks(user_id: str, parent_id: str) -> list:
    docs = (
        await get_db()[TODOS]
        .find({"user_id": user_id, "parent_id": parent_id})
        .sort("created_at", 1)
        .to_list(length=200)
    )
    return [_serialize(d) for d in docs]
