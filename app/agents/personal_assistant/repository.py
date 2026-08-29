"""MongoDB to-do and memory helpers for the personal-assistant agent."""

import logging
import re
import uuid
from datetime import datetime, timezone, date, timedelta
from typing import Optional, List

from bson import ObjectId
from dateutil import parser as date_parser
from dateutil.relativedelta import relativedelta

from app.database import get_db

MEMORIES = "memories"

logger = logging.getLogger(__name__)

TODOS = "todos"
NOTES_KEY = "pa_notes"

# Filter value meaning "tasks with no originating agent" — the ones the user
# created here, rather than ones another agent handed to the assistant.
SOURCE_SELF = "self"

# Fields `GET /tasks` may sort on, each with the direction that is useful when
# the caller doesn't name one: newest first for timestamps, soonest first for a
# due date, most urgent first for priority, A-Z for a title.
TODO_SORT_FIELDS = {
    "created_at": "desc",
    "updated_at": "desc",
    "due_at": "asc",
    "priority": "desc",
    "title": "asc",
}

# Priority is stored as a word, so sorting the field itself orders it
# high < low < medium. Rank it numerically instead, ascending with urgency, so
# "desc" means for priority what it means for every other field: biggest first.
_PRIORITY_RANK = {
    "$switch": {
        "branches": [
            {"case": {"$eq": ["$priority", "high"]}, "then": 3},
            {"case": {"$eq": ["$priority", "medium"]}, "then": 2},
            {"case": {"$eq": ["$priority", "low"]}, "then": 1},
        ],
        "default": 0,
    }
}

INDEXES = (
    # Every task read is scoped by user_id and then sorted. Without these the
    # sort runs in memory over a collection scan — fine at ten tasks, and past a
    # few thousand both slow and liable to hit the server's 32MB sort limit,
    # which fails the request outright rather than degrading.
    (TODOS, [("user_id", 1), ("status", 1), ("created_at", -1)], "pa_todos_by_created"),
    (TODOS, [("user_id", 1), ("status", 1), ("due_at", 1)], "pa_todos_by_due"),
    (TODOS, [("user_id", 1), ("status", 1), ("updated_at", -1)], "pa_todos_by_updated"),
    (TODOS, [("user_id", 1), ("parent_id", 1)], "pa_todos_by_parent"),
    # Backs the source filter and the by_source counts behind it. Sparse would
    # not do: `source=self` looks for the absence of the field, and a sparse
    # index cannot answer that.
    (TODOS, [("user_id", 1), ("source", 1), ("created_at", -1)], "pa_todos_by_source"),
)


async def ensure_indexes() -> None:
    """Create the personal-assistant task indexes. Idempotent and best-effort:
    `create_index` on an existing index is a no-op, and a failure is logged
    rather than raised — a missing index is a slow app, not a broken one, and
    not a reason to refuse to boot."""
    db = get_db()
    for collection, keys, name in INDEXES:
        try:
            await db[collection].create_index(keys, name=name, background=True)
        except Exception as e:
            logger.error("pa index %s failed: %s", name, e)


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
        # Provenance when another agent created this task (see service.create_tasks).
        "source": data.get("source"),
        "source_ref": data.get("source_ref"),
        "status": "pending",
        "created_at": now,
        "updated_at": now,
    }
    res = await get_db()[TODOS].insert_one(doc)
    doc["_id"] = res.inserted_id
    return _serialize(doc)


def _todo_query(
    user_id: str,
    status: Optional[str] = None,
    priority: Optional[str] = None,
    q: Optional[str] = None,
    source: Optional[str] = None,
) -> dict:
    """The Mongo filter behind both the whole-list fetch and one page of it."""
    query: dict = {"user_id": user_id}
    if status:
        query["status"] = status
    if priority:
        query["priority"] = priority
    if source:
        # Provenance has been written on every cross-agent task since
        # `service.create_tasks` shipped and read by nothing, which is how a
        # roadmap's worth of "Learn:" tasks ends up indistinguishable from work
        # the user entered themselves. `self` is the useful case: a null source
        # matches both the missing field on old rows and an explicit null.
        query["source"] = None if source == SOURCE_SELF else source
    if q and q.strip():
        # Escaped because this is whatever the user typed into a search box: an
        # unescaped `(` is a Mongo error and an unescaped `.*` is a scan of the
        # whole collection.
        needle = re.escape(q.strip())
        query["$or"] = [
            {"title": {"$regex": needle, "$options": "i"}},
            {"details": {"$regex": needle, "$options": "i"}},
        ]
    return query


async def fetch_todos(
    user_id: str,
    status: Optional[str] = None,
    priority: Optional[str] = None,
    source: Optional[str] = None,
) -> list:
    """Every matching task, newest first. For the internal callers — the digest,
    the planner, the agenda — that want the whole set rather than a page of it."""
    cursor = (
        get_db()[TODOS]
        .find(_todo_query(user_id, status, priority, source=source))
        .sort("created_at", -1)
    )
    docs = await cursor.to_list(length=500)
    return [_serialize(d) for d in docs]


async def fetch_todos_page(
    user_id: str,
    status: Optional[str] = None,
    priority: Optional[str] = None,
    q: Optional[str] = None,
    source: Optional[str] = None,
    sort: str = "created_at",
    order: Optional[str] = None,
    limit: int = 500,
    offset: int = 0,
) -> dict:
    """One page of a user's tasks, plus the total it was taken from.

    Searching and sorting happen here rather than in the client because the
    client only ever holds the rows it happened to have fetched. A search box
    filtering that slice silently misses everything past it, and "sort by due
    date" reorders the slice rather than the list — both of which look like
    working features right up until the list is longer than one fetch.

    Returns `{"items": [...], "total": n}`.
    """
    col = get_db()[TODOS]
    query = _todo_query(user_id, status, priority, q, source)
    descending = (order or TODO_SORT_FIELDS.get(sort, "desc")) == "desc"
    direction = -1 if descending else 1

    stages: list = [{"$match": query}]
    sort_spec: dict = {}
    computed: Optional[str] = None

    if sort == "priority":
        computed = "_rank"
        stages.append({"$addFields": {computed: _PRIORITY_RANK}})
        sort_spec[computed] = direction
    elif sort == "due_at":
        # Undated tasks trail in both directions. They are not the earliest —
        # they are unscheduled, which is the distinction categorize_agenda draws
        # by giving them a bucket of their own.
        computed = "_undated"
        stages.append(
            {
                "$addFields": {
                    computed: {"$cond": [{"$ifNull": ["$due_at", False]}, 0, 1]}
                }
            }
        )
        sort_spec[computed] = 1
        sort_spec["due_at"] = direction
    else:
        sort_spec[sort] = direction

    # Ties broken on a unique key, so a row can't slide between two pages — or
    # onto both — as the client walks the offsets. Every sort field here except
    # created_at has duplicates in normal use.
    sort_spec["_id"] = 1
    stages += [{"$sort": sort_spec}, {"$skip": offset}, {"$limit": limit}]
    if computed:
        stages.append({"$project": {computed: 0}})

    docs = await col.aggregate(stages).to_list(length=limit)
    total = await col.count_documents(query)
    return {"items": [_serialize(d) for d in docs], "total": total}


async def find_pending_todos(
    user_id: str, title: Optional[str], match_all: bool
) -> list:
    """Resolve a TaskSelector to the matching pending todos (serialized).

    A title always narrows, including when `match_all` is set. The branches used
    to run the other way round, and `match_all` returned every pending task
    without ever consulting the title — so "delete all my Learn: tasks" resolved
    to the entire list. The qualifier that made the request *more* specific was
    the very word that set `match_all`, which meant the more carefully someone
    scoped a bulk delete, the more likely it swept up everything they owned.
    `match_all` now only widens when there is no filter to widen past.
    """
    docs = (
        await get_db()[TODOS]
        .find({"user_id": user_id, "status": "pending"})
        .to_list(length=500)
    )
    if title:
        needle = title.lower()
        matched = [d for d in docs if needle in (d.get("title", "").lower())]
    elif match_all:
        matched = docs
    else:
        matched = []
    return [_serialize(d) for d in matched]


def describe_selection(title: Optional[str], match_all: bool, count: int) -> str:
    """Plain-English account of what a delete actually matched, for the approval
    card. The interrupt is only as good as what it shows: a list of titles with
    no statement of the rule that produced them gives the user nothing to check
    the rule against."""
    noun = "task" if count == 1 else "tasks"
    if title:
        return f'{count} {noun} whose title contains "{title}"'
    if match_all:
        return f"all {count} pending {noun}"
    return f"{count} {noun}"


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


async def _spawn_next_occurrence(user_id: str, task: dict) -> Optional[dict]:
    """Queue the next instance of a recurring task, so completing one never ends
    the series. Returns the created task, or None when the task doesn't repeat.

    Shared by every path that can complete a task. It used to live inline in
    `complete_todo` — the agent's path — which meant ticking a recurring task off
    in the app went through `update_todo_by_id` and silently ended the
    recurrence, while asking the assistant to complete the same task kept it
    going. Same user action, two different outcomes depending on which surface
    they used.
    """
    recurrence = task.get("recurrence")
    if not recurrence:
        return None
    return await insert_todo(
        user_id,
        {
            "title": task["title"],
            "details": task.get("details"),
            "priority": task.get("priority", "medium"),
            "due_at": _next_due(task.get("due_at"), recurrence),
            "recurrence": recurrence,
            # Provenance is inherited so a repeating task created by another
            # agent keeps saying where it came from.
            "source": task.get("source"),
            "source_ref": task.get("source_ref"),
        },
    )


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
    follow_up = await _spawn_next_occurrence(user_id, target)
    if follow_up:
        target = {**target, "next_occurrence": follow_up}
    return target


async def complete_todo_by_id(user_id: str, task_id: str) -> Optional[dict]:
    """Complete one task by id, honouring recurrence. Used by bulk actions."""
    target = await get_todo_by_id(user_id, task_id)
    if not target or target.get("status") == "done":
        return None
    await get_db()[TODOS].update_one(
        {"_id": ObjectId(task_id), "user_id": user_id},
        {"$set": {"status": "done", "updated_at": datetime.now().isoformat()}},
    )
    follow_up = await _spawn_next_occurrence(user_id, target)
    return {**target, "status": "done", **({"next_occurrence": follow_up} if follow_up else {})}


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
    """Update a task by its MongoDB ID (direct API path).

    A change of status to `done` gets the same recurrence follow-up the agent's
    complete path performs — this is the endpoint the app's tick box calls, and
    before this it was the only way to complete a task that quietly ended the
    series.
    """
    try:
        before = await get_todo_by_id(user_id, task_id)
        if not before:
            return None

        updates["updated_at"] = datetime.now().isoformat()
        await get_db()[TODOS].update_one(
            {"_id": ObjectId(task_id), "user_id": user_id},
            {"$set": updates},
        )
        doc = await get_db()[TODOS].find_one({"_id": ObjectId(task_id)})
        if not doc:
            return None
        updated = _serialize(doc)

        # Only on the pending -> done transition. Re-saving an already-completed
        # task, or reopening one, must not mint another occurrence.
        if updates.get("status") == "done" and before.get("status") != "done":
            follow_up = await _spawn_next_occurrence(user_id, before)
            if follow_up:
                updated["next_occurrence"] = follow_up
        return updated
    except Exception:
        return None


async def bulk_complete(user_id: str, ids: List[str]) -> dict:
    """Complete many tasks at once, honouring each one's recurrence.

    Deliberately per-task rather than one `update_many`: a repeating task owes
    the user its next occurrence, and a bulk write would skip that for every
    task in the batch. Returns what happened, since some ids may be unknown or
    already done.
    """
    completed: List[dict] = []
    spawned: List[dict] = []
    for task_id in ids:
        done = await complete_todo_by_id(user_id, task_id)
        if not done:
            continue
        completed.append(done)
        if done.get("next_occurrence"):
            spawned.append(done["next_occurrence"])
    return {
        "completed": len(completed),
        "next_occurrences": spawned,
        "tasks": completed,
    }


async def delete_completed(user_id: str) -> int:
    """Clear out every finished task. Returns how many went."""
    res = await get_db()[TODOS].delete_many({"user_id": user_id, "status": "done"})
    return res.deleted_count


async def fetch_overdue_ids(user_id: str) -> List[str]:
    """Ids of pending tasks whose due date has passed, for a bulk sweep.

    Bucketed the same way the agenda is — on the date portion only — so a task
    due later today is not swept up as overdue.
    """
    todos = await fetch_todos(user_id, status="pending")
    return [t["id"] for t in categorize_agenda(todos)["overdue"]]


async def fetch_completed_since(user_id: str, since_iso: str) -> list:
    """Tasks finished on or after `since_iso`, newest first.

    `updated_at` is the completion time in practice: the only thing that writes
    it on a done task is the write that completed it.
    """
    cursor = (
        get_db()[TODOS]
        .find({"user_id": user_id, "status": "done", "updated_at": {"$gte": since_iso}})
        .sort("updated_at", -1)
    )
    docs = await cursor.to_list(length=200)
    return [_serialize(d) for d in docs]


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
        logger.error("pa remember error: %s", e)


async def _memory_value(user_id: str, key: str) -> list:
    """Read one list-valued memory field from the user's `memories` doc."""
    try:
        doc = await get_db()[MEMORIES].find_one({"user_id": user_id})
        if doc:
            return list((doc.get("data") or {}).get(key, []) or [])
    except Exception as e:
        logger.error("pa memory read error key=%s: %s", key, e)
    return []


async def append_memory_list(user_id: str, key: str, item: str, cap: int = 50):
    """Append an item to a list-valued memory entry, de-duplicated and capped."""
    existing = await _memory_value(user_id, key)
    merged = list(dict.fromkeys(existing + [item]))[-cap:]
    await remember(user_id, key, merged)
    return merged


async def fetch_notes(user_id: str) -> list:
    """The user's remembered facts, every one carrying an `id`.

    Notes are a plain list inside the shared memory document rather than their
    own collection, so until now nothing addressed an individual note — which is
    why they could only ever be appended to. Editing or deleting one needs a
    stable handle, so notes written from here on carry a uuid, and any note
    stored before that gets one the first time it is read.

    The backfill is written straight back, so it happens once per user rather
    than on every read, and an id handed to a client stays valid.
    """
    notes = await _memory_value(user_id, NOTES_KEY)
    missing = [n for n in notes if isinstance(n, dict) and not n.get("id")]
    if not missing:
        return notes
    for note in missing:
        note["id"] = str(uuid.uuid4())
    await remember(user_id, NOTES_KEY, notes)
    logger.info("backfilled %d note ids for user=%s", len(missing), user_id)
    return notes


async def add_note(
    user_id: str, content: str, category: Optional[str] = None, cap: int = 100
) -> dict:
    """Append a timestamped personal fact, newest-capped."""
    existing = await fetch_notes(user_id)
    note = {
        "id": str(uuid.uuid4()),
        "content": content,
        "category": category,
        "created_at": datetime.now().isoformat(),
    }
    merged = (existing + [note])[-cap:]
    await remember(user_id, NOTES_KEY, merged)
    return note


async def update_note(
    user_id: str, note_id: str, content: Optional[str] = None, category: Optional[str] = None
) -> Optional[dict]:
    """Edit one remembered fact. Returns the updated note, or None if unknown.

    A fact the assistant misheard used to be uncorrectable — it would keep
    feeding the wrong thing into every reply, and the only recourse was to add a
    second, contradictory note next to it.
    """
    notes = await fetch_notes(user_id)
    for note in notes:
        if note.get("id") != note_id:
            continue
        if content is not None:
            note["content"] = content
        # An explicit empty category clears it; `None` means "leave alone".
        if category is not None:
            note["category"] = category or None
        note["updated_at"] = datetime.now().isoformat()
        await remember(user_id, NOTES_KEY, notes)
        return note
    return None


async def delete_note(user_id: str, note_id: str) -> bool:
    """Forget one fact. Returns whether there was one to forget."""
    notes = await fetch_notes(user_id)
    remaining = [n for n in notes if n.get("id") != note_id]
    if len(remaining) == len(notes):
        return False
    await remember(user_id, NOTES_KEY, remaining)
    return True


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
async def insert_subtasks(user_id: str, parent_id: str, subtasks: List[dict]) -> list:
    """Persist a goal's steps as child tasks.

    Takes whole specs rather than bare titles. A subtask used to be a title and
    nothing else, so a breakdown told to "schedule these across next week" had
    nowhere to put the schedule except the title string — leaving every step
    undated at medium priority, invisible to the agenda, and unaffected by any
    instruction about which one mattered most.
    """
    created = []
    for spec in subtasks:
        title = (spec.get("title") or "").strip()
        if not title:
            continue
        created.append(
            await insert_todo(
                user_id,
                {
                    **{k: v for k, v in spec.items() if v is not None},
                    "title": title,
                    "parent_id": parent_id,
                },
            )
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
