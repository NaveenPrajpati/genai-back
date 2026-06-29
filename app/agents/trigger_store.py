"""Shared MongoDB-backed trigger store + scheduling gate.

Every agent's scheduled job (learning digest, meal plan, PA digest) opts users
in via a single Mongo `triggers` collection keyed by (user_id, action_type). Each
row carries delivery settings — schedule_hour, timezone, and optional
schedule_dow — so the hourly scheduler sweeps fire per user at their chosen local
time rather than one fixed server-wide time.

This is the single source of truth for trigger reads/writes; agents and routers
should go through here instead of touching the collection (or Supabase) directly.
"""

import logging
from datetime import datetime, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from app.database import get_db

logger = logging.getLogger(__name__)


def _zone(trig: dict) -> ZoneInfo:
    name = trig.get("timezone") or "UTC"
    try:
        return ZoneInfo(name)
    except ZoneInfoNotFoundError:
        logger.warning(
            "unknown timezone %r for user=%s, using UTC", name, trig.get("user_id")
        )
        return ZoneInfo("UTC")


def is_due(trig: dict, now: datetime) -> bool:
    """True when `now` matches the trigger's local schedule_hour (and schedule_dow,
    if set) in its timezone, and it hasn't already fired during the user's current
    local day. The same-day check dedupes against restarts and overlapping runs."""
    tz = _zone(trig)
    local = now.astimezone(tz)
    if local.hour != trig.get("schedule_hour", 9):
        return False
    # Mon=0 .. Sun=6 (datetime.weekday()). None => every day.
    dow = trig.get("schedule_dow")
    if dow is not None and local.weekday() != dow:
        return False
    last = trig.get("last_run_at")
    if last:
        try:
            if datetime.fromisoformat(last).astimezone(tz).date() == local.date():
                return False
        except ValueError:
            pass
    return True


async def due_triggers(action_type: str, now: datetime | None = None) -> list[dict]:
    """Enabled triggers of `action_type` that are due to fire right now."""
    now = now or datetime.now(timezone.utc)
    cursor = get_db()["triggers"].find({"action_type": action_type, "enabled": True})
    return [t for t in await cursor.to_list(None) if is_due(t, now)]


async def mark_ran(trig: dict, now: datetime | None = None) -> None:
    """Stamp last_run_at so is_due won't refire this trigger today."""
    now = now or datetime.now(timezone.utc)
    await get_db()["triggers"].update_one(
        {"_id": trig["_id"]}, {"$set": {"last_run_at": now.isoformat()}}
    )


async def toggle(user_id: str, action_type: str, defaults: dict | None = None) -> bool:
    """Flip `enabled` for (user_id, action_type); create an enabled row if absent.
    `defaults` seeds extra fields (schedule_hour, schedule_dow, name, …) on first
    create. Returns the resulting enabled state."""
    col = get_db()["triggers"]
    existing = await col.find_one({"user_id": user_id, "action_type": action_type})
    if existing:
        enabled = not existing.get("enabled", True)
        await col.update_one(
            {"_id": existing["_id"]},
            {
                "$set": {
                    "enabled": enabled,
                    "updatedAt": datetime.now(timezone.utc).isoformat(),
                }
            },
        )
        return enabled

    doc = {
        "user_id": user_id,
        "action_type": action_type,
        "enabled": True,
        "schedule_hour": 9,
        "timezone": "UTC",
        "createdAt": datetime.now(timezone.utc).isoformat(),
    }
    if defaults:
        doc.update(defaults)
    await col.insert_one(doc)
    return True
