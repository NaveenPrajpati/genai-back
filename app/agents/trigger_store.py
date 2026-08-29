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
from datetime import datetime, timedelta, timezone
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


def _ran_on(trig: dict, tz: ZoneInfo, day) -> bool:
    """Whether this trigger already fired on `day` in its own timezone."""
    last = trig.get("last_run_at")
    if not last:
        return False
    try:
        return datetime.fromisoformat(last).astimezone(tz).date() == day
    except ValueError:
        return False


def schedule_dows(trig: dict) -> list[int] | None:
    """The weekdays this trigger may fire on (Mon=0 .. Sun=6), or None for any day.

    Accepts either a single day or a list of them. The field started out as one
    int — the meal planner still writes a bare `6` for its Sunday plan — but the
    case worth having for a work digest is "weekdays only", which is five days,
    not one. Old rows keep meaning exactly what they always did.
    """
    dow = trig.get("schedule_dow")
    if dow is None:
        return None
    if isinstance(dow, int):
        return [dow]
    days = sorted({d for d in dow if isinstance(d, int) and 0 <= d <= 6})
    # An empty or unparseable list would otherwise mean "never fires", which is
    # not something a schedule field should be able to express by accident.
    return days or None


def is_due(trig: dict, now: datetime) -> bool:
    """True when `now` matches the trigger's local schedule_hour (and schedule_dow,
    if set) in its timezone, and it hasn't already fired during the user's current
    local day. The same-day check dedupes against restarts and overlapping runs."""
    tz = _zone(trig)
    local = now.astimezone(tz)
    if local.hour != trig.get("schedule_hour", 9):
        return False
    dows = schedule_dows(trig)
    if dows is not None and local.weekday() not in dows:
        return False
    return not _ran_on(trig, tz, local.date())


def next_run_at(trig: dict, now: datetime | None = None) -> str | None:
    """When this trigger will next fire, as a UTC ISO timestamp.

    The mirror image of `is_due`: same schedule_hour, timezone, schedule_dow and
    already-ran-today rules, read forwards instead of tested against now. None
    when the trigger is switched off — there is no next run to count down to.
    """
    if not trig.get("enabled", True):
        return None

    now = now or datetime.now(timezone.utc)
    tz = _zone(trig)
    local = now.astimezone(tz)
    hour = trig.get("schedule_hour", 9)

    candidate = local.replace(hour=hour, minute=0, second=0, microsecond=0)
    # Today's slot is gone if the hour has passed or it already fired today.
    if candidate <= local or _ran_on(trig, tz, candidate.date()):
        candidate += timedelta(days=1)

    dows = schedule_dows(trig)
    if dows is not None:
        # At most a week out; the loop can't run away.
        for _ in range(7):
            if candidate.weekday() in dows:
                break
            candidate += timedelta(days=1)

    return candidate.astimezone(timezone.utc).isoformat()


async def user_zone(user_id: str, action_type: str) -> ZoneInfo:
    """The timezone this user's `action_type` trigger is scheduled in, or UTC.

    The trigger row is the only place a learner's timezone is recorded, and it is
    already what decides when their day's work arrives. Anything else that has to
    answer "which day did this happen on" — a streak, a daily cap — has to ask
    the same question of the same source, or it will disagree with the schedule
    the user actually set for users far from UTC.
    """
    try:
        trig = await get_db()["triggers"].find_one(
            {"user_id": user_id, "action_type": action_type},
            {"timezone": 1, "user_id": 1},
        )
    except Exception as e:
        logger.error("user_zone lookup failed user=%s: %s", user_id, e)
        return ZoneInfo("UTC")
    # A user who never opted in has no row, and UTC is the same default the
    # trigger itself would have been created with.
    return _zone(trig or {})


async def get_trigger(user_id: str, action_type: str) -> dict | None:
    """This user's `action_type` trigger row, or None if they never opted in.

    The counterpart to `toggle`, which until now was write-only: a client could
    flip a digest on but had no way to ask whether it was on. Every caller of
    `toggle` therefore had to *assume* a starting state, and a switch rendered
    from an assumption is wrong for anyone who set it in an earlier session.
    """
    try:
        return await get_db()["triggers"].find_one(
            {"user_id": user_id, "action_type": action_type}
        )
    except Exception as e:
        logger.error(
            "get_trigger lookup failed user=%s action=%s: %s", user_id, action_type, e
        )
        return None


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


async def update_trigger(
    user_id: str,
    action_type: str,
    updates: dict,
    defaults: dict | None = None,
    unset: list[str] | None = None,
) -> dict:
    """Set trigger fields directly, creating the row if it doesn't exist yet.

    `toggle` flips a boolean and nothing else, so the delivery time every trigger
    row already carries — `schedule_hour`, `timezone`, `schedule_dow` — could
    only ever be whatever the router hard-coded when the row was first created.
    The scheduler has always read those fields per user; this is what lets a
    user actually set them.

    `unset` names fields to clear outright. A None in `updates` means "leave
    alone", so that a partial update needn't restate every field — which leaves
    no way to say "back to the default" without saying so separately.

    Returns the row as it now stands.
    """
    col = get_db()["triggers"]
    now = datetime.now(timezone.utc).isoformat()
    clean = {k: v for k, v in updates.items() if v is not None}
    dropped = [k for k in (unset or []) if k not in clean]

    update: dict = {
        "$set": {**clean, "updatedAt": now},
        "$setOnInsert": {
            "user_id": user_id,
            "action_type": action_type,
            "createdAt": now,
            # Only applied on insert, and only for fields the caller did not
            # already specify — otherwise a create would overwrite the very
            # values being set. A field being cleared is skipped for the same
            # reason, and because Mongo rejects touching one path twice.
            **{
                k: v
                for k, v in {
                    "enabled": True,
                    "schedule_hour": 9,
                    "timezone": "UTC",
                    **(defaults or {}),
                }.items()
                if k not in clean and k not in dropped
            },
        },
    }
    if dropped:
        update["$unset"] = {k: "" for k in dropped}

    await col.update_one(
        {"user_id": user_id, "action_type": action_type}, update, upsert=True
    )
    return await col.find_one({"user_id": user_id, "action_type": action_type}) or {}


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
