"""Expo push-notification delivery.

A user is named by two ids: the app-level `uid` every agent collection is keyed
by (`current_user["uid"]`), and the Mongo `_id` that is the JWT `sub`. The
schedulers hold a `uid` — it is what the trigger row stores — while the token
endpoint (PATCH /api/user/me/expo-push-token) writes by `_id`. Both address the
same `users` document, and the token lives on it, so the lookup here resolves
either rather than making every caller convert first.
"""

import logging
import os

import httpx
from bson import ObjectId

from app.database import get_db

logger = logging.getLogger(__name__)

EXPO_PUSH_URL = "https://exp.host/--/api/v2/push/send"

# The Android channel the client registers at startup, at MAX importance. Naming
# it is what earns a heads-up banner and a sound: a message sent with no
# channelId lands on the fallback channel at default importance, which shows
# nothing and plays nothing — from the user's side, indistinguishable from a
# notification that never arrived.
ANDROID_CHANNEL_ID = "default"

# One day. Everything sent from here is tied to a moment — today's digest, next
# week's plan — so an undelivered message should expire rather than sit in FCM's
# queue for the default four weeks and then surface as yesterday's news the next
# time the device comes online.
DEFAULT_TTL_SECONDS = 86_400

# Only needed if push security is enabled on the Expo project, in which case
# unauthenticated sends are rejected outright. Unset is the normal case and sends
# the request exactly as before.
EXPO_ACCESS_TOKEN = os.getenv("EXPO_ACCESS_TOKEN", "")


async def get_expo_push_token(user_id: str) -> str | None:
    """Look up the Expo push token stored on the user document, or None.

    Accepts either id the callers hold. `uid` is a uuid4 and `_id` an ObjectId,
    so the two can't be confused for one another — matching on both is a lookup
    by "whichever of these you have", not an ambiguity.
    """
    if not user_id:
        return None
    query: dict = {"uid": user_id}
    if ObjectId.is_valid(user_id):
        query = {"$or": [{"uid": user_id}, {"_id": ObjectId(user_id)}]}
    user = await get_db()["users"].find_one(query, {"expo_push_token": 1})
    return (user or {}).get("expo_push_token")


async def send_push_notification(
    user_id: str,
    title: str,
    body: str,
    data: dict | None = None,
    ttl: int = DEFAULT_TTL_SECONDS,
) -> bool:
    """Send a single push notification to a user via the Expo push service.

    Returns True if Expo accepted the message, False if the user has no token
    or delivery failed. Never raises — scheduler jobs must not abort on a
    notification error.
    """
    token = await get_expo_push_token(user_id)
    if not token:
        logger.info("push skipped: no expo token for user=%s", user_id)
        return False

    message = {
        "to": token,
        "title": title,
        "body": body,
        "channelId": ANDROID_CHANNEL_ID,
        # iOS only. On Android the sound is a property of the channel named
        # above, which is why sending one without the other is silent there.
        "sound": "default",
        # FCM holds `normal` priority messages back under Doze, which is how a
        # digest scheduled for 9am arrives at noon. These already fire at an hour
        # the user picked; delivering them at that hour is the entire point.
        "priority": "high",
        "ttl": ttl,
    }
    if data:
        message["data"] = data

    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    if EXPO_ACCESS_TOKEN:
        headers["Authorization"] = f"Bearer {EXPO_ACCESS_TOKEN}"

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(EXPO_PUSH_URL, json=message, headers=headers)
        resp.raise_for_status()
        ticket = resp.json().get("data", {})
        if isinstance(ticket, dict) and ticket.get("status") == "error":
            logger.error("expo push error user=%s: %s", user_id, ticket.get("message"))
            return False
        logger.info("push sent user=%s title=%s", user_id, title)
        return True
    except Exception as e:
        logger.error("push send failed user=%s: %s", user_id, e)
        return False
