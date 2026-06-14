"""Expo push-notification delivery.

All schedulers identify a user by their Mongo `_id` (the JWT `sub`). The Expo
push token is stored on that same `users` document (see PATCH
/api/user/me/expo-push-token), so a single helper keyed on the user id can be
reused by every trigger job.
"""

import logging

import httpx
from bson import ObjectId

from app.database import get_db

logger = logging.getLogger(__name__)

EXPO_PUSH_URL = "https://exp.host/--/api/v2/push/send"


async def get_expo_push_token(userId: str) -> str | None:
    """Look up the Expo push token stored on the user document, or None."""
    if not userId or not ObjectId.is_valid(userId):
        return None
    user = await get_db()["users"].find_one(
        {"_id": ObjectId(userId)}, {"expo_push_token": 1}
    )
    return (user or {}).get("expo_push_token")


async def send_push_notification(
    userId: str,
    title: str,
    body: str,
    data: dict | None = None,
) -> bool:
    """Send a single push notification to a user via the Expo push service.

    Returns True if Expo accepted the message, False if the user has no token
    or delivery failed. Never raises — scheduler jobs must not abort on a
    notification error.
    """
    token = await get_expo_push_token(userId)
    if not token:
        logger.info("push skipped: no expo token for user=%s", userId)
        return False

    message = {"to": token, "title": title, "body": body, "sound": "default"}
    if data:
        message["data"] = data

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                EXPO_PUSH_URL,
                json=message,
                headers={
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                },
            )
        resp.raise_for_status()
        ticket = resp.json().get("data", {})
        if isinstance(ticket, dict) and ticket.get("status") == "error":
            logger.error(
                "expo push error user=%s: %s", userId, ticket.get("message")
            )
            return False
        logger.info("push sent user=%s title=%s", userId, title)
        return True
    except Exception as e:
        logger.error("push send failed user=%s: %s", userId, e)
        return False
