"""Tests for Expo push delivery.

This module had no coverage, and what that hid was total: every scheduler passes
the app-level `uid`, the token lookup accepted only a Mongo `_id`, and a uuid4 is
never a valid ObjectId — so every notification the product has ever tried to send
returned early at `no expo token`, logged at INFO, for months.

So the lookup identity is the first thing pinned here, followed by the delivery
fields that decide whether an accepted message is actually shown: on Android a
message with no channelId lands on the fallback channel at default importance,
which raises no banner and plays no sound.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from bson import ObjectId

import app.services.push_service as push

_UID = "f5216e91-94f2-479a-be91-2b12d90b9f1a"  # what the schedulers hold
_OID = "507f1f77bcf86cd799439011"  # the JWT `sub`


def _patch_users(monkeypatch, user=None):
    """A users collection double that records the filter it was queried with."""
    col = MagicMock()
    col.find_one = AsyncMock(return_value=user)
    monkeypatch.setattr(push, "get_db", lambda: {"users": col})
    return col


def _patch_expo(monkeypatch, payload=None, status="ok"):
    """Capture the message posted to Expo without going near the network."""
    sent = {}
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json = MagicMock(return_value=payload or {"data": {"status": status, "id": "x"}})

    async def _post(url, json=None, headers=None):
        sent["url"], sent["message"], sent["headers"] = url, json, headers
        return resp

    client = MagicMock()
    client.post = _post
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    monkeypatch.setattr(push.httpx, "AsyncClient", lambda **kw: client)
    return sent


# --------------------------------------------------------------------------- #
# the identity the callers actually hold
# --------------------------------------------------------------------------- #
async def test_token_is_found_by_the_uid_the_schedulers_pass(monkeypatch):
    col = _patch_users(monkeypatch, {"expo_push_token": "ExponentPushToken[abc]"})

    assert await push.get_expo_push_token(_UID) == "ExponentPushToken[abc]"
    assert col.find_one.await_args.args[0] == {"uid": _UID}


async def test_token_is_still_found_by_the_mongo_id(monkeypatch):
    col = _patch_users(monkeypatch, {"expo_push_token": "ExponentPushToken[abc]"})

    assert await push.get_expo_push_token(_OID) == "ExponentPushToken[abc]"
    # An id that could be either is looked up as either — the token endpoint
    # writes by _id, the schedulers read by uid, one document.
    assert col.find_one.await_args.args[0] == {
        "$or": [{"uid": _OID}, {"_id": ObjectId(_OID)}]
    }


@pytest.mark.parametrize("user_id", ["", None])
async def test_no_user_id_never_reaches_the_database(monkeypatch, user_id):
    col = _patch_users(monkeypatch)

    assert await push.get_expo_push_token(user_id) is None
    col.find_one.assert_not_awaited()


async def test_a_user_without_a_registered_token_is_skipped(monkeypatch):
    _patch_users(monkeypatch, {"_id": ObjectId(_OID)})
    sent = _patch_expo(monkeypatch)

    assert await push.send_push_notification(_UID, "t", "b") is False
    assert sent == {}


# --------------------------------------------------------------------------- #
# what actually goes over the wire
# --------------------------------------------------------------------------- #
async def test_message_carries_the_fields_android_needs_to_show_it(monkeypatch):
    _patch_users(monkeypatch, {"expo_push_token": "ExponentPushToken[abc]"})
    sent = _patch_expo(monkeypatch)

    assert await push.send_push_notification(
        _UID, "Today's tips: Rust", "Ownership moves by default.", {"type": "d"}
    ) is True

    msg = sent["message"]
    assert msg["to"] == "ExponentPushToken[abc]"
    assert msg["title"] == "Today's tips: Rust"
    assert msg["body"] == "Ownership moves by default."
    assert msg["data"] == {"type": "d"}
    # Without this the notification is delivered and shows nothing.
    assert msg["channelId"] == push.ANDROID_CHANNEL_ID
    # Without this FCM batches it under Doze and the 9am digest lands at noon.
    assert msg["priority"] == "high"
    # Without this an undelivered digest resurfaces weeks later.
    assert msg["ttl"] == push.DEFAULT_TTL_SECONDS


async def test_ttl_is_overridable_per_message(monkeypatch):
    _patch_users(monkeypatch, {"expo_push_token": "ExponentPushToken[abc]"})
    sent = _patch_expo(monkeypatch)

    await push.send_push_notification(_UID, "t", "b", ttl=60)

    assert sent["message"]["ttl"] == 60


async def test_the_access_token_is_only_sent_when_one_is_configured(monkeypatch):
    _patch_users(monkeypatch, {"expo_push_token": "ExponentPushToken[abc]"})
    sent = _patch_expo(monkeypatch)

    await push.send_push_notification(_UID, "t", "b")
    assert "Authorization" not in sent["headers"]

    monkeypatch.setattr(push, "EXPO_ACCESS_TOKEN", "secret")
    await push.send_push_notification(_UID, "t", "b")
    assert sent["headers"]["Authorization"] == "Bearer secret"


# --------------------------------------------------------------------------- #
# failures are reported, never raised — a scheduler must survive them
# --------------------------------------------------------------------------- #
async def test_an_expo_error_ticket_is_reported_as_failure(monkeypatch):
    _patch_users(monkeypatch, {"expo_push_token": "ExponentPushToken[abc]"})
    _patch_expo(
        monkeypatch,
        payload={"data": {"status": "error", "message": "DeviceNotRegistered"}},
    )

    assert await push.send_push_notification(_UID, "t", "b") is False


async def test_a_transport_failure_does_not_escape(monkeypatch):
    _patch_users(monkeypatch, {"expo_push_token": "ExponentPushToken[abc]"})

    def _boom(**kw):
        raise RuntimeError("connection reset")

    monkeypatch.setattr(push.httpx, "AsyncClient", _boom)

    assert await push.send_push_notification(_UID, "t", "b") is False
