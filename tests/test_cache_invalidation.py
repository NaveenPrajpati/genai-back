"""Offline tests for cache.invalidate_user — no Redis server, no network.

A tiny fake Redis stands in for the async client so we can assert that ingest/
delete invalidation clears exactly the acting user's scopes and never raises.
"""

import fnmatch

from app.services import cache


class _FakeRedis:
    """Minimal async stand-in: index_key -> set(member keys)."""

    def __init__(self, sets: dict):
        self.sets = sets
        self.deleted: list[str] = []

    async def scan_iter(self, match=None):
        for key in list(self.sets.keys()):
            if match is None or fnmatch.fnmatch(key, match):
                yield key

    async def smembers(self, key):
        return set(self.sets.get(key, set()))

    async def delete(self, *keys):
        for k in keys:
            self.deleted.append(k)
            self.sets.pop(k, None)
        return len(keys)


def _seed():
    p = cache.CACHE_INDEX_PREFIX
    return {
        f"{p}userA::__all__": {"rag:cache:1", "rag:cache:2"},
        f"{p}userA::doc1": {"rag:cache:3"},
        f"{p}userB::__all__": {"rag:cache:4"},
    }


async def test_invalidate_clears_only_that_users_scopes(monkeypatch):
    p = cache.CACHE_INDEX_PREFIX
    fake = _FakeRedis(_seed())
    monkeypatch.setattr(cache, "redis_client", fake)

    cleared = await cache.invalidate_user("userA")

    assert cleared == 2  # both of userA's scopes
    assert f"{p}userA::__all__" not in fake.sets
    assert f"{p}userA::doc1" not in fake.sets
    assert f"{p}userB::__all__" in fake.sets  # other user untouched

    # payloads: userA's cleared, userB's kept
    assert {"rag:cache:1", "rag:cache:2", "rag:cache:3"}.issubset(set(fake.deleted))
    assert "rag:cache:4" not in fake.deleted


async def test_invalidate_no_scopes_returns_zero(monkeypatch):
    fake = _FakeRedis({})
    monkeypatch.setattr(cache, "redis_client", fake)
    assert await cache.invalidate_user("nobody") == 0


async def test_invalidate_never_raises_on_redis_error(monkeypatch):
    class _Boom:
        def scan_iter(self, match=None):
            raise RuntimeError("redis down")

    monkeypatch.setattr(cache, "redis_client", _Boom())
    # Must degrade to 0, not propagate — a cache-clear failure can't break ingest.
    assert await cache.invalidate_user("userA") == 0
