"""Redis-backed fixed-window rate limiting for auth endpoints.

Protects brute-forceable surfaces (login, password reset) and abusable ones
(signup, guest creation). Uses a simple INCR-with-expiry fixed window.

Fails OPEN — if Redis is unavailable the request is allowed — so a cache outage
never locks every user out. For auth, availability is preferred over strictness;
the JWT/password layers are the real security boundary, this is defense-in-depth.
"""

import logging
from typing import Optional

from fastapi import HTTPException, Request

from app.core.config import redis_client

logger = logging.getLogger(__name__)


def client_ip(request: Request) -> str:
    """Best-effort client IP. Behind a proxy (AWS ALB), the real client
    is the first entry in X-Forwarded-For."""
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


async def check(key: str, limit: int, window_seconds: int) -> Optional[int]:
    """Count one hit against `key`. Returns None while under `limit`, or the
    seconds to wait once over it.

    The counting half of `enforce`, split out for callers that must not raise. A
    limit reached below the HTTP layer has to become that caller's own kind of
    answer: an HTTPException thrown inside an agent's tool loop reaches the
    learner as a broken turn rather than as a reason.

    Fails open — a Redis error counts nothing and returns None (allowed).
    """
    full = f"rl:{key}"
    try:
        count = await redis_client.incr(full)
        if count == 1:
            await redis_client.expire(full, window_seconds)
    except Exception:
        logger.warning("rate-limit check failed for %s — allowing", key)
        return None

    if count <= limit:
        return None

    try:
        ttl = await redis_client.ttl(full)
    except Exception:
        ttl = window_seconds
    return max(ttl, 1)


async def enforce(key: str, limit: int, window_seconds: int) -> None:
    """Count one hit against `key`; raise 429 if it exceeds `limit` within the
    window. No-op on any Redis error (fail open)."""
    retry = await check(key, limit, window_seconds)
    if retry is None:
        return
    raise HTTPException(
        status_code=429,
        detail=f"Too many attempts. Please try again in {retry}s.",
        headers={"Retry-After": str(retry)},
    )


async def limit_ip(
    request: Request, name: str, limit: int, window_seconds: int
) -> None:
    """Rate-limit by client IP, namespaced by `name` (the endpoint)."""
    await enforce(f"{name}:ip:{client_ip(request)}", limit, window_seconds)


async def limit_key(
    name: str, identifier: str, limit: int, window_seconds: int
) -> None:
    """Rate-limit by an arbitrary identifier (e.g. email), namespaced by `name`."""
    await enforce(f"{name}:key:{identifier.lower()}", limit, window_seconds)


async def limit_user(
    name: str, user_id: str, limit: int, window_seconds: int
) -> None:
    """Rate-limit by authenticated user id, namespaced by `name` (the endpoint).
    Used on the cost-bearing RAG endpoints — the per-user complement to the
    global daily spend cap (the cap bounds total $/day; this bounds one user's
    request rate). Fails open on a Redis outage, like the auth limiters."""
    await enforce(f"{name}:user:{user_id}", limit, window_seconds)


async def check_user(
    name: str, user_id: str, limit: int, window_seconds: int
) -> Optional[int]:
    """Non-raising `limit_user`: None while under the cap, else the seconds to
    wait. Same key, so a call site using this and one using `limit_user` spend
    the same allowance rather than getting one each."""
    return await check(f"{name}:user:{user_id}", limit, window_seconds)
