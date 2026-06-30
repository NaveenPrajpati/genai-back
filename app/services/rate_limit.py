"""Redis-backed fixed-window rate limiting for auth endpoints.

Protects brute-forceable surfaces (login, password reset) and abusable ones
(signup, guest creation). Uses a simple INCR-with-expiry fixed window.

Fails OPEN — if Redis is unavailable the request is allowed — so a cache outage
never locks every user out. For auth, availability is preferred over strictness;
the JWT/password layers are the real security boundary, this is defense-in-depth.
"""

import logging

from fastapi import HTTPException, Request

from app.core.config import redis_client

logger = logging.getLogger(__name__)


def client_ip(request: Request) -> str:
    """Best-effort client IP. Behind a proxy (Vercel/AWS ALB), the real client
    is the first entry in X-Forwarded-For."""
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


async def enforce(key: str, limit: int, window_seconds: int) -> None:
    """Count one hit against `key`; raise 429 if it exceeds `limit` within the
    window. No-op on any Redis error (fail open)."""
    full = f"rl:{key}"
    try:
        count = await redis_client.incr(full)
        if count == 1:
            await redis_client.expire(full, window_seconds)
    except HTTPException:
        raise
    except Exception:
        logger.warning("rate-limit check failed for %s — allowing", key)
        return

    if count > limit:
        try:
            ttl = await redis_client.ttl(full)
        except Exception:
            ttl = window_seconds
        retry = max(ttl, 1)
        raise HTTPException(
            status_code=429,
            detail=f"Too many attempts. Please try again in {retry}s.",
            headers={"Retry-After": str(retry)},
        )


async def limit_ip(request: Request, name: str, limit: int, window_seconds: int) -> None:
    """Rate-limit by client IP, namespaced by `name` (the endpoint)."""
    await enforce(f"{name}:ip:{client_ip(request)}", limit, window_seconds)


async def limit_key(name: str, identifier: str, limit: int, window_seconds: int) -> None:
    """Rate-limit by an arbitrary identifier (e.g. email), namespaced by `name`."""
    await enforce(f"{name}:key:{identifier.lower()}", limit, window_seconds)
