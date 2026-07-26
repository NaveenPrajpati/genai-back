"""Mounting the meal MCP server inside the main FastAPI app.

Two things need care when an MCP server rides along in another ASGI app:

  1. LIFESPAN. FastAPI does not run a mounted sub-app's lifespan, and the
     streamable-HTTP transport needs its session manager running or every
     request fails. `session_lifespan()` is entered from app.main's own
     lifespan to cover that.
  2. AUTH. The mount is reachable by anything that can reach the API, while the
     MCP protocol itself carries no auth. `guarded_app()` wraps it in a bearer
     check when MCP_AUTH_TOKEN is set.
"""

import logging
from contextlib import asynccontextmanager

from app.core.config import MCP_AUTH_TOKEN
from .meal_server import mcp

logger = logging.getLogger(__name__)

# Serve at the mount root, so mounting at /mcp gives the endpoint /mcp/ rather
# than /mcp/mcp. Must be set before streamable_http_app() builds the routes.
mcp.settings.streamable_http_path = "/"


class _BearerGuard:
    """Pure-ASGI bearer check. Sits outside the MCP app so an unauthorized
    caller is rejected before any protocol handling happens."""

    def __init__(self, app, token: str):
        self.app = app
        self.token = token

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        headers = dict(scope.get("headers") or [])
        provided = headers.get(b"authorization", b"").decode()
        if provided != f"Bearer {self.token}":
            await send(
                {
                    "type": "http.response.start",
                    "status": 401,
                    "headers": [(b"content-type", b"application/json")],
                }
            )
            await send(
                {"type": "http.response.body", "body": b'{"detail":"unauthorized"}'}
            )
            return
        await self.app(scope, receive, send)


def guarded_app():
    """The MCP ASGI app, bearer-guarded when a token is configured."""
    app = mcp.streamable_http_app()
    if MCP_AUTH_TOKEN:
        return _BearerGuard(app, MCP_AUTH_TOKEN)
    logger.warning(
        "MCP_AUTH_TOKEN unset — /mcp is unauthenticated. Acceptable only when "
        "the port is not reachable from outside the host."
    )
    return app


@asynccontextmanager
async def session_lifespan():
    """Run the streamable-HTTP session manager for as long as the app lives."""
    async with mcp.session_manager.run():
        yield
