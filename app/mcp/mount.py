"""Mounting MCP servers inside the main FastAPI app.

Server-agnostic: pass any FastMCP instance. Two things need care when an MCP
server rides along in another ASGI app:

  1. LIFESPAN. FastAPI does not run a mounted sub-app's lifespan, and the
     streamable-HTTP transport needs its session manager running or every
     request fails. `session_lifespan()` is entered from app.main's own lifespan
     to cover that, and takes as many servers as you mount.
  2. AUTH. A mount is reachable by anything that can reach the API, while the
     MCP protocol itself carries no auth. `guarded_app()` wraps it in a bearer
     check when MCP_AUTH_TOKEN is set.

Adding a second server is: build it in its own package, then mount it in
app.main with `guarded_app(<pkg>.mcp)` and add it to the `session_lifespan(...)`
call.
"""

import logging
from contextlib import AsyncExitStack, asynccontextmanager

from mcp.server.fastmcp import FastMCP

from app.core.config import MCP_AUTH_TOKEN

logger = logging.getLogger(__name__)


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


def guarded_app(server: FastMCP):
    """`server` as an ASGI app, bearer-guarded when a token is configured.

    The server is set to serve at its own root, so mounting at /meal gives the
    endpoint /meal/ rather than /meal/mcp. Must happen before the routes are
    built, which is why it lives here rather than at the server's definition.
    """
    server.settings.streamable_http_path = "/"
    app = server.streamable_http_app()
    if MCP_AUTH_TOKEN:
        return _BearerGuard(app, MCP_AUTH_TOKEN)
    logger.warning(
        "MCP_AUTH_TOKEN unset — the %s MCP mount is unauthenticated. Acceptable "
        "only when the port is not reachable from outside the host.",
        server.name,
    )
    return app


@asynccontextmanager
async def session_lifespan(*servers: FastMCP):
    """Run the streamable-HTTP session manager of each server for as long as the
    app lives. Call once with every mounted server."""
    async with AsyncExitStack() as stack:
        for server in servers:
            await stack.enter_async_context(server.session_manager.run())
        yield
