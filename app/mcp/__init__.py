"""Model Context Protocol surface for this backend.

One package per server. `meal_planner` is the first: it publishes the meal
planner as a real MCP server and holds the client bindings this app uses to
consume it.

Shared, server-agnostic pieces live here:
  • `mount`  — mounting a server into the FastAPI app (bearer guard + lifespan)
  • `client` — building a per-user client and loading its tools
"""
