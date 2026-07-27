"""Entry point: `python -m app.mcp.meal_planner`.

Lives here rather than in server.py because the package `__init__` imports
`.server`; running `python -m app.mcp.meal_planner.server` directly would load
that module a second time as `__main__`, leaving two FastMCP instances in the
process. Going through the package imports it exactly once.
"""

from .server import main

main()
