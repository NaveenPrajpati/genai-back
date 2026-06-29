"""LangSmith tracing setup.

LangChain/LangGraph emit traces to LangSmith automatically when the
`LANGSMITH_TRACING` / `LANGSMITH_API_KEY` env vars are set — no per-call code
needed. This module just makes the integration explicit and opt-in: it confirms
the toggle at startup, defaults a project name, and logs whether tracing is live
so it's obvious in the boot logs rather than silently on/off.

Enable by setting (e.g. in .env):
    LANGSMITH_TRACING=true
    LANGSMITH_API_KEY=ls__...
    LANGSMITH_PROJECT=aiengineer-agents   # optional; defaulted below
"""

import logging
import os

logger = logging.getLogger(__name__)

DEFAULT_PROJECT = "aiengineer-agents"


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def init_tracing() -> bool:
    """Activate LangSmith tracing if configured. Returns whether it's enabled.

    Safe to call once at startup. Accepts the modern `LANGSMITH_*` names and the
    legacy `LANGCHAIN_*` aliases that older LangChain versions read."""
    enabled = _truthy(os.getenv("LANGSMITH_TRACING")) or _truthy(
        os.getenv("LANGCHAIN_TRACING_V2")
    )
    if not enabled:
        logger.info("LangSmith tracing disabled (set LANGSMITH_TRACING=true to enable)")
        return False

    if not (os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")):
        logger.warning(
            "LANGSMITH_TRACING is on but no LANGSMITH_API_KEY is set — traces "
            "will not be sent."
        )
        return False

    # Normalize the env so LangChain picks it up regardless of which name was set.
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    project = os.getenv("LANGSMITH_PROJECT") or os.getenv("LANGCHAIN_PROJECT") or DEFAULT_PROJECT
    os.environ["LANGSMITH_PROJECT"] = project
    os.environ["LANGCHAIN_PROJECT"] = project

    logger.info("LangSmith tracing enabled — project=%s", project)
    return True
