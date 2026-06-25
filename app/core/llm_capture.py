"""
core/llm_capture.py
===================
Best-effort capture of every LLM call into JSONL, for building a distillation
dataset to fine-tune a local model (Llama 3.1 8B) from the current teacher
(gpt-4o-mini).

How it works
------------
We attach a LangChain callback handler to the single shared `llm` in
core/llm.py. Because callbacks propagate to the underlying chat model, this
captures EVERY call — plain, `with_structured_output(...)`, and
`bind_tools(...)` — across all agents, with zero changes to any agent code.

Each record pairs:
  * the exact messages the model received (the rendered prompt),
  * the schema/tools it was offered (so we know the "call site"),
  * the model's output (tool-call args for structured output / tool use, or text).

That is exactly an SFT training pair: teacher input -> teacher output.

Enable it with env vars (off by default, so production is untouched):
  LLM_CAPTURE=1
  LLM_CAPTURE_PATH=captures/llm_calls.jsonl   # optional, this is the default

Capture is wrapped in try/except everywhere — it must never break a real call.
The output may contain user data; the captures/ dir is git-ignored.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone
from typing import Any, Optional

from langchain_core.callbacks import BaseCallbackHandler

logger = logging.getLogger(__name__)

_DEFAULT_PATH = "captures/llm_calls.jsonl"


def capture_enabled() -> bool:
    return os.getenv("LLM_CAPTURE", "").lower() in ("1", "true", "yes", "on")


def _serialize_message(m: Any) -> dict:
    """Minimal, JSON-safe view of a LangChain message."""
    out = {"role": getattr(m, "type", m.__class__.__name__), "content": m.content}
    tool_calls = getattr(m, "tool_calls", None)
    if tool_calls:
        out["tool_calls"] = tool_calls
    fn_call = (getattr(m, "additional_kwargs", {}) or {}).get("function_call")
    if fn_call:
        out["function_call"] = fn_call
    return out


def _offered_tools(invocation_params: dict) -> list[dict]:
    """Normalize the tools / response_format the model was offered."""
    tools = []
    for t in invocation_params.get("tools") or []:
        fn = t.get("function", t) if isinstance(t, dict) else {}
        tools.append(
            {
                "name": fn.get("name"),
                "description": fn.get("description"),
                "parameters": fn.get("parameters"),
            }
        )
    # `with_structured_output(method="json_schema")` uses response_format instead.
    rf = invocation_params.get("response_format")
    if isinstance(rf, dict) and rf.get("type") == "json_schema":
        js = rf.get("json_schema", {})
        tools.append(
            {
                "name": js.get("name"),
                "description": js.get("description"),
                "parameters": js.get("schema"),
            }
        )
    return tools


def _forced_tool_name(invocation_params: dict) -> Optional[str]:
    tc = invocation_params.get("tool_choice")
    if isinstance(tc, dict):
        return tc.get("function", {}).get("name") or tc.get("name")
    return None


class DistillationCapture(BaseCallbackHandler):
    """Pairs chat-model start (prompt) with end (completion) and appends JSONL."""

    raise_error = False  # never let a capture error bubble into the LLM call

    def __init__(self, path: str = _DEFAULT_PATH):
        self.path = path
        self._pending: dict[str, dict] = {}
        self._lock = threading.Lock()
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        except Exception as e:  # pragma: no cover - filesystem edge
            logger.warning("llm_capture: could not create dir for %s: %s", path, e)

    # -- start: stash the prompt + offered schema keyed by run_id -------------
    def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs):
        try:
            params = kwargs.get("invocation_params") or {}
            tools = _offered_tools(params)
            forced = _forced_tool_name(params)
            call_site = forced or (tools[0]["name"] if tools else "text")
            prompt = [_serialize_message(m) for m in (messages[0] if messages else [])]
            self._pending[str(run_id)] = {
                "model": params.get("model") or params.get("model_name"),
                "call_site": call_site,
                "tools": tools,
                "messages": prompt,
            }
        except Exception as e:  # pragma: no cover - defensive
            logger.debug("llm_capture start error: %s", e)

    # -- end: attach output and write the record ------------------------------
    def on_llm_end(self, response, *, run_id, **kwargs):
        rec = self._pending.pop(str(run_id), None)
        if rec is None:
            return
        try:
            gen = response.generations[0][0]
            msg = getattr(gen, "message", None)
            output: dict = {}
            kind = "text"
            if msg is not None:
                output["content"] = msg.content
                tcs = getattr(msg, "tool_calls", None)
                if tcs:
                    output["tool_calls"] = [
                        {"name": tc.get("name"), "args": tc.get("args")} for tc in tcs
                    ]
                    kind = "tool_call"
                fc = (getattr(msg, "additional_kwargs", {}) or {}).get("function_call")
                if fc:
                    output["function_call"] = fc
                    kind = "tool_call"
            else:
                output["text"] = getattr(gen, "text", "")
            rec.update(
                {
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "kind": kind,
                    "output": output,
                }
            )
            self._write(rec)
        except Exception as e:  # pragma: no cover - defensive
            logger.debug("llm_capture end error: %s", e)

    def on_llm_error(self, error, *, run_id, **kwargs):
        self._pending.pop(str(run_id), None)

    def _write(self, record: dict):
        line = json.dumps(record, ensure_ascii=False, default=str)
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")


def build_capture_callbacks() -> list:
    """Return the capture callback list when enabled, else an empty list."""
    if not capture_enabled():
        return []
    path = os.getenv("LLM_CAPTURE_PATH", _DEFAULT_PATH)
    logger.info("LLM capture ENABLED -> %s", path)
    return [DistillationCapture(path)]
