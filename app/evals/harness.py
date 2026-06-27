"""
evals/harness.py
================
A schema-agnostic eval harness for the app's structured-output LLM calls.

It consumes the SAME JSONL records that core/llm_capture.py produces, so the
workflow is: capture from the teacher (gpt-4o-mini) -> split into train/eval ->
fine-tune Llama on train -> run this harness on eval against the candidate model.

Each record carries the prompt (`messages`), the offered JSON schema (`tools`),
and the teacher output (`output`). The harness:
  1. re-runs the *currently configured* `llm` on the prompt with that schema,
  2. scores the prediction vs the teacher's output (the gold label).

Because the schema travels inside each record, the harness needs no registry of
Pydantic models and is immune to the IntentOutput/ResearchOutput name clashes
across agents.

Metrics (per call_site and overall):
  * valid       — prediction is a dict containing every `required` schema field
  * field_match — mean over gold keys of (prediction[key] == gold[key])

Task-specific scorers (e.g. "is the quiz answer actually correct") can be
registered in CUSTOM_SCORERS without changing the core loop.

Usage:
  # offline sanity check of the scoring logic (no API calls):
  python -m app.evals.harness --data captures/llm_calls.jsonl --self-test

  # real eval against whatever core.llm.llm currently points at:
  python -m app.evals.harness --data eval.jsonl --limit 50
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import defaultdict
from typing import Any, Callable, Optional

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

def score_quiz(gold: dict, pred: dict) -> float:
    """QuizOutput correctness, gold-free.

    Comparing a generated quiz field-by-field to the teacher's quiz is
    meaningless (different questions), so instead we check the property that
    actually matters and that small models get wrong: every question's `answer`
    must be a valid index into its own `options`. Returns the fraction of
    well-formed, in-range questions.
    """
    questions = pred.get("quiz") if isinstance(pred, dict) else None
    if not questions:
        return 0.0
    good = 0
    for q in questions:
        if not isinstance(q, dict):
            continue
        options = q.get("options") or []
        answer = q.get("answer")
        if isinstance(answer, int) and len(options) >= 2 and 0 <= answer < len(options):
            good += 1
    return good / len(questions)


# Per-call-site scorers: name -> fn(gold: dict, pred: dict) -> float in [0, 1].
# When a call_site is registered here, its scorer replaces the default
# field-match metric. Add more (e.g. plan diet-adherence) the same way.
CUSTOM_SCORERS: dict[str, Callable[[dict, dict], float]] = {
    "QuizOutput": score_quiz,
}

_ROLE_TO_MESSAGE = {
    "system": SystemMessage,
    "human": HumanMessage,
    "ai": AIMessage,
}


# --------------------------------------------------------------------------- #
# record helpers
# --------------------------------------------------------------------------- #
def load_records(path: str) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def gold_output(record: dict) -> Optional[dict]:
    """The teacher's structured args for this record, or None if it wasn't
    a structured/tool call."""
    out = record.get("output") or {}
    tcs = out.get("tool_calls")
    if tcs:
        return tcs[0].get("args") or {}
    fc = out.get("function_call")
    if fc and isinstance(fc.get("arguments"), str):
        try:
            return json.loads(fc["arguments"])
        except json.JSONDecodeError:
            return None
    # json_schema / response_format channel: with_structured_output(method=
    # "json_schema") returns the result as a JSON string in `content` with no
    # tool_calls. Parse it so those records aren't silently dropped. Plain-text
    # generations (prose) don't start with { or [, so they stay non-structured.
    content = out.get("content") or out.get("text")
    if isinstance(content, str):
        s = content.strip()
        if s[:1] in ("{", "["):
            try:
                return json.loads(s)
            except json.JSONDecodeError:
                return None
    return None


def record_schema(record: dict) -> Optional[dict]:
    """A JSON schema usable with `llm.with_structured_output`, rebuilt from the
    offered tool that matches this record's call_site."""
    tools = record.get("tools") or []
    if not tools:
        return None
    match = next(
        (t for t in tools if t.get("name") == record.get("call_site")), tools[0]
    )
    params = match.get("parameters")
    if not isinstance(params, dict):
        return None
    return {
        "title": match.get("name") or "Output",
        "description": match.get("description") or "",
        **params,
    }


def rebuild_messages(record: dict) -> list:
    msgs = []
    for m in record.get("messages") or []:
        role = m.get("role", "human")
        content = m.get("content", "")
        cls = _ROLE_TO_MESSAGE.get(role)
        if cls is not None:
            msgs.append(cls(content=content))
        elif role == "tool":
            msgs.append(ToolMessage(content=content, tool_call_id="0"))
        else:
            msgs.append(HumanMessage(content=content))
    return msgs


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #
def score_record(record: dict, prediction: Any) -> dict:
    """Score a prediction (dict) against a record's gold output + schema."""
    gold = gold_output(record) or {}
    schema = record_schema(record) or {}
    required = schema.get("required") or list(gold.keys())

    valid = isinstance(prediction, dict) and all(k in prediction for k in required)

    if record.get("call_site") in CUSTOM_SCORERS and isinstance(prediction, dict):
        field_match = CUSTOM_SCORERS[record["call_site"]](gold, prediction)
    else:
        keys = list(gold.keys())
        if not keys:
            field_match = 1.0 if valid else 0.0
        elif not isinstance(prediction, dict):
            field_match = 0.0
        else:
            hits = sum(1 for k in keys if prediction.get(k) == gold.get(k))
            field_match = hits / len(keys)

    return {
        "call_site": record.get("call_site", "?"),
        "valid": bool(valid),
        "field_match": round(field_match, 4),
    }


def aggregate(rows: list[dict]) -> dict:
    by_site: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_site[r["call_site"]].append(r)

    def summarize(items: list[dict]) -> dict:
        n = len(items)
        return {
            "n": n,
            "valid": round(sum(i["valid"] for i in items) / n, 4),
            "field_match": round(sum(i["field_match"] for i in items) / n, 4),
        }

    return {
        "overall": summarize(rows) if rows else {"n": 0},
        "by_call_site": {site: summarize(items) for site, items in by_site.items()},
    }


# --------------------------------------------------------------------------- #
# running
# --------------------------------------------------------------------------- #
async def predict(record: dict) -> Any:
    """Run the currently-configured llm on the record's prompt + schema."""
    from app.core.llm import llm  # imported lazily so --self-test needs no key

    schema = record_schema(record)
    messages = rebuild_messages(record)
    if schema is None or not messages:
        return None
    return await llm.with_structured_output(schema).ainvoke(messages)


async def run_eval(records: list[dict], self_test: bool = False) -> dict:
    rows = []
    for rec in records:
        if gold_output(rec) is None:
            continue  # skip non-structured (plain-text) records
        if self_test:
            prediction = gold_output(rec)  # gold-as-prediction → validates scoring
        else:
            try:
                prediction = await predict(rec)
            except Exception as e:  # keep going; a failed call scores as invalid
                prediction = {"__error__": str(e)}
        rows.append(score_record(rec, prediction))
    return aggregate(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Structured-output eval harness")
    ap.add_argument("--data", required=True, help="JSONL of capture records")
    ap.add_argument("--limit", type=int, default=0, help="cap number of records")
    ap.add_argument(
        "--self-test",
        action="store_true",
        help="score gold-as-prediction (no API calls) to sanity-check scoring",
    )
    args = ap.parse_args()

    records = load_records(args.data)
    if args.limit:
        records = records[: args.limit]

    report = asyncio.run(run_eval(records, self_test=args.self_test))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
