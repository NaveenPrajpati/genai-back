"""
evals/rag_eval.py
=================
OFFLINE EVALUATION for the RAG answer pipeline: grade generated answers against a
hand-verified golden set, and (optionally) FAIL a build when quality regresses.

This complements evals/harness.py — that one grades agent structured-output
distillation (field_match); THIS one grades RAG *answer quality*: the metrics that
matter for the grounding work in step6_grounding.py.

──────────────────────────────────────────────────────────────────────────────
MODE — frozen context
──────────────────────────────────────────────────────────────────────────────
Each golden row stores the retrieved chunks, so the eval grades GENERATION only:
it runs the real grounding gate + the real RAG_ANSWER prompt over those chunks,
WITHOUT touching Pinecone. That makes it cheap, low-flake, and it isolates
generation regressions from retrieval ones. (End-to-end retrieval — question →
Pinecone → rerank → generate — is a separate, later concern; see
RAG_EVALUATION_PLAN.md.)

──────────────────────────────────────────────────────────────────────────────
METRICS
──────────────────────────────────────────────────────────────────────────────
  • correctness      — deterministic: fraction of a row's `key_facts` present in
                       the answer. No LLM ⇒ stable in CI.
  • faithfulness     — LLM-judge: are the answer's claims supported by the context?
                       1.0 = fully grounded. Default backend reuses
                       services/evaluation; `--scorer ragas` swaps in RAGAS.
  • refusal_accuracy — for `should_refuse` rows: did the system correctly decline?
  • over_refusal     — for answerable rows: did it WRONGLY refuse? Guards the
                       grounding gate against becoming too strict.

──────────────────────────────────────────────────────────────────────────────
USAGE
──────────────────────────────────────────────────────────────────────────────
  # offline sanity-check of scoring/aggregation (NO API keys, NO network):
  python -m app.services.rag.eval_harness --self-test

  # real eval against the currently-configured llm:
  python -m app.services.rag.eval_harness --data app/services/rag/datasets/rag_golden.jsonl

  # same, but exit non-zero when below threshold (flip on in CI when ready):
  python -m app.services.rag.eval_harness --data app/services/rag/datasets/rag_golden.jsonl --gate

  # use RAGAS for faithfulness instead of the built-in judge (needs `pip install ragas`):
  python -m app.services.rag.eval_harness --data ... --scorer ragas

IMPORTANT: every `app.*` import in this file is LAZY (inside a function), so the
pure scorers, the tests, and `--self-test` run with no env vars and no network —
exactly like evals/harness.py.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

# ── Pass/fail thresholds (the CI gate's bar) ─────────────────────────────────
# Set the bar BELOW the score you actually observe, so normal LLM-judge noise
# never false-blocks a PR — only a real regression does.
#
# Calibrated 2026-07-22 against the golden set (13 rows: 9 answerable, 4 refusal),
# after the eval_hallucination judge fix. Observed all four metrics at ceiling:
# correctness 1.00, faithfulness 1.00, refusal_accuracy 1.00, over_refusal 0.00.
# The set is small, so one flipped row moves a metric a lot — bars are set to
# absorb a single noisy row but catch a genuine (≥2-row) regression:
#   • correctness      9 rows → one miss = 0.889; 0.85 tolerates 1, fails at 2.
#   • faithfulness     9 rows, LLM judge → 0.90 absorbs judge noise, catches real ungrounding.
#   • refusal_accuracy 4 rows → one miss = 0.75 < 0.90; every refusal must hold.
#   • over_refusal     9 rows → one false refusal = 0.111 < 0.15; fails at 2.
# Re-run report-only and re-tune whenever the golden set or the judges change.
THRESHOLDS: dict[str, float] = {
    "correctness": 0.85,       # ≥
    "faithfulness": 0.90,      # ≥
    "refusal_accuracy": 0.90,  # ≥
    "max_over_refusal": 0.15,  # ≤
}

DEFAULT_DATA = "app/services/rag/datasets/rag_golden.jsonl"

# Types for the injectable strategy fns (real | stub | ragas) — keeps run_eval
# pure and offline-testable.
GenerateFn = Callable[[dict], Awaitable[tuple[str, bool]]]      # record -> (answer, refused)
FaithfulnessFn = Callable[[dict, str], Awaitable[float]]        # (record, answer) -> 0..1


# --------------------------------------------------------------------------- #
# dataset
# --------------------------------------------------------------------------- #
_REQUIRED_KEYS = ("id", "question")


def load_golden(path: str) -> list[dict]:
    """Load + validate the golden JSONL. Raises ValueError on a malformed row."""
    records: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            missing = [k for k in _REQUIRED_KEYS if k not in rec]
            if missing:
                raise ValueError(f"{path}:{i} golden row missing {missing}: {rec!r}")
            if not rec.get("should_refuse") and not rec.get("key_facts"):
                # An answerable row with no key_facts can't be graded for correctness.
                raise ValueError(
                    f"{path}:{i} answerable row '{rec['id']}' needs non-empty key_facts"
                )
            records.append(rec)
    return records


# --------------------------------------------------------------------------- #
# pure scorers (no LLM, no imports — safe in tests / self-test)
# --------------------------------------------------------------------------- #
def format_context(chunks: list[str]) -> str:
    """Number chunks [1] [2] … exactly like generation.prepare_context, so the
    prompt sees the same shape and citations line up."""
    return "\n\n".join(f"[{i}] {c}" for i, c in enumerate(chunks, start=1))


def key_fact_recall(answer: str, key_facts: list) -> float:
    """Fraction of `key_facts` present in the answer (case-insensitive substring).
    Deterministic proxy for correctness — no answer key round-tripped through an LLM."""
    if not key_facts:
        return 1.0
    lowered = answer.lower()
    hits = sum(1 for fact in key_facts if str(fact).lower() in lowered)
    return round(hits / len(key_facts), 4)


def gate_pass(metrics: dict) -> bool:
    """True if every populated metric clears its threshold. A metric of None
    (its group had no rows) is skipped, not failed."""
    checks: list[bool] = []
    if metrics.get("correctness") is not None:
        checks.append(metrics["correctness"] >= THRESHOLDS["correctness"])
    if metrics.get("faithfulness") is not None:
        checks.append(metrics["faithfulness"] >= THRESHOLDS["faithfulness"])
    if metrics.get("refusal_accuracy") is not None:
        checks.append(metrics["refusal_accuracy"] >= THRESHOLDS["refusal_accuracy"])
    if metrics.get("over_refusal") is not None:
        checks.append(metrics["over_refusal"] <= THRESHOLDS["max_over_refusal"])
    return all(checks)


def _mean(xs: list[float]) -> Optional[float]:
    return round(sum(xs) / len(xs), 4) if xs else None


# --------------------------------------------------------------------------- #
# core loop (pure — strategies injected)
# --------------------------------------------------------------------------- #
async def run_eval(
    records: list[dict],
    *,
    generate_fn: GenerateFn,
    faithfulness_fn: FaithfulnessFn,
) -> dict:
    """Grade every record and aggregate. `generate_fn` produces (answer, refused);
    `faithfulness_fn` scores grounding. Both are injected so this stays offline-
    testable (the CLI wires the real ones; tests/self-test wire stubs)."""
    answerable: list[dict] = []
    refusal: list[dict] = []

    for rec in records:
        answer, refused = await generate_fn(rec)
        if rec.get("should_refuse"):
            refusal.append({"id": rec["id"], "refused": refused})
            continue

        row: dict[str, Any] = {"id": rec["id"], "over_refused": refused}
        if refused:
            # Wrongly refused an answerable question: 0 correctness, faithfulness N/A.
            row["correctness"] = 0.0
            row["faithfulness"] = None
        else:
            row["correctness"] = key_fact_recall(answer, rec.get("key_facts") or [])
            row["faithfulness"] = await faithfulness_fn(rec, answer)
        answerable.append(row)

    metrics = {
        "correctness": _mean([r["correctness"] for r in answerable]),
        "faithfulness": _mean([r["faithfulness"] for r in answerable if r["faithfulness"] is not None]),
        "over_refusal": _mean([1.0 if r["over_refused"] else 0.0 for r in answerable]),
        "refusal_accuracy": _mean([1.0 if r["refused"] else 0.0 for r in refusal]),
    }
    return {
        "n": len(records),
        "answerable": {"n": len(answerable), "rows": answerable},
        "refusal": {"n": len(refusal), "rows": refusal},
        "metrics": metrics,
        "thresholds": THRESHOLDS,
        "pass": gate_pass(metrics),
    }


# --------------------------------------------------------------------------- #
# real strategies (lazy app.* imports — only touched on a real run)
# --------------------------------------------------------------------------- #
async def real_generate(record: dict) -> tuple[str, bool]:
    """Reproduce the route's grounding behaviour over the row's frozen context:
    gate → generate → sentinel check. Returns (answer, refused)."""
    from app.core.llm import llm
    from app.core.prompts import RAG_ANSWER, REFUSAL_MESSAGE
    from app.services.rag.step6_grounding import is_answerable, is_refusal

    context = format_context(record.get("context") or [])
    question = record["question"]

    if not await is_answerable(question, context):
        return REFUSAL_MESSAGE, True

    response = await (RAG_ANSWER | llm).ainvoke({"context": context, "question": question})
    answer = response.content
    if is_refusal(answer):
        return REFUSAL_MESSAGE, True
    return answer, False


async def own_faithfulness(record: dict, answer: str) -> float:
    """Faithfulness via the built-in judge (services/evaluation): 1 − hallucination."""
    from app.services.rag.step7_evaluation import _score_hallucination

    context = format_context(record.get("context") or [])
    return round(1.0 - await _score_hallucination(context, answer), 4)


async def ragas_faithfulness(record: dict, answer: str) -> float:
    """Faithfulness via RAGAS (optional). Requires `pip install ragas` — kept off
    the runtime lockfile on purpose; this import is lazy so the default path never
    needs it."""
    try:
        from ragas.dataset_schema import SingleTurnSample
        from ragas.metrics import Faithfulness
        from ragas.llms import LangchainLLMWrapper
    except ImportError as exc:  # pragma: no cover - depends on optional dep
        raise SystemExit(
            "RAGAS is not installed. Install it to use `--scorer ragas`:\n"
            "    pip install ragas\n"
            "(kept out of requirements to avoid bloating the runtime bundle)."
        ) from exc

    from app.core.llm import llm

    metric = Faithfulness(llm=LangchainLLMWrapper(llm))
    sample = SingleTurnSample(
        user_input=record["question"],
        response=answer,
        retrieved_contexts=record.get("context") or [],
    )
    return round(float(await metric.single_turn_ascore(sample)), 4)


# --------------------------------------------------------------------------- #
# offline stubs (self-test: perfect system, no network)
# --------------------------------------------------------------------------- #
async def _stub_generate(record: dict) -> tuple[str, bool]:
    if record.get("should_refuse"):
        return "REFUSED", True
    facts = " ".join(str(f) for f in (record.get("key_facts") or []))
    return f"{facts} [1]", False


async def _stub_faithfulness(record: dict, answer: str) -> float:
    return 1.0


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def _summary_line(report: dict) -> str:
    m = report["metrics"]
    def pct(v):
        return "  n/a" if v is None else f"{v * 100:5.1f}%"
    verdict = "PASS" if report["pass"] else "FAIL"
    return (
        f"[{verdict}] n={report['n']}  "
        f"correctness={pct(m['correctness'])}  "
        f"faithfulness={pct(m['faithfulness'])}  "
        f"refusal_acc={pct(m['refusal_accuracy'])}  "
        f"over_refusal={pct(m['over_refusal'])}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Offline RAG answer-quality eval")
    ap.add_argument("--data", default=DEFAULT_DATA, help="golden JSONL path")
    ap.add_argument("--limit", type=int, default=0, help="cap number of rows")
    ap.add_argument("--scorer", choices=["own", "ragas"], default="own",
                    help="faithfulness backend (default: built-in judge)")
    ap.add_argument("--gate", action="store_true",
                    help="exit non-zero if any metric is below threshold")
    ap.add_argument("--self-test", action="store_true",
                    help="score a perfect stub system offline (no API calls)")
    args = ap.parse_args()

    records = load_golden(args.data)
    if args.limit:
        records = records[: args.limit]

    if args.self_test:
        generate_fn, faithfulness_fn = _stub_generate, _stub_faithfulness
    else:
        # Group this run's traces separately in LangSmith (if tracing is enabled).
        import os
        os.environ.setdefault("LANGSMITH_PROJECT", "aiengineer-rag-eval")
        from app.core.observability import init_tracing
        init_tracing()
        generate_fn = real_generate
        faithfulness_fn = ragas_faithfulness if args.scorer == "ragas" else own_faithfulness

    report = asyncio.run(
        run_eval(records, generate_fn=generate_fn, faithfulness_fn=faithfulness_fn)
    )

    print(json.dumps(report, indent=2))
    print(_summary_line(report), file=sys.stderr)

    if args.gate and not report["pass"]:
        print(
            "\nRAG eval gate FAILED — a metric dropped below threshold "
            f"({THRESHOLDS}).",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
