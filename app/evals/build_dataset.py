"""
evals/build_dataset.py
======================
Turn raw capture JSONL (from core/llm_capture.py) into balanced train/eval
splits for fine-tuning Llama 3.1 8B.

Captures are dominated by the easy, high-frequency call sites (every request
hits `classify_intent`), while the hard sites that actually need fine-tuning
(`PlanOutput`, `RoadmapOutput`, `QuizOutput`, `BreakdownOutput`, nested
`ResearchOutput`, the `get_nutrition` tool calls) are comparatively rare. Left
as-is, training would over-fit trivial classification and under-serve the hard
generation. This script rebalances by **tier**:

  * Tier A (keep all)   — the hard tasks; never downsampled.
  * Tier B (keep ~60%)  — extraction + routing with edge cases.
  * Tier C (keep ~25%)  — trivial classification; just enough as format anchors.

It also de-duplicates identical prompts and holds out a stratified eval split so
every call site is represented in eval.

Two output files, each in the shape its consumer needs:
  * train.jsonl — SFT chat format `{"messages": [system, user, assistant]}` where
    the assistant turn is the teacher's structured JSON. This is what the Llama
    fine-tuner ingests.
  * eval.jsonl  — the RAW capture records (prompt + offered schema + teacher
    output), because evals/harness.py re-runs the candidate model on the prompt
    and scores it against the teacher gold. (You never eval on the train split.)

Name collisions: `IntentOutput`/`ResearchOutput` exist in several agents with
different schemas but capture only records the class name. We resolve ambiguous
names to the *stronger* tier (keep more data) — over-keeping is safe.

Usage:
  .venv/bin/python -m app.evals.build_dataset --data captures/llm_calls.jsonl \\
      --out-dir captures/dataset --eval-frac 0.15 --cap 400
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict
from typing import Optional

from app.evals.harness import gold_output, load_records

# Capture roles -> OpenAI/Llama SFT chat roles.
_PROMPT_ROLE = {"system": "system", "human": "user", "ai": "assistant", "tool": "tool"}

# call_site -> tier. Unknown sites default to "B".
TIERS: dict[str, str] = {
    # ---- Tier A: hard generation / tool-calling / correctness-critical ----
    "PlanOutput": "A",
    "RoadmapOutput": "A",
    "QuizOutput": "A",
    "BreakdownOutput": "A",
    "ResearchOutput": "A",  # ambiguous; meal-planner's nested variant is hard
    "get_nutrition": "A",   # tool-calling traces
    # ---- Tier B: extraction with subtle reasoning / routing / NL quality ----
    "IntentOutput": "B",    # ambiguous router; keep generously
    "LogOutput": "B",
    "RecipeOutput": "B",
    "TutorOutput": "B",
    "MemoryExtract": "B",
    "TopicTipsOutput": "B",
    "TaskInput": "B",
    "TaskUpdateInput": "B",
    "NoteInput": "B",
    "SynthesisOutput": "B",
    # ---- Tier C: trivial classification / simple slot extraction ----
    "QueryOutput": "C",
    "UpdateProgressOutput": "C",
    "TaskSelector": "C",
}

TIER_KEEP = {"A": 1.0, "B": 0.6, "C": 0.25}


def tier_of(call_site: str) -> str:
    return TIERS.get(call_site, "B")


def _dedupe(records: list[dict]) -> list[dict]:
    seen: set = set()
    unique = []
    for r in records:
        key = (
            r.get("call_site"),
            json.dumps(r.get("messages"), sort_keys=True, ensure_ascii=False),
        )
        if key not in seen:
            seen.add(key)
            unique.append(r)
    return unique


def build_splits(
    records: list[dict],
    *,
    tier_keep: dict[str, float] = TIER_KEEP,
    cap: int = 400,
    eval_frac: float = 0.15,
    seed: int = 0,
) -> tuple[list[dict], list[dict], dict]:
    """Return (train, eval, summary). Pure — no file IO — so it's unit-testable."""
    rng = random.Random(seed)

    by_site: dict[str, list[dict]] = defaultdict(list)
    for r in _dedupe(records):
        by_site[r.get("call_site", "?")].append(r)

    train: list[dict] = []
    eval_: list[dict] = []
    summary: dict[str, dict] = {}

    for site, items in sorted(by_site.items()):
        rng.shuffle(items)
        keep_frac = tier_keep.get(tier_of(site), 0.6)
        keep_n = min(cap, max(1, round(len(items) * keep_frac)))
        kept = items[:keep_n]

        if len(kept) >= 2:
            eval_n = min(len(kept) - 1, max(1, round(len(kept) * eval_frac)))
        else:
            eval_n = 0  # too few to hold out; keep it for training

        eval_.extend(kept[:eval_n])
        train.extend(kept[eval_n:])
        summary[site] = {
            "tier": tier_of(site),
            "total": len(items),
            "kept": len(kept),
            "train": len(kept) - eval_n,
            "eval": eval_n,
        }

    rng.shuffle(train)
    rng.shuffle(eval_)
    return train, eval_, summary


def to_sft_example(record: dict) -> Optional[dict]:
    """Convert a raw capture record into one SFT chat example, or None if it has
    no usable target. The assistant turn is the teacher's structured JSON (for
    structured-output / tool calls) or the raw text (for plain generations).

    Tool-loop traces (e.g. the meal-planner get_nutrition turns) carry the
    prompt's prior ai/tool messages through unchanged, so multi-turn context is
    preserved and the final assistant turn is the teacher's answer.
    """
    gold = gold_output(record)
    if gold is not None:
        target = json.dumps(gold, ensure_ascii=False)
    else:
        out = record.get("output") or {}
        target = out.get("content") or out.get("text")
    if not target:
        return None

    messages = [
        {"role": _PROMPT_ROLE.get(m.get("role"), "user"), "content": m.get("content", "")}
        for m in (record.get("messages") or [])
    ]
    if not messages:
        return None
    messages.append({"role": "assistant", "content": target})
    return {"messages": messages}


def _write_jsonl(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build balanced train/eval splits")
    ap.add_argument("--data", required=True, help="raw capture JSONL")
    ap.add_argument("--out-dir", default="captures/dataset")
    ap.add_argument("--eval-frac", type=float, default=0.15)
    ap.add_argument("--cap", type=int, default=400, help="max kept per call_site")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    records = load_records(args.data)
    train, eval_, summary = build_splits(
        records, cap=args.cap, eval_frac=args.eval_frac, seed=args.seed
    )

    # train.jsonl -> SFT chat format for the fine-tuner; eval.jsonl -> raw
    # records for the harness to re-run the candidate against teacher gold.
    sft_train = [ex for r in train if (ex := to_sft_example(r)) is not None]
    skipped = len(train) - len(sft_train)

    _write_jsonl(os.path.join(args.out_dir, "train.jsonl"), sft_train)
    _write_jsonl(os.path.join(args.out_dir, "eval.jsonl"), eval_)

    print(f"{'call_site':24} {'tier':4} {'total':>6} {'kept':>6} {'train':>6} {'eval':>6}")
    for site, s in sorted(summary.items(), key=lambda kv: (kv[1]["tier"], kv[0])):
        print(
            f"{site:24} {s['tier']:4} {s['total']:6} {s['kept']:6} "
            f"{s['train']:6} {s['eval']:6}"
        )
    print(
        f"\nTOTAL  train(SFT)={len(sft_train)}  eval(raw)={len(eval_)}"
        + (f"  [skipped {skipped} train records with no target]" if skipped else "")
        + f"  -> {args.out_dir}/"
    )


if __name__ == "__main__":
    main()
