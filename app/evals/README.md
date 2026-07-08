# LLM distillation capture + eval harness

Tooling to fine-tune a local model (Llama 3.1 8B) to replace gpt-4o-mini across
all three agents (meal_planner, personal_assistant, learning_tracker), and to
*measure* whether the swap holds up before you ship it.

Both pieces speak the **same JSONL record format**, so capture feeds eval directly.

## 1. Capture teacher data (gpt-4o-mini)

Capture is a callback on the single shared `llm` (see `app/core/llm_capture.py`),
so it records **every** call — plain, `with_structured_output`, and `bind_tools`
(tool calling) — across all agents with **no changes to agent code**.

Turn it on with env vars (off by default):

```bash
export LLM_CAPTURE=1
export LLM_CAPTURE_PATH=captures/llm_calls.jsonl   # default
```

Then run the app / your tests / a backfill script normally. Every LLM call appends
one JSON line:

```json
{
  "ts": "...", "model": "gpt-4o-mini",
  "call_site": "PlanOutput",
  "tools": [{"name": "PlanOutput", "description": "...", "parameters": { /* JSON schema */ }}],
  "messages": [{"role": "system", "content": "..."}, {"role": "human", "content": "..."}],
  "kind": "tool_call",
  "output": {"tool_calls": [{"name": "PlanOutput", "args": { /* the gold label */ }}]}
}
```

`call_site` tells you which of the 24 LLM call sites it is (the structured-output
schema name, or the tool name for the meal-planner nutrition tool). That is your
distillation pair: `messages` → `output`.

> ⚠️ Captures contain real user prompts. The `captures/` directory is git-ignored.

## 2. Build the fine-tune dataset

- Group records by `call_site` and **rebalance** toward the hard tasks (Tier A:
  `PlanOutput`, `RoadmapOutput`, `QuizOutput`, `BreakdownOutput`, nested
  `ResearchOutput`, and the `get_nutrition` tool-calling traces).
- **Verify the labels** on correctness-critical sites (quiz answers, plan protein,
  roadmap prerequisites) — the student copies the teacher, so wrong labels teach
  wrong behavior.
- Convert to your trainer's chat format. Keep tool-calling traces in Llama 3.1's
  tool template.
- Hold out ~10–20% per `call_site` as the eval split.

## 3. Eval a candidate model

The harness re-runs whatever `app.core.llm.llm` currently points at, on each
record's prompt + schema, and scores it against the teacher's output.

```bash
# offline: validate the scoring logic on the sample data, no API calls
.venv/bin/python -m app.evals.harness --data app/evals/datasets/sample.jsonl --self-test

# real: point core.llm at your candidate (e.g. ChatOllama llama3.1) and run
.venv/bin/python -m app.evals.harness --data eval.jsonl --limit 200
```

Output is per-`call_site` and overall:

```json
{ "overall": {"n": 200, "valid": 0.97, "field_match": 0.91},
  "by_call_site": { "PlanOutput": {"n": 20, "valid": 0.9, "field_match": 0.72}, ... } }
```

---

# RAG answer-quality eval (`rag_eval.py`)

A **separate** harness for the RAG pipeline. Where `harness.py` grades agent
structured-output distillation, `rag_eval.py` grades **RAG answers**: are they
faithful to the retrieved context, do they contain the right facts, and does the
system correctly refuse when the docs don't cover the question? See the full
rationale in [`../../RAG_EVALUATION_PLAN.md`](../../RAG_EVALUATION_PLAN.md).

**Golden set** — `datasets/rag_golden.jsonl`, one hand-verified row per line:

```jsonc
{"id":"rev-q3","question":"What was Q3 revenue?","should_refuse":false,
 "key_facts":["$4.2M"],"source_doc":"financials_q3.pdf",
 "context":["Total revenue for the third quarter reached $4.2M ..."]}
{"id":"refund-policy","question":"What is the refund policy?","should_refuse":true,
 "key_facts":[],"source_doc":null,"context":["<near-but-unrelated chunks>"]}
```

`context` is the **frozen retrieved chunks** — the eval grades *generation* over
them without touching Pinecone (cheap, low-flake). Keep ~20–30% `should_refuse`
rows so the grounding gate can't regress into over- or under-refusal unnoticed.

**Metrics:** `correctness` (fraction of `key_facts` in the answer — deterministic),
`faithfulness` (LLM-judge, or RAGAS via `--scorer ragas`), `refusal_accuracy`, and
`over_refusal`.

```bash
# offline: validate scoring + the golden set, no API keys
.venv/bin/python -m app.evals.rag_eval --self-test

# real eval against the configured llm
.venv/bin/python -m app.evals.rag_eval --data app/evals/datasets/rag_golden.jsonl

# enforce thresholds (CI hard gate); exits non-zero on a regression
.venv/bin/python -m app.evals.rag_eval --data app/evals/datasets/rag_golden.jsonl --gate
```

CI runs this **report-only** on every PR (`.github/workflows/eval.yml`). Flip it to
a blocking gate by adding `--gate` there once thresholds are calibrated.

- **valid** — prediction is a dict with every required schema field.
- **field_match** — mean agreement with the teacher on the gold fields.

Add task-specific metrics (e.g. quiz-answer correctness, plan diet-adherence) by
registering a scorer in `CUSTOM_SCORERS` in `harness.py` — no core changes needed.

Gate the model swap on these numbers per `call_site`, not on vibes.
