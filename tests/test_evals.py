"""Tests for the eval harness scoring + the capture record plumbing.

All offline — no LLM calls (the harness's --self-test path scores gold against
itself, and score_record is pure)."""

import json
from pathlib import Path

from app.evals import harness
from app.evals import build_dataset
from app.core import llm_capture


_RECORD = {
    "call_site": "TaskInput",
    "tools": [
        {
            "name": "TaskInput",
            "description": "A task",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "priority": {"type": "string"},
                },
                "required": ["title"],
            },
        }
    ],
    "messages": [
        {"role": "system", "content": "Extract a task."},
        {"role": "human", "content": "buy milk, important"},
    ],
    "kind": "tool_call",
    "output": {"tool_calls": [{"name": "TaskInput", "args": {"title": "Buy milk", "priority": "high"}}]},
}


# --------------------------------------------------------------------------- #
# record helpers
# --------------------------------------------------------------------------- #
def test_gold_output_and_schema():
    assert harness.gold_output(_RECORD) == {"title": "Buy milk", "priority": "high"}
    schema = harness.record_schema(_RECORD)
    assert schema["title"] == "TaskInput"
    assert schema["required"] == ["title"]


def test_gold_output_none_for_text_record():
    assert harness.gold_output({"output": {"content": "hello"}}) is None


def test_rebuild_messages_roles():
    msgs = harness.rebuild_messages(_RECORD)
    assert [m.type for m in msgs] == ["system", "human"]


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #
def test_score_exact_match():
    pred = {"title": "Buy milk", "priority": "high"}
    row = harness.score_record(_RECORD, pred)
    assert row["valid"] is True
    assert row["field_match"] == 1.0


def test_score_partial_match():
    pred = {"title": "Buy milk", "priority": "low"}  # priority wrong
    row = harness.score_record(_RECORD, pred)
    assert row["valid"] is True  # required field present
    assert row["field_match"] == 0.5


def test_score_missing_required_is_invalid():
    pred = {"priority": "high"}  # no title
    row = harness.score_record(_RECORD, pred)
    assert row["valid"] is False


def test_custom_scorer_is_used(monkeypatch):
    monkeypatch.setitem(harness.CUSTOM_SCORERS, "TaskInput", lambda gold, pred: 0.42)
    row = harness.score_record(_RECORD, {"title": "x"})
    assert row["field_match"] == 0.42


# --------------------------------------------------------------------------- #
# QuizOutput scorer (answer-index correctness)
# --------------------------------------------------------------------------- #
def test_quiz_scorer_all_valid():
    pred = {
        "quiz": [
            {"question": "2+2?", "options": ["3", "4", "5"], "answer": 1},
            {"question": "Sky?", "options": ["blue", "green"], "answer": 0},
        ]
    }
    assert harness.score_quiz({}, pred) == 1.0


def test_quiz_scorer_out_of_range_and_too_few_options():
    pred = {
        "quiz": [
            {"question": "a", "options": ["x", "y"], "answer": 5},   # out of range
            {"question": "b", "options": ["only"], "answer": 0},     # <2 options
            {"question": "c", "options": ["x", "y"], "answer": 1},   # ok
        ]
    }
    assert abs(harness.score_quiz({}, pred) - 1 / 3) < 1e-9


def test_quiz_scorer_empty():
    assert harness.score_quiz({}, {"quiz": []}) == 0.0


def test_quiz_scorer_registered():
    assert harness.CUSTOM_SCORERS.get("QuizOutput") is harness.score_quiz


# --------------------------------------------------------------------------- #
# dataset builder
# --------------------------------------------------------------------------- #
def _rec(call_site, content):
    return {
        "call_site": call_site,
        "messages": [{"role": "human", "content": content}],
        "output": {"tool_calls": [{"name": call_site, "args": {"x": content}}]},
    }


def test_tier_lookup_defaults_to_b():
    assert build_dataset.tier_of("PlanOutput") == "A"
    assert build_dataset.tier_of("QueryOutput") == "C"
    assert build_dataset.tier_of("SomethingNew") == "B"


def test_builder_dedupes_identical_prompts():
    records = [_rec("PlanOutput", "same"), _rec("PlanOutput", "same")]
    train, eval_, summary = build_dataset.build_splits(records, eval_frac=0.0)
    assert summary["PlanOutput"]["total"] == 1


def test_builder_downsamples_tier_c_keeps_tier_a():
    records = [_rec("QueryOutput", f"q{i}") for i in range(100)]  # Tier C
    records += [_rec("PlanOutput", f"p{i}") for i in range(100)]  # Tier A
    train, eval_, summary = build_dataset.build_splits(records, eval_frac=0.0, seed=1)
    # Tier A kept in full; Tier C downsampled to ~25%.
    assert summary["PlanOutput"]["kept"] == 100
    assert summary["QueryOutput"]["kept"] == 25


def test_builder_caps_per_site():
    records = [_rec("PlanOutput", f"p{i}") for i in range(50)]
    _, _, summary = build_dataset.build_splits(records, cap=10, eval_frac=0.0)
    assert summary["PlanOutput"]["kept"] == 10


def test_builder_holds_out_stratified_eval():
    records = [_rec("PlanOutput", f"p{i}") for i in range(20)]
    records += [_rec("LogOutput", f"l{i}") for i in range(20)]
    train, eval_, summary = build_dataset.build_splits(records, eval_frac=0.25, seed=2)
    # every site contributes to eval, and train+eval == kept
    for site in ("PlanOutput", "LogOutput"):
        assert summary[site]["eval"] >= 1
        assert summary[site]["train"] + summary[site]["eval"] == summary[site]["kept"]
    assert len(train) + len(eval_) == sum(s["kept"] for s in summary.values())


# --------------------------------------------------------------------------- #
# end-to-end self-test (no network)
# --------------------------------------------------------------------------- #
async def test_run_eval_self_test():
    report = await harness.run_eval([_RECORD], self_test=True)
    assert report["overall"]["n"] == 1
    assert report["overall"]["field_match"] == 1.0
    assert "TaskInput" in report["by_call_site"]


async def test_run_eval_skips_text_records():
    text_only = {"call_site": "x", "output": {"content": "hi"}}
    report = await harness.run_eval([_RECORD, text_only], self_test=True)
    assert report["overall"]["n"] == 1  # the text record is skipped


def test_sample_dataset_self_scores(tmp_path):
    sample = Path("app/evals/datasets/sample.jsonl")
    records = harness.load_records(str(sample))
    assert len(records) >= 2
    for rec in records:
        row = harness.score_record(rec, harness.gold_output(rec))
        assert row["valid"] and row["field_match"] == 1.0


# --------------------------------------------------------------------------- #
# capture handler
# --------------------------------------------------------------------------- #
def test_capture_disabled_by_default(monkeypatch):
    monkeypatch.delenv("LLM_CAPTURE", raising=False)
    assert llm_capture.build_capture_callbacks() == []


def test_capture_writes_jsonl(tmp_path, monkeypatch):
    path = tmp_path / "cap.jsonl"
    handler = llm_capture.DistillationCapture(str(path))

    params = {
        "model": "gpt-4o-mini",
        "tools": [
            {"function": {"name": "TaskInput", "description": "d", "parameters": {"type": "object"}}}
        ],
        "tool_choice": {"type": "function", "function": {"name": "TaskInput"}},
    }

    class _Msg:
        type = "human"
        content = "buy milk"
        additional_kwargs: dict = {}
        tool_calls = None

    handler.on_chat_model_start(
        {}, [[_Msg()]], run_id="r1", invocation_params=params
    )

    class _OutMsg:
        content = ""
        additional_kwargs: dict = {}
        tool_calls = [{"name": "TaskInput", "args": {"title": "Buy milk"}}]

    gen = type("G", (), {"message": _OutMsg(), "text": ""})()
    response = type("R", (), {"generations": [[gen]]})()
    handler.on_llm_end(response, run_id="r1")

    line = path.read_text(encoding="utf-8").strip()
    rec = json.loads(line)
    assert rec["call_site"] == "TaskInput"
    assert rec["kind"] == "tool_call"
    assert rec["output"]["tool_calls"][0]["args"] == {"title": "Buy milk"}
    assert rec["messages"][0]["content"] == "buy milk"


# ── RAG eval: judge-reply parsing (services/evaluation._to_unit_float) ────────
# Regression guard for the "scores look unreliable" bug: judges are told to
# return a bare decimal, but the old strict float() scored 0.0 the instant a
# reply carried any prose, silently dragging recall/hallucination down.
def test_to_unit_float_parses_bare_and_chatty_replies():
    from app.services.rag.step7_evaluation import _to_unit_float

    assert _to_unit_float("1.0") == 1.0
    assert _to_unit_float("0.0") == 0.0
    assert _to_unit_float("  0.95 ") == 0.95
    # prose around the number must not zero the score
    assert _to_unit_float("0.8 — mostly covered") == 0.8
    assert _to_unit_float("score: 0.3") == 0.3
    # clamped to [0, 1]
    assert _to_unit_float("1.7") == 1.0
    assert _to_unit_float("-0.5") == 0.0
    # genuinely no number → 0.0
    assert _to_unit_float("N/A") == 0.0
