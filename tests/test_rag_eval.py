"""Offline tests for the RAG answer-quality eval harness.

All offline — no LLM calls. The harness injects generate/faithfulness strategies,
so we drive it with stubs, and the pure scorers need no env or network (every
app.* import in rag_eval is lazy)."""

from pathlib import Path

import pytest

from app.evals import rag_eval


# --------------------------------------------------------------------------- #
# pure scorers
# --------------------------------------------------------------------------- #
def test_format_context_numbers_chunks():
    ctx = rag_eval.format_context(["alpha", "beta"])
    assert ctx == "[1] alpha\n\n[2] beta"


def test_key_fact_recall_full_partial_empty():
    assert rag_eval.key_fact_recall("Revenue was $4.2M, up 12%.", ["$4.2M", "12%"]) == 1.0
    assert rag_eval.key_fact_recall("Revenue was $4.2M.", ["$4.2M", "12%"]) == 0.5
    assert rag_eval.key_fact_recall("anything", []) == 1.0  # nothing to check


def test_key_fact_recall_is_case_insensitive():
    assert rag_eval.key_fact_recall("led by ceo jane doe", ["Jane Doe"]) == 1.0


def test_gate_pass_thresholds():
    good = {"correctness": 0.9, "faithfulness": 0.9, "refusal_accuracy": 1.0, "over_refusal": 0.0}
    assert rag_eval.gate_pass(good) is True
    bad = {**good, "faithfulness": 0.5}
    assert rag_eval.gate_pass(bad) is False
    too_refusey = {**good, "over_refusal": 0.5}
    assert rag_eval.gate_pass(too_refusey) is False


def test_gate_pass_skips_none_metrics():
    # A group with no rows yields None — must be skipped, not treated as failure.
    only_refusals = {"correctness": None, "faithfulness": None, "refusal_accuracy": 1.0, "over_refusal": None}
    assert rag_eval.gate_pass(only_refusals) is True


# --------------------------------------------------------------------------- #
# run_eval with injected stubs
# --------------------------------------------------------------------------- #
_ANSWERABLE = {"id": "a1", "question": "q", "key_facts": ["x"], "context": ["x is true"], "should_refuse": False}
_REFUSAL = {"id": "r1", "question": "q2", "key_facts": [], "context": ["unrelated"], "should_refuse": True}


async def test_perfect_system_passes():
    report = await rag_eval.run_eval(
        [_ANSWERABLE, _REFUSAL],
        generate_fn=rag_eval._stub_generate,
        faithfulness_fn=rag_eval._stub_faithfulness,
    )
    assert report["pass"] is True
    assert report["metrics"]["correctness"] == 1.0
    assert report["metrics"]["refusal_accuracy"] == 1.0
    assert report["metrics"]["over_refusal"] == 0.0
    assert report["answerable"]["n"] == 1 and report["refusal"]["n"] == 1


async def test_over_refusal_fails_gate():
    # System wrongly refuses an answerable question.
    async def refuse_everything(rec):
        return "REFUSED", True

    report = await rag_eval.run_eval(
        [_ANSWERABLE],
        generate_fn=refuse_everything,
        faithfulness_fn=rag_eval._stub_faithfulness,
    )
    assert report["metrics"]["over_refusal"] == 1.0
    assert report["metrics"]["correctness"] == 0.0
    assert report["metrics"]["faithfulness"] is None  # not scored on a refusal
    assert report["pass"] is False


async def test_missed_facts_fail_gate():
    async def empty_answer(rec):
        return "I have no idea.", False

    report = await rag_eval.run_eval(
        [_ANSWERABLE],
        generate_fn=empty_answer,
        faithfulness_fn=rag_eval._stub_faithfulness,
    )
    assert report["metrics"]["correctness"] == 0.0
    assert report["pass"] is False


async def test_missed_refusal_fails_gate():
    # System answers a question it should have refused.
    async def answer_everything(rec):
        return "The refund policy is 30 days.", False

    report = await rag_eval.run_eval(
        [_REFUSAL],
        generate_fn=answer_everything,
        faithfulness_fn=rag_eval._stub_faithfulness,
    )
    assert report["metrics"]["refusal_accuracy"] == 0.0
    assert report["pass"] is False


# --------------------------------------------------------------------------- #
# the shipped golden dataset
# --------------------------------------------------------------------------- #
def test_golden_dataset_loads_and_validates():
    records = rag_eval.load_golden(rag_eval.DEFAULT_DATA)
    assert len(records) >= 8
    # every answerable row must carry key_facts; refusal rows must not need them
    for rec in records:
        if not rec.get("should_refuse"):
            assert rec.get("key_facts"), f"{rec['id']} answerable but no key_facts"


def test_load_golden_rejects_answerable_without_key_facts(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"id": "x", "question": "q", "should_refuse": false, "key_facts": []}\n')
    with pytest.raises(ValueError):
        rag_eval.load_golden(str(bad))


async def test_golden_dataset_self_scores_perfect():
    records = rag_eval.load_golden(rag_eval.DEFAULT_DATA)
    report = await rag_eval.run_eval(
        records,
        generate_fn=rag_eval._stub_generate,
        faithfulness_fn=rag_eval._stub_faithfulness,
    )
    assert report["pass"] is True
    assert report["metrics"]["correctness"] == 1.0
    assert report["metrics"]["refusal_accuracy"] == 1.0
