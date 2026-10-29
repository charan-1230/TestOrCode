"""``scripts/review_runs.py`` on the committed Day-2 live pilot records (no model, no repository)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "results" / "day2" / "pilot-full-b1" / "runs.jsonl"


@pytest.fixture(scope="module")
def review():
    spec = importlib.util.spec_from_file_location("review_runs", ROOT / "scripts" / "review_runs.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def records(review):
    records, problems = review.load([PILOT, PILOT.with_name("config.json")])  # config.json is skipped, not an error
    assert problems == [] and len(records) == 6
    return {(r.episode_id, r.condition.value): r for r in records}


def test_escalation_cause_reads_the_guard_reason_not_the_review_verdict(review, records) -> None:
    # dev-06 FULL: "...the semantic review found it consistent, but the updated tests detect none of the mutants".
    assert review.escalation_cause(records[("dev-06", "FULL")], "g1") == "guard: test strength not shown by mutants"


def test_run_review_shows_trace_scores_and_errors(review, records) -> None:
    text = "\n".join(review.review(records[("dev-04", "FULL")], tools=True))
    assert "decision=FIX_CODE guard=approve disposition=accepted" in text
    assert "gold=FIX_CODE OK" in text and "run_mutants by integrity_guard" in text
    failed = "\n".join(review.review(records[("dev-07", "FULL")], tools=False))
    assert "error: rate_limit at repair:g1" in failed


def test_aggregate_counts_failures_and_escalation_causes(review, records) -> None:
    text = "\n".join(review.aggregate(list(records.values())))
    assert "rate_limit at repair" in text
    assert "guard: test strength not shown by mutants" in text
