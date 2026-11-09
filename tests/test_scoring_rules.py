"""Scoring rules (A1 / direction source, Guard escalation after a correct repair) and the hypothesis check.

Checked on the saved Day-3 dev runs (``results/day3/pilot-dev-5cond``): FULL's dev-08 ``test_bytes2human`` is the one test
the Guard escalated after a correct, verified UPDATE_TEST repair, and A1 escalated every test.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from testorcode.dataset import episode_path, load_episodes
from testorcode.evaluation import (
    ScoringRules,
    escalated_after_correct_repair,
    evaluate_run,
    summarize,
)
from testorcode.experiment import dataset_version, read_records
from testorcode.freeze import FrozenConfig, load_rules
from testorcode.hypothesis import HypothesisThresholds, check_hypothesis, markdown
from testorcode.schemas.result_schemas import ExperimentCondition as C
from testorcode.stats import paired_bootstrap

ROOT = Path(__file__).resolve().parents[1]
DEV_RUNS = ROOT / "results" / "day3" / "pilot-dev-5cond" / "runs.jsonl"


@pytest.fixture(scope="module")
def dev():
    return read_records(DEV_RUNS), load_episodes(episode_path("dev"))


def metrics(dev, **rules):
    records, episodes = dev
    return {s.condition: s.metrics for s in summarize(records, episodes, dataset_version=dataset_version("dev"),
                                                      rules=ScoringRules(**rules))}


def test_default_rules_reproduce_the_pilot_tables(dev) -> None:
    m = metrics(dev)
    assert round(m[C.FULL]["accuracy"], 4) == 0.5625 and m[C.FULL]["direction_macro_f1"] == 1.0  # docs/day3-human3.md
    records, episodes = dev
    by_id = {e.episode_id: e for e in episodes}
    raw = [d.correct for r in records if r.condition is C.FULL
           for d in evaluate_run(r, by_id[r.episode_id]).decisions]
    assert m[C.FULL]["accuracy"] == sum(raw) / len(raw)


def test_the_guard_escalation_after_a_correct_repair_is_found_and_scored_by_the_rule(dev) -> None:
    records, _ = dev
    full_dev08 = next(r for r in records if r.condition is C.FULL and r.episode_id == "dev-08")
    test_id = "tests/test_strutils.py::test_bytes2human"
    from testorcode.schemas.episode_schemas import RepairDirection

    assert escalated_after_correct_repair(full_dev08, test_id, RepairDirection.UPDATE_TEST)
    assert not escalated_after_correct_repair(full_dev08, test_id, RepairDirection.FIX_CODE)  # wrong side: not it
    found = [(r.episode_id, p.test_id) for r in records for p in r.predictions for d in RepairDirection
             if escalated_after_correct_repair(r, p.test_id, d)]
    assert found == [("dev-08", test_id)]
    wrong, correct, separate = (metrics(dev, escalate_after_correct_repair=r)[C.FULL]
                                for r in ("wrong", "correct", "separate"))
    assert wrong["n_escalated_after_correct_repair"] == correct["n_escalated_after_correct_repair"] == 1.0
    assert (wrong["n_tests_scored"], correct["n_tests_scored"], separate["n_tests_scored"]) == (16.0, 16.0, 15.0)
    assert correct["accuracy"] == pytest.approx(10 / 16) and separate["accuracy"] == pytest.approx(9 / 15)
    assert wrong["direction_macro_f1"] == correct["direction_macro_f1"]  # direction metrics are unaffected


def test_direction_from_action_gives_escalations_no_side(dev) -> None:
    hypothesis, action = metrics(dev)[C.A1_NO_INTENT_RETRIEVAL], metrics(dev, direction_from="action")[C.A1_NO_INTENT_RETRIEVAL]
    assert hypothesis["direction_macro_f1"] > 0 and action["direction_macro_f1"] == 0.0  # A1 escalated every test
    assert action["n_direction_undecided"] == action["n_direction_scored"]
    assert action["accuracy"] == hypothesis["accuracy"]  # final-action metrics do not depend on it


def test_statistics_score_with_the_same_rules(dev) -> None:
    records, episodes = dev
    default = paired_bootstrap(records, C.FULL, C.A1_NO_INTENT_RETRIEVAL, episodes, samples=200)
    action = paired_bootstrap(records, C.FULL, C.A1_NO_INTENT_RETRIEVAL, episodes, samples=200,
                              rules=ScoringRules(direction_from="action"))
    m = metrics(dev, direction_from="action")
    assert action["difference"] == pytest.approx(m[C.FULL]["direction_macro_f1"] - m[C.A1_NO_INTENT_RETRIEVAL]["direction_macro_f1"])
    assert action["difference"] != default["difference"]


def test_hypothesis_check_reports_every_criterion_and_applies_the_rejection_rule(dev) -> None:
    records, episodes = dev
    result = check_hypothesis(records, episodes, dataset_version=dataset_version("dev"), samples=200,
                              audit_jsonl=DEV_RUNS.parent / "integrity_audit.jsonl")
    ci_test, b2_test = result["rejection_tests"]
    # FULL 1.0 vs B1 0.944 direction macro-F1 on 9 episodes: the CI includes 0, which alone rejects (proposal section 7),
    # even though the B2 test cannot be computed (no B2 run).
    assert ci_test["fires"] is True and b2_test["fires"] is None and result["verdict"] == "rejected"
    gain = next(c for c in result["criteria"] if c["criterion"].startswith("direction macro-F1 gain"))
    assert gain["observed"] == pytest.approx(1.0 - metrics(dev)[C.B1]["direction_macro_f1"]) and gain["met"] is False
    coarse = next(c for c in result["criteria"] if "coarse" in c["criterion"])
    assert coarse["observed"] is None and coarse["met"] is None  # A4 did not run: not computable, never 0
    assert "Verdict: rejected" in "\n".join(markdown(result))
    json.dumps(result)


def test_without_b2_the_verdict_is_not_decidable_unless_the_ci_rejects(dev) -> None:
    records, episodes = dev
    better = [r for r in records if r.condition is not C.B1]
    as_b1 = [r.model_copy(update={"provenance": r.provenance.model_copy(update={"condition": C.B1})})
             for r in records if r.condition is C.TRIVIAL_ALWAYS_FIX_CODE]  # a much worse "B1"
    result = check_hypothesis([*better, *as_b1], episodes, dataset_version=dataset_version("dev"), samples=500)
    assert result["rejection_tests"][0]["fires"] is False and result["verdict"] == "not decidable"


def test_b2_closing_the_gap_rejects_the_hypothesis(dev) -> None:
    records, episodes = dev
    as_b2 = [r.model_copy(update={"provenance": r.provenance.model_copy(update={"condition": C.B2})})
             for r in records if r.condition is C.FULL]  # a B2 exactly as good as FULL
    result = check_hypothesis([*records, *as_b2], episodes, dataset_version=dataset_version("dev"), samples=200)
    assert result["rejection_tests"][1]["fires"] is True and result["verdict"] == "rejected"


def test_frozen_rules_and_thresholds_load_and_are_validated(tmp_path: Path) -> None:
    path = tmp_path / "f.json"
    path.write_text(json.dumps({"name": "x", "scoring": {"direction_from": "action"},
                                "hypothesis": {"min_direction_f1_gain": 0.1}}))
    rules, thresholds, frozen = load_rules(path)
    assert rules.direction_from == "action" and rules.escalate_after_correct_repair == "wrong"
    assert thresholds.min_direction_f1_gain == 0.1 and thresholds.b2_gap == HypothesisThresholds().b2_gap
    assert frozen is not None and load_rules(None)[0] == ScoringRules()
    with pytest.raises(ValidationError):
        FrozenConfig(name="x", scoring={"direction_from": "vibes"})
