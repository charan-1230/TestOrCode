"""Metrics: one run against its gold labels (evaluate_run) and per-condition aggregates (summarize).

Records are built by hand against the real dev episodes, so the expected numbers can be checked by reading the gold labels.
"""

from __future__ import annotations

import pytest

from testorcode.dataset import episode_path, load_episodes
from testorcode.evaluation import evaluate_run, summarize
from testorcode.schemas.episode_schemas import DecisionLabel as D
from testorcode.schemas.episode_schemas import HypothesisKind, PatchScope
from testorcode.schemas.result_schemas import (
    AgentRole,
    EvidenceFreshness,
    ExperimentCondition,
    ExperimentRunRecord,
    GuardDecision,
    IntegrityReport,
    PatchApplicationStatus,
    PatchRecord,
    PredictedDecision,
    RetrievedEvidence,
    RunOutcome,
    RunProvenance,
    RunStatus,
    TestRunPhase,
    TestRunResult,
    TestRunStatus,
    TestWeakeningFindings,
    TokenUsage,
)

DEV = {e.episode_id: e for e in load_episodes(episode_path("dev"))}
B1 = ExperimentCondition.B1


def gold_tests(eid: str) -> list[str]:
    return [lbl.test_id for lbl in DEV[eid].ground_truth.per_test_labels]


def record(eid: str, actions: dict[str, D] | D | None, *, condition=B1, status=RunStatus.COMPLETED, **extra) -> ExperimentRunRecord:
    """A run of ``eid``.  ``actions`` is one decision for every failing test, or a per-test map, or None (no decision)."""
    if isinstance(actions, D):
        actions = dict.fromkeys(gold_tests(eid), actions)
    pred_fields = extra.pop("pred", {})
    preds = [
        PredictedDecision(test_id=t, predicted_action=a, decided_by=AgentRole.SINGLE_AGENT, **pred_fields)
        for t, a in (actions or {}).items()
    ]
    return ExperimentRunRecord(
        provenance=RunProvenance(run_id=f"{eid}-r", episode_id=eid, condition=condition, seed=0, model_id="test-model-1"),
        predictions=preds,
        outcome=RunOutcome(status=status),
        **extra,
    )


def metrics(runs, condition=B1) -> dict:
    [summary] = [s for s in summarize(runs, list(DEV.values())) if s.condition is condition]
    return summary.metrics


# --------------------------------------------------------------------------- #
# One run
# --------------------------------------------------------------------------- #


def test_wrong_side_edits_are_told_apart() -> None:
    masked = evaluate_run(record("dev-08", D.UPDATE_TEST), DEV["dev-08"])  # gold: 1 UPDATE_TEST + 3 FIX_CODE
    assert [d.wrong_side for d in masked.decisions] == [None, "masked_regression", "masked_regression", "masked_regression"]
    assert [d.correct for d in masked.decisions] == [True, False, False, False]
    reverted = evaluate_run(record("dev-05", D.FIX_CODE), DEV["dev-05"])
    assert [d.wrong_side for d in reverted.decisions] == ["reverted_intent"]


def test_escalation_is_never_a_wrong_side_edit() -> None:
    wrong = evaluate_run(record("dev-05", D.ESCALATE), DEV["dev-05"])
    assert not wrong.decisions[0].correct and wrong.decisions[0].wrong_side is None
    assert wrong.unnecessary_escalation is True and wrong.escalation_correct is False
    right = evaluate_run(record("dev-01", D.ESCALATE), DEV["dev-01"])  # S5: gold is ESCALATE
    assert all(d.correct for d in right.decisions)
    assert right.escalation_correct is True and right.unnecessary_escalation is False
    guessed = evaluate_run(record("dev-01", D.UPDATE_TEST), DEV["dev-01"])
    assert guessed.escalation_correct is False and guessed.decisions[0].wrong_side is None


def test_a_run_without_a_decision_scores_as_not_decided() -> None:
    ev = evaluate_run(record("dev-02", None, status=RunStatus.ORCHESTRATOR_ERROR), DEV["dev-02"])
    assert [d.predicted_action for d in ev.decisions] == [None] and not ev.decisions[0].correct
    assert ev.escalation_correct is None and ev.unnecessary_escalation is None  # nothing was decided either way


def test_hypothesis_is_scored_against_the_side_that_is_wrong() -> None:
    good = record("dev-05", D.UPDATE_TEST, pred={"hypothesis": HypothesisKind.OUTDATED_TEST})
    bad = record("dev-05", D.UPDATE_TEST, pred={"hypothesis": HypothesisKind.REGRESSION})
    assert evaluate_run(good, DEV["dev-05"]).decisions[0].hypothesis_correct is True
    assert evaluate_run(bad, DEV["dev-05"]).decisions[0].hypothesis_correct is False
    assert evaluate_run(record("dev-05", D.UPDATE_TEST), DEV["dev-05"]).decisions[0].hypothesis_correct is None


def test_citations_to_nothing_are_reported_and_real_ones_are_not() -> None:
    real = DEV["dev-05"].ground_truth.per_test_labels[0].gold_evidence_ids[0]
    run = record("dev-05", D.UPDATE_TEST, pred={"cited_evidence_ids": [real, "msg:doesnotexist"]})
    assert evaluate_run(run, DEV["dev-05"]).evidence.invalid_citation_ids == ["msg:doesnotexist"]


def test_the_wrong_episode_is_refused() -> None:
    with pytest.raises(ValueError, match="dev-05"):
        evaluate_run(record("dev-05", D.UPDATE_TEST), DEV["dev-02"])


# --- repair: a green suite alone is never a successful repair ---------------------------------------------------


def repaired(eid: str, action: D, *, scope: PatchScope, files: list[str], integrity: IntegrityReport | None = None,
             final=TestRunStatus.PASSED) -> ExperimentRunRecord:
    ids = gold_tests(eid)
    return record(
        eid, action,
        patches=[PatchRecord(patch_id="p1", scope=scope, test_ids=ids, target_files=files,
                             application_status=PatchApplicationStatus.APPLIED)],
        test_runs=[TestRunResult(run_id="t1", phase=TestRunPhase.FINAL, status=final, passed=1 if final is TestRunStatus.PASSED else 0)],
        final_test_status=final,
        integrity=integrity or IntegrityReport(),
    )  # fmt: skip


APPROVED = IntegrityReport(
    guard_ran=True, guard_decision=GuardDecision.APPROVE,
    weakening=TestWeakeningFindings(checks_run=True, weakening_detected=False),
)  # fmt: skip


def test_repair_needs_direction_suite_and_a_verified_integrity_check() -> None:
    kwargs = {"scope": PatchScope.TEST, "files": ["tests/test_tbutils_parsed_exc.py"]}
    green_only = evaluate_run(repaired("dev-05", D.UPDATE_TEST, **kwargs), DEV["dev-05"]).repair
    assert green_only.repair_attempted and green_only.repair_successful is None  # no Guard: integrity unverified
    ok = evaluate_run(repaired("dev-05", D.UPDATE_TEST, integrity=APPROVED, **kwargs), DEV["dev-05"])
    assert ok.repair.repair_successful is True and len(ok.repair.verified_dimensions) == 3
    wrong = evaluate_run(repaired("dev-05", D.FIX_CODE, scope=PatchScope.SRC, files=["boltons/tbutils.py"], integrity=APPROVED), DEV["dev-05"])
    assert wrong.repair.repair_successful is False  # green suite, wrong side
    red = evaluate_run(repaired("dev-05", D.UPDATE_TEST, integrity=APPROVED, final=TestRunStatus.FAILED, **kwargs), DEV["dev-05"])
    assert red.repair.repair_successful is False


def test_no_patch_means_no_repair_not_a_failed_repair() -> None:
    repair = evaluate_run(record("dev-05", D.ESCALATE), DEV["dev-05"]).repair
    assert repair.repair_attempted is False and repair.repair_successful is None


def test_editing_tests_to_fix_code_is_flagged_and_blocks_success() -> None:
    run = repaired("dev-02", D.FIX_CODE, scope=PatchScope.TEST, files=["tests/test_tableutils.py"], integrity=APPROVED)
    ev = evaluate_run(run, DEV["dev-02"])
    assert ev.integrity.tests_touched_on_fix_code is True and ev.repair.repair_successful is False
    clean = evaluate_run(repaired("dev-02", D.FIX_CODE, scope=PatchScope.SRC, files=["boltons/tableutils.py"], integrity=APPROVED), DEV["dev-02"])
    assert clean.integrity.tests_touched_on_fix_code is False and clean.repair.repair_successful is True


# --------------------------------------------------------------------------- #
# One condition
# --------------------------------------------------------------------------- #


def test_macro_f1_and_accuracy_match_a_hand_computed_example() -> None:
    ids = gold_tests("dev-08")  # gold: UPDATE_TEST, FIX_CODE, FIX_CODE, FIX_CODE
    preds = dict(zip(ids, [D.UPDATE_TEST, D.UPDATE_TEST, D.FIX_CODE, D.FIX_CODE], strict=True))
    m = metrics([record("dev-08", preds)])
    # UPDATE_TEST: tp 1, fp 1, fn 0 -> 2/3.  FIX_CODE: tp 2, fp 0, fn 1 -> 4/5.  ESCALATE absent everywhere: skipped.
    assert m["macro_f1"] == pytest.approx((2 / 3 + 4 / 5) / 2)
    assert m["accuracy"] == pytest.approx(3 / 4) and m["n_tests_scored"] == 4
    assert m["masked_regression_rate"] == pytest.approx(1 / 3) and m["reverted_intent_rate"] == 0
    assert m["accuracy[S3]"] == pytest.approx(3 / 4)


def test_escalation_precision_recall_and_unnecessary_rate() -> None:
    runs = [record("dev-01", D.ESCALATE), record("dev-02", D.ESCALATE), record("dev-05", D.UPDATE_TEST)]
    m = metrics(runs)
    # predicted ESCALATE on 3 tests (2 gold-ESCALATE in dev-01, 1 wrong on dev-02); gold ESCALATE tests: 2
    assert m["escalation_precision"] == pytest.approx(2 / 3) and m["escalation_recall"] == 1.0
    assert m["unnecessary_escalation_rate"] == pytest.approx(1 / 2)  # of the 2 non-ESCALATE-gold tests, dev-02 was escalated


def test_empty_denominators_are_none_not_zero() -> None:
    m = metrics([record("dev-05", D.UPDATE_TEST)])
    assert m["escalation_precision"] is None and m["escalation_recall"] is None  # nothing escalated, no gold ESCALATE
    assert m["masked_regression_rate"] is None  # no FIX_CODE gold in dev-05
    assert m["repair_success_rate"] is None and m["citation_existence_rate"] is None
    assert m["gold_evidence_recall@3"] is None  # nothing was retrieved: not a recall of 0


def test_execution_failures_are_counted_apart_and_never_scored() -> None:
    runs = [record("dev-05", D.UPDATE_TEST), record("dev-02", None, status=RunStatus.API_FAILURE)]
    m = metrics(runs)
    assert m["n_runs"] == 2 and m["n_execution_failures"] == 1 and m["n_tests_scored"] == 1 and m["accuracy"] == 1.0


def test_gold_evidence_recall_uses_the_retrieved_ranking() -> None:
    gold = DEV["dev-05"].ground_truth.per_test_labels[0].gold_evidence_ids[0]

    def item(eid: str, rank: int) -> RetrievedEvidence:
        return RetrievedEvidence(evidence_id=eid, source_type="commit_message", source_ref="x", rank=rank,
                                 freshness=EvidenceFreshness.UNKNOWN)  # fmt: skip

    at4 = [item("msg:a", 1), item("msg:b", 2), item("msg:c", 3), item(gold, 4)]
    m = metrics([record("dev-05", D.UPDATE_TEST, retrieved_evidence=at4)])
    assert (m["gold_evidence_recall@3"], m["gold_evidence_recall@5"], m["gold_evidence_recall@8"]) == (0.0, 1.0, 1.0)
    m = metrics([record("dev-02", D.FIX_CODE, retrieved_evidence=at4)])
    assert m["gold_evidence_recall@5"] is None  # dev-02 has no gold evidence (a regression has none to find)


def test_citation_existence_rate_counts_only_retrieved_citations() -> None:
    ev = RetrievedEvidence(evidence_id="msg:a", source_type="commit_message", source_ref="x", rank=1)
    run = record("dev-05", D.UPDATE_TEST, retrieved_evidence=[ev], pred={"cited_evidence_ids": ["msg:a", "msg:zzz"]})
    assert metrics([run])["citation_existence_rate"] == 0.5


def test_a_usable_memory_record_is_a_citable_id_and_a_blocked_one_is_not() -> None:
    from testorcode.schemas.result_schemas import MemoryReadEvent, ProbeStatus

    ev = RetrievedEvidence(evidence_id="msg:a", source_type="commit_message", source_ref="x", rank=1)
    reads = [MemoryReadEvent(record_id="mem-ok", probe_status=ProbeStatus.PASSED, used_as_evidence=True),
             MemoryReadEvent(record_id="mem-bad", probe_status=ProbeStatus.FAILED, blocked_reason="probe_failed")]
    cited = {"cited_evidence_ids": ["msg:a", "mem-ok", "mem-bad"]}
    run = record("dev-07", D.FIX_CODE, retrieved_evidence=[ev], memory_reads=reads, pred=cited)
    assert metrics([run])["citation_existence_rate"] == pytest.approx(2 / 3)


def test_tokens_are_averaged_only_over_runs_that_reported_them() -> None:
    reported = record("dev-05", D.UPDATE_TEST, usage=TokenUsage(reported=True, total_tokens=100, input_tokens=60, output_tokens=40))
    unreported = record("dev-06", D.UPDATE_TEST)
    assert metrics([reported, unreported])["mean_total_tokens"] == 100.0
    assert metrics([unreported])["mean_total_tokens"] is None


def test_summaries_are_per_condition_and_unknown_episodes_are_refused() -> None:
    a2 = record("dev-05", D.UPDATE_TEST, condition=ExperimentCondition.A2_NO_MEMORY)
    sums = summarize([record("dev-05", D.FIX_CODE), a2], list(DEV.values()))
    assert [s.condition for s in sums] == [ExperimentCondition.A2_NO_MEMORY, ExperimentCondition.B1]
    assert sums[0].metrics["accuracy"] == 1.0 and sums[1].metrics["accuracy"] == 0.0
    assert any("mutation_fixtures" in n for n in sums[0].notes)  # what is not computed is stated, not hidden
    with pytest.raises(ValueError, match="unknown episode"):
        summarize([record("dev-05", D.UPDATE_TEST)], [DEV["dev-02"]])


def test_a_trivial_baseline_has_no_recall_and_zero_cost() -> None:
    from testorcode.baselines import trivial_record
    from testorcode.dataset import agent_view_path, load_agent_inputs

    views = {e.episode_id: e for e in load_agent_inputs(agent_view_path("dev"))}
    runs = [trivial_record(v, ExperimentCondition.TRIVIAL_ALWAYS_FIX_CODE, seed=0) for v in views.values()]
    m = metrics(runs, ExperimentCondition.TRIVIAL_ALWAYS_FIX_CODE)
    assert m["accuracy"] == pytest.approx(10 / 16) and m["escalation_recall"] == 0.0  # dev gold: 10 FIX_CODE, 2 ESCALATE
    assert m["gold_evidence_recall@3"] is None and m["mean_tool_calls"] == 0 and m["mean_total_tokens"] == 0


# --------------------------------------------------------------------------- #
# Direction macro-F1 (the proposal's per-test UPDATE_TEST vs FIX_CODE choice)
# --------------------------------------------------------------------------- #


def decision(test_id: str, action: D, hypothesis: HypothesisKind | None) -> PredictedDecision:
    return PredictedDecision(test_id=test_id, predicted_action=action, hypothesis=hypothesis,
                             decided_by=AgentRole.INTENT_ANALYST)  # fmt: skip


def test_direction_macro_f1_matches_a_hand_computed_example() -> None:
    t = gold_tests("dev-08")  # gold actions: UPDATE_TEST, FIX_CODE, FIX_CODE, FIX_CODE
    run = record("dev-08", None).model_copy(update={"predictions": [
        decision(t[0], D.ESCALATE, HypothesisKind.OUTDATED_TEST),  # unverified repair: the side blamed was right
        decision(t[1], D.FIX_CODE, HypothesisKind.REGRESSION),     # right
        decision(t[2], D.ESCALATE, HypothesisKind.UNKNOWN),        # undecided: a miss, no false positive
        decision(t[3], D.UPDATE_TEST, HypothesisKind.OUTDATED_TEST),  # wrong side
    ]})
    m = metrics([run])
    # UPDATE_TEST: tp 1, fp 1, fn 0 -> 2/3.  FIX_CODE: tp 1, fp 0, fn 2 -> 2/4.  Macro = (2/3 + 1/2) / 2 = 7/12.
    assert m["direction_macro_f1"] == pytest.approx(7 / 12)
    assert m["direction_accuracy"] == pytest.approx(2 / 4)
    assert (m["n_direction_scored"], m["n_direction_undecided"]) == (4, 1)
    assert m["accuracy"] == pytest.approx(1 / 4)  # final actions: only the FIX_CODE one matches; scored separately


def test_gold_escalate_cases_are_not_direction_cases() -> None:
    run = record("dev-01", None).model_copy(update={"predictions": [
        decision(t, D.ESCALATE, HypothesisKind.OUTDATED_TEST) for t in gold_tests("dev-01")]})  # S5: gold ESCALATE
    m = metrics([run])
    assert m["n_direction_scored"] == 0 and m["direction_macro_f1"] is None and m["direction_accuracy"] is None
    assert m["escalation_recall"] == 1.0


# --------------------------------------------------------------------------- #
# Records from another dataset revision are set aside, never scored as zero
# --------------------------------------------------------------------------- #


def with_version(run: ExperimentRunRecord, version: str | None) -> ExperimentRunRecord:
    return run.model_copy(update={"provenance": run.provenance.model_copy(update={"dataset_version": version})})


def test_records_from_another_dataset_revision_are_set_aside() -> None:
    current = with_version(record("dev-05", D.UPDATE_TEST), "pilot@current")
    old = with_version(record("dev-02", D.FIX_CODE), "pilot@old")
    gone = record("dev-02", None)  # an episode id the current split no longer has
    gone = gone.model_copy(update={"provenance": gone.provenance.model_copy(
        update={"episode_id": "dev-99-gone", "run_id": "dev-99-gone-r"})})
    unknown = with_version(gone, "pilot@old")
    [summary] = summarize([current, old, unknown], list(DEV.values()), dataset_version="pilot@current")
    assert summary.metrics["n_runs"] == 1 and summary.metrics["n_incompatible_records"] == 2
    assert summary.metrics["accuracy"] == 1.0  # only the compatible record is scored
    assert set(summary.incompatible_run_ids) == {"dev-02-r", "dev-99-gone-r"}
    assert any("pilot@old" in n and "pilot@current" in n for n in summary.notes)


def test_a_condition_with_only_incompatible_records_reports_nothing_as_measured() -> None:
    old = with_version(record("dev-05", D.UPDATE_TEST), None)
    [summary] = summarize([old], list(DEV.values()), dataset_version="pilot@current")
    m = summary.metrics
    assert m["n_runs"] == 0 and m["n_incompatible_records"] == 1
    assert m["accuracy"] is None and m["direction_macro_f1"] is None and m["repair_success_rate"] is None


def test_predictions_for_tests_the_episode_does_not_have_are_set_aside() -> None:
    stale = record("dev-05", {"tests/test_cacheutils.py::test_unscoped_cached_dec": D.UPDATE_TEST})  # old dev-05
    [summary] = summarize([stale], list(DEV.values()))
    assert summary.metrics["n_incompatible_records"] == 1 and summary.metrics["n_tests_scored"] == 0
    assert any("does not have" in n for n in summary.notes)


def test_a_completed_run_missing_a_prediction_counts_the_test_as_wrong() -> None:
    tests = gold_tests("dev-03")  # two FIX_CODE tests
    assert len(tests) == 2
    m = metrics([record("dev-03", {tests[0]: D.FIX_CODE})])  # no decision for the second test
    assert m["n_tests_scored"] == 2 and m["accuracy"] == 0.5
    assert m["n_direction_scored"] == 2 and m["n_direction_undecided"] == 1 and m["direction_accuracy"] == 0.5


def test_a_test_the_harness_found_unstable_is_not_scored_as_missed() -> None:
    tests = gold_tests("dev-03")
    m = metrics([record("dev-03", {tests[0]: D.FIX_CODE}, unstable_test_ids=[tests[1]])])
    assert m["n_tests_scored"] == 1 and m["accuracy"] == 1.0
