"""Tests for ``testorcode.result_schemas``.  Runs, tokens and traces are synthetic;
no experiment was executed to produce them."""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import ValidationError

from testorcode.schemas.tests.conftest import T_NORM, T_SLUG, make_s3
from testorcode.schemas.episode_schemas import (
    DecisionLabel, EvidenceSourceType, HypothesisKind, PatchScope,
    RepairDirection, ToolFaultKind, ToolName,
)
from testorcode.schemas.result_schemas import (
    CORE_CONDITIONS, METRIC_SUPPORT, AgentRole, ErrorCategory, ErrorRecord, EscalationSummary,
    EvidenceEvaluation, ExperimentCondition, ExperimentRunRecord, GuardDecision, IntegrityEvaluation,
    IntegrityReport, MemoryReadEvent, MemoryWriteAttempt, MutationResult, PatchApplicationStatus,
    PatchRecord, PredictedDecision, ProbeStatus, ProhibitedEditCheck, RepairEvaluation,
    RepairVerificationDimension as Dim, RetrievalMethod, RetrievedEvidence, RunEvaluation,
    RunOutcome, RunProvenance, RunStatus, SupportStatus, TestRunPhase, TestRunResult, TestRunStatus,
    TestWeakeningFindings, TokenUsage, ToolCallStatus, ToolEvent, UnitDecisionScore,
)

MODEL = "qwen/qwen3-coder-480b-a35b-instruct"
C = ExperimentCondition


def prov(run_id: str = "run-1", condition: ExperimentCondition = C.FULL, seed: int | None = 0,
         episode_id: str = "ep-0001", **kw: Any) -> RunProvenance:
    kw.setdefault("model_id", MODEL)
    return RunProvenance(run_id=run_id, episode_id=episode_id, condition=condition, seed=seed, **kw)


def decision(test_id: str = T_NORM, action: DecisionLabel = DecisionLabel.UPDATE_TEST, **kw: Any) -> PredictedDecision:
    kw.setdefault("cited_evidence_ids", ["ev-msg"])
    return PredictedDecision(test_id=test_id, predicted_action=action, hypothesis=HypothesisKind.OUTDATED_TEST,
                             predicted_behaviour="normalise('ab') == 'Ab'", rationale="Commit says intended.", **kw)


def evidence() -> RetrievedEvidence:
    return RetrievedEvidence(evidence_id="ev-msg", source_type=EvidenceSourceType.COMMIT_MESSAGE, source_ref="a1b2c3d",
                             retrieval_query="normalise intent", retrieval_method=RetrievalMethod.BM25, rank=1,
                             score=12.4, support=SupportStatus.SUPPORTS)


def passed_run(run_id: str = "tr-2", phase: TestRunPhase = TestRunPhase.POST_REPAIR) -> TestRunResult:
    return TestRunResult(run_id=run_id, phase=phase, status=TestRunStatus.PASSED, passed=40, failed=0)


def complete_run() -> ExperimentRunRecord:
    post = passed_run()
    return ExperimentRunRecord(
        provenance=prov(dataset_version="pilot-v1", split="pilot", code_revision="deadbee",
                        prompt_template_version="p3", retrieval_config_version="bm25-k8"),
        predictions=[decision()], retrieved_evidence=[evidence()],
        patches=[PatchRecord(patch_id="p-1", scope=PatchScope.TEST, test_ids=[T_NORM],
                             target_files=["tests/test_synthlib.py"], unified_diff="@@ ...",
                             application_status=PatchApplicationStatus.APPLIED)],
        test_runs=[TestRunResult(run_id="tr-1", phase=TestRunPhase.PRE_REPAIR, status=TestRunStatus.FAILED,
                                 passed=39, failed=1, failing_test_ids=[T_NORM]), post],
        final_test_status=TestRunStatus.PASSED,
        integrity=IntegrityReport(
            guard_ran=True, guard_decision=GuardDecision.APPROVE,
            weakening=TestWeakeningFindings(checks_run=True, assertions_before=3, assertions_after=3,
                                            removed_assertions=0, added_skip_markers=0, weakening_detected=False),
            prohibited_edits=ProhibitedEditCheck(checks_run=True, files_modified=["tests/test_synthlib.py"],
                                                 test_files_modified=["tests/test_synthlib.py"]),
            mutation=MutationResult(status=TestRunStatus.PASSED, mutants_total=4,
                                    killed_by_reference_tests=3, killed_by_repaired_tests=3),
            evidence_requirement_met=True),
        memory_reads=[MemoryReadEvent(record_id="mem-1", probe_status=ProbeStatus.PASSED, used_as_evidence=True)],
        memory_writes=[MemoryWriteAttempt(record_id="mem-2", approved=True, written=True)],
        tool_events=[ToolEvent(event_id="te-1", step_index=0, tool=ToolName.SEARCH_REPO,
                               called_by=AgentRole.INTENT_ANALYST, arguments={"query": "normalise"})],
        usage=TokenUsage(reported=True, input_tokens=18_000, output_tokens=2_100, llm_calls=12),
        errors=[ErrorRecord(error_id="e-1", category=ErrorCategory.RATE_LIMIT, message="429", retries=2,
                            recovered=True, counted_against_agent=False)],
        outcome=RunOutcome(status=RunStatus.COMPLETED, maintainer_approved=True))


# 10/12. serialisation and completeness ------------------------------------- #

def test_round_trip_and_required_fields_present_in_one_run() -> None:
    run = complete_run()
    line = run.to_jsonl()
    assert "\n" not in line and ExperimentRunRecord.model_validate_json(line) == run
    d = json.loads(line)
    assert (d["provenance"]["episode_id"], d["provenance"]["condition"], d["provenance"]["seed"],
            d["provenance"]["model_id"]) == ("ep-0001", "FULL", 0, MODEL)
    assert d["predictions"][0]["predicted_action"] == "UPDATE_TEST"
    assert d["retrieved_evidence"][0]["rank"] == 1 and d["retrieved_evidence"][0]["retrieval_method"] == "bm25"
    assert d["patches"][0]["scope"] == "test" and d["patches"][0]["application_status"] == "applied"
    assert d["final_test_status"] == "passed"
    assert d["integrity"]["guard_decision"] == "approve"
    assert d["usage"]["input_tokens"] == 18_000 and len(d["tool_events"]) == 1
    assert d["errors"][0]["category"] == "rate_limit" and d["outcome"]["status"] == "completed"


def test_core_conditions_cover_the_five_day_plan() -> None:
    assert set(CORE_CONDITIONS) == {C.FULL, C.B1, C.A1_NO_INTENT_RETRIEVAL, C.A2_NO_MEMORY, C.A3_NO_INTEGRITY_GUARD}
    assert {"Direction accuracy / macro-F1", "Mutant-kill retention", "Per-test hypothesis accuracy"} <= set(METRIC_SUPPORT)


# provenance ---------------------------------------------------------------- #

def test_unseeded_differs_from_seed_zero_and_seed_is_required() -> None:
    assert prov(seed=None).seed is None and prov(seed=0).seed == 0
    assert json.loads(prov(seed=None).model_dump_json())["seed"] is None
    with pytest.raises(ValidationError):
        RunProvenance(run_id="r", episode_id="e", condition=C.FULL, model_id=MODEL)  # type: ignore[call-arg]


def test_model_id_rules() -> None:
    with pytest.raises(ValidationError, match="provider name"):
        prov(model_id="mistral")
    with pytest.raises(ValidationError, match="model_id is required"):
        prov(model_id=None)
    assert prov(condition=C.TRIVIAL_ALWAYS_FIX_CODE, model_id=None, seed=None).model_id is None


def test_same_episode_across_conditions_and_seeds() -> None:
    runs = [ExperimentRunRecord(provenance=prov(run_id=f"{c.value}-{s}", condition=c, seed=s),
                                predictions=[decision()], retrieved_evidence=[evidence()])
            for c in CORE_CONDITIONS for s in (0, 1)]
    assert {r.episode_id for r in runs} == {"ep-0001"} and len({r.provenance.run_id for r in runs}) == 10


# decisions / evidence ------------------------------------------------------ #

def test_mixed_episode_has_one_decision_per_test_with_different_actions() -> None:
    ep = make_s3()
    run = ExperimentRunRecord(provenance=prov(run_id="r3", episode_id=ep.episode_id),
                              predictions=[decision(T_NORM, DecisionLabel.UPDATE_TEST),
                                           decision(T_SLUG, DecisionLabel.FIX_CODE, cited_evidence_ids=[])],
                              retrieved_evidence=[evidence()])
    assert run.predicted_actions() == ep.gold_actions()
    with pytest.raises(ValidationError, match="more than one decision"):
        ExperimentRunRecord(provenance=prov(), predictions=[decision(), decision()])


def test_hallucinated_citation_is_recorded_not_rejected() -> None:
    run = ExperimentRunRecord(provenance=prov(), predictions=[decision(cited_evidence_ids=["ev-ghost", "ev-msg"])],
                              retrieved_evidence=[evidence()])
    assert run.unretrieved_citations() == {T_NORM: ["ev-ghost"]}
    # also detected when retrieval returned nothing at all (A1 / B0 style)
    assert ExperimentRunRecord(provenance=prov(), predictions=[decision()]).unretrieved_citations() == {T_NORM: ["ev-msg"]}


def test_missing_evidence_score_is_not_zero() -> None:
    e = RetrievedEvidence(evidence_id="e", source_type=EvidenceSourceType.DOCUMENTATION, source_ref="docs/a.md")
    assert e.score is None and e != e.model_copy(update={"score": 0.0}) and e.support is SupportStatus.UNKNOWN


# 11. unrun vs passed -------------------------------------------------------- #

def test_final_test_status_must_match_test_runs() -> None:
    assert ExperimentRunRecord(provenance=prov()).final_test_status is TestRunStatus.NOT_RUN
    with pytest.raises(ValidationError, match="disagrees with the recorded test runs"):
        ExperimentRunRecord(provenance=prov(), final_test_status=TestRunStatus.PASSED)  # claims pass, ran nothing
    with pytest.raises(ValidationError, match="disagrees with the recorded test runs"):
        ExperimentRunRecord(provenance=prov(), test_runs=[passed_run()])  # ran, but claims NOT_RUN
    pre_only = TestRunResult(run_id="t", phase=TestRunPhase.PRE_REPAIR, status=TestRunStatus.FAILED, failed=1)
    assert ExperimentRunRecord(provenance=prov(), test_runs=[pre_only]).final_test_status is TestRunStatus.NOT_RUN


def test_last_post_repair_run_decides_final_status() -> None:
    rev = TestRunResult(run_id="t3", phase=TestRunPhase.POST_REVISION, status=TestRunStatus.FAILED, failed=1)
    run = ExperimentRunRecord(provenance=prov(), test_runs=[passed_run(), rev], final_test_status=TestRunStatus.FAILED)
    assert run.final_test_status is TestRunStatus.FAILED


def test_test_run_internal_consistency() -> None:
    with pytest.raises(ValidationError, match="NOT_RUN must not carry counts"):
        TestRunResult(run_id="t", phase=TestRunPhase.POST_REPAIR, failed=0)
    with pytest.raises(ValidationError, match="PASSED contradicts"):
        TestRunResult(run_id="t", phase=TestRunPhase.POST_REPAIR, status=TestRunStatus.PASSED, failed=2)
    with pytest.raises(ValidationError, match="FAILED contradicts"):
        TestRunResult(run_id="t", phase=TestRunPhase.POST_REPAIR, status=TestRunStatus.FAILED, failed=0)


def test_unrun_checks_are_not_passing_checks() -> None:
    w = TestWeakeningFindings()
    assert w.checks_run is False and w.weakening_detected is None
    with pytest.raises(ValidationError, match="all be None when checks_run is False"):
        TestWeakeningFindings(checks_run=False, removed_assertions=0)
    with pytest.raises(ValidationError, match="all be None"):
        TestWeakeningFindings(checks_run=False, weakening_detected=False)
    with pytest.raises(ValidationError, match="requires weakening_detected"):
        TestWeakeningFindings(checks_run=True)
    with pytest.raises(ValidationError, match="contradicts a positive weakening count"):
        TestWeakeningFindings(checks_run=True, added_skip_markers=1, weakening_detected=False)


def test_assertion_count_apparent_decrease_consistency() -> None:
    with pytest.raises(ValidationError, match="weakening_detected=False contradicts an apparent decrease"):
        TestWeakeningFindings(checks_run=True, assertions_before=5, assertions_after=3, weakening_detected=False)
    with pytest.raises(ValidationError, match="removed_assertions=0 contradicts an apparent decrease"):
        TestWeakeningFindings(checks_run=True, assertions_before=5, assertions_after=3, removed_assertions=0, weakening_detected=True)
    valid = TestWeakeningFindings(checks_run=True, assertions_before=5, assertions_after=3, removed_assertions=1, weakening_detected=True)
    assert valid.weakening_detected is True
    equal = TestWeakeningFindings(checks_run=True, assertions_before=3, assertions_after=3, removed_assertions=0, weakening_detected=False)
    assert equal.weakening_detected is False
    increased = TestWeakeningFindings(checks_run=True, assertions_before=3, assertions_after=4, removed_assertions=0, weakening_detected=False)
    assert increased.weakening_detected is False
    with pytest.raises(ValidationError, match="edit lists must be empty"):
        ProhibitedEditCheck(checks_run=False, files_modified=["a.py"])
    with pytest.raises(ValidationError, match="NOT_RUN mutation result"):
        MutationResult(killed_by_repaired_tests=0)
    assert MutationResult().status is TestRunStatus.NOT_RUN and MutationResult().killed_by_repaired_tests is None


def test_a3_guard_not_run_leaves_everything_unrun() -> None:
    r = IntegrityReport()
    assert r.guard_decision is GuardDecision.NOT_RUN and not r.guard_ran
    with pytest.raises(ValidationError, match="Guard did not run"):
        IntegrityReport(tests_untouched_for_fix_code=True)
    with pytest.raises(ValidationError, match="guard_ran must be True exactly"):
        IntegrityReport(guard_ran=True)


# 12. integrity / mutation / repair ----------------------------------------- #

def test_guard_rejection_and_contradictory_approval() -> None:
    with pytest.raises(ValidationError, match="at least one reason"):
        IntegrityReport(guard_ran=True, guard_decision=GuardDecision.REJECT)
    assert IntegrityReport(guard_ran=True, guard_decision=GuardDecision.REJECT, rejection_reasons=["lost assertion"])
    weak = TestWeakeningFindings(checks_run=True, removed_assertions=2, weakening_detected=True)
    with pytest.raises(ValidationError, match="APPROVE contradicts"):
        IntegrityReport(guard_ran=True, guard_decision=GuardDecision.APPROVE, weakening=weak)
    with pytest.raises(ValidationError, match="APPROVE contradicts"):
        IntegrityReport(guard_ran=True, guard_decision=GuardDecision.APPROVE, tests_untouched_for_fix_code=False)
    edits = ProhibitedEditCheck(checks_run=True, files_modified=["tests/conftest.py"],
                                out_of_scope_files=["tests/conftest.py"])
    with pytest.raises(ValidationError, match="APPROVE contradicts"):
        IntegrityReport(guard_ran=True, guard_decision=GuardDecision.APPROVE, prohibited_edits=edits)


def test_mutation_counts_bounded() -> None:
    with pytest.raises(ValidationError, match="exceeds mutants_total"):
        MutationResult(status=TestRunStatus.PASSED, mutants_total=2, killed_by_repaired_tests=3)


def test_green_suite_alone_is_not_a_successful_repair() -> None:
    with pytest.raises(ValidationError, match="passing suite alone"):
        RepairEvaluation(repair_attempted=True, repair_successful=True, verified_dimensions=[Dim.SUITE_PASSED],
                         final_test_status=TestRunStatus.PASSED)
    with pytest.raises(ValidationError, match="final_test_status PASSED"):
        RepairEvaluation(repair_attempted=True, repair_successful=True,
                         verified_dimensions=[Dim.DIRECTION_CORRECT, Dim.SUITE_PASSED, Dim.INTEGRITY_PRESERVED])
    ok = RepairEvaluation(repair_attempted=True, repair_successful=True, final_test_status=TestRunStatus.PASSED,
                          verified_dimensions=[Dim.DIRECTION_CORRECT, Dim.SUITE_PASSED, Dim.INTEGRITY_PRESERVED])
    assert ok.repair_successful


def test_repair_success_contradicting_integrity_findings_is_rejected() -> None:
    ok = RepairEvaluation(repair_attempted=True, repair_successful=True, final_test_status=TestRunStatus.PASSED,
                          verified_dimensions=[Dim.DIRECTION_CORRECT, Dim.SUITE_PASSED, Dim.INTEGRITY_PRESERVED])
    with pytest.raises(ValidationError, match="test_weakening_detected=True"):
        RunEvaluation(repair=ok, integrity=IntegrityEvaluation(test_weakening_detected=True))
    with pytest.raises(ValidationError, match="out-of-scope edits"):
        RunEvaluation(repair=ok, integrity=IntegrityEvaluation(out_of_scope_edits=["pyproject.toml"]))


def test_repair_successful_requires_harness_integrity_evaluation() -> None:
    ok = RepairEvaluation(repair_attempted=True, repair_successful=True, final_test_status=TestRunStatus.PASSED,
                          verified_dimensions=[Dim.DIRECTION_CORRECT, Dim.SUITE_PASSED, Dim.INTEGRITY_PRESERVED])
    with pytest.raises(ValidationError, match="independent harness integrity evaluation"):
        RunEvaluation(repair=ok, integrity=None)


def test_repair_successful_contradicting_integrity_dimensions_is_rejected() -> None:
    ok = RepairEvaluation(repair_attempted=True, repair_successful=True, final_test_status=TestRunStatus.PASSED,
                          verified_dimensions=[Dim.DIRECTION_CORRECT, Dim.SUITE_PASSED, Dim.INTEGRITY_PRESERVED])
    with pytest.raises(ValidationError, match="tests_touched_on_fix_code=True"):
        RunEvaluation(repair=ok, integrity=IntegrityEvaluation(tests_touched_on_fix_code=True))
    with pytest.raises(ValidationError, match="unapproved_memory_write=True"):
        RunEvaluation(repair=ok, integrity=IntegrityEvaluation(unapproved_memory_write=True))
    with pytest.raises(ValidationError, match="stale_memory_used_as_evidence=True"):
        RunEvaluation(repair=ok, integrity=IntegrityEvaluation(stale_memory_used_as_evidence=True))


def test_a3_successful_repair_without_guard_is_valid() -> None:
    ok = RepairEvaluation(repair_attempted=True, repair_successful=True, final_test_status=TestRunStatus.PASSED,
                          verified_dimensions=[Dim.DIRECTION_CORRECT, Dim.SUITE_PASSED, Dim.INTEGRITY_PRESERVED])
    harness_integrity = IntegrityEvaluation(test_weakening_detected=False, out_of_scope_edits=[])
    run = ExperimentRunRecord(
        provenance=prov(condition=ExperimentCondition.A3_NO_INTEGRITY_GUARD),
        integrity=IntegrityReport(guard_ran=False, guard_decision=GuardDecision.NOT_RUN),
        test_runs=[passed_run()],
        final_test_status=TestRunStatus.PASSED,
        outcome=RunOutcome(status=RunStatus.COMPLETED),
        evaluation=RunEvaluation(repair=ok, integrity=harness_integrity),
    )
    assert run.evaluation.repair.repair_successful is True
    assert run.integrity.guard_ran is False


def test_correct_decision_is_separate_from_repair_success() -> None:
    ev = RunEvaluation(
        decisions=[UnitDecisionScore(test_id=T_NORM, predicted_action=DecisionLabel.UPDATE_TEST,
                                     expected_action=DecisionLabel.UPDATE_TEST,
                                     gold_direction=RepairDirection.UPDATE_TEST, correct=True, hypothesis_correct=True)],
        repair=RepairEvaluation(repair_attempted=True, repair_successful=False, final_test_status=TestRunStatus.PASSED),
        integrity=IntegrityEvaluation(test_weakening_detected=True, mutants_killed_after=2, mutants_killed_reference=4),
        evidence=EvidenceEvaluation(unsupported_citation_ids=["ev-9"]))
    assert ev.decisions[0].correct and ev.repair and ev.repair.repair_successful is False


def test_unit_score_consistency() -> None:
    kw: dict[str, Any] = dict(test_id="t", gold_direction=RepairDirection.FIX_CODE, expected_action=DecisionLabel.FIX_CODE)
    with pytest.raises(ValidationError, match="correct must equal"):
        UnitDecisionScore(predicted_action=DecisionLabel.UPDATE_TEST, correct=True, **kw)
    with pytest.raises(ValidationError, match="cannot also be a wrong-side"):
        UnitDecisionScore(predicted_action=DecisionLabel.FIX_CODE, correct=True, wrong_side="masked_regression", **kw)
    assert UnitDecisionScore(predicted_action=DecisionLabel.UPDATE_TEST, correct=False,
                             wrong_side="masked_regression", **kw).wrong_side == "masked_regression"
    assert UnitDecisionScore(predicted_action=None, correct=False, **kw).predicted_action is None  # no decision reached


def test_evaluation_must_agree_with_run_record() -> None:
    score = UnitDecisionScore(test_id=T_NORM, predicted_action=DecisionLabel.FIX_CODE,
                              expected_action=DecisionLabel.UPDATE_TEST, gold_direction=RepairDirection.UPDATE_TEST,
                              correct=False)
    with pytest.raises(ValidationError, match="disagrees with the recorded prediction"):
        ExperimentRunRecord(provenance=prov(), predictions=[decision()], evaluation=RunEvaluation(decisions=[score]))
    with pytest.raises(ValidationError, match="repair.final_test_status disagrees"):
        ExperimentRunRecord(provenance=prov(), evaluation=RunEvaluation(
            repair=RepairEvaluation(final_test_status=TestRunStatus.PASSED)))


# patches -------------------------------------------------------------------- #

def test_rejected_patch_differs_from_unapplicable_patch() -> None:
    PatchRecord(patch_id="a", scope=PatchScope.TEST, application_status=PatchApplicationStatus.REJECTED_BY_GUARD)
    with pytest.raises(ValidationError, match="must record apply_error"):
        PatchRecord(patch_id="b", scope=PatchScope.SRC, application_status=PatchApplicationStatus.FAILED_TO_APPLY)
    with pytest.raises(ValidationError, match="must list the offending files"):
        PatchRecord(patch_id="c", scope=PatchScope.SRC, application_status=PatchApplicationStatus.BLOCKED_OUT_OF_SCOPE)
    assert PatchRecord(patch_id="d", scope=PatchScope.SRC, application_status=PatchApplicationStatus.BLOCKED_OUT_OF_SCOPE,
                       out_of_scope_files=["tests/conftest.py"]).out_of_scope_files


# 16. tokens ------------------------------------------------------------------ #

def test_unreported_tokens_are_not_zero() -> None:
    assert TokenUsage().input_tokens is None and TokenUsage() != TokenUsage(reported=True, input_tokens=0, output_tokens=0)
    with pytest.raises(ValidationError, match="leave token counts None"):
        TokenUsage(reported=False, input_tokens=0)
    with pytest.raises(ValidationError, match="at least one token count"):
        TokenUsage(reported=True)


# 18. tools / errors ---------------------------------------------------------- #

def test_injected_fault_and_recovery() -> None:
    failed = ToolEvent(event_id="te-1", step_index=3, tool=ToolName.APPLY_PATCH, status=ToolCallStatus.INJECTED_FAULT,
                       fault_kind=ToolFaultKind.PATCH_DOES_NOT_APPLY, error_message="patch does not apply", recovered=True)
    ok = ToolEvent(event_id="te-2", step_index=4, tool=ToolName.APPLY_PATCH)
    run = ExperimentRunRecord(provenance=prov(), tool_events=[failed, ok])
    assert [e.status for e in run.tool_events] == [ToolCallStatus.INJECTED_FAULT, ToolCallStatus.OK]


def test_tool_event_rules() -> None:
    with pytest.raises(ValidationError, match="fault_kind must be set exactly"):
        ToolEvent(event_id="t", step_index=0, tool=ToolName.RUN_TESTS, status=ToolCallStatus.INJECTED_FAULT, error_message="x")
    with pytest.raises(ValidationError, match="fault_kind must be set exactly"):
        ToolEvent(event_id="t", step_index=0, tool=ToolName.RUN_TESTS, fault_kind=ToolFaultKind.TIMEOUT)
    with pytest.raises(ValidationError, match="must record an error_message"):
        ToolEvent(event_id="t", step_index=0, tool=ToolName.RUN_TESTS, status=ToolCallStatus.TIMEOUT)
    with pytest.raises(ValidationError, match="failed calls only"):
        ToolEvent(event_id="t", step_index=0, tool=ToolName.RUN_TESTS, recovered=True)


def test_provider_errors_not_counted_against_agent() -> None:
    assert ErrorRecord(error_id="e", category=ErrorCategory.PROVIDER_OUTAGE, message="503", counted_against_agent=False)
    with pytest.raises(ValidationError, match="must not be counted against the agent"):
        ErrorRecord(error_id="e", category=ErrorCategory.RATE_LIMIT, message="429")
    with pytest.raises(ValidationError, match="both recovered and fatal"):
        ErrorRecord(error_id="e", category=ErrorCategory.TOOL_ERROR, message="x", recovered=True, fatal=True)


# outcomes -------------------------------------------------------------------- #

def test_escalation_is_not_an_execution_failure() -> None:
    esc = EscalationSummary(trigger="insufficient_evidence", failing_test_ids=[T_NORM],
                            evidence_missing=["commit message"], competing_hypotheses=["outdated test", "regression"],
                            summary="No evidence either way.")
    run = ExperimentRunRecord(provenance=prov(), outcome=RunOutcome(status=RunStatus.ESCALATED, escalation=esc),
                              predictions=[decision(action=DecisionLabel.ESCALATE, cited_evidence_ids=[])])
    assert run.outcome.status is RunStatus.ESCALATED
    with pytest.raises(ValidationError, match="requires an EscalationSummary"):
        RunOutcome(status=RunStatus.ESCALATED)
    with pytest.raises(ValidationError, match="every prediction to be ESCALATE"):
        ExperimentRunRecord(provenance=prov(), predictions=[decision()], outcome=RunOutcome(status=RunStatus.ESCALATED, escalation=esc))
    with pytest.raises(ValidationError, match="requires at least one ESCALATE prediction"):
        ExperimentRunRecord(provenance=prov(), predictions=[decision()], outcome=RunOutcome(status=RunStatus.COMPLETED, escalation=esc))


@pytest.mark.parametrize("status", [RunStatus.INCOMPLETE, RunStatus.BUDGET_EXHAUSTED, RunStatus.TOOL_FAILURE,
                                    RunStatus.API_FAILURE, RunStatus.ORCHESTRATOR_ERROR])
def test_failed_and_incomplete_runs_are_preserved(status: RunStatus) -> None:
    run = ExperimentRunRecord(provenance=prov(), outcome=RunOutcome(status=status))
    assert ExperimentRunRecord.model_validate_json(run.to_jsonl()).outcome.status is status


# memory ---------------------------------------------------------------------- #

def test_stale_memory_cannot_be_used_as_evidence() -> None:
    with pytest.raises(ValidationError, match="probe passed"):
        MemoryReadEvent(record_id="m", probe_status=ProbeStatus.FAILED, used_as_evidence=True)
    with pytest.raises(ValidationError, match="probe passed"):
        MemoryReadEvent(record_id="m", used_as_evidence=True)  # probe never run
    assert MemoryReadEvent(record_id="m", probe_status=ProbeStatus.FAILED, blocked_reason="probe_failed")


def test_memory_write_needs_approval_and_the_guard() -> None:
    with pytest.raises(ValidationError, match="only after maintainer approval"):
        MemoryWriteAttempt(record_id="m", written=True)
    with pytest.raises(ValidationError, match="only the Integrity Guard"):
        MemoryWriteAttempt(record_id="m", approved=True, written=True, written_by=AgentRole.REPAIRER)
    with pytest.raises(ValidationError, match="blocked_reason"):
        MemoryWriteAttempt(record_id="m")
    assert MemoryWriteAttempt(record_id="m", blocked_reason="not_approved").written is False


def test_schemas_compatibility_alias() -> None:
    from testorcode.schemas import schemas as compat_schemas
    assert "testorcode.schemas.episode_schemas" in (compat_schemas.__doc__ or "")
    assert hasattr(compat_schemas, "Episode")
    assert hasattr(compat_schemas, "SCHEMA_VERSION")
