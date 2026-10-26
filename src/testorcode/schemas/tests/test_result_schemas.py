"""Tests for ``testorcode.result_schemas``.

All runs, token counts and tool traces below are synthetic.  No real experiment
has been executed to produce them.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from testorcode.result_schemas import (
    AgentRole,
    BudgetUsage,
    ConfidenceReport,
    DecisionRationale,
    ErrorCategory,
    ErrorRecord,
    EscalationSummary,
    EvidenceEvaluation,
    ExperimentCondition,
    ExperimentRunRecord,
    GuardDecision,
    Hypothesis,
    IntegrityCheck,
    IntegrityEvaluation,
    IntegrityReport,
    CheckStatus,
    MemoryReadEvent,
    MemoryWriteAttempt,
    MutantOutcome,
    MutationResult,
    OperationType,
    PatchApplicationStatus,
    PatchRecord,
    PredictedDecision,
    ProbeRun,
    ProbeStatus,
    ProhibitedEditCheck,
    RepairEvaluation,
    RepairVerificationDimension,
    RetrievalMethod,
    RetrievedEvidence,
    RunEvaluation,
    RunOutcome,
    RunProvenance,
    RunStatus,
    StopReason,
    SupportStatus,
    TestRunPhase,
    TestRunResult,
    TestRunStatus,
    TestWeakeningFindings,
    TokenUsage,
    ToolCallStatus,
    ToolCallSummary,
    ToolEvent,
    TrajectoryStep,
    UnitDecisionScore,
    METRIC_SUPPORT,
)
from testorcode.episode_schemas import (
    DecisionLabel,
    Episode,
    EvidenceSourceType,
    HypothesisKind,
    PatchScope,
    RepairDirection,
    ToolFaultKind,
    ToolName,
)

MODEL_ID = "qwen/qwen3-coder-480b-a35b-instruct"


def provenance(
    *,
    run_id: str = "run-1",
    episode_id: str = "ep-s1",
    condition: ExperimentCondition = ExperimentCondition.FULL,
    seed: int | None = 0,
) -> RunProvenance:
    return RunProvenance(
        run_id=run_id,
        episode_id=episode_id,
        condition=condition,
        seed=seed,
        model_id=MODEL_ID,
        provider="nvidia-nim",
        dataset_id="pilot",
        dataset_version="2026-10-25",
        split="pilot",
        code_revision="deadbee",
        prompt_template_version="prompts-v3",
        agent_config_version="full-v1",
        retrieval_config_version="bm25-k8-v1",
        index_version="bm25-v1",
        evaluation_protocol_version="eval-v1",
    )


def evidence_item() -> RetrievedEvidence:
    return RetrievedEvidence(
        evidence_id="ev-commit-msg-1",
        source_type=EvidenceSourceType.COMMIT_MESSAGE,
        source_ref="a1b2c3d",
        retrieval_query="normalise capitalise intent",
        retrieval_method=RetrievalMethod.BM25,
        rank=1,
        score=12.4,
        used_by_agent=True,
        cited=True,
        support=SupportStatus.SUPPORTS,
    )


def decision(
    action: DecisionLabel = DecisionLabel.UPDATE_TEST,
    test_id: str = "tests/test_synthlib.py::test_normalise",
    decision_id: str = "d-1",
) -> PredictedDecision:
    return PredictedDecision(
        decision_id=decision_id,
        unit_kind="test",
        test_ids=[test_id],
        group_id="g-normalise",
        function_qualname="synthlib.core.normalise",
        predicted_action=action,
        predicted_direction=(
            None if action is DecisionLabel.ESCALATE else RepairDirection(action.value)
        ),
        hypotheses=[
            Hypothesis(
                hypothesis_id="h-1",
                kind=HypothesisKind.OUTDATED_TEST,
                statement="normalise() was intentionally changed to capitalise.",
                predicted_behaviour="normalise('ab') == 'Ab'",
                supporting_evidence_ids=["ev-commit-msg-1"],
                checked=True,
                check_outcome="confirmed",
                selected=True,
            )
        ],
        selected_hypothesis_id="h-1",
        cited_evidence_ids=["ev-commit-msg-1"],
        rationale=DecisionRationale(
            summary="Commit message states the capitalisation is intended.",
            escalation_reason=(
                "No positive evidence found" if action is DecisionLabel.ESCALATE else None
            ),
        ),
    )


# --------------------------------------------------------------------------- #
# 10. Serialisation round-trip
# --------------------------------------------------------------------------- #


def test_run_record_json_round_trip() -> None:
    run = ExperimentRunRecord(
        provenance=provenance(),
        predictions=[decision()],
        retrieved_evidence=[evidence_item()],
    )
    line = run.to_jsonl()
    assert "\n" not in line
    restored = ExperimentRunRecord.model_validate_json(line)
    assert restored == run
    assert json.loads(line)["provenance"]["condition"] == "FULL"


# --------------------------------------------------------------------------- #
# 15. One run records every mandatory field
# --------------------------------------------------------------------------- #


def full_run_record() -> ExperimentRunRecord:
    post = TestRunResult(
        run_id="tr-2",
        phase=TestRunPhase.POST_REPAIR,
        status=TestRunStatus.PASSED,
        passed=40,
        failed=0,
    )
    return ExperimentRunRecord(
        provenance=provenance(),
        predictions=[decision()],
        retrieved_evidence=[evidence_item()],
        trajectory=[
            TrajectoryStep(
                step_id="s-1",
                index=0,
                operation=OperationType.RETRIEVAL,
                actor=AgentRole.INTENT_ANALYST,
                parameters={"query": "normalise capitalise", "k": 8},
                evidence_ids_returned=["ev-commit-msg-1"],
                stopping_reason=StopReason.SUFFICIENT_EVIDENCE,
            ),
            TrajectoryStep(
                step_id="s-2",
                index=1,
                operation=OperationType.INTEGRITY_VERIFICATION,
                actor=AgentRole.INTEGRITY_GUARD,
                changed_decision=False,
                stopping_reason=StopReason.COMPLETED,
            ),
        ],
        patches=[
            PatchRecord(
                patch_id="p-1",
                scope=PatchScope.TEST,
                declared_for_decision_id="d-1",
                target_files=["tests/test_synthlib.py"],
                unified_diff="@@\n-    assert normalise('ab') == 'ab'\n+    assert normalise('ab') == 'Ab'",
                application_status=PatchApplicationStatus.APPLIED,
            )
        ],
        test_runs=[
            TestRunResult(
                run_id="tr-1",
                phase=TestRunPhase.PRE_REPAIR,
                status=TestRunStatus.FAILED,
                passed=39,
                failed=1,
                failing_test_ids=["tests/test_synthlib.py::test_normalise"],
            ),
            post,
        ],
        final_test_run=post,
        integrity=IntegrityReport(
            guard_ran=True,
            guard_decision=GuardDecision.APPROVE,
            checks=[
                IntegrityCheck(name="assertion_count", status=CheckStatus.PASSED),
                IntegrityCheck(name="scope", status=CheckStatus.PASSED),
            ],
            weakening=TestWeakeningFindings(
                checks_run=True,
                assertions_before=3,
                assertions_after=3,
                removed_assertions=0,
                added_skip_markers=0,
                weakening_detected=False,
            ),
            prohibited_edits=ProhibitedEditCheck(
                checks_run=True,
                files_modified=["tests/test_synthlib.py"],
                test_files_modified=["tests/test_synthlib.py"],
                ci_config_touched=False,
                dependency_files_touched=False,
                conftest_touched=False,
            ),
            mutation=MutationResult(
                status=TestRunStatus.PASSED,
                function_qualname="synthlib.core.normalise",
                catalogue_id="cat-normalise",
                mutants_total=4,
                killed_by_reference_tests=3,
                killed_by_repaired_tests=3,
                per_mutant=[
                    MutantOutcome(
                        mutant_id="mut-1",
                        killed_by_reference=True,
                        killed_after_repair=True,
                        run_status=TestRunStatus.PASSED,
                    )
                ],
            ),
            evidence_requirement_met=True,
        ),
        memory_reads=[
            MemoryReadEvent(
                record_id="mem-1",
                function_qualname="synthlib.core.normalise",
                probe=ProbeRun(probe_id="probe-mem-1", status=ProbeStatus.PASSED),
                used_as_evidence=True,
            )
        ],
        memory_writes=[
            MemoryWriteAttempt(
                record_id="mem-2",
                function_qualname="synthlib.core.normalise",
                behaviour_signature="normalise() capitalises",
                direction=RepairDirection.UPDATE_TEST,
                probe_id="probe-mem-2",
                evidence_ids=["ev-commit-msg-1"],
                approved=True,
                approval_source="simulated_maintainer",
                written=True,
            )
        ],
        tool_events=[
            ToolEvent(
                event_id="te-1",
                step_index=0,
                tool=ToolName.SEARCH_REPO,
                called_by=AgentRole.INTENT_ANALYST,
                arguments={"query": "normalise"},
                status=ToolCallStatus.OK,
                duration_ms=35.0,
            )
        ],
        tool_summary=ToolCallSummary(
            total_calls=7, calls_by_tool={"search_repo": 2, "run_tests": 3}
        ),
        usage=TokenUsage(
            reported=True, input_tokens=18_000, output_tokens=2_100, llm_calls=12
        ),
        budget=BudgetUsage(
            max_tool_calls=40, tool_calls_used=7, max_revisions=1, revisions_used=0
        ),
        errors=[
            ErrorRecord(
                error_id="e-1",
                category=ErrorCategory.RATE_LIMIT,
                message="429 from provider",
                operation="llm_call",
                retries=2,
                recovered=True,
                counted_against_agent=False,
            )
        ],
        outcome=RunOutcome(
            status=RunStatus.COMPLETED,
            final_action=DecisionLabel.UPDATE_TEST,
            terminated_by=StopReason.COMPLETED,
        ),
    )


def test_single_run_records_all_mandatory_fields() -> None:
    run = full_run_record()
    data = json.loads(run.to_jsonl())
    prov = data["provenance"]
    assert prov["episode_id"] == "ep-s1"
    assert prov["condition"] == "FULL"
    assert prov["seed"] == 0
    assert prov["model_id"] == MODEL_ID
    assert data["predictions"][0]["predicted_action"] == "UPDATE_TEST"
    assert data["retrieved_evidence"][0]["evidence_id"] == "ev-commit-msg-1"
    assert data["patches"][0]["application_status"] == "applied"
    assert data["final_test_run"]["status"] == "passed"
    assert data["integrity"]["guard_decision"] == "approve"
    assert data["usage"]["input_tokens"] == 18_000
    assert data["tool_summary"]["total_calls"] == 7
    assert data["errors"][0]["category"] == "rate_limit"
    assert run.final_test_status is TestRunStatus.PASSED


# --------------------------------------------------------------------------- #
# 11. Test pass status vs. repair success
# --------------------------------------------------------------------------- #


def test_green_suite_alone_does_not_establish_repair_success() -> None:
    with pytest.raises(ValidationError, match="passing suite alone"):
        RepairEvaluation(
            repair_attempted=True,
            repair_successful=True,
            verified_dimensions=[RepairVerificationDimension.SUITE_PASSED],
            final_test_status=TestRunStatus.PASSED,
        )


def test_repair_success_requires_all_three_dimensions() -> None:
    ok = RepairEvaluation(
        repair_attempted=True,
        repair_successful=True,
        verified_dimensions=[
            RepairVerificationDimension.DIRECTION_CORRECT,
            RepairVerificationDimension.SUITE_PASSED,
            RepairVerificationDimension.INTEGRITY_PRESERVED,
            RepairVerificationDimension.SCOPE_RESPECTED,
        ],
        final_test_status=TestRunStatus.PASSED,
    )
    assert ok.repair_successful


def test_correct_decision_does_not_imply_successful_repair() -> None:
    evaluation = RunEvaluation(
        decisions=[
            UnitDecisionScore(
                test_id="tests/test_synthlib.py::test_normalise",
                predicted_action=DecisionLabel.UPDATE_TEST,
                expected_action=DecisionLabel.UPDATE_TEST,
                gold_direction=RepairDirection.UPDATE_TEST,
                correct=True,
            )
        ],
        repair=RepairEvaluation(
            repair_attempted=True,
            repair_successful=False,
            final_test_status=TestRunStatus.PASSED,
            reason="tests pass but mutant-kill retention dropped",
        ),
        integrity=IntegrityEvaluation(
            test_weakening_detected=True,
            mutant_kill_retention=0.5,
            mutants_killed_after=2,
            mutants_killed_reference=4,
        ),
    )
    assert evaluation.decisions[0].correct
    assert evaluation.repair is not None and evaluation.repair.repair_successful is False
    assert evaluation.integrity is not None
    assert evaluation.integrity.test_weakening_detected


# --------------------------------------------------------------------------- #
# 12. Mutation and integrity outcomes
# --------------------------------------------------------------------------- #


def test_mutation_results_are_recorded_with_counts() -> None:
    result = MutationResult(
        status=TestRunStatus.PASSED,
        mutants_total=5,
        killed_by_reference_tests=4,
        killed_by_repaired_tests=2,
        per_mutant=[
            MutantOutcome(mutant_id="m1", killed_by_reference=True, killed_after_repair=False),
        ],
    )
    assert result.killed_by_repaired_tests == 2
    with pytest.raises(ValidationError, match="exceeds mutants_total"):
        MutationResult(
            status=TestRunStatus.PASSED, mutants_total=2, killed_by_repaired_tests=3
        )


def test_unrun_mutation_is_not_zero_kills() -> None:
    result = MutationResult()
    assert result.status is TestRunStatus.NOT_RUN
    assert result.killed_by_repaired_tests is None
    with pytest.raises(ValidationError, match="NOT_RUN mutation result"):
        MutationResult(killed_by_repaired_tests=0)


def test_unchecked_weakening_is_not_no_weakening() -> None:
    findings = TestWeakeningFindings()
    assert findings.checks_run is False
    assert findings.weakening_detected is None
    with pytest.raises(ValidationError, match="must stay None"):
        TestWeakeningFindings(checks_run=False, weakening_detected=False)


def test_guard_rejection_requires_a_reason() -> None:
    with pytest.raises(ValidationError, match="rejection must record"):
        IntegrityReport(guard_ran=True, guard_decision=GuardDecision.REJECT)
    report = IntegrityReport(
        guard_ran=True,
        guard_decision=GuardDecision.REJECT,
        rejection_reasons=["assertion removed"],
        revision_requested=True,
    )
    assert report.revision_requested


def test_a3_ablation_leaves_integrity_not_run() -> None:
    report = IntegrityReport()
    assert report.guard_decision is GuardDecision.NOT_RUN
    assert report.mutation.status is TestRunStatus.NOT_RUN
    assert report.weakening.weakening_detected is None


# --------------------------------------------------------------------------- #
# 16. Unavailable token usage and tests that were not run
# --------------------------------------------------------------------------- #


def test_unavailable_token_usage_is_not_zero() -> None:
    unavailable = TokenUsage(reported=False)
    assert unavailable.input_tokens is None
    zero = TokenUsage(reported=True, input_tokens=0, output_tokens=0, total_tokens=0)
    assert zero.total_tokens == 0
    assert unavailable != zero
    with pytest.raises(ValidationError, match="returned no usage"):
        TokenUsage(reported=False, input_tokens=0)
    with pytest.raises(ValidationError, match="requires at least one token count"):
        TokenUsage(reported=True)


def test_tests_not_run_is_not_tests_passed() -> None:
    not_run = TestRunResult(run_id="tr", phase=TestRunPhase.POST_REPAIR)
    assert not_run.status is TestRunStatus.NOT_RUN
    assert not_run.passed is None
    with pytest.raises(ValidationError, match="must not carry counts"):
        TestRunResult(
            run_id="tr", phase=TestRunPhase.POST_REPAIR, status=TestRunStatus.NOT_RUN, failed=0
        )
    run = ExperimentRunRecord(provenance=provenance())
    assert run.final_test_run is None
    assert run.final_test_status is TestRunStatus.NOT_RUN


def test_missing_seed_differs_from_seed_zero() -> None:
    unseeded = provenance(run_id="r-a", seed=None)
    seeded = provenance(run_id="r-b", seed=0)
    assert unseeded.seed is None and seeded.seed == 0
    assert json.loads(unseeded.model_dump_json())["seed"] is None
    with pytest.raises(ValidationError):
        RunProvenance(  # seed is required and must be given explicitly
            run_id="r", episode_id="e", condition=ExperimentCondition.FULL,
            model_id=MODEL_ID,
        )


def test_model_id_must_not_be_a_bare_provider_name() -> None:
    with pytest.raises(ValidationError, match="names a provider"):
        RunProvenance(
            run_id="r",
            episode_id="e",
            condition=ExperimentCondition.FULL,
            seed=1,
            model_id="mistral",
        )


# --------------------------------------------------------------------------- #
# 17. Several runs of the same episode across conditions
# --------------------------------------------------------------------------- #


def test_same_episode_across_conditions_and_seeds() -> None:
    runs = [
        ExperimentRunRecord(
            provenance=provenance(
                run_id=f"run-{condition.value}-{seed}", condition=condition, seed=seed
            ),
            predictions=[decision()],
        )
        for condition in (
            ExperimentCondition.FULL,
            ExperimentCondition.B1,
            ExperimentCondition.A1_NO_INTENT_RETRIEVAL,
            ExperimentCondition.A2_NO_MEMORY,
            ExperimentCondition.A3_NO_INTEGRITY_GUARD,
        )
        for seed in (0, 1)
    ]
    assert {r.episode_id for r in runs} == {"ep-s1"}
    assert len({r.provenance.run_id for r in runs}) == 10
    grouped: dict[ExperimentCondition, int] = {}
    for run in runs:
        grouped[run.condition] = grouped.get(run.condition, 0) + 1
    assert set(grouped.values()) == {2}


def test_incomplete_runs_are_representable() -> None:
    run = ExperimentRunRecord(
        provenance=provenance(run_id="run-x"),
        outcome=RunOutcome(
            status=RunStatus.BUDGET_EXHAUSTED,
            terminated_by=StopReason.BUDGET_EXHAUSTED,
        ),
        budget=BudgetUsage(max_tool_calls=40, tool_calls_used=40, budget_exhausted=True),
    )
    assert run.outcome.escalated is False
    assert run.final_test_status is TestRunStatus.NOT_RUN


# --------------------------------------------------------------------------- #
# 18. Retries and recovered tool failures
# --------------------------------------------------------------------------- #


def test_injected_fault_retry_and_recovery() -> None:
    failed = ToolEvent(
        event_id="te-1",
        step_index=3,
        tool=ToolName.APPLY_PATCH,
        status=ToolCallStatus.INJECTED_FAULT,
        injected_fault=True,
        fault_kind=ToolFaultKind.PATCH_DOES_NOT_APPLY,
        fault_id="f-1",
        error_type=ErrorCategory.INJECTED_TOOL_FAULT,
        error_message="patch does not apply",
        recovered=True,
    )
    retry = ToolEvent(
        event_id="te-2",
        step_index=4,
        tool=ToolName.APPLY_PATCH,
        status=ToolCallStatus.OK,
        retry_index=1,
        retry_of_event_id="te-1",
    )
    summary = ToolCallSummary(
        total_calls=2,
        calls_by_tool={"apply_patch": 2},
        injected_faults=1,
        retries=1,
        recovered_failures=1,
    )
    run = ExperimentRunRecord(
        provenance=provenance(run_id="run-fault"),
        tool_events=[failed, retry],
        tool_summary=summary,
    )
    assert run.tool_events[0].recovered is True
    assert run.tool_events[1].retry_of_event_id == "te-1"
    assert run.tool_summary.recovered_failures == 1


def test_failed_tool_call_requires_an_error_message() -> None:
    with pytest.raises(ValidationError, match="must record an error_message"):
        ToolEvent(
            event_id="te",
            step_index=0,
            tool=ToolName.RUN_TESTS,
            status=ToolCallStatus.TIMEOUT,
        )


def test_injected_fault_is_distinguished_from_api_error() -> None:
    with pytest.raises(ValidationError, match="logged separately"):
        ToolEvent(
            event_id="te",
            step_index=0,
            tool=ToolName.RUN_TESTS,
            status=ToolCallStatus.ERROR,
            injected_fault=True,
            fault_kind=ToolFaultKind.TIMEOUT,
            error_type=ErrorCategory.API_ERROR,
            error_message="boom",
        )
    api_error = ErrorRecord(
        error_id="e",
        category=ErrorCategory.PROVIDER_OUTAGE,
        message="503",
        counted_against_agent=False,
    )
    assert api_error.counted_against_agent is False
    with pytest.raises(ValidationError, match="must not be counted against the agent"):
        ErrorRecord(error_id="e", category=ErrorCategory.RATE_LIMIT, message="429")


# --------------------------------------------------------------------------- #
# Required distinctions
# --------------------------------------------------------------------------- #


def test_rejected_patch_differs_from_unapplicable_patch() -> None:
    rejected = PatchRecord(
        patch_id="p",
        scope=PatchScope.TEST,
        application_status=PatchApplicationStatus.REJECTED_BY_GUARD,
    )
    unapplicable = PatchRecord(
        patch_id="p2",
        scope=PatchScope.TEST,
        application_status=PatchApplicationStatus.FAILED_TO_APPLY,
        apply_error="hunk #1 failed",
    )
    assert rejected.application_status is not unapplicable.application_status
    with pytest.raises(ValidationError, match="must record apply_error"):
        PatchRecord(
            patch_id="p3",
            scope=PatchScope.SRC,
            application_status=PatchApplicationStatus.FAILED_TO_APPLY,
        )


def test_out_of_scope_patch_must_list_offending_files() -> None:
    with pytest.raises(ValidationError, match="must list the offending files"):
        PatchRecord(
            patch_id="p",
            scope=PatchScope.SRC,
            application_status=PatchApplicationStatus.BLOCKED_OUT_OF_SCOPE,
        )
    blocked = PatchRecord(
        patch_id="p",
        scope=PatchScope.SRC,
        application_status=PatchApplicationStatus.BLOCKED_OUT_OF_SCOPE,
        out_of_scope_files=["tests/conftest.py"],
    )
    assert blocked.out_of_scope_files == ["tests/conftest.py"]


def test_escalation_is_not_a_tool_failure() -> None:
    escalation = EscalationSummary(
        trigger="insufficient_evidence",
        failing_test_ids=["tests/test_synthlib.py::test_normalise"],
        evidence_missing=["no commit message", "no changelog entry"],
        summary="Neither side has positive evidence.",
    )
    ok = RunOutcome(
        status=RunStatus.ESCALATED,
        final_action=DecisionLabel.ESCALATE,
        escalated=True,
        escalation=escalation,
        terminated_by=StopReason.ESCALATION,
    )
    assert ok.escalated
    with pytest.raises(ValidationError, match="not an escalation"):
        RunOutcome(
            status=RunStatus.TOOL_FAILURE,
            escalated=True,
            escalation=escalation,
            terminated_by=StopReason.TOOL_FAILURE,
        )


def test_escalate_decision_requires_an_escalation_reason() -> None:
    with pytest.raises(ValidationError, match="escalation_reason"):
        PredictedDecision(
            decision_id="d",
            test_ids=["t"],
            predicted_action=DecisionLabel.ESCALATE,
            rationale=DecisionRationale(summary="unclear"),
        )


def test_stale_memory_cannot_be_used_as_evidence() -> None:
    with pytest.raises(ValidationError, match="probe passed"):
        MemoryReadEvent(
            record_id="mem-1",
            probe=ProbeRun(probe_id="p", status=ProbeStatus.FAILED),
            used_as_evidence=True,
        )
    blocked = MemoryReadEvent(
        record_id="mem-1",
        probe=ProbeRun(probe_id="p", status=ProbeStatus.FAILED),
        used_as_evidence=False,
        blocked_reason="probe_failed",
    )
    assert blocked.blocked_reason == "probe_failed"


def test_memory_write_requires_approval_and_the_guard() -> None:
    with pytest.raises(ValidationError, match="after maintainer approval"):
        MemoryWriteAttempt(record_id="m", written=True, approved=False)
    with pytest.raises(ValidationError, match="only the Integrity Guard"):
        MemoryWriteAttempt(
            record_id="m",
            written=True,
            approved=True,
            approval_source="simulated_maintainer",
            written_by=AgentRole.REPAIRER,
        )
    blocked = MemoryWriteAttempt(record_id="m", blocked_reason="memory_disabled")
    assert blocked.written is False


def test_confidence_is_not_a_calibrated_probability() -> None:
    report = ConfidenceReport(value=0.8, source="model_self_report")
    assert report.calibrated is False
    with pytest.raises(ValidationError, match="calibration_method"):
        ConfidenceReport(value=0.8, source="model_self_report", calibrated=True)
    with pytest.raises(ValidationError, match="explicit source"):
        ConfidenceReport(value=0.8)


def test_evidence_score_absence_is_not_zero() -> None:
    no_score = RetrievedEvidence(
        evidence_id="ev",
        source_type=EvidenceSourceType.DOCUMENTATION,
        source_ref="docs/api.md",
        retrieval_method=RetrievalMethod.BM25,
    )
    assert no_score.score is None
    assert no_score.support is SupportStatus.UNKNOWN
    assert no_score != no_score.model_copy(update={"score": 0.0})


def test_citation_must_be_marked_used() -> None:
    with pytest.raises(ValidationError, match="must be marked used_by_agent"):
        RetrievedEvidence(
            evidence_id="ev",
            source_type=EvidenceSourceType.CHANGELOG,
            source_ref="CHANGELOG.md",
            cited=True,
        )


def test_run_rejects_citations_of_unretrieved_evidence() -> None:
    bad = decision()
    with pytest.raises(ValidationError, match="cites evidence that is not in"):
        ExperimentRunRecord(
            provenance=provenance(),
            predictions=[bad],
            retrieved_evidence=[
                RetrievedEvidence(
                    evidence_id="ev-other",
                    source_type=EvidenceSourceType.DOCUMENTATION,
                    source_ref="docs/api.md",
                )
            ],
        )


# --------------------------------------------------------------------------- #
# Mixed episode: several predictions with different actions
# --------------------------------------------------------------------------- #


def test_mixed_episode_predictions_expand_per_test(s3_episode: Episode) -> None:
    tests = [t.test_id for t in s3_episode.agent_input.ci_event.failing_tests]
    run = ExperimentRunRecord(
        provenance=provenance(run_id="run-s3", episode_id="ep-s3"),
        predictions=[
            decision(DecisionLabel.UPDATE_TEST, tests[0], "d-1"),
            decision(DecisionLabel.FIX_CODE, tests[1], "d-2"),
        ],
    )
    predicted = run.predicted_action_by_test()
    assert predicted == {
        tests[0]: DecisionLabel.UPDATE_TEST,
        tests[1]: DecisionLabel.FIX_CODE,
    }
    gold = s3_episode.gold_actions()
    scores = [
        UnitDecisionScore(
            test_id=test_id,
            predicted_action=predicted[test_id],
            expected_action=gold[test_id],
            gold_direction=RepairDirection(gold[test_id].value),
            correct=predicted[test_id] is gold[test_id],
        )
        for test_id in tests
    ]
    assert all(score.correct for score in scores)


def test_group_level_decision_expands_to_each_test() -> None:
    group_decision = PredictedDecision(
        decision_id="d-group",
        unit_kind="group",
        group_id="g-normalise",
        test_ids=["tests/t.py::a", "tests/t.py::b"],
        predicted_action=DecisionLabel.FIX_CODE,
    )
    run = ExperimentRunRecord(
        provenance=provenance(run_id="run-g"), predictions=[group_decision]
    )
    assert set(run.predicted_action_by_test()) == {"tests/t.py::a", "tests/t.py::b"}


def test_test_level_decision_must_name_exactly_one_test() -> None:
    with pytest.raises(ValidationError, match="exactly one test_id"):
        PredictedDecision(
            decision_id="d",
            unit_kind="test",
            test_ids=["a", "b"],
            predicted_action=DecisionLabel.FIX_CODE,
        )


# --------------------------------------------------------------------------- #
# Evaluation dimensions stay separate
# --------------------------------------------------------------------------- #


def test_evaluation_dimensions_are_independent() -> None:
    evaluation = RunEvaluation(
        decisions=[
            UnitDecisionScore(
                test_id="t",
                predicted_action=DecisionLabel.UPDATE_TEST,
                expected_action=DecisionLabel.FIX_CODE,
                gold_direction=RepairDirection.FIX_CODE,
                correct=False,
                wrong_side="masked_regression",
            )
        ],
        evidence=EvidenceEvaluation(
            gold_evidence_ids=["ev-1"],
            retrieved_ranked_ids=["ev-9", "ev-1"],
            cited_evidence_ids=["ev-9"],
            unsupported_citation_ids=["ev-9"],
            stale_evidence_used_ids=["ev-9"],
            stale_evidence_caused_wrong_hypothesis=True,
        ),
        integrity=IntegrityEvaluation(memory_over_generalised=True),
    )
    assert evaluation.decisions[0].wrong_side == "masked_regression"
    assert evaluation.evidence is not None
    assert evaluation.evidence.retrieved_ranked_ids.index("ev-1") == 1


def test_correct_decision_cannot_be_a_wrong_side_edit() -> None:
    with pytest.raises(ValidationError, match="cannot also be a wrong-side edit"):
        UnitDecisionScore(
            test_id="t",
            predicted_action=DecisionLabel.FIX_CODE,
            expected_action=DecisionLabel.FIX_CODE,
            gold_direction=RepairDirection.FIX_CODE,
            correct=True,
            wrong_side="masked_regression",
        )


def test_evaluation_must_agree_with_recorded_predictions() -> None:
    run = full_run_record()
    with pytest.raises(ValidationError, match="disagrees with the recorded prediction"):
        run.evaluation = RunEvaluation(
            decisions=[
                UnitDecisionScore(
                    test_id="tests/test_synthlib.py::test_normalise",
                    predicted_action=DecisionLabel.FIX_CODE,
                    expected_action=DecisionLabel.UPDATE_TEST,
                    gold_direction=RepairDirection.UPDATE_TEST,
                    correct=False,
                )
            ]
        )


def test_metric_support_map_covers_the_stated_metrics() -> None:
    expected = {
        "Per-test direction accuracy / macro-F1",
        "Correct-repair rate",
        "Masked-regression rate",
        "Reverted-intent rate",
        "Test-weakening rate",
        "Mutant-kill retention",
        "Out-of-scope-edit rate",
        "Escalation precision / recall",
        "Gold-evidence Recall@k and citation faithfulness",
        "Stale-evidence error rate",
        "Memory over-generalisation on decoys",
        "Tool-call accuracy and error recovery",
        "Tool calls and tokens per episode",
    }
    assert expected <= set(METRIC_SUPPORT)


def test_trajectory_records_ordered_steps_with_stop_reasons() -> None:
    run = full_run_record()
    assert [s.index for s in run.trajectory] == [0, 1]
    assert run.trajectory[0].stopping_reason is StopReason.SUFFICIENT_EVIDENCE
    with pytest.raises(ValidationError, match="must record an error_type"):
        TrajectoryStep(
            step_id="s", index=0, operation=OperationType.RETRIEVAL, status="failed"
        )
