"""Shared contracts: construction, validation and JSON round-trips (synthetic data only)."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from testorcode.errors import (
    BudgetExceededError,
    LLMProviderOutageError,
    LLMRateLimitError,
    PatchError,
    ToolTimeoutError,
)
from testorcode.schemas.contracts import (
    EpisodeContext,
    EscalationRequest,
    FailureGroupDetail,
    GuardResult,
    Hypothesis,
    IntentAnalysis,
    MemoryApproval,
    MemoryReadResult,
    MemoryRecordCheck,
    PatchProposal,
    PatchValidation,
    PytestExecution,
    TriageResult,
    unknown_evidence_ids,
)
from testorcode.schemas.episode_schemas import (
    DecisionLabel,
    ExecutableProbe,
    FailingTest,
    FailureGroup,
    HypothesisKind,
    MemoryRecord,
    PatchScope,
    RepairDirection,
)
from testorcode.schemas.result_schemas import (
    ErrorCategory,
    EscalationSummary,
    GuardDecision,
    IntegrityReport,
    ProbeStatus,
    RunStatus,
    TestRunPhase,
    TestRunStatus,
)

SHA = "a1b2c3d4e5f60718293a4b5c6d7e8f9012345678"
T1 = "tests/test_x.py::test_a"


def roundtrip(model):
    return type(model).model_validate_json(model.model_dump_json())


def passing_run() -> PytestExecution:
    return PytestExecution(
        selectors=(T1,), command=("pytest", T1), status=TestRunStatus.PASSED, exit_code=0, passed=1, failed=0
    )


def analysis(**kw) -> IntentAnalysis:
    base = {
        "group_id": "g1",
        "hypothesis": Hypothesis(kind=HypothesisKind.OUTDATED_TEST, explanation="changelog says so"),
        "cited_evidence_ids": ("ev-1",),
        "confidence": 0.8,
        "recommended_decision": DecisionLabel.UPDATE_TEST,
    }
    return IntentAnalysis(**{**base, **kw})


def proposal(**kw) -> PatchProposal:
    base = {
        "group_id": "g1",
        "decision": RepairDirection.UPDATE_TEST,
        "scope": PatchScope.TEST,
        "target_files": ("tests/test_x.py",),
        "unified_diff": "--- a/tests/test_x.py\n+++ b/tests/test_x.py\n",
        "rationale": "behaviour changed on purpose",
        "evidence_ids": ("ev-1",),
    }
    return PatchProposal(**{**base, **kw})


def guard(decision: GuardDecision, **kw) -> GuardResult:
    integrity = IntegrityReport(
        guard_ran=True,
        guard_decision=decision,
        rejection_reasons=["weakened"] if decision is GuardDecision.REJECT else [],
    )
    base = {"decision": decision, "integrity": integrity, "reasons": ("r",), "post_repair_run": passing_run()}
    return GuardResult(**{**base, **kw})


def test_episode_context_roundtrip() -> None:
    from testorcode.schemas.episode_schemas import RepoRef

    ctx = EpisodeContext(episode_id="e", repo=RepoRef(repo_id="r"), worktree_path=Path("/w"), commit_sha=SHA)
    assert roundtrip(ctx) == ctx
    with pytest.raises(ValidationError):
        EpisodeContext(episode_id="e", repo=RepoRef(repo_id="r"), worktree_path=Path("/w"), commit_sha="zz")


class TestPytestExecution:
    __test__ = True

    def test_roundtrip_and_record(self) -> None:
        run = passing_run()
        assert roundtrip(run) == run
        rec = run.to_record("r1", TestRunPhase.PRE_REPAIR)
        assert rec.status is TestRunStatus.PASSED and rec.passed == 1

    def test_distinguishes_failure_error_timeout(self) -> None:
        failing = FailingTest(test_id=T1, test_file="tests/test_x.py", test_name="test_a")
        failed = PytestExecution(
            selectors=(T1,), command=("pytest",), status=TestRunStatus.FAILED, exit_code=1,
            passed=0, failed=1, failures=(failing,),
        )
        assert failed.failing_test_ids == [T1]
        timeout = PytestExecution(
            selectors=(T1,), command=("pytest",), status=TestRunStatus.TIMEOUT, timed_out=True,
            timeout_seconds=5,
        )
        assert timeout.failed is None  # unknown, not zero
        with pytest.raises(ValidationError, match="error_message"):
            PytestExecution(selectors=(), command=("pytest",), status=TestRunStatus.ERROR, exit_code=4)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"status": TestRunStatus.NOT_RUN},
            {"status": TestRunStatus.TIMEOUT, "timed_out": False},
            {"status": TestRunStatus.PASSED, "timed_out": True},
            {"status": TestRunStatus.PASSED, "exit_code": 1},
        ],
    )
    def test_inconsistent_runs_rejected(self, kwargs) -> None:
        with pytest.raises(ValidationError):
            PytestExecution(selectors=(), command=("pytest",), **kwargs)


def test_failure_group_detail_must_match_group() -> None:
    t = FailingTest(test_id=T1, test_file="tests/test_x.py", test_name="test_a")
    group = FailureGroup(group_id="g1", test_ids=[T1])
    detail = FailureGroupDetail(group=group, failing_tests=(t,))
    assert roundtrip(detail) == detail
    other = FailingTest(test_id="tests/test_x.py::test_b", test_file="tests/test_x.py", test_name="test_b")
    with pytest.raises(ValidationError, match="match group.test_ids"):
        FailureGroupDetail(group=group, failing_tests=(t, other))
    triage = TriageResult(groups=(detail,), baseline_run=passing_run())
    assert roundtrip(triage) == triage


class TestIntentAnalysis:
    __test__ = True

    def test_valid_and_roundtrip(self) -> None:
        a = analysis()
        assert roundtrip(a) == a

    def test_update_test_needs_evidence(self) -> None:
        with pytest.raises(ValidationError, match="cited evidence"):
            analysis(cited_evidence_ids=())

    def test_hypothesis_decision_consistency(self) -> None:
        with pytest.raises(ValidationError, match="cannot recommend"):
            analysis(recommended_decision=DecisionLabel.FIX_CODE)
        with pytest.raises(ValidationError, match="cannot recommend"):
            analysis(
                hypothesis=Hypothesis(kind=HypothesisKind.UNKNOWN, explanation="?"),
                recommended_decision=DecisionLabel.FIX_CODE,
            )
        analysis(  # uncertain evidence may always escalate
            hypothesis=Hypothesis(kind=HypothesisKind.UNKNOWN, explanation="?"),
            recommended_decision=DecisionLabel.ESCALATE,
            cited_evidence_ids=(),
        )

    def test_confidence_bounds_and_decision_enum(self) -> None:
        with pytest.raises(ValidationError):
            analysis(confidence=1.5)
        with pytest.raises(ValidationError):
            analysis(recommended_decision="MAYBE")

    def test_invented_citations_are_detected(self) -> None:
        assert unknown_evidence_ids(["ev-1", "ev-fake"], ["ev-1", "ev-2"]) == ["ev-fake"]
        assert unknown_evidence_ids([], []) == []


class TestPatchProposal:
    __test__ = True

    def test_valid(self) -> None:
        p = proposal()
        assert roundtrip(p) == p

    def test_scope_must_match_direction(self) -> None:
        with pytest.raises(ValidationError, match="requires scope"):
            proposal(scope=PatchScope.SRC)
        fix = proposal(decision=RepairDirection.FIX_CODE, scope=PatchScope.SRC, evidence_ids=())
        assert fix.scope is PatchScope.SRC

    def test_update_test_needs_evidence(self) -> None:
        with pytest.raises(ValidationError, match="evidence_ids"):
            proposal(evidence_ids=())

    def test_escalate_is_not_a_patch_direction(self) -> None:
        with pytest.raises(ValidationError):
            proposal(decision="ESCALATE")

    def test_escalation_request_trigger_matches_summary(self) -> None:
        for trigger in ("insufficient_evidence", "contradictory_evidence", "guard_rejection_after_revision"):
            EscalationRequest(group_id="g1", trigger=trigger, summary="s")
            EscalationSummary(trigger=trigger, failing_test_ids=[T1], summary="s")
        with pytest.raises(ValidationError):
            EscalationRequest(group_id="g1", trigger="because", summary="s")


def test_patch_validation() -> None:
    ok = PatchValidation(valid=True, allowed=True, changed_paths=("src/a.py",))
    assert ok.ok and roundtrip(ok) == ok
    with pytest.raises(ValidationError, match="rejection_reasons"):
        PatchValidation(valid=True, allowed=False)
    bad = PatchValidation(valid=True, allowed=False, rejection_reasons=("touches pyproject.toml",))
    assert not bad.ok


class TestGuardResult:
    __test__ = True

    def test_approve_reject_escalate(self) -> None:
        approve = guard(GuardDecision.APPROVE, reasons=())
        assert roundtrip(approve) == approve
        guard(GuardDecision.REJECT)
        guard(GuardDecision.ESCALATE)

    def test_not_run_is_not_a_guard_result(self) -> None:
        with pytest.raises(ValidationError):
            guard(GuardDecision.NOT_RUN)

    def test_approve_requires_passing_run(self) -> None:
        with pytest.raises(ValidationError, match="passing post-repair"):
            guard(GuardDecision.APPROVE, reasons=(), post_repair_run=None)

    def test_reject_requires_reasons_and_matching_report(self) -> None:
        with pytest.raises(ValidationError, match="require reasons"):
            guard(GuardDecision.REJECT, reasons=())
        with pytest.raises(ValidationError, match="same decision"):
            guard(GuardDecision.REJECT, integrity=IntegrityReport())


class TestMemory:
    __test__ = True

    @staticmethod
    def record(**kw) -> MemoryRecord:
        base = {
            "record_id": "m1", "repo_id": "r", "function_qualname": "pkg.f", "behaviour_signature": "f() -> 'A'",
            "direction": RepairDirection.UPDATE_TEST, "probe": ExecutableProbe(probe_id="p", probe_code="assert 1"),
            "approved_at_commit": SHA,
        }
        return MemoryRecord(**{**base, **kw})

    def test_only_passing_current_records_are_usable(self) -> None:
        good = MemoryRecordCheck(record=self.record(), probe_status=ProbeStatus.PASSED)
        failed = MemoryRecordCheck(
            record=self.record(record_id="m2"), probe_status=ProbeStatus.FAILED, blocked_reason="probe_failed"
        )
        unrun = MemoryRecordCheck(record=self.record(record_id="m3"))
        stale = MemoryRecordCheck(
            record=self.record(record_id="m4", flagged_stale=True), probe_status=ProbeStatus.PASSED
        )
        superseded = MemoryRecordCheck(
            record=self.record(record_id="m5"), probe_status=ProbeStatus.PASSED, blocked_reason="superseded"
        )
        result = MemoryReadResult(checks=(good, failed, unrun, stale, superseded))
        assert [r.record_id for r in result.usable] == ["m1"]
        assert len(result.unusable) == 4
        assert roundtrip(result) == result

    def test_passed_probe_cannot_be_blocked_for_probe_reasons(self) -> None:
        with pytest.raises(ValidationError):
            MemoryRecordCheck(record=self.record(), probe_status=ProbeStatus.PASSED, blocked_reason="probe_failed")

    def test_approval_requires_guard_approve_and_maintainer(self) -> None:
        ok = MemoryApproval(
            record_id="m1", guard_decision=GuardDecision.APPROVE, maintainer_approved=True, approved_at_commit=SHA
        )
        assert roundtrip(ok) == ok
        with pytest.raises(ValidationError, match="Guard APPROVE"):
            MemoryApproval(
                record_id="m1", guard_decision=GuardDecision.REJECT, maintainer_approved=True, approved_at_commit=SHA
            )
        with pytest.raises(ValidationError, match="maintainer"):
            MemoryApproval(
                record_id="m1", guard_decision=GuardDecision.APPROVE, maintainer_approved=False, approved_at_commit=SHA
            )


def test_error_hierarchy_maps_to_run_vocabulary() -> None:
    assert LLMRateLimitError("429").category is ErrorCategory.RATE_LIMIT
    rec = LLMRateLimitError("429").to_error_record("e1")
    assert rec.counted_against_agent is False
    assert LLMProviderOutageError("down").to_error_record("e2").counted_against_agent is False
    assert PatchError("bad").to_error_record("e3").category is ErrorCategory.PATCH_ERROR
    assert ToolTimeoutError("slow").category is ErrorCategory.TIMEOUT
    assert BudgetExceededError.run_status is RunStatus.BUDGET_EXHAUSTED
    assert LLMRateLimitError.run_status is RunStatus.API_FAILURE
