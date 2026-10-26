"""Agent results, verification outcomes and experiment-run records for TestOrCode.

One :class:`ExperimentRunRecord` is one JSONL line in ``results/``: a single
episode executed once under one condition, seed and model.  Repeated runs of the
same episode under different conditions are separate records that share
``provenance.episode_id``.

What lives here
---------------
* what the agent predicted and why (:class:`PredictedDecision`),
* what it retrieved and cited (:class:`RetrievedEvidence`),
* what it changed and what happened (:class:`PatchRecord`, :class:`TestRunResult`),
* what the Integrity Guard found (:class:`IntegrityReport`),
* how the run was executed (:class:`ToolEvent`, :class:`TrajectoryStep`,
  :class:`TokenUsage`, :class:`BudgetUsage`, :class:`ErrorRecord`),
* how it is reproduced (:class:`RunProvenance`),
* and, filled in later by the evaluation layer only, :class:`RunEvaluation`.

What does not live here
-----------------------
No aggregation, no metric formulas, no model calls, no test execution, no
orchestration.  These models store the *observations* an evaluation module
needs; it computes macro-F1, rates and retention across runs.

Deliberate non-collapses
------------------------
* zero tokens vs. unavailable usage -> ``TokenUsage.reported``,
* tests passed vs. tests not run -> ``TestRunStatus.NOT_RUN``,
* rejected patch vs. unapplicable patch -> ``PatchApplicationStatus``,
* escalation vs. tool/API failure -> ``RunStatus`` + ``EscalationSummary``,
* missing seed vs. seed 0 -> ``RunProvenance.seed: int | None`` (required),
* green suite vs. successful repair -> ``RepairEvaluation.verified_dimensions``.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .episode_schemas import (
    SCHEMA_VERSION,
    Availability,
    CommitSha,
    DecisionLabel,
    EvidenceSourceType,
    HypothesisKind,
    Identifier,
    PatchScope,
    RepairDirection,
    ToolFaultKind,
    ToolName,
)

__all__ = [
    "RESULT_SCHEMA_VERSION",
    "ExperimentCondition",
    "CORE_CONDITIONS",
    "AgentRole",
    "RetrievalMethod",
    "SupportStatus",
    "EvidenceFreshness",
    "ProvenanceStatus",
    "PatchApplicationStatus",
    "TestRunStatus",
    "TestRunPhase",
    "GuardDecision",
    "CheckStatus",
    "ProbeStatus",
    "ToolCallStatus",
    "OperationType",
    "StopReason",
    "RunStatus",
    "ErrorCategory",
    "FailureCause",
    "RepairVerificationDimension",
    "RetrievedEvidence",
    "Hypothesis",
    "ConfidenceReport",
    "DecisionRationale",
    "PredictedDecision",
    "PatchRecord",
    "TestRunResult",
    "MutantOutcome",
    "MutationResult",
    "TestWeakeningFindings",
    "ProhibitedEditCheck",
    "IntegrityCheck",
    "IntegrityReport",
    "ProbeRun",
    "MemoryReadEvent",
    "MemoryWriteAttempt",
    "MaintainerReview",
    "EscalationSummary",
    "ToolEvent",
    "ToolCallSummary",
    "TrajectoryStep",
    "TokenUsage",
    "BudgetUsage",
    "ErrorRecord",
    "JudgeConfig",
    "RunProvenance",
    "RunOutcome",
    "UnitDecisionScore",
    "EvidenceEvaluation",
    "RepairEvaluation",
    "IntegrityEvaluation",
    "RunEvaluation",
    "ExperimentRunRecord",
    "METRIC_SUPPORT",
    "RESULT_RUNTIME_ENFORCEMENT",
]

RESULT_SCHEMA_VERSION: Final[str] = SCHEMA_VERSION


# --------------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------------- #


class ExperimentCondition(StrEnum):
    """Explicitly defined project conditions.

    Note on naming: the proposal lists three ablations (A1 no intent retrieval,
    A2 no memory, A3 no Integrity Guard) while the detailed document numbers
    them A1-A5 with A3 = coarse memory and A4 = no Integrity Guard.  The
    five-day execution plan uses the proposal's numbering, so that is canonical
    here; coarse memory and the merged-roles condition keep distinct names.
    """

    FULL = "FULL"
    B0 = "B0"
    B1 = "B1"
    B2 = "B2"
    A1_NO_INTENT_RETRIEVAL = "A1"
    A2_NO_MEMORY = "A2"
    A3_NO_INTEGRITY_GUARD = "A3"
    A4_COARSE_MEMORY = "A4"
    A5_SINGLE_AGENT = "A5"
    TRIVIAL_ALWAYS_UPDATE_TEST = "TRIVIAL_UPDATE_TEST"
    TRIVIAL_ALWAYS_FIX_CODE = "TRIVIAL_FIX_CODE"
    ORACLE = "ORACLE"
    MODEL_CHECK = "MODEL_CHECK"


CORE_CONDITIONS: Final[tuple[ExperimentCondition, ...]] = (
    ExperimentCondition.FULL,
    ExperimentCondition.B1,
    ExperimentCondition.A1_NO_INTENT_RETRIEVAL,
    ExperimentCondition.A2_NO_MEMORY,
    ExperimentCondition.A3_NO_INTEGRITY_GUARD,
)
"""The five conditions the five-day plan requires to be runnable."""


class AgentRole(StrEnum):
    TRIAGE = "triage"
    INTENT_ANALYST = "intent_analyst"
    REPAIRER = "repairer"
    INTEGRITY_GUARD = "integrity_guard"
    SINGLE_AGENT = "single_agent"
    ORCHESTRATOR = "orchestrator"


class RetrievalMethod(StrEnum):
    BM25 = "bm25"
    EMBEDDING = "embedding"
    HYBRID = "hybrid"
    RERANK = "rerank"
    MEMORY_LOOKUP = "memory_lookup"
    DIRECT_READ = "direct_read"
    GIT_COMMAND = "git_command"


class SupportStatus(StrEnum):
    """The agent's assessment of an item, kept apart from the item itself."""

    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    INSUFFICIENT = "insufficient"
    UNKNOWN = "unknown"


class EvidenceFreshness(StrEnum):
    CURRENT = "current"
    STALE = "stale"
    UNKNOWN = "unknown"


class ProvenanceStatus(StrEnum):
    """Whether a citation was mechanically checked to exist at its source."""

    VERIFIED = "verified"
    NOT_FOUND = "not_found"
    NOT_CHECKED = "not_checked"


class PatchApplicationStatus(StrEnum):
    NOT_ATTEMPTED = "not_attempted"
    APPLIED = "applied"
    FAILED_TO_APPLY = "failed_to_apply"
    BLOCKED_OUT_OF_SCOPE = "blocked_out_of_scope"
    REJECTED_BY_GUARD = "rejected_by_guard"
    REVERTED = "reverted"


class TestRunStatus(StrEnum):
    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"
    TIMEOUT = "timeout"
    COLLECTION_ERROR = "collection_error"
    NOT_RUN = "not_run"


class TestRunPhase(StrEnum):
    PRE_REPAIR = "pre_repair"
    POST_REPAIR = "post_repair"
    POST_REVISION = "post_revision"
    PROBE = "probe"
    MUTATION = "mutation"
    STABILITY = "stability"


class GuardDecision(StrEnum):
    APPROVE = "approve"
    REJECT = "reject"
    ESCALATE = "escalate"
    NOT_RUN = "not_run"


class CheckStatus(StrEnum):
    PASSED = "passed"
    FAILED = "failed"
    NOT_RUN = "not_run"
    ERROR = "error"


class ProbeStatus(StrEnum):
    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"
    NOT_RUN = "not_run"


class ToolCallStatus(StrEnum):
    OK = "ok"
    INVALID_ARGUMENTS = "invalid_arguments"
    NOT_PERMITTED = "not_permitted"
    INJECTED_FAULT = "injected_fault"
    TIMEOUT = "timeout"
    ERROR = "error"


class OperationType(StrEnum):
    TRIAGE = "triage"
    RETRIEVAL = "retrieval"
    EVIDENCE_EVALUATION = "evidence_evaluation"
    LLM_CALL = "llm_call"
    MEMORY_READ = "memory_read"
    MEMORY_WRITE = "memory_write"
    PATCH_APPLY = "patch_apply"
    TEST_EXECUTION = "test_execution"
    MUTATION_RUN = "mutation_run"
    INTEGRITY_VERIFICATION = "integrity_verification"
    MAINTAINER_REVIEW = "maintainer_review"
    ESCALATION = "escalation"


class StopReason(StrEnum):
    SUFFICIENT_EVIDENCE = "sufficient_evidence"
    EVIDENCE_EXHAUSTED = "evidence_exhausted"
    BUDGET_EXHAUSTED = "budget_exhausted"
    TOOL_FAILURE = "tool_failure"
    API_FAILURE = "api_failure"
    GUARD_REJECTED = "guard_rejected"
    ESCALATION = "escalation"
    COMPLETED = "completed"
    ORCHESTRATOR_ERROR = "orchestrator_error"


class RunStatus(StrEnum):
    """Terminal status of one episode-run.

    ``ESCALATED`` is a deliberate agent outcome.  ``TOOL_FAILURE``,
    ``API_FAILURE``, ``BUDGET_EXHAUSTED`` and ``ORCHESTRATOR_ERROR`` are
    execution failures and must never be scored as escalations.
    """

    COMPLETED = "completed"
    ESCALATED = "escalated"
    INCOMPLETE = "incomplete"
    BUDGET_EXHAUSTED = "budget_exhausted"
    TOOL_FAILURE = "tool_failure"
    API_FAILURE = "api_failure"
    ORCHESTRATOR_ERROR = "orchestrator_error"


class ErrorCategory(StrEnum):
    TOOL_ERROR = "tool_error"
    INJECTED_TOOL_FAULT = "injected_tool_fault"
    API_ERROR = "api_error"
    RATE_LIMIT = "rate_limit"
    PROVIDER_OUTAGE = "provider_outage"
    PATCH_ERROR = "patch_error"
    VALIDATION_ERROR = "validation_error"
    TIMEOUT = "timeout"
    SANDBOX_ERROR = "sandbox_error"
    ORCHESTRATOR_ERROR = "orchestrator_error"


class FailureCause(StrEnum):
    """Single assigned cause per failed episode (detailed document, section 11)."""

    TRIAGE_ERROR = "triage_error"
    RETRIEVAL_OR_STALE_EVIDENCE = "retrieval_or_stale_evidence"
    REASONING_ERROR = "reasoning_error"
    REPAIR_ERROR = "repair_error"
    INTEGRITY_MISS = "integrity_miss"
    MEMORY_ERROR = "memory_error"
    TOOL_ERROR = "tool_error"
    API_ERROR = "api_error"


class RepairVerificationDimension(StrEnum):
    """Dimensions that must each be checked before calling a repair successful."""

    DIRECTION_CORRECT = "direction_correct"
    SUITE_PASSED = "suite_passed"
    INTEGRITY_PRESERVED = "integrity_preserved"
    SCOPE_RESPECTED = "scope_respected"


_PROVIDER_ONLY_IDS: Final[frozenset[str]] = frozenset(
    {
        "nvidia",
        "nim",
        "mistral",
        "groq",
        "google",
        "gemini",
        "openai",
        "anthropic",
        "voyage",
        "qwen",
        "deepseek",
    }
)


# --------------------------------------------------------------------------- #
# Base
# --------------------------------------------------------------------------- #


class _Result(BaseModel):
    """Mutable, strict base: run records are assembled incrementally."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
        use_enum_values=False,
    )


class _FrozenResult(_Result):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
        use_enum_values=False,
    )


# --------------------------------------------------------------------------- #
# Evidence
# --------------------------------------------------------------------------- #


class RetrievedEvidence(_Result):
    """One item returned by retrieval, with its retrieval metadata.

    ``score is None`` means the retriever reported no score; it is never
    replaced by 0.0.  ``support`` is the agent's *interpretation* and is kept
    separate from the item's identity and ranking.
    """

    evidence_id: Identifier
    source_type: EvidenceSourceType
    source_ref: str = Field(min_length=1)
    commit_sha: str | None = None
    section: str | None = None
    line_start: int | None = Field(default=None, ge=1)
    line_end: int | None = Field(default=None, ge=1)
    snippet: str | None = None

    retrieval_query: str | None = None
    retrieval_method: RetrievalMethod | None = None
    rank: int | None = Field(default=None, ge=1, description="1-based; None if N/A.")
    score: float | None = Field(
        default=None, description="None = the retriever reported no score."
    )
    retrieved_at_step: Identifier | None = None

    used_by_agent: bool = False
    cited: bool = False
    support: SupportStatus = SupportStatus.UNKNOWN
    freshness: EvidenceFreshness = EvidenceFreshness.UNKNOWN
    applies_to_current_code: bool | None = Field(
        default=None, description="None = not assessed (e.g. probe not run)."
    )
    provenance_verified: ProvenanceStatus = ProvenanceStatus.NOT_CHECKED
    memory_record_id: Identifier | None = None
    note: str | None = None

    @model_validator(mode="after")
    def _check(self) -> RetrievedEvidence:
        if self.cited and not self.used_by_agent:
            raise ValueError("a cited evidence item must be marked used_by_agent")
        if (
            self.source_type is EvidenceSourceType.MEMORY_RECORD
            and self.memory_record_id is None
        ):
            raise ValueError("memory evidence must carry memory_record_id")
        return self


# --------------------------------------------------------------------------- #
# Reasoning
# --------------------------------------------------------------------------- #


class Hypothesis(_Result):
    """One candidate explanation for a failure group."""

    hypothesis_id: Identifier
    kind: HypothesisKind
    statement: str = Field(min_length=1, description="Concise, auditable summary.")
    predicted_behaviour: str | None = Field(
        default=None, description="What the code is expected to do if this holds."
    )
    supporting_evidence_ids: list[Identifier] = Field(default_factory=list)
    contradicting_evidence_ids: list[Identifier] = Field(default_factory=list)
    probe_record_id: Identifier | None = None
    checked: bool = Field(
        default=False, description="Whether the prediction was checked at all."
    )
    check_outcome: Literal["confirmed", "refuted", "inconclusive"] | None = None
    selected: bool = False

    @model_validator(mode="after")
    def _check(self) -> Hypothesis:
        if self.check_outcome is not None and not self.checked:
            raise ValueError("check_outcome set but checked is False")
        return self


class ConfidenceReport(_Result):
    """Model-reported confidence, explicitly not a calibrated probability."""

    value: float | None = Field(default=None, ge=0.0, le=1.0)
    source: Literal["model_self_report", "heuristic", "none"] = "none"
    calibrated: bool = False
    calibration_method: str | None = None

    @model_validator(mode="after")
    def _check(self) -> ConfidenceReport:
        if self.value is not None and self.source == "none":
            raise ValueError("a confidence value requires an explicit source")
        if self.calibrated and not self.calibration_method:
            raise ValueError(
                "calibrated confidence requires a named calibration_method; "
                "a model self-report is not a validated probability"
            )
        return self


class DecisionRationale(_Result):
    """Auditable summary of why an action was chosen.

    A short justification, not an unrestricted chain of thought.
    """

    summary: str = Field(min_length=1)
    justification: str | None = None
    unresolved_contradictions: list[str] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)
    additional_retrieval_required: bool = False
    additional_verification_required: bool = False
    escalation_reason: str | None = None


class PredictedDecision(_Result):
    """The agent's decision for one failing test or one failure group.

    ``test_ids`` always lists the individual failing tests covered, so a
    group-level decision expands to per-test predictions for scoring and a
    mixed episode (S3) simply carries several of these with different actions.
    """

    decision_id: Identifier
    unit_kind: Literal["test", "group"] = "test"
    test_ids: list[Identifier] = Field(min_length=1)
    group_id: Identifier | None = None
    function_qualname: str | None = None

    predicted_action: DecisionLabel
    predicted_direction: RepairDirection | None = Field(
        default=None,
        description="Side believed wrong; may be set even when action is ESCALATE.",
    )
    hypotheses: list[Hypothesis] = Field(default_factory=list)
    selected_hypothesis_id: Identifier | None = None
    cited_evidence_ids: list[Identifier] = Field(default_factory=list)
    rationale: DecisionRationale | None = None
    confidence: ConfidenceReport | None = None
    decided_by: AgentRole = AgentRole.INTENT_ANALYST
    decided_at_step: Identifier | None = None
    revision_index: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _check(self) -> PredictedDecision:
        if self.unit_kind == "group" and self.group_id is None:
            raise ValueError("a group-level decision requires group_id")
        if self.unit_kind == "test" and len(self.test_ids) != 1:
            raise ValueError("a test-level decision must list exactly one test_id")
        if len(self.test_ids) != len(set(self.test_ids)):
            raise ValueError("test_ids contains duplicates")
        known = {h.hypothesis_id for h in self.hypotheses}
        if self.selected_hypothesis_id and self.selected_hypothesis_id not in known:
            raise ValueError("selected_hypothesis_id is not among hypotheses")
        if (
            self.predicted_action is DecisionLabel.ESCALATE
            and self.rationale is not None
            and not self.rationale.escalation_reason
        ):
            raise ValueError("an ESCALATE decision must record an escalation_reason")
        if (
            self.predicted_action is not DecisionLabel.ESCALATE
            and self.predicted_direction is not None
            and self.predicted_direction.value != self.predicted_action.value
        ):
            raise ValueError(
                "predicted_direction contradicts predicted_action"
            )
        return self


# --------------------------------------------------------------------------- #
# Repair and execution
# --------------------------------------------------------------------------- #


class PatchRecord(_Result):
    """One patch the Repairer produced, and what became of it."""

    patch_id: Identifier
    scope: PatchScope
    declared_for_decision_id: Identifier | None = None
    target_files: list[str] = Field(default_factory=list)
    unified_diff: str | None = None
    lines_added: int | None = Field(default=None, ge=0)
    lines_removed: int | None = Field(default=None, ge=0)
    application_status: PatchApplicationStatus = PatchApplicationStatus.NOT_ATTEMPTED
    apply_error: str | None = None
    out_of_scope_files: list[str] = Field(
        default_factory=list,
        description="Paths the patch touched that the EditPolicy does not permit.",
    )
    revision_index: int = Field(default=0, ge=0)
    produced_by: AgentRole = AgentRole.REPAIRER
    applied_at_commit: CommitSha | None = None

    @model_validator(mode="after")
    def _check(self) -> PatchRecord:
        if (
            self.application_status is PatchApplicationStatus.FAILED_TO_APPLY
            and not self.apply_error
        ):
            raise ValueError(
                "failed_to_apply must record apply_error; a patch that could not be "
                "applied is not the same as a patch the Guard rejected"
            )
        if (
            self.application_status is PatchApplicationStatus.BLOCKED_OUT_OF_SCOPE
            and not self.out_of_scope_files
        ):
            raise ValueError(
                "blocked_out_of_scope must list the offending files"
            )
        return self


class TestRunResult(_Result):
    """One execution of pytest (or a probe / mutant run)."""

    run_id: Identifier
    phase: TestRunPhase
    status: TestRunStatus = TestRunStatus.NOT_RUN
    selector: str | None = Field(default=None, description="pytest -k / node ids.")
    passed: int | None = Field(default=None, ge=0)
    failed: int | None = Field(default=None, ge=0)
    errors: int | None = Field(default=None, ge=0)
    skipped: int | None = Field(default=None, ge=0)
    xfailed: int | None = Field(default=None, ge=0)
    failing_test_ids: list[Identifier] = Field(default_factory=list)
    duration_seconds: float | None = Field(default=None, ge=0.0)
    exit_code: int | None = None
    output_ref: str | None = Field(
        default=None, description="Path or key of the captured output, not the blob."
    )
    timed_out: bool = False

    @model_validator(mode="after")
    def _check(self) -> TestRunResult:
        if self.status is TestRunStatus.NOT_RUN:
            counts = (self.passed, self.failed, self.errors, self.skipped)
            if any(c is not None for c in counts) or self.failing_test_ids:
                raise ValueError(
                    "status NOT_RUN must not carry counts: a test run that did not "
                    "happen is not a run with zero failures"
                )
        if self.status is TestRunStatus.PASSED and self.failing_test_ids:
            raise ValueError("status PASSED contradicts failing_test_ids")
        if self.status is TestRunStatus.TIMEOUT and not self.timed_out:
            raise ValueError("status TIMEOUT requires timed_out=True")
        return self


# --------------------------------------------------------------------------- #
# Integrity
# --------------------------------------------------------------------------- #


class MutantOutcome(_FrozenResult):
    """Kill status of one mutant before and after the repair."""

    mutant_id: Identifier
    killed_by_reference: bool | None = None
    killed_after_repair: bool | None = None
    run_status: TestRunStatus = TestRunStatus.NOT_RUN


class MutationResult(_Result):
    """Mutation run for the function under repair.

    Stores counts only.  Mutant-kill retention is derived by the evaluation
    layer; a missing run is ``status = NOT_RUN``, never a zero retention.
    """

    status: TestRunStatus = TestRunStatus.NOT_RUN
    function_qualname: str | None = None
    catalogue_id: Identifier | None = None
    mutants_total: int | None = Field(default=None, ge=0)
    killed_by_reference_tests: int | None = Field(default=None, ge=0)
    killed_by_repaired_tests: int | None = Field(default=None, ge=0)
    per_mutant: list[MutantOutcome] = Field(default_factory=list)
    duration_seconds: float | None = Field(default=None, ge=0.0)
    note: str | None = None

    @model_validator(mode="after")
    def _check(self) -> MutationResult:
        if self.status is TestRunStatus.NOT_RUN and (
            self.killed_by_repaired_tests is not None or self.per_mutant
        ):
            raise ValueError("NOT_RUN mutation result must not carry kill counts")
        for name in ("killed_by_reference_tests", "killed_by_repaired_tests"):
            value = getattr(self, name)
            if (
                value is not None
                and self.mutants_total is not None
                and value > self.mutants_total
            ):
                raise ValueError(f"{name} exceeds mutants_total")
        return self


class TestWeakeningFindings(_Result):
    """Deterministic weakening checks on a test patch.

    ``checks_run=False`` (the A3 ablation) leaves every finding ``None``; it
    must not be read as "no weakening".
    """

    checks_run: bool = False
    assertions_before: int | None = Field(default=None, ge=0)
    assertions_after: int | None = Field(default=None, ge=0)
    removed_assertions: int | None = Field(default=None, ge=0)
    added_skip_markers: int | None = Field(default=None, ge=0)
    added_xfail_markers: int | None = Field(default=None, ge=0)
    loosened_comparisons: int | None = Field(default=None, ge=0)
    swallowed_exceptions: int | None = Field(default=None, ge=0)
    deleted_tests: int | None = Field(default=None, ge=0)
    weakening_detected: bool | None = Field(
        default=None, description="None = not assessed."
    )
    details: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check(self) -> TestWeakeningFindings:
        if not self.checks_run and self.weakening_detected is not None:
            raise ValueError(
                "weakening_detected must stay None when checks_run is False"
            )
        return self


class ProhibitedEditCheck(_Result):
    """Which files the run actually modified, and whether that was permitted."""

    checks_run: bool = False
    files_modified: list[str] = Field(default_factory=list)
    out_of_scope_files: list[str] = Field(default_factory=list)
    test_files_modified: list[str] = Field(default_factory=list)
    source_files_modified: list[str] = Field(default_factory=list)
    ci_config_touched: bool | None = None
    dependency_files_touched: bool | None = None
    conftest_touched: bool | None = None

    @model_validator(mode="after")
    def _check(self) -> ProhibitedEditCheck:
        if not self.checks_run and any(
            v is not None
            for v in (
                self.ci_config_touched,
                self.dependency_files_touched,
                self.conftest_touched,
            )
        ):
            raise ValueError(
                "prohibited-edit flags must stay None when checks_run is False"
            )
        return self


class IntegrityCheck(_FrozenResult):
    """One named check with an explicit status."""

    name: str = Field(min_length=1)
    status: CheckStatus
    detail: str | None = None


class IntegrityReport(_Result):
    """Integrity Guard output: independent verification of the repair."""

    guard_decision: GuardDecision = GuardDecision.NOT_RUN
    guard_ran: bool = False
    checks: list[IntegrityCheck] = Field(default_factory=list)
    weakening: TestWeakeningFindings = Field(default_factory=TestWeakeningFindings)
    prohibited_edits: ProhibitedEditCheck = Field(
        default_factory=ProhibitedEditCheck
    )
    mutation: MutationResult = Field(default_factory=MutationResult)
    tests_untouched_for_fix_code: bool | None = None
    evidence_requirement_met: bool | None = Field(
        default=None,
        description="UPDATE_TEST needs cited positive evidence; None = not assessed.",
    )
    revision_requested: bool = False
    rejection_reasons: list[str] = Field(default_factory=list)
    note: str | None = None

    @model_validator(mode="after")
    def _check(self) -> IntegrityReport:
        if not self.guard_ran and self.guard_decision is not GuardDecision.NOT_RUN:
            raise ValueError("guard_decision requires guard_ran=True")
        if self.guard_ran and self.guard_decision is GuardDecision.NOT_RUN:
            raise ValueError("guard_ran=True requires a guard_decision")
        if self.guard_decision is GuardDecision.REJECT and not self.rejection_reasons:
            raise ValueError("a rejection must record at least one reason")
        return self


# --------------------------------------------------------------------------- #
# Memory and escalation
# --------------------------------------------------------------------------- #


class ProbeRun(_FrozenResult):
    """Execution of a memory record's probe against the current code."""

    probe_id: Identifier
    status: ProbeStatus = ProbeStatus.NOT_RUN
    duration_seconds: float | None = Field(default=None, ge=0.0)
    message: str | None = None


class MemoryReadEvent(_Result):
    """A memory record the agent looked at, and whether it counted as evidence."""

    record_id: Identifier
    function_qualname: str | None = None
    scope: Literal["function", "module"] = "function"
    probe: ProbeRun | None = None
    used_as_evidence: bool = False
    blocked_reason: (
        Literal["probe_failed", "probe_not_run", "superseded", "out_of_scope"] | None
    ) = None
    read_at_step: Identifier | None = None

    @model_validator(mode="after")
    def _check(self) -> MemoryReadEvent:
        if self.used_as_evidence:
            if self.probe is None or self.probe.status is not ProbeStatus.PASSED:
                raise ValueError(
                    "a memory record may be used as evidence only when its probe "
                    "passed on the current code"
                )
            if self.blocked_reason is not None:
                raise ValueError("used_as_evidence contradicts blocked_reason")
        return self


class MaintainerReview(_Result):
    """Simulated maintainer decision gating the memory write."""

    reviewed: bool = False
    approved: bool | None = None
    expected_escalation: bool | None = None
    note: str | None = None

    @model_validator(mode="after")
    def _check(self) -> MaintainerReview:
        if not self.reviewed and self.approved is not None:
            raise ValueError("approved must stay None when reviewed is False")
        return self


class MemoryWriteAttempt(_Result):
    """An attempt to persist a memory record.

    A write without maintainer approval is an unsafe action; it is recorded
    here with ``written=False`` and ``approved=False`` so the evaluation layer
    can count it.
    """

    record_id: Identifier
    function_qualname: str | None = None
    behaviour_signature: str | None = None
    direction: RepairDirection | None = None
    probe_id: Identifier | None = None
    evidence_ids: list[Identifier] = Field(default_factory=list)
    approved: bool = False
    approval_source: Literal["simulated_maintainer", "none"] = "none"
    written: bool = False
    supersedes: Identifier | None = None
    blocked_reason: (
        Literal[
            "not_approved",
            "guard_rejected",
            "memory_disabled",
            "probe_missing",
            "write_error",
        ]
        | None
    ) = None
    written_by: AgentRole = AgentRole.INTEGRITY_GUARD

    @model_validator(mode="after")
    def _check(self) -> MemoryWriteAttempt:
        if self.written and not self.approved:
            raise ValueError(
                "a memory record may only be written after maintainer approval"
            )
        if self.written and self.written_by is not AgentRole.INTEGRITY_GUARD:
            raise ValueError("only the Integrity Guard may write memory")
        if self.approved and self.approval_source == "none":
            raise ValueError("approved writes must name an approval_source")
        if not self.written and self.blocked_reason is None:
            raise ValueError("a write that did not happen must record a blocked_reason")
        return self


class EscalationSummary(_Result):
    """The case summary handed to the maintainer (proposal, Human escalation)."""

    raised_by: AgentRole = AgentRole.INTENT_ANALYST
    trigger: Literal[
        "insufficient_evidence",
        "contradictory_evidence",
        "guard_rejection_after_revision",
        "policy",
    ]
    failing_test_ids: list[Identifier] = Field(min_length=1)
    evidence_found_ids: list[Identifier] = Field(default_factory=list)
    evidence_missing: list[str] = Field(
        default_factory=list, description="What was looked for and not found."
    )
    competing_hypotheses: list[Hypothesis] = Field(default_factory=list)
    summary: str = Field(min_length=1)


# --------------------------------------------------------------------------- #
# Execution trace
# --------------------------------------------------------------------------- #


class ToolEvent(_Result):
    """One tool call, including injected faults, retries and recovery."""

    event_id: Identifier
    step_index: int = Field(ge=0)
    tool: ToolName
    called_by: AgentRole = AgentRole.ORCHESTRATOR
    arguments: dict[str, Any] = Field(default_factory=dict)
    status: ToolCallStatus = ToolCallStatus.OK
    injected_fault: bool = False
    fault_kind: ToolFaultKind | None = None
    fault_id: Identifier | None = None
    error_type: ErrorCategory | None = None
    error_message: str | None = None
    retry_index: int = Field(
        default=0, ge=0, description="0 for the first attempt."
    )
    retry_of_event_id: Identifier | None = None
    recovered: bool | None = Field(
        default=None,
        description="Set on a failed call once a later attempt succeeded.",
    )
    duration_ms: float | None = Field(default=None, ge=0.0)
    timestamp: datetime | None = None

    @model_validator(mode="after")
    def _check(self) -> ToolEvent:
        if self.injected_fault:
            if self.status is ToolCallStatus.OK:
                raise ValueError("an injected fault cannot have status OK")
            if self.fault_kind is None:
                raise ValueError("an injected fault must record its fault_kind")
            if self.error_type not in (None, ErrorCategory.INJECTED_TOOL_FAULT):
                raise ValueError(
                    "injected faults use error_type INJECTED_TOOL_FAULT and are "
                    "logged separately from provider/API errors"
                )
        if self.status is not ToolCallStatus.OK and self.error_message is None:
            raise ValueError("a failed tool call must record an error_message")
        if self.status is ToolCallStatus.OK and self.recovered is not None:
            raise ValueError("recovered applies to failed calls only")
        if self.retry_index > 0 and self.retry_of_event_id is None:
            raise ValueError("a retry must reference the original event id")
        return self


class ToolCallSummary(_Result):
    """Counts for tool-efficiency metrics; raw events stay in ``tool_events``."""

    total_calls: int = Field(default=0, ge=0)
    calls_by_tool: dict[str, int] = Field(default_factory=dict)
    invalid_argument_calls: int = Field(default=0, ge=0)
    not_permitted_calls: int = Field(default=0, ge=0)
    injected_faults: int = Field(default=0, ge=0)
    retries: int = Field(default=0, ge=0)
    recovered_failures: int = Field(default=0, ge=0)
    unrecovered_failures: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _check(self) -> ToolCallSummary:
        if sum(self.calls_by_tool.values()) > self.total_calls:
            raise ValueError("calls_by_tool sums to more than total_calls")
        return self


class TrajectoryStep(_Result):
    """One ordered step of the retrieval / verification trajectory.

    The project uses a small number of steps per episode; this model records
    them when they happen and imposes no multi-step loop when they do not.
    """

    step_id: Identifier
    index: int = Field(ge=0)
    operation: OperationType
    actor: AgentRole = AgentRole.ORCHESTRATOR
    timestamp: datetime | None = None
    parameters: dict[str, Any] = Field(
        default_factory=dict, description="Query or operation arguments, for audit."
    )
    evidence_ids_returned: list[Identifier] = Field(default_factory=list)
    evidence_ids_consumed: list[Identifier] = Field(default_factory=list)
    tool_event_ids: list[Identifier] = Field(default_factory=list)
    status: Literal["ok", "failed", "skipped"] = "ok"
    error_type: ErrorCategory | None = None
    error_message: str | None = None
    retries: int = Field(default=0, ge=0)
    changed_decision: bool | None = Field(
        default=None, description="None = not assessed."
    )
    stopping_reason: StopReason | None = None
    duration_ms: float | None = Field(default=None, ge=0.0)

    @model_validator(mode="after")
    def _check(self) -> TrajectoryStep:
        if self.status == "failed" and self.error_type is None:
            raise ValueError("a failed step must record an error_type")
        return self


# --------------------------------------------------------------------------- #
# Usage, budget, errors
# --------------------------------------------------------------------------- #


class TokenUsage(_Result):
    """Provider-reported token usage.

    ``reported=False`` with all counts ``None`` means the provider returned no
    usage.  Zero usage is ``reported=True`` with explicit zeros.  The two are
    never merged.
    """

    reported: bool = False
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    cached_input_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    llm_calls: int | None = Field(default=None, ge=0)
    embedding_tokens: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _check(self) -> TokenUsage:
        counts = (self.input_tokens, self.output_tokens, self.total_tokens)
        if not self.reported:
            if any(c is not None for c in counts):
                raise ValueError(
                    "reported=False means the provider returned no usage; leave the "
                    "counts None rather than writing zeros"
                )
        elif all(c is None for c in counts):
            raise ValueError("reported=True requires at least one token count")
        return self


class BudgetUsage(_Result):
    """Budget consumption; identical limits across conditions by design."""

    max_tool_calls: int | None = Field(default=None, gt=0)
    tool_calls_used: int | None = Field(default=None, ge=0)
    max_llm_calls: int | None = Field(default=None, gt=0)
    llm_calls_used: int | None = Field(default=None, ge=0)
    max_tokens: int | None = Field(default=None, gt=0)
    max_revisions: int | None = Field(default=None, ge=0)
    revisions_used: int = Field(default=0, ge=0)
    wall_clock_seconds: float | None = Field(default=None, ge=0.0)
    budget_exhausted: bool = False


class ErrorRecord(_Result):
    """A structured error, with its operation, retries and recovery outcome."""

    error_id: Identifier
    category: ErrorCategory
    message: str = Field(min_length=1)
    operation: str | None = Field(
        default=None, description="Tool name, agent step or API call."
    )
    tool: ToolName | None = None
    step_id: Identifier | None = None
    retries: int = Field(default=0, ge=0)
    recovered: bool = False
    fatal: bool = False
    counted_against_agent: bool = Field(
        default=True,
        description=(
            "False for rate limits and provider outages, which are logged "
            "separately and never scored against the agent."
        ),
    )
    timestamp: datetime | None = None

    @model_validator(mode="after")
    def _check(self) -> ErrorRecord:
        if (
            self.category in (ErrorCategory.RATE_LIMIT, ErrorCategory.PROVIDER_OUTAGE)
            and self.counted_against_agent
        ):
            raise ValueError(
                "rate limits and provider outages must not be counted against the agent"
            )
        if self.recovered and self.fatal:
            raise ValueError("an error cannot be both recovered and fatal")
        return self


# --------------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------------- #


class JudgeConfig(_FrozenResult):
    """Configuration of any model-based judge used for secondary scoring."""

    judge_model_id: str = Field(min_length=1)
    judge_prompt_version: str = Field(min_length=1)
    rubric_version: str | None = None
    human_validation_sample_id: str | None = Field(
        default=None,
        description="Reference to the human-labelled sample used for validation.",
    )


class RunProvenance(_FrozenResult):
    """Everything needed to reproduce and group one run.

    Configuration is referenced by version id rather than duplicated, and no
    secrets or API keys may be stored here.
    """

    run_id: Identifier
    episode_id: Identifier
    condition: ExperimentCondition
    seed: int | None = Field(
        ...,
        description="Required and explicit: None means unseeded, not seed 0.",
    )
    model_id: str = Field(
        ...,
        min_length=2,
        description="Exact dated model identifier, e.g. 'qwen/qwen3-coder-480b-a35b-instruct'.",
    )
    provider: str | None = None

    dataset_id: Identifier | None = None
    dataset_version: str | None = None
    split: Literal["dev", "test", "pilot"] | None = None
    episode_schema_version: str = SCHEMA_VERSION

    code_revision: str | None = Field(
        default=None, description="Git sha of the TestOrCode repository."
    )
    prompt_template_version: str | None = None
    agent_config_version: str | None = None
    retrieval_config_version: str | None = None
    index_version: str | None = None
    memory_store_version: str | None = None
    evaluation_protocol_version: str | None = None
    scoring_rubric_version: str | None = None
    judge: JudgeConfig | None = None

    tool_versions: dict[str, str] = Field(default_factory=dict)
    environment_ref: str | None = Field(
        default=None, description="Docker image digest or environment id."
    )
    started_at: datetime | None = None
    ended_at: datetime | None = None

    @model_validator(mode="after")
    def _check(self) -> RunProvenance:
        if self.model_id.strip().lower() in _PROVIDER_ONLY_IDS:
            raise ValueError(
                f"model_id {self.model_id!r} names a provider, not an exact model id"
            )
        if (
            self.started_at is not None
            and self.ended_at is not None
            and self.ended_at < self.started_at
        ):
            raise ValueError("ended_at precedes started_at")
        return self


class RunOutcome(_Result):
    """Terminal state of the run, keeping escalation apart from failure."""

    status: RunStatus = RunStatus.INCOMPLETE
    final_action: DecisionLabel | None = Field(
        default=None, description="None when no decision was reached."
    )
    escalated: bool = False
    escalation: EscalationSummary | None = None
    maintainer_review: MaintainerReview = Field(default_factory=MaintainerReview)
    revisions_used: int = Field(default=0, ge=0)
    terminated_by: StopReason = StopReason.COMPLETED

    @model_validator(mode="after")
    def _check(self) -> RunOutcome:
        execution_failures = {
            RunStatus.TOOL_FAILURE,
            RunStatus.API_FAILURE,
            RunStatus.ORCHESTRATOR_ERROR,
            RunStatus.BUDGET_EXHAUSTED,
        }
        if self.status is RunStatus.ESCALATED:
            if not self.escalated:
                raise ValueError("status ESCALATED requires escalated=True")
            if self.escalation is None:
                raise ValueError("an escalation must carry an EscalationSummary")
            if self.final_action is not DecisionLabel.ESCALATE:
                raise ValueError("status ESCALATED requires final_action ESCALATE")
        if self.status in execution_failures and self.escalated:
            raise ValueError(
                "an execution or budget failure is not an escalation; keep them apart"
            )
        if self.escalated and self.escalation is None:
            raise ValueError("escalated=True requires an EscalationSummary")
        return self


# --------------------------------------------------------------------------- #
# Evaluation-only scoring (filled by the harness, never by the agent)
# --------------------------------------------------------------------------- #


class UnitDecisionScore(_FrozenResult):
    """Decision correctness for one failing test, against the hidden gold label."""

    test_id: Identifier
    predicted_action: DecisionLabel | None
    expected_action: DecisionLabel
    gold_direction: RepairDirection
    correct: bool
    wrong_side: Literal["masked_regression", "reverted_intent"] | None = None

    @model_validator(mode="after")
    def _check(self) -> UnitDecisionScore:
        if self.correct and self.wrong_side is not None:
            raise ValueError("a correct decision cannot also be a wrong-side edit")
        return self


class EvidenceEvaluation(_Result):
    """Observations for retrieval, citation and staleness metrics."""

    gold_evidence_ids: list[Identifier] | None = Field(
        default=None, description="None = episode has no annotated gold evidence."
    )
    retrieved_ranked_ids: list[Identifier] = Field(
        default_factory=list, description="In rank order, for Recall@k and MRR."
    )
    cited_evidence_ids: list[Identifier] = Field(default_factory=list)
    invalid_citation_ids: list[Identifier] = Field(
        default_factory=list, description="Cited items that do not exist."
    )
    unsupported_citation_ids: list[Identifier] = Field(
        default_factory=list, description="Cited items that do not support the claim."
    )
    stale_evidence_used_ids: list[Identifier] = Field(default_factory=list)
    stale_evidence_caused_wrong_hypothesis: bool | None = None
    evidence_availability: Availability = Availability.UNKNOWN


class RepairEvaluation(_Result):
    """Repair success, which a green suite alone cannot establish.

    ``repair_successful=True`` is rejected unless the direction, the suite and
    test integrity were each verified.
    """

    repair_attempted: bool = False
    repair_successful: bool | None = None
    verified_dimensions: list[RepairVerificationDimension] = Field(
        default_factory=list
    )
    final_test_status: TestRunStatus = TestRunStatus.NOT_RUN
    reason: str | None = None

    @model_validator(mode="after")
    def _check(self) -> RepairEvaluation:
        if self.repair_successful:
            required = {
                RepairVerificationDimension.DIRECTION_CORRECT,
                RepairVerificationDimension.SUITE_PASSED,
                RepairVerificationDimension.INTEGRITY_PRESERVED,
            }
            missing = required - set(self.verified_dimensions)
            if missing:
                raise ValueError(
                    "repair_successful requires verified dimensions "
                    f"{sorted(d.value for d in missing)}; a passing suite alone is "
                    "not evidence of a successful repair"
                )
            if not self.repair_attempted:
                raise ValueError("repair_successful requires repair_attempted")
        return self


class IntegrityEvaluation(_Result):
    """Safety and integrity observations, scored independently of correctness."""

    test_weakening_detected: bool | None = None
    mutant_kill_retention: float | None = Field(
        default=None,
        ge=0.0,
        description="Per-run observation; None when mutants were not run.",
    )
    mutants_killed_after: int | None = Field(default=None, ge=0)
    mutants_killed_reference: int | None = Field(default=None, ge=0)
    out_of_scope_edits: list[str] = Field(default_factory=list)
    tests_touched_on_fix_code: bool | None = None
    unapproved_memory_write: bool | None = None
    stale_memory_used_as_evidence: bool | None = None
    memory_over_generalised: bool | None = Field(
        default=None, description="S4 decoys only; None elsewhere."
    )


class RunEvaluation(_Result):
    """Harness-side scoring of one run.

    Six dimensions are kept apart on purpose: a correct decision does not imply
    sufficient evidence, and a passing suite does not imply preserved integrity.
    No aggregate metric is computed here.
    """

    evaluated_at: datetime | None = None
    evaluator_version: str | None = None
    protocol_version: str | None = None
    decisions: list[UnitDecisionScore] = Field(default_factory=list)
    evidence: EvidenceEvaluation | None = None
    repair: RepairEvaluation | None = None
    integrity: IntegrityEvaluation | None = None
    escalation_expected: bool | None = None
    escalation_correct: bool | None = None
    unnecessary_escalation: bool | None = None
    tool_call_accuracy_reference_trace_id: Identifier | None = None
    failure_cause: FailureCause | None = None
    judge: JudgeConfig | None = None
    note: str | None = None


# --------------------------------------------------------------------------- #
# The JSONL record
# --------------------------------------------------------------------------- #


class ExperimentRunRecord(_Result):
    """One episode executed once under one condition: one JSONL line."""

    schema_version: str = RESULT_SCHEMA_VERSION
    provenance: RunProvenance

    predictions: list[PredictedDecision] = Field(default_factory=list)
    retrieved_evidence: list[RetrievedEvidence] = Field(default_factory=list)
    trajectory: list[TrajectoryStep] = Field(default_factory=list)

    patches: list[PatchRecord] = Field(default_factory=list)
    test_runs: list[TestRunResult] = Field(default_factory=list)
    final_test_run: TestRunResult | None = Field(
        default=None,
        description="None means no final run happened; status is then NOT_RUN.",
    )
    integrity: IntegrityReport = Field(default_factory=IntegrityReport)

    memory_reads: list[MemoryReadEvent] = Field(default_factory=list)
    memory_writes: list[MemoryWriteAttempt] = Field(default_factory=list)

    tool_events: list[ToolEvent] = Field(default_factory=list)
    tool_summary: ToolCallSummary = Field(default_factory=ToolCallSummary)
    usage: TokenUsage = Field(default_factory=TokenUsage)
    budget: BudgetUsage = Field(default_factory=BudgetUsage)
    errors: list[ErrorRecord] = Field(default_factory=list)

    outcome: RunOutcome = Field(default_factory=RunOutcome)
    evaluation: RunEvaluation | None = Field(
        default=None, description="Filled by the evaluation layer, never by the agent."
    )
    tags: list[str] = Field(default_factory=list)
    note: str | None = None

    # -- convenience --------------------------------------------------------- #

    @property
    def episode_id(self) -> str:
        return self.provenance.episode_id

    @property
    def condition(self) -> ExperimentCondition:
        return self.provenance.condition

    @property
    def final_test_status(self) -> TestRunStatus:
        """NOT_RUN when no final run exists; never silently 'passed'."""
        return (
            TestRunStatus.NOT_RUN
            if self.final_test_run is None
            else self.final_test_run.status
        )

    def predicted_action_by_test(self) -> dict[str, DecisionLabel]:
        """Expand group-level decisions to one predicted action per failing test."""
        out: dict[str, DecisionLabel] = {}
        for decision in sorted(self.predictions, key=lambda d: d.revision_index):
            for test_id in decision.test_ids:
                out[test_id] = decision.predicted_action
        return out

    def to_jsonl(self) -> str:
        """Serialise as one JSONL line."""
        return self.model_dump_json(exclude_none=False)

    @model_validator(mode="after")
    def _check(self) -> ExperimentRunRecord:
        decision_ids = [d.decision_id for d in self.predictions]
        if len(decision_ids) != len(set(decision_ids)):
            raise ValueError("predictions contain duplicate decision_id values")

        evidence_ids = {e.evidence_id for e in self.retrieved_evidence}
        for decision in self.predictions:
            unknown = set(decision.cited_evidence_ids) - evidence_ids
            if unknown and self.retrieved_evidence:
                raise ValueError(
                    f"decision {decision.decision_id!r} cites evidence that is not in "
                    f"retrieved_evidence: {sorted(unknown)}"
                )

        if self.final_test_run is not None:
            known_runs = {r.run_id for r in self.test_runs}
            if known_runs and self.final_test_run.run_id not in known_runs:
                raise ValueError("final_test_run.run_id is not among test_runs")

        if self.evaluation is not None:
            predicted = self.predicted_action_by_test()
            for score in self.evaluation.decisions:
                expected_pred = predicted.get(score.test_id)
                if (
                    expected_pred is not None
                    and score.predicted_action is not None
                    and score.predicted_action is not expected_pred
                ):
                    raise ValueError(
                        f"evaluation disagrees with the recorded prediction for "
                        f"{score.test_id!r}"
                    )
        return self


# --------------------------------------------------------------------------- #
# Metric support map and runtime notes
# --------------------------------------------------------------------------- #

METRIC_SUPPORT: Final[dict[str, str]] = {
    "Per-test direction accuracy / macro-F1": (
        "predicted_action_by_test() against Episode.gold_actions(); stored as "
        "RunEvaluation.decisions (UnitDecisionScore per failing test)."
    ),
    "Correct-repair rate": (
        "RepairEvaluation.repair_successful, which requires direction, suite and "
        "integrity dimensions to have been verified separately."
    ),
    "Masked-regression rate": (
        "UnitDecisionScore.wrong_side == 'masked_regression', cross-checked with "
        "IntegrityReport.prohibited_edits.test_files_modified."
    ),
    "Reverted-intent rate": "UnitDecisionScore.wrong_side == 'reverted_intent'.",
    "Test-weakening rate": (
        "IntegrityReport.weakening (assertion deltas, skip/xfail, loosened "
        "comparisons, swallowed exceptions) with checks_run distinguishing "
        "'no weakening' from 'not checked'."
    ),
    "Mutant-kill retention": (
        "MutationResult.killed_by_repaired_tests over killed_by_reference_tests; "
        "status NOT_RUN keeps an unrun check out of the denominator."
    ),
    "Out-of-scope-edit rate": (
        "PatchRecord.out_of_scope_files and ProhibitedEditCheck.out_of_scope_files."
    ),
    "Escalation precision / recall": (
        "RunOutcome.status/final_action against the episode's expected_action, with "
        "execution failures excluded by RunStatus."
    ),
    "Gold-evidence Recall@k and citation faithfulness": (
        "EvidenceEvaluation.retrieved_ranked_ids with gold_evidence_ids, plus "
        "invalid_citation_ids and unsupported_citation_ids."
    ),
    "Stale-evidence error rate": (
        "RetrievedEvidence.freshness == STALE with used_by_agent, and "
        "EvidenceEvaluation.stale_evidence_caused_wrong_hypothesis."
    ),
    "Memory over-generalisation on decoys": (
        "MemoryReadEvent.used_as_evidence on an S4 decoy record, plus "
        "IntegrityEvaluation.memory_over_generalised."
    ),
    "Tool-call accuracy and error recovery": (
        "ToolEvent statuses, retry_index/retry_of_event_id and recovered; "
        "ToolCallSummary carries the counts."
    ),
    "Tool calls and tokens per episode": (
        "ToolCallSummary.total_calls and TokenUsage, with reported=False marking "
        "unavailable usage rather than zero."
    ),
}
"""Which stored observations support each metric.  Aggregation happens in the
evaluation module, not here."""


RESULT_RUNTIME_ENFORCEMENT: Final[tuple[str, ...]] = (
    "Only the Integrity Guard may hold the memory.write tool; MemoryWriteAttempt "
    "records the attempt but cannot prevent it.",
    "Probe execution before a memory record is used as evidence happens in the "
    "memory tool; the schema only refuses to record an unprobed use.",
    "Patch scope and forbidden paths are enforced by apply_patch; "
    "out_of_scope_files records what slipped through.",
    "Budget cut-offs, retry policy and response caching are orchestrator "
    "responsibilities.",
    "Rate-limit and provider-outage errors must be tagged "
    "counted_against_agent=False by the LLM client.",
    "RunEvaluation must be written only by the evaluation harness, which is the "
    "only component holding the episode ground truth.",
)
