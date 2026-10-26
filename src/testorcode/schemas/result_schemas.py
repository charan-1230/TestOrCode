"""Run records for TestOrCode.

One ``ExperimentRunRecord`` is one JSONL line: one episode, executed once under
one condition, seed and model.  Repeated runs of an episode are separate records
sharing ``provenance.episode_id``.

These models store raw observations only.  No metric is computed here: the
evaluation module derives macro-F1, rates, Recall@k, kill retention and tool
counts across runs (see ``METRIC_SUPPORT``).

Distinctions enforced by validators, not comments:
unseeded vs seed 0 | zero tokens vs unreported | tests passed vs not run |
check passed vs check not run | patch failed to apply vs rejected by the Guard |
escalation vs tool/API/budget failure | green suite vs successful repair.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from testorcode.schemas.episode_schemas import (
    SCHEMA_VERSION,
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

RESULT_SCHEMA_VERSION: Final[str] = SCHEMA_VERSION


# --------------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------------- #


class ExperimentCondition(StrEnum):
    """Conditions defined by the project documents.

    The proposal and the five-day plan number the ablations A1 (no intent
    retrieval), A2 (no memory), A3 (no Integrity Guard); the detailed document
    renumbers them and adds coarse memory and merged roles.  The five-day plan
    numbering is canonical; the extra conditions keep distinct names.
    ``CORE_CONDITIONS`` are the five the plan requires to be runnable.
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


CORE_CONDITIONS: Final[tuple[ExperimentCondition, ...]] = (
    ExperimentCondition.FULL,
    ExperimentCondition.B1,
    ExperimentCondition.A1_NO_INTENT_RETRIEVAL,
    ExperimentCondition.A2_NO_MEMORY,
    ExperimentCondition.A3_NO_INTEGRITY_GUARD,
)
_NO_LLM_CONDITIONS: Final[frozenset[ExperimentCondition]] = frozenset(
    {
        ExperimentCondition.TRIVIAL_ALWAYS_UPDATE_TEST,
        ExperimentCondition.TRIVIAL_ALWAYS_FIX_CODE,
    }
)


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
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    INSUFFICIENT = "insufficient"
    UNKNOWN = "unknown"


class EvidenceFreshness(StrEnum):
    CURRENT = "current"
    STALE = "stale"
    UNKNOWN = "unknown"


class PatchApplicationStatus(StrEnum):
    NOT_ATTEMPTED = "not_attempted"
    APPLIED = "applied"
    FAILED_TO_APPLY = "failed_to_apply"
    BLOCKED_OUT_OF_SCOPE = "blocked_out_of_scope"
    REJECTED_BY_GUARD = "rejected_by_guard"
    REVERTED = "reverted"


class TestRunStatus(StrEnum):
    __test__ = False  # not a pytest class

    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"
    TIMEOUT = "timeout"
    COLLECTION_ERROR = "collection_error"
    NOT_RUN = "not_run"


class TestRunPhase(StrEnum):
    __test__ = False

    PRE_REPAIR = "pre_repair"
    POST_REPAIR = "post_repair"
    POST_REVISION = "post_revision"


_FINAL_PHASES: Final[frozenset[TestRunPhase]] = frozenset(
    {TestRunPhase.POST_REPAIR, TestRunPhase.POST_REVISION}
)


class GuardDecision(StrEnum):
    APPROVE = "approve"
    REJECT = "reject"
    ESCALATE = "escalate"
    NOT_RUN = "not_run"


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


class ErrorCategory(StrEnum):
    TOOL_ERROR = "tool_error"
    INJECTED_TOOL_FAULT = "injected_tool_fault"
    API_ERROR = "api_error"
    RATE_LIMIT = "rate_limit"
    PROVIDER_OUTAGE = "provider_outage"
    PATCH_ERROR = "patch_error"
    VALIDATION_ERROR = "validation_error"
    TIMEOUT = "timeout"
    ORCHESTRATOR_ERROR = "orchestrator_error"


class RunStatus(StrEnum):
    """Terminal status.  ESCALATED means every failing test was escalated.

    A mixed run with some escalated tests is COMPLETED; the per-test
    predictions say which.  TOOL_FAILURE, API_FAILURE, BUDGET_EXHAUSTED and
    ORCHESTRATOR_ERROR are execution failures and are never escalations.
    """

    COMPLETED = "completed"
    ESCALATED = "escalated"
    INCOMPLETE = "incomplete"
    BUDGET_EXHAUSTED = "budget_exhausted"
    TOOL_FAILURE = "tool_failure"
    API_FAILURE = "api_failure"
    ORCHESTRATOR_ERROR = "orchestrator_error"


class FailureCause(StrEnum):
    """One assigned cause per failed episode (detailed document, failure analysis)."""

    TRIAGE_ERROR = "triage_error"
    RETRIEVAL_OR_STALE_EVIDENCE = "retrieval_or_stale_evidence"
    REASONING_ERROR = "reasoning_error"
    REPAIR_ERROR = "repair_error"
    INTEGRITY_MISS = "integrity_miss"
    MEMORY_ERROR = "memory_error"
    TOOL_ERROR = "tool_error"
    API_ERROR = "api_error"


class RepairVerificationDimension(StrEnum):
    DIRECTION_CORRECT = "direction_correct"
    SUITE_PASSED = "suite_passed"
    INTEGRITY_PRESERVED = "integrity_preserved"
    SCOPE_RESPECTED = "scope_respected"


_PROVIDER_ONLY_IDS: Final[frozenset[str]] = frozenset(
    {"nvidia", "nim", "mistral", "groq", "google", "gemini", "openai", "anthropic",
     "voyage", "qwen", "deepseek"}
)


class _Result(BaseModel):
    """Mutable base: run records are assembled incrementally."""

    model_config = ConfigDict(
        extra="forbid", validate_assignment=True, str_strip_whitespace=True,
        use_enum_values=False,
    )


class _FrozenResult(_Result):
    model_config = ConfigDict(
        extra="forbid", frozen=True, str_strip_whitespace=True, use_enum_values=False
    )


# --------------------------------------------------------------------------- #
# Evidence and decisions
# --------------------------------------------------------------------------- #


class RetrievedEvidence(_Result):
    """One retrieved item.  ``score is None`` means no score was reported, never 0.0.

    ``support`` is the agent's reading of the item and is kept apart from the
    item's identity and rank.  Which items a decision relies on is stored on the
    decision (``cited_evidence_ids``), not duplicated here.  For memory items,
    ``source_ref`` is the memory record id.
    """

    evidence_id: Identifier
    source_type: EvidenceSourceType
    source_ref: str = Field(min_length=1)
    commit_sha: str | None = None
    line_start: int | None = Field(default=None, ge=1)
    line_end: int | None = Field(default=None, ge=1)
    snippet: str | None = None
    retrieval_query: str | None = None
    retrieval_method: RetrievalMethod | None = None
    rank: int | None = Field(default=None, ge=1)
    score: float | None = None
    support: SupportStatus = SupportStatus.UNKNOWN
    freshness: EvidenceFreshness = EvidenceFreshness.UNKNOWN


class PredictedDecision(_Result):
    """The agent's decision for one failing test (final decision after any revision)."""

    test_id: Identifier
    group_id: Identifier | None = None
    predicted_action: DecisionLabel
    hypothesis: HypothesisKind | None = Field(
        default=None, description="None = no hypothesis produced (e.g. B0, trivial)."
    )
    predicted_behaviour: str | None = None
    cited_evidence_ids: list[Identifier] = Field(
        default_factory=list, description="Evidence the decision relies on."
    )
    contradicting_evidence_ids: list[Identifier] = Field(default_factory=list)
    rationale: str | None = Field(default=None, description="Short auditable justification.")
    decided_by: AgentRole = AgentRole.INTENT_ANALYST


# --------------------------------------------------------------------------- #
# Patches and test runs
# --------------------------------------------------------------------------- #


class PatchRecord(_Result):
    patch_id: Identifier
    scope: PatchScope = Field(description="Declared scope of the patch.")
    test_ids: list[Identifier] = Field(default_factory=list, description="Tests it addresses.")
    target_files: list[str] = Field(default_factory=list)
    unified_diff: str | None = None
    application_status: PatchApplicationStatus = PatchApplicationStatus.NOT_ATTEMPTED
    apply_error: str | None = None
    out_of_scope_files: list[str] = Field(default_factory=list)
    revision_index: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _check(self) -> PatchRecord:
        if self.application_status is PatchApplicationStatus.FAILED_TO_APPLY and not self.apply_error:
            raise ValueError(
                "failed_to_apply must record apply_error (a patch that could not be "
                "applied is not a patch the Guard rejected)"
            )
        if (
            self.application_status is PatchApplicationStatus.BLOCKED_OUT_OF_SCOPE
            and not self.out_of_scope_files
        ):
            raise ValueError("blocked_out_of_scope must list the offending files")
        return self


class TestRunResult(_Result):
    """One pytest execution.  Counts stay ``None`` when unknown or not run."""

    __test__ = False

    run_id: Identifier
    phase: TestRunPhase
    status: TestRunStatus = TestRunStatus.NOT_RUN
    selector: str | None = None
    passed: int | None = Field(default=None, ge=0)
    failed: int | None = Field(default=None, ge=0)
    errors: int | None = Field(default=None, ge=0)
    skipped: int | None = Field(default=None, ge=0)
    failing_test_ids: list[Identifier] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check(self) -> TestRunResult:
        counts = (self.passed, self.failed, self.errors, self.skipped)
        if self.status is TestRunStatus.NOT_RUN:
            if any(c is not None for c in counts) or self.failing_test_ids:
                raise ValueError(
                    "status NOT_RUN must not carry counts or failing ids: a run that "
                    "did not happen is not a run with zero failures"
                )
        if self.status is TestRunStatus.PASSED and (
            self.failing_test_ids or (self.failed or 0) > 0 or (self.errors or 0) > 0
        ):
            raise ValueError("status PASSED contradicts failed/errors counts or failing ids")
        if (
            self.status is TestRunStatus.FAILED
            and self.failed == 0
            and not (self.errors or 0)
            and not self.failing_test_ids
        ):
            raise ValueError("status FAILED contradicts failed == 0")
        return self


# --------------------------------------------------------------------------- #
# Integrity
# --------------------------------------------------------------------------- #


class TestWeakeningFindings(_Result):
    """Deterministic weakening checks.

    ``checks_run=False`` (A3) leaves every other field empty: unrun is not
    'no weakening'.
    """

    __test__ = False

    checks_run: bool = False
    assertions_before: int | None = Field(default=None, ge=0)
    assertions_after: int | None = Field(default=None, ge=0)
    removed_assertions: int | None = Field(default=None, ge=0)
    added_skip_markers: int | None = Field(default=None, ge=0)
    added_xfail_markers: int | None = Field(default=None, ge=0)
    loosened_comparisons: int | None = Field(default=None, ge=0)
    swallowed_exceptions: int | None = Field(default=None, ge=0)
    deleted_tests: int | None = Field(default=None, ge=0)
    weakening_detected: bool | None = None

    @model_validator(mode="after")
    def _check(self) -> TestWeakeningFindings:
        counts = [
            self.assertions_before, self.assertions_after, self.removed_assertions,
            self.added_skip_markers, self.added_xfail_markers, self.loosened_comparisons,
            self.swallowed_exceptions, self.deleted_tests,
        ]
        if not self.checks_run:
            if self.weakening_detected is not None or any(c is not None for c in counts):
                raise ValueError(
                    "weakening findings must all be None when checks_run is False"
                )
            return self
        if self.weakening_detected is None:
            raise ValueError("checks_run=True requires weakening_detected True/False")
        positive = any(
            (c or 0) > 0 for c in counts[2:]
        )
        if self.weakening_detected is False and positive:
            raise ValueError(
                "weakening_detected=False contradicts a positive weakening count"
            )
        if self.assertions_before is not None and self.assertions_after is not None:
            if self.assertions_after < self.assertions_before:
                if self.weakening_detected is False:
                    raise ValueError(
                        f"weakening_detected=False contradicts an apparent decrease in assertions "
                        f"({self.assertions_before} -> {self.assertions_after})"
                    )
                if self.removed_assertions == 0:
                    raise ValueError(
                        f"removed_assertions=0 contradicts an apparent decrease in assertions "
                        f"({self.assertions_before} -> {self.assertions_after})"
                    )
        return self


class ProhibitedEditCheck(_Result):
    """Files actually modified, against the EditPolicy (``out_of_scope_files`` covers
    CI config, dependency files and conftest.py)."""

    checks_run: bool = False
    files_modified: list[str] = Field(default_factory=list)
    out_of_scope_files: list[str] = Field(default_factory=list)
    test_files_modified: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check(self) -> ProhibitedEditCheck:
        lists = (self.files_modified, self.out_of_scope_files, self.test_files_modified)
        if not self.checks_run and any(lists):
            raise ValueError("edit lists must be empty when checks_run is False")
        modified = set(self.files_modified)
        for name, sub in (
            ("out_of_scope_files", self.out_of_scope_files),
            ("test_files_modified", self.test_files_modified),
        ):
            if not set(sub) <= modified:
                raise ValueError(f"{name} must be a subset of files_modified")
        return self


class MutationResult(_Result):
    """Aggregate mutation counts.  Kill retention is derived by the evaluation layer;
    an unrun check is ``NOT_RUN``, never zero kills."""

    status: TestRunStatus = TestRunStatus.NOT_RUN
    mutants_total: int | None = Field(default=None, ge=0)
    killed_by_reference_tests: int | None = Field(default=None, ge=0)
    killed_by_repaired_tests: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _check(self) -> MutationResult:
        if self.status is TestRunStatus.NOT_RUN and any(
            v is not None
            for v in (self.mutants_total, self.killed_by_reference_tests, self.killed_by_repaired_tests)
        ):
            raise ValueError("a NOT_RUN mutation result must not carry counts")
        for name in ("killed_by_reference_tests", "killed_by_repaired_tests"):
            v = getattr(self, name)
            if v is not None and self.mutants_total is not None and v > self.mutants_total:
                raise ValueError(f"{name} exceeds mutants_total")
        return self


class IntegrityReport(_Result):
    """Integrity Guard output.  With the Guard disabled (A3) everything stays unrun."""

    guard_ran: bool = False
    guard_decision: GuardDecision = GuardDecision.NOT_RUN
    weakening: TestWeakeningFindings = Field(default_factory=TestWeakeningFindings)
    prohibited_edits: ProhibitedEditCheck = Field(default_factory=ProhibitedEditCheck)
    mutation: MutationResult = Field(default_factory=MutationResult)
    tests_untouched_for_fix_code: bool | None = None
    evidence_requirement_met: bool | None = Field(
        default=None, description="UPDATE_TEST needs cited positive evidence; None = not assessed."
    )
    rejection_reasons: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check(self) -> IntegrityReport:
        if self.guard_ran == (self.guard_decision is GuardDecision.NOT_RUN):
            raise ValueError("guard_ran must be True exactly when guard_decision is not NOT_RUN")
        if not self.guard_ran:
            if (
                self.weakening.checks_run
                or self.prohibited_edits.checks_run
                or self.mutation.status is not TestRunStatus.NOT_RUN
                or self.tests_untouched_for_fix_code is not None
                or self.evidence_requirement_met is not None
            ):
                raise ValueError("Guard did not run, so no integrity check may be recorded")
        if self.guard_decision is GuardDecision.REJECT and not self.rejection_reasons:
            raise ValueError("a rejection must record at least one reason")
        if self.guard_decision is GuardDecision.APPROVE and (
            self.weakening.weakening_detected is True
            or self.prohibited_edits.out_of_scope_files
            or self.tests_untouched_for_fix_code is False
            or self.evidence_requirement_met is False
        ):
            raise ValueError("Guard APPROVE contradicts its own recorded findings")
        return self


# --------------------------------------------------------------------------- #
# Memory and escalation
# --------------------------------------------------------------------------- #


class MemoryReadEvent(_Result):
    """A memory record the agent consulted and whether it counted as evidence."""

    record_id: Identifier
    probe_status: ProbeStatus = ProbeStatus.NOT_RUN
    used_as_evidence: bool = False
    blocked_reason: Literal["probe_failed", "probe_not_run", "superseded", "out_of_scope"] | None = None

    @model_validator(mode="after")
    def _check(self) -> MemoryReadEvent:
        if self.used_as_evidence:
            if self.probe_status is not ProbeStatus.PASSED:
                raise ValueError(
                    "a memory record may be used as evidence only when its probe passed "
                    "on the current code"
                )
            if self.blocked_reason is not None:
                raise ValueError("used_as_evidence contradicts blocked_reason")
        return self


class MemoryWriteAttempt(_Result):
    """An attempt to persist a record.  An unapproved attempt is an unsafe action and
    is logged with ``written=False`` so the evaluation layer can count it."""

    record_id: Identifier
    approved: bool = False
    written: bool = False
    written_by: AgentRole = AgentRole.INTEGRITY_GUARD
    blocked_reason: Literal["not_approved", "guard_rejected", "memory_disabled", "write_error"] | None = None

    @model_validator(mode="after")
    def _check(self) -> MemoryWriteAttempt:
        if self.written and not self.approved:
            raise ValueError("a memory record may be written only after maintainer approval")
        if self.written and self.written_by is not AgentRole.INTEGRITY_GUARD:
            raise ValueError("only the Integrity Guard may write memory")
        if not self.written and self.blocked_reason is None:
            raise ValueError("a write that did not happen must record a blocked_reason")
        return self


class EscalationSummary(_Result):
    """Case summary for the maintainer (proposal, Human escalation)."""

    trigger: Literal[
        "insufficient_evidence", "contradictory_evidence", "guard_rejection_after_revision"
    ]
    failing_test_ids: list[Identifier] = Field(min_length=1)
    evidence_found_ids: list[Identifier] = Field(default_factory=list)
    evidence_missing: list[str] = Field(default_factory=list)
    competing_hypotheses: list[str] = Field(default_factory=list)
    summary: str = Field(min_length=1)


# --------------------------------------------------------------------------- #
# Tool calls, usage, errors
# --------------------------------------------------------------------------- #


class ToolEvent(_Result):
    """One tool call, in order.  Reconstructs the trajectory and tool metrics."""

    event_id: Identifier
    step_index: int = Field(ge=0)
    tool: ToolName
    called_by: AgentRole = AgentRole.ORCHESTRATOR
    arguments: dict[str, Any] = Field(default_factory=dict)
    status: ToolCallStatus = ToolCallStatus.OK
    fault_kind: ToolFaultKind | None = Field(
        default=None, description="Set exactly when status is INJECTED_FAULT."
    )
    error_message: str | None = None
    recovered: bool | None = Field(
        default=None, description="On a failed call: did the run later succeed at the task?"
    )
    duration_ms: float | None = Field(default=None, ge=0.0)

    @model_validator(mode="after")
    def _check(self) -> ToolEvent:
        if (self.status is ToolCallStatus.INJECTED_FAULT) != (self.fault_kind is not None):
            raise ValueError("fault_kind must be set exactly when status is INJECTED_FAULT")
        if self.status is not ToolCallStatus.OK and not self.error_message:
            raise ValueError("a failed tool call must record an error_message")
        if self.status is ToolCallStatus.OK and self.recovered is not None:
            raise ValueError("recovered applies to failed calls only")
        return self


class TokenUsage(_Result):
    """``reported=False`` = the provider returned no usage (counts stay None).
    Zero usage is ``reported=True`` with explicit zeros."""

    reported: bool = False
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    llm_calls: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _check(self) -> TokenUsage:
        counts = (self.input_tokens, self.output_tokens, self.total_tokens)
        if not self.reported and any(c is not None for c in counts):
            raise ValueError("reported=False: leave token counts None rather than writing zeros")
        if self.reported and all(c is None for c in counts):
            raise ValueError("reported=True requires at least one token count")
        return self


class ErrorRecord(_Result):
    error_id: Identifier
    category: ErrorCategory
    message: str = Field(min_length=1)
    operation: str | None = None
    tool: ToolName | None = None
    retries: int = Field(default=0, ge=0)
    recovered: bool = False
    fatal: bool = False
    counted_against_agent: bool = Field(
        default=True,
        description="False for rate limits and provider outages: logged, never scored against the agent.",
    )

    @model_validator(mode="after")
    def _check(self) -> ErrorRecord:
        if (
            self.category in (ErrorCategory.RATE_LIMIT, ErrorCategory.PROVIDER_OUTAGE)
            and self.counted_against_agent
        ):
            raise ValueError("rate limits and provider outages must not be counted against the agent")
        if self.recovered and self.fatal:
            raise ValueError("an error cannot be both recovered and fatal")
        return self


# --------------------------------------------------------------------------- #
# Provenance and outcome
# --------------------------------------------------------------------------- #


class RunProvenance(_FrozenResult):
    """What is needed to group and reproduce one run.  Store versions, never secrets."""

    run_id: Identifier
    episode_id: Identifier
    condition: ExperimentCondition
    seed: int | None = Field(..., description="Required: None = unseeded, not seed 0.")
    model_id: str | None = Field(
        default=None,
        description="Exact dated model id.  Required unless the condition makes no LLM calls.",
    )
    provider: str | None = None
    dataset_version: str | None = None
    split: Literal["dev", "test", "pilot"] | None = None
    code_revision: str | None = Field(default=None, description="Git sha of this repository.")
    prompt_template_version: str | None = None
    agent_config_version: str | None = None
    retrieval_config_version: str | None = None
    index_version: str | None = None
    environment_ref: str | None = None
    started_at: datetime | None = None
    ended_at: datetime | None = None

    @model_validator(mode="after")
    def _check(self) -> RunProvenance:
        if self.model_id is None:
            if self.condition not in _NO_LLM_CONDITIONS:
                raise ValueError(
                    f"model_id is required for condition {self.condition.value}; only the "
                    "trivial always-UPDATE_TEST / always-FIX_CODE baselines make no LLM calls"
                )
        else:
            if len(self.model_id) < 2 or self.model_id.lower() in _PROVIDER_ONLY_IDS:
                raise ValueError(f"model_id {self.model_id!r} is a provider name, not an exact model id")
        if self.started_at and self.ended_at and self.ended_at < self.started_at:
            raise ValueError("ended_at precedes started_at")
        return self


class RunOutcome(_Result):
    status: RunStatus = RunStatus.INCOMPLETE
    escalation: EscalationSummary | None = None
    maintainer_approved: bool | None = Field(
        default=None, description="Simulated maintainer verdict; None = not reviewed."
    )
    revisions_used: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _check(self) -> RunOutcome:
        if self.status is RunStatus.ESCALATED and self.escalation is None:
            raise ValueError("status ESCALATED requires an EscalationSummary")
        return self


# --------------------------------------------------------------------------- #
# Harness-side evaluation (never written by the agent)
# --------------------------------------------------------------------------- #


class UnitDecisionScore(_FrozenResult):
    """Decision correctness for one failing test against the hidden gold label."""

    test_id: Identifier
    predicted_action: DecisionLabel | None
    expected_action: DecisionLabel
    gold_direction: RepairDirection
    correct: bool
    hypothesis_correct: bool | None = Field(default=None, description="None = not assessed.")
    wrong_side: Literal["masked_regression", "reverted_intent"] | None = None

    @model_validator(mode="after")
    def _check(self) -> UnitDecisionScore:
        if self.correct != (self.predicted_action is self.expected_action):
            raise ValueError("correct must equal (predicted_action == expected_action)")
        if self.correct and self.wrong_side is not None:
            raise ValueError("a correct decision cannot also be a wrong-side edit")
        return self


class EvidenceEvaluation(_Result):
    """Citation and staleness observations.  Recall@k comes from retrieved_evidence ranks
    plus the episode's gold ids; no gold data is copied here."""

    invalid_citation_ids: list[Identifier] = Field(
        default_factory=list, description="Cited ids that do not exist."
    )
    unsupported_citation_ids: list[Identifier] = Field(
        default_factory=list, description="Cited items that do not support the claim."
    )
    stale_evidence_used_ids: list[Identifier] = Field(default_factory=list)
    stale_evidence_caused_wrong_hypothesis: bool | None = None


class RepairEvaluation(_Result):
    """``repair_successful=True`` needs direction, suite and integrity each verified;
    a green suite alone is rejected."""

    repair_attempted: bool = False
    repair_successful: bool | None = None
    verified_dimensions: list[RepairVerificationDimension] = Field(default_factory=list)
    final_test_status: TestRunStatus = TestRunStatus.NOT_RUN

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
                    f"repair_successful requires verified dimensions "
                    f"{sorted(d.value for d in missing)}; a passing suite alone is not "
                    "evidence of a successful repair"
                )
            if not self.repair_attempted:
                raise ValueError("repair_successful requires repair_attempted")
            if self.final_test_status is not TestRunStatus.PASSED:
                raise ValueError("repair_successful requires final_test_status PASSED")
        return self


class IntegrityEvaluation(_Result):
    """Harness-measured safety observations, independent of the Guard's own report
    (so A3, which has no Guard, can still be scored)."""

    test_weakening_detected: bool | None = None
    mutants_killed_after: int | None = Field(default=None, ge=0)
    mutants_killed_reference: int | None = Field(default=None, ge=0)
    out_of_scope_edits: list[str] = Field(default_factory=list)
    tests_touched_on_fix_code: bool | None = None
    unapproved_memory_write: bool | None = None
    stale_memory_used_as_evidence: bool | None = None
    memory_over_generalised: bool | None = Field(default=None, description="S4 only.")


class RunEvaluation(_Result):
    """Six dimensions, kept apart on purpose: decision, evidence, citations, repair,
    integrity, scope/escalation.  None is inferred from another."""

    protocol_version: str | None = None
    decisions: list[UnitDecisionScore] = Field(default_factory=list)
    evidence: EvidenceEvaluation | None = None
    repair: RepairEvaluation | None = None
    integrity: IntegrityEvaluation | None = None
    escalation_correct: bool | None = None
    unnecessary_escalation: bool | None = None
    failure_cause: FailureCause | None = None

    @model_validator(mode="after")
    def _check(self) -> RunEvaluation:
        if self.repair and self.repair.repair_successful:
            if self.integrity is None:
                raise ValueError(
                    "repair_successful requires an independent harness integrity evaluation"
                )
            if self.integrity.test_weakening_detected is True:
                raise ValueError("repair_successful contradicts test_weakening_detected=True")
            if self.integrity.out_of_scope_edits:
                raise ValueError("repair_successful contradicts recorded out-of-scope edits")
            if self.integrity.tests_touched_on_fix_code is True:
                raise ValueError("repair_successful contradicts tests_touched_on_fix_code=True")
            if self.integrity.unapproved_memory_write is True:
                raise ValueError("repair_successful contradicts unapproved_memory_write=True")
            if self.integrity.stale_memory_used_as_evidence is True:
                raise ValueError("repair_successful contradicts stale_memory_used_as_evidence=True")
        return self


# --------------------------------------------------------------------------- #
# The JSONL record
# --------------------------------------------------------------------------- #


class ExperimentRunRecord(_Result):
    """One episode executed once: one JSONL line."""

    schema_version: str = RESULT_SCHEMA_VERSION
    provenance: RunProvenance
    predictions: list[PredictedDecision] = Field(default_factory=list)
    retrieved_evidence: list[RetrievedEvidence] = Field(default_factory=list)
    patches: list[PatchRecord] = Field(default_factory=list)
    test_runs: list[TestRunResult] = Field(default_factory=list)
    final_test_status: TestRunStatus = Field(
        default=TestRunStatus.NOT_RUN,
        description="Must equal the last post-repair test run, or NOT_RUN if there is none.",
    )
    integrity: IntegrityReport = Field(default_factory=IntegrityReport)
    memory_reads: list[MemoryReadEvent] = Field(default_factory=list)
    memory_writes: list[MemoryWriteAttempt] = Field(default_factory=list)
    tool_events: list[ToolEvent] = Field(default_factory=list)
    usage: TokenUsage = Field(default_factory=TokenUsage)
    errors: list[ErrorRecord] = Field(default_factory=list)
    outcome: RunOutcome = Field(default_factory=RunOutcome)
    evaluation: RunEvaluation | None = Field(
        default=None, description="Filled by the evaluation harness only."
    )

    @property
    def episode_id(self) -> str:
        return self.provenance.episode_id

    @property
    def condition(self) -> ExperimentCondition:
        return self.provenance.condition

    def predicted_actions(self) -> dict[str, DecisionLabel]:
        return {p.test_id: p.predicted_action for p in self.predictions}

    def unretrieved_citations(self) -> dict[str, list[str]]:
        """Per test, cited ids absent from ``retrieved_evidence``.

        Not rejected at validation: a hallucinated citation is an observation the
        citation-faithfulness metric needs, and rejecting it would drop the run.
        """
        known = {e.evidence_id for e in self.retrieved_evidence}
        out = {
            p.test_id: sorted(set(p.cited_evidence_ids) - known)
            for p in self.predictions
        }
        return {k: v for k, v in out.items() if v}

    def to_jsonl(self) -> str:
        return self.model_dump_json()

    @model_validator(mode="after")
    def _check(self) -> ExperimentRunRecord:
        ids = [p.test_id for p in self.predictions]
        if len(ids) != len(set(ids)):
            raise ValueError("predictions contain more than one decision for the same test_id")

        run_ids = [r.run_id for r in self.test_runs]
        if len(run_ids) != len(set(run_ids)):
            raise ValueError("duplicate test run_id")
        final_runs = [r for r in self.test_runs if r.phase in _FINAL_PHASES]
        expected = final_runs[-1].status if final_runs else TestRunStatus.NOT_RUN
        if self.final_test_status is not expected:
            raise ValueError(
                f"final_test_status {self.final_test_status.value!r} disagrees with the "
                f"recorded test runs (expected {expected.value!r})"
            )

        if self.outcome.status is RunStatus.ESCALATED:
            if not self.predictions or any(
                p.predicted_action is not DecisionLabel.ESCALATE for p in self.predictions
            ):
                raise ValueError("status ESCALATED requires every prediction to be ESCALATE")
        if self.outcome.escalation is not None and not any(
            p.predicted_action is DecisionLabel.ESCALATE for p in self.predictions
        ):
            raise ValueError("an escalation summary requires at least one ESCALATE prediction")

        if self.evaluation is not None:
            predicted = self.predicted_actions()
            for s in self.evaluation.decisions:
                if s.predicted_action != predicted.get(s.test_id):
                    raise ValueError(
                        f"evaluation disagrees with the recorded prediction for {s.test_id!r}"
                    )
            if (
                self.evaluation.repair is not None
                and self.evaluation.repair.final_test_status is not self.final_test_status
            ):
                raise ValueError("evaluation.repair.final_test_status disagrees with the run record")
        return self


# --------------------------------------------------------------------------- #
# Metric support and runtime notes
# --------------------------------------------------------------------------- #

METRIC_SUPPORT: Final[dict[str, str]] = {
    "Direction accuracy / macro-F1": "predictions[].predicted_action vs Episode.gold_actions(); evaluation.decisions",
    "Correct-repair rate": "evaluation.repair (direction + suite + integrity verified separately)",
    "Masked-regression / reverted-intent rate": "evaluation.decisions[].wrong_side; patches[].scope; integrity.prohibited_edits.test_files_modified",
    "Test-weakening rate": "integrity.weakening (Guard) and evaluation.integrity.test_weakening_detected (harness)",
    "Mutant-kill retention": "integrity.mutation or evaluation.integrity.mutants_killed_after / mutants_killed_reference",
    "Out-of-scope-edit rate": "patches[].out_of_scope_files; integrity.prohibited_edits.out_of_scope_files",
    "Escalation precision / recall": "predictions[].predicted_action and outcome.status vs gold; execution failures excluded by RunStatus",
    "Gold-evidence Recall@k, citation faithfulness": "retrieved_evidence[].rank + episode gold ids; evaluation.evidence.invalid/unsupported_citation_ids",
    "Stale-evidence error rate": "retrieved_evidence[].freshness with predictions[].cited_evidence_ids; evaluation.evidence",
    "Memory over-generalisation (decoys)": "memory_reads[].used_as_evidence on the S4 decoy; evaluation.integrity.memory_over_generalised",
    "Tool-call accuracy / invalid-call / recovery": "tool_events[].status, .arguments, .recovered (reference traces: see limitations)",
    "Tool calls and tokens per episode": "len(tool_events) and usage (reported=False means unavailable, not zero)",
    "Per-test hypothesis accuracy": "predictions[].hypothesis; evaluation.decisions[].hypothesis_correct",
}

RESULT_RUNTIME_ENFORCEMENT: Final[tuple[str, ...]] = (
    "Memory write authority and probe execution are enforced by the memory tool; the schema only refuses to record an invalid use.",
    "Patch scope and forbidden paths are enforced by apply_patch; out_of_scope_files records what slipped through.",
    "Budgets, retries and response caching are orchestrator/LLM-client duties.",
    "The LLM client must tag rate limits and provider outages counted_against_agent=False.",
    "Only the evaluation harness, which holds the ground truth, may fill RunRecord.evaluation.",
)
