"""Runtime contracts exchanged between agents, tools and the orchestrator.

``episode_schemas`` (dataset side) and ``result_schemas`` (run-log side) already
define the shared vocabulary: ``DecisionLabel``, ``FailureGroup``,
``EvidenceItem``, ``RetrievedEvidence``, ``MemoryRecord``, ``IntegrityReport``,
``ExperimentRunRecord`` and ``RunEvaluation``.  This module adds only the
agent-to-agent messages that were missing, and reuses those types instead of
redefining them.

Mapping of the planned contract names onto the existing types::

    Decision          -> episode_schemas.DecisionLabel
    Hypothesis        -> episode_schemas.HypothesisKind (+ text, see ``Hypothesis``)
    Evidence          -> episode_schemas.EvidenceItem (corpus) /
                         result_schemas.RetrievedEvidence (retrieval result)
    EpisodeResult     -> result_schemas.ExperimentRunRecord
    EvaluationResult  -> result_schemas.RunEvaluation

Pydantic checks shapes and cross-field consistency only.  Sandboxing, path
policy, memory-write authority and "the cited id really was retrieved" are
runtime duties; ``unknown_evidence_ids`` is the helper for the last one.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from testorcode.schemas.episode_schemas import (
    CommitSha,
    DecisionLabel,
    EditPolicy,
    EpisodeBudget,
    EpisodeInput,
    FailingTest,
    FailureGroup,
    HypothesisKind,
    Identifier,
    MemoryRecord,
    PatchScope,
    RepairDirection,
    RepoRef,
    ToolName,
)
from testorcode.schemas.result_schemas import (
    GuardDecision,
    IntegrityReport,
    ProbeStatus,
    TestRunPhase,
    TestRunResult,
    TestRunStatus,
)


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


# --------------------------------------------------------------------------- #
# Episode context
# --------------------------------------------------------------------------- #


class EpisodeContext(_Contract):
    """What every component of one episode needs to know; built once by the orchestrator."""

    episode_id: Identifier
    repo: RepoRef
    worktree_path: Path = Field(description="Isolated checkout for this episode.")
    commit_sha: CommitSha = Field(description="Current commit; also the history cutoff.")
    edit_policy: EditPolicy = Field(default_factory=EditPolicy)
    budget: EpisodeBudget = Field(default_factory=EpisodeBudget)
    allowed_tools: tuple[ToolName, ...] = ()

    @classmethod
    def from_episode_input(cls, episode: EpisodeInput, worktree_path: Path) -> EpisodeContext:
        return cls(
            episode_id=episode.episode_id,
            repo=episode.repo,
            worktree_path=worktree_path,
            commit_sha=episode.ci_event.commit.sha,
            edit_policy=episode.edit_policy,
            budget=episode.budget,
            allowed_tools=tuple(episode.allowed_tools),
        )


# --------------------------------------------------------------------------- #
# Test execution (tools/pytest_runner output)
# --------------------------------------------------------------------------- #


class PytestExecution(_Contract):
    """Outcome of one pytest invocation, as returned by ``tools.pytest_runner``.

    Separates the three ways a run can go wrong: tests *failed* (``FAILED``),
    pytest could not run them (``ERROR`` / ``COLLECTION_ERROR``), and the run
    was killed (``TIMEOUT``).  ``to_record`` converts it to the run-log form.
    """

    __test__ = False

    selectors: tuple[str, ...] = Field(description="Node ids or paths passed to pytest.")
    command: tuple[str, ...] = Field(description="Argument vector actually executed.")
    status: TestRunStatus
    exit_code: int | None = Field(default=None, description="None when killed or never started.")
    timed_out: bool = False
    timeout_seconds: int | None = Field(default=None, gt=0)
    duration_seconds: float | None = Field(default=None, ge=0)
    passed: int | None = Field(default=None, ge=0)
    failed: int | None = Field(default=None, ge=0)
    errors: int | None = Field(default=None, ge=0)
    skipped: int | None = Field(default=None, ge=0)
    failures: tuple[FailingTest, ...] = ()
    error_message: str | None = Field(
        default=None, description="Runner-level problem (collection error, bad selector, crash)."
    )

    @model_validator(mode="after")
    def _check(self) -> PytestExecution:
        if self.status is TestRunStatus.NOT_RUN:
            raise ValueError("a PytestExecution describes a run that happened; NOT_RUN is invalid")
        if self.timed_out != (self.status is TestRunStatus.TIMEOUT):
            raise ValueError("timed_out must be True exactly when status is TIMEOUT")
        if self.status is TestRunStatus.TIMEOUT and self.exit_code == 0:
            raise ValueError("a timed-out run cannot have exit_code 0")
        if self.status is TestRunStatus.PASSED and self.exit_code not in (None, 0):
            raise ValueError("status PASSED contradicts a non-zero exit_code")
        if self.status in (TestRunStatus.ERROR, TestRunStatus.COLLECTION_ERROR) and not self.error_message:
            raise ValueError("ERROR and COLLECTION_ERROR runs must record an error_message")
        return self

    @property
    def failing_test_ids(self) -> list[str]:
        return [f.test_id for f in self.failures]

    def to_record(self, run_id: str, phase: TestRunPhase) -> TestRunResult:
        """Run-log form (``result_schemas.TestRunResult``)."""
        return TestRunResult(
            run_id=run_id,
            phase=phase,
            status=self.status,
            selector=" ".join(self.selectors) or None,
            passed=self.passed,
            failed=self.failed,
            errors=self.errors,
            skipped=self.skipped,
            failing_test_ids=self.failing_test_ids,
        )


# --------------------------------------------------------------------------- #
# Git tools
# --------------------------------------------------------------------------- #


class CommitInfo(_Contract):
    """One commit visible to the episode (never newer than the history cutoff)."""

    sha: CommitSha
    message: str
    committed_at: datetime | None = None
    changed_files: tuple[str, ...] = ()


# --------------------------------------------------------------------------- #
# Triage -> Intent Analyst
# --------------------------------------------------------------------------- #


class FailureGroupDetail(_Contract):
    """A ``FailureGroup`` plus everything the Intent Analyst needs about it."""

    group: FailureGroup
    failing_tests: tuple[FailingTest, ...] = Field(min_length=1)
    affected_files: tuple[str, ...] = ()
    context_commits: tuple[CommitInfo, ...] = Field(
        default=(), description="Relevant history, at or before the current commit."
    )
    diff_context: str | None = Field(default=None, description="Relevant part of the CI commit diff.")

    @model_validator(mode="after")
    def _check(self) -> FailureGroupDetail:
        known = {t.test_id for t in self.failing_tests}
        if known != set(self.group.test_ids):
            raise ValueError("failing_tests must match group.test_ids exactly")
        return self


class TriageResult(_Contract):
    groups: tuple[FailureGroupDetail, ...]
    unstable_test_ids: tuple[Identifier, ...] = Field(
        default=(), description="Failed the stability check; excluded from groups."
    )
    baseline_run: PytestExecution


# --------------------------------------------------------------------------- #
# Intent Analyst
# --------------------------------------------------------------------------- #


class Hypothesis(_Contract):
    kind: HypothesisKind
    explanation: str = Field(min_length=1, max_length=2000)


_ALLOWED_DECISIONS: dict[HypothesisKind, frozenset[DecisionLabel]] = {
    HypothesisKind.OUTDATED_TEST: frozenset({DecisionLabel.UPDATE_TEST, DecisionLabel.ESCALATE}),
    HypothesisKind.REGRESSION: frozenset({DecisionLabel.FIX_CODE, DecisionLabel.ESCALATE}),
    HypothesisKind.UNKNOWN: frozenset({DecisionLabel.ESCALATE}),
}


class IntentAnalysis(_Contract):
    """Intent Analyst output for one failure group.

    ``cited_evidence_ids`` must be ids the retriever or memory tool actually
    returned; the orchestrator checks that with ``unknown_evidence_ids`` (a
    model can write any string).  An ``UPDATE_TEST`` recommendation without a
    cited id is invalid.
    """

    group_id: Identifier
    hypothesis: Hypothesis
    expected_behaviour: str | None = Field(default=None, description="Behaviour the code should have.")
    cited_evidence_ids: tuple[Identifier, ...] = ()
    contradicting_evidence_ids: tuple[Identifier, ...] = ()
    confidence: float = Field(ge=0.0, le=1.0)
    recommended_decision: DecisionLabel

    @model_validator(mode="after")
    def _check(self) -> IntentAnalysis:
        if self.recommended_decision not in _ALLOWED_DECISIONS[self.hypothesis.kind]:
            raise ValueError(
                f"hypothesis {self.hypothesis.kind.value!r} cannot recommend "
                f"{self.recommended_decision.value}"
            )
        if self.recommended_decision is DecisionLabel.UPDATE_TEST and not self.cited_evidence_ids:
            raise ValueError("UPDATE_TEST requires at least one cited evidence id")
        return self


def unknown_evidence_ids(cited: Iterable[str], retrieved: Iterable[str]) -> list[str]:
    """Cited ids that were never retrieved (invented or stale).  Empty = all traceable."""
    known = set(retrieved)
    return sorted({c for c in cited if c not in known})


# --------------------------------------------------------------------------- #
# Repairer
# --------------------------------------------------------------------------- #

_SCOPE_FOR_DIRECTION = {
    RepairDirection.UPDATE_TEST: PatchScope.TEST,
    RepairDirection.FIX_CODE: PatchScope.SRC,
}


class PatchProposal(_Contract):
    """A scoped repair.  Proposing is not applying: nothing here touches the worktree."""

    group_id: Identifier
    decision: RepairDirection = Field(description="UPDATE_TEST edits tests; FIX_CODE edits source.")
    scope: PatchScope
    target_files: tuple[str, ...] = Field(min_length=1)
    unified_diff: str = Field(min_length=1)
    rationale: str = Field(min_length=1)
    evidence_ids: tuple[Identifier, ...] = ()
    revision_index: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _check(self) -> PatchProposal:
        if self.scope is not _SCOPE_FOR_DIRECTION[self.decision]:
            raise ValueError(f"{self.decision.value} requires scope {_SCOPE_FOR_DIRECTION[self.decision].value}")
        if self.decision is RepairDirection.UPDATE_TEST and not self.evidence_ids:
            raise ValueError("UPDATE_TEST requires evidence_ids")
        return self


EscalationTrigger = Literal[
    "insufficient_evidence", "contradictory_evidence", "guard_rejection_after_revision"
]


class EscalationRequest(_Contract):
    """The Repairer (or orchestrator) declines to repair.  Feeds ``EscalationSummary``."""

    group_id: Identifier
    trigger: EscalationTrigger
    summary: str = Field(min_length=1)


RepairerOutput = PatchProposal | EscalationRequest


class PatchValidation(_Contract):
    """Result of ``tools.patch_tools.validate_patch`` (static checks, nothing applied)."""

    valid: bool = Field(description="The diff is well-formed and applies to the worktree.")
    allowed: bool = Field(description="Every changed path is permitted by the EditPolicy.")
    changed_paths: tuple[str, ...] = ()
    rejection_reasons: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _check(self) -> PatchValidation:
        if not (self.valid and self.allowed) and not self.rejection_reasons:
            raise ValueError("an invalid or disallowed patch must give rejection_reasons")
        return self

    @property
    def ok(self) -> bool:
        return self.valid and self.allowed


# --------------------------------------------------------------------------- #
# Integrity Guard and memory
# --------------------------------------------------------------------------- #


class GuardResult(_Contract):
    """Integrity Guard verdict.  ``integrity`` carries the detailed checks."""

    decision: GuardDecision
    reasons: tuple[str, ...] = ()
    post_repair_run: PytestExecution | None = None
    integrity: IntegrityReport

    @model_validator(mode="after")
    def _check(self) -> GuardResult:
        if self.decision is GuardDecision.NOT_RUN:
            raise ValueError("a GuardResult is an explicit APPROVE, REJECT or ESCALATE")
        if self.integrity.guard_decision is not self.decision or not self.integrity.guard_ran:
            raise ValueError("integrity report must record the same decision and guard_ran=True")
        if self.decision is not GuardDecision.APPROVE and not self.reasons:
            raise ValueError("REJECT and ESCALATE require reasons")
        if self.decision is GuardDecision.APPROVE and (
            self.post_repair_run is None or self.post_repair_run.status is not TestRunStatus.PASSED
        ):
            raise ValueError("APPROVE requires a passing post-repair run (necessary, not sufficient)")
        return self


class MemoryApproval(_Contract):
    """Evidence that a memory write was approved.

    Built by the Integrity Guard's workflow after a ``GuardResult`` of APPROVE
    and a (simulated) maintainer approval.  ``MemoryStore.write`` requires one.
    This is a convention the store checks, not a security boundary: any code
    can construct this object.
    """

    record_id: Identifier
    guard_decision: GuardDecision
    maintainer_approved: bool
    approved_at_commit: CommitSha

    @model_validator(mode="after")
    def _check(self) -> MemoryApproval:
        if self.guard_decision is not GuardDecision.APPROVE:
            raise ValueError("memory may only be written after a Guard APPROVE")
        if not self.maintainer_approved:
            raise ValueError("memory may only be written after maintainer approval")
        return self


class MemoryRecordCheck(_Contract):
    """One stored record and whether it may be used as evidence right now."""

    record: MemoryRecord
    probe_status: ProbeStatus = ProbeStatus.NOT_RUN
    blocked_reason: Literal["probe_failed", "probe_not_run", "superseded", "out_of_scope"] | None = None

    @property
    def valid_as_evidence(self) -> bool:
        """True only if the probe passed on the current code and the record is not stale."""
        return (
            self.probe_status is ProbeStatus.PASSED
            and not self.record.flagged_stale
            and self.blocked_reason is None
        )

    @model_validator(mode="after")
    def _check(self) -> MemoryRecordCheck:
        if self.probe_status is ProbeStatus.PASSED and self.blocked_reason in ("probe_failed", "probe_not_run"):
            raise ValueError("blocked_reason contradicts a passed probe")
        return self


class MemoryReadResult(_Contract):
    checks: tuple[MemoryRecordCheck, ...] = ()

    @property
    def usable(self) -> list[MemoryRecord]:
        return [c.record for c in self.checks if c.valid_as_evidence]

    @property
    def unusable(self) -> list[MemoryRecordCheck]:
        return [c for c in self.checks if not c.valid_as_evidence]
