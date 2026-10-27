"""Episode orchestration: agent order, budgets, revision limit, escalation.

Planned flow for one episode::

    Triage -> for each failure group:
        Intent Analyst -> (ESCALATE?) -> Repairer -> validate_patch -> apply
            -> Integrity Guard --APPROVE--> (maintainer approval) -> memory write
                               --REJECT---> one revision, then ESCALATE

Responsibilities: build ``EpisodeContext``; share one ``LLMClient`` with all
agents; enforce budgets (``BudgetExceededError``); allow at most
``settings.max_repair_revisions`` revisions per rejected repair; turn exceptions
into ``ErrorRecord`` entries; never let an execution failure masquerade as an
escalation; return the complete ``ExperimentRunRecord``.

The flow is a small explicit state machine.  The episode goes
``TRIAGE -> (groups) -> DONE`` or ``FAILED``; each failure group goes::

    INTENT_ANALYSIS -> REPAIR -> PATCH_VALIDATION -> PATCH_APPLICATION -> INTEGRITY_GUARD
           |              |            |                   |                  |
           v              v            +--- rejected ------+------ REJECT ----+--> REPAIR (one revision)
       ESCALATED      ESCALATED                                                 or ESCALATED
                                                                  APPROVE ----> DECIDED

Agents and patch tools are injected (``Agents``, ``PatchTools``) so every stage
can be tested with fakes.  Defaults are the real modules; any that are still
stubs fail the run explicitly as ``ORCHESTRATOR_ERROR`` ("not implemented").

Scaffolding limits: maintainer approval and memory writes are not performed
(``RunOutcome.maintainer_approved`` stays None), tool-call budgets are not
counted (agents call tools directly), and only the last Guard verdict is kept
in ``ExperimentRunRecord.integrity``.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

from testorcode.agents import integrity_guard as _integrity_guard
from testorcode.agents import intent_analyst as _intent_analyst
from testorcode.agents import repairer as _repairer
from testorcode.agents import triage as _triage
from testorcode.config import Settings
from testorcode.errors import (
    BudgetExceededError,
    ConfigError,
    LLMStructuredOutputError,
    MemoryAuthorizationError,
    PatchError,
    ScopeViolationError,
    TestOrCodeError,
)
from testorcode.llm_client import LLMClient, LLMResponse, merge_usage
from testorcode.logging_config import episode_logging, get_logger
from testorcode.memory import MemoryStore
from testorcode.retrieval import Retriever, build_retriever
from testorcode.schemas.contracts import (
    EpisodeContext,
    EscalationRequest,
    EscalationTrigger,
    FailureGroupDetail,
    GuardResult,
    IntentAnalysis,
    MemoryApproval,
    MemoryReadResult,
    PatchProposal,
    TriageResult,
    unknown_evidence_ids,
)
from testorcode.schemas.episode_schemas import (
    DecisionLabel,
    EpisodeBudget,
    EpisodeInput,
    MemoryRecord,
)
from testorcode.schemas.result_schemas import (
    AgentRole,
    ErrorCategory,
    ErrorRecord,
    EscalationSummary,
    ExperimentRunRecord,
    GuardDecision,
    IntegrityReport,
    MemoryReadEvent,
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
    TokenUsage,
)
from testorcode.tools import patch_tools as _patch_tools

logger = get_logger(__name__)


class Stage(StrEnum):
    TRIAGE = "triage"
    INTENT_ANALYSIS = "intent_analysis"
    REPAIR = "repair"
    PATCH_VALIDATION = "patch_validation"
    PATCH_APPLICATION = "patch_application"
    INTEGRITY_GUARD = "integrity_guard"
    DECIDED = "decided"
    ESCALATED = "escalated"
    DONE = "done"
    FAILED = "failed"


_GROUP_TERMINAL = frozenset({Stage.DECIDED, Stage.ESCALATED})
_FINAL_PHASES = (TestRunPhase.POST_REPAIR, TestRunPhase.POST_REVISION)


@dataclass(frozen=True)
class Agents:
    """The four agent functions (signatures as in ``docs/interfaces.md``)."""

    triage: Callable[..., TriageResult] = _triage.run_triage
    intent_analyst: Callable[..., IntentAnalysis] = _intent_analyst.analyse_intent
    repairer: Callable[..., PatchProposal | EscalationRequest] = _repairer.propose_repair
    integrity_guard: Callable[..., GuardResult] = _integrity_guard.verify_repair


@dataclass(frozen=True)
class PatchTools:
    """Patch validation/application owned by the sandbox branch."""

    validate: Callable[..., Any] = _patch_tools.validate_patch
    apply: Callable[..., Any] = _patch_tools.apply_patch
    revert: Callable[..., None] = _patch_tools.revert_patch


@dataclass
class GroupState:
    """Everything known about one failure group as it moves through the stages."""

    detail: FailureGroupDetail
    stage: Stage = Stage.INTENT_ANALYSIS
    analysis: IntentAnalysis | None = None
    proposal: PatchProposal | None = None
    patch: PatchRecord | None = None
    guard: GuardResult | None = None
    previous_rejection: GuardResult | None = None
    revision: int = 0
    decision: DecisionLabel | None = None
    decided_by: AgentRole | None = None
    escalation: tuple[EscalationTrigger, str] | None = None

    @property
    def group_id(self) -> str:
        return self.detail.group.group_id


# --------------------------------------------------------------------------- #
# Wrappers handed to agents
# --------------------------------------------------------------------------- #


class _MeteredLLM:
    """Shares one client, counts calls and tokens, and enforces the LLM budget."""

    def __init__(self, llm: LLMClient, budget: EpisodeBudget) -> None:
        self._llm = llm
        self._budget = budget
        self.calls = 0
        self.usage = TokenUsage()
        self.model_id: str | None = None

    def complete_structured[T: BaseModel](
        self, *, role: AgentRole, system: str, user: str, response_model: type[T]
    ) -> LLMResponse[T]:
        if self._budget.max_llm_calls is not None and self.calls >= self._budget.max_llm_calls:
            raise BudgetExceededError(f"LLM call budget of {self._budget.max_llm_calls} exhausted")
        self.calls += 1
        try:
            response = self._llm.complete_structured(
                role=role, system=system, user=user, response_model=response_model
            )
        except TestOrCodeError as exc:
            # Replies that arrived but failed validation still cost tokens.
            partial = getattr(exc, "usage", None)
            if isinstance(partial, TokenUsage):
                self.usage = merge_usage(self.usage, partial)
            raise
        self.usage = merge_usage(self.usage, response.usage)
        self.model_id = response.model_id
        limit = self._budget.max_tokens
        if limit is not None and self.usage.reported and (self.usage.total_tokens or 0) > limit:
            raise BudgetExceededError(f"token budget of {limit} exhausted")
        return response


class _RecordingRetriever:
    """Remembers every returned item so citations can be checked against it."""

    def __init__(self, inner: Retriever) -> None:
        self._inner = inner
        self.evidence: list[RetrievedEvidence] = []

    def search(self, query: str, *, top_k: int) -> list[RetrievedEvidence]:
        results = self._inner.search(query, top_k=top_k)
        self.evidence.extend(results)
        return results


class _ReadOnlyMemory:
    """The view of memory given to agents: reads are logged, writes are refused."""

    def __init__(self, inner: MemoryStore) -> None:
        self._inner = inner
        self.events: list[MemoryReadEvent] = []
        self.usable_ids: set[str] = set()

    def read(self, context: EpisodeContext, function_qualname: str) -> MemoryReadResult:
        result = self._inner.read(context, function_qualname)
        for check in result.checks:
            self.events.append(
                MemoryReadEvent(
                    record_id=check.record.record_id,
                    probe_status=check.probe_status,
                    blocked_reason=check.blocked_reason,
                )
            )
            if check.valid_as_evidence:
                self.usable_ids.add(check.record.record_id)
        return result

    def write(self, record: MemoryRecord, approval: MemoryApproval) -> None:
        raise MemoryAuthorizationError("agents may not write memory; only the Guard workflow can")

    def mark_cited(self, cited: set[str]) -> None:
        for event in self.events:
            if event.record_id in cited and event.record_id in self.usable_ids:
                event.used_as_evidence = True


class _ContractViolation(TestOrCodeError):
    """An agent returned a result that does not belong to the current group/stage."""


# --------------------------------------------------------------------------- #
# State machine
# --------------------------------------------------------------------------- #


class EpisodeRun:
    """One episode, executed once.  ``execute()`` returns the run record; ``trace``
    lists every ``(group_id or None, Stage)`` entered, in order."""

    def __init__(
        self,
        episode: EpisodeInput,
        *,
        worktree_path: Path,
        settings: Settings,
        llm: LLMClient,
        memory: MemoryStore | None = None,
        retriever: Retriever | None = None,
        agents: Agents | None = None,
        patch_tools: PatchTools | None = None,
    ) -> None:
        if llm is None:
            raise ConfigError("run_episode requires an LLMClient")
        self.episode = episode
        self.settings = settings
        self.context = EpisodeContext.from_episode_input(episode, worktree_path)
        self.agents = agents or Agents()
        self.tools = patch_tools or PatchTools()
        self.llm = _MeteredLLM(llm, self.context.budget)
        self.memory = _ReadOnlyMemory(memory) if memory is not None else None
        self.retriever = _RecordingRetriever(retriever) if retriever is not None else None
        self.revision_limit = min(settings.max_repair_revisions, self.context.budget.max_revisions)
        self.model_id = settings.llm_model or getattr(llm, "model_id", None)
        if not self.model_id:
            raise ConfigError("LLM_MODEL is required to record which model ran the episode")

        self.trace: list[tuple[str | None, Stage]] = []
        self.triage_result: TriageResult | None = None
        self.groups: list[GroupState] = []
        self.patches: list[PatchRecord] = []
        self.test_runs: list[TestRunResult] = []
        self.integrity: IntegrityReport | None = None
        self.errors: list[ErrorRecord] = []
        self.failure_status: RunStatus | None = None
        self.revisions_used = 0
        self._current: tuple[str | None, Stage] = (None, Stage.TRIAGE)
        self._applied_diff: str | None = None
        self._started = 0.0

    # -- driver ----------------------------------------------------------- #

    def execute(self) -> ExperimentRunRecord:
        started_at = datetime.now(UTC)
        self._started = time.monotonic()
        with episode_logging(self.episode.episode_id):
            try:
                self._enter(None, Stage.TRIAGE)
                result = self.agents.triage(
                    self.context, self.episode.ci_event, llm=self.llm, settings=self.settings
                )
                self.triage_result = result
                self.test_runs.append(result.baseline_run.to_record("baseline", TestRunPhase.PRE_REPAIR))
                self._check_triage(result)
                for detail in result.groups:
                    group = GroupState(detail=detail)
                    self.groups.append(group)
                    self._run_group(group)
                self._enter(None, Stage.DONE)
            except Exception as exc:  # noqa: BLE001 - every failure becomes an ErrorRecord
                self._fail(exc)
            return self._build_record(started_at)

    def _check_triage(self, result: TriageResult) -> None:
        """Every CI failure must be in exactly one group or explicitly marked unstable."""
        grouped = [t for d in result.groups for t in d.group.test_ids]
        duplicated = sorted({t for t in grouped if grouped.count(t) > 1})
        _require(not duplicated, f"triage put test(s) in more than one group: {', '.join(duplicated)}")
        group_ids = [d.group.group_id for d in result.groups]
        _require(len(group_ids) == len(set(group_ids)), "triage returned duplicate group ids")
        accounted = set(grouped) | set(result.unstable_test_ids)
        missing = sorted(
            t.test_id for t in self.episode.ci_event.failing_tests if t.test_id not in accounted
        )
        _require(
            not missing,
            f"triage left {len(missing)} failing test(s) ungrouped and not marked unstable: "
            + ", ".join(missing),
        )

    def _run_group(self, g: GroupState) -> None:
        handlers: dict[Stage, Callable[[GroupState], Stage]] = {
            Stage.INTENT_ANALYSIS: self._intent_analysis,
            Stage.REPAIR: self._repair,
            Stage.PATCH_VALIDATION: self._patch_validation,
            Stage.PATCH_APPLICATION: self._patch_application,
            Stage.INTEGRITY_GUARD: self._integrity_guard,
        }
        self._enter(g.group_id, g.stage)
        while g.stage not in _GROUP_TERMINAL:
            g.stage = handlers[g.stage](g)
            self._enter(g.group_id, g.stage)

    def _enter(self, group_id: str | None, stage: Stage) -> None:
        self._current = (group_id, stage)
        self.trace.append((group_id, stage))
        logger.info("stage=%s group=%s", stage, group_id or "-")
        limit = self.context.budget.wall_clock_seconds
        if (
            limit is not None
            and stage not in (Stage.DONE, Stage.FAILED)
            and time.monotonic() - self._started > limit
        ):
            raise BudgetExceededError(f"wall-clock budget of {limit}s exhausted")

    # -- group stages ----------------------------------------------------- #

    def _intent_analysis(self, g: GroupState) -> Stage:
        if self.retriever is None:
            self.retriever = _RecordingRetriever(
                build_retriever(self.episode.corpus, self.context.worktree_path)
            )
        analysis = self.agents.intent_analyst(
            self.context, g.detail, retriever=self.retriever, memory=self.memory, llm=self.llm
        )
        _require(isinstance(analysis, IntentAnalysis), "intent analyst must return IntentAnalysis")
        _require(analysis.group_id == g.group_id, "intent analysis is for a different group")
        g.analysis = analysis

        cited = [*analysis.cited_evidence_ids, *analysis.contradicting_evidence_ids]
        unknown = unknown_evidence_ids(cited, self._known_evidence_ids())
        if unknown:
            return self._escalate(
                g, "insufficient_evidence",
                f"analysis cited {len(unknown)} evidence id(s) that were never retrieved",
                AgentRole.ORCHESTRATOR,
            )
        if self.memory is not None:
            self.memory.mark_cited(set(analysis.cited_evidence_ids))
        if analysis.recommended_decision is DecisionLabel.ESCALATE:
            trigger = (
                "contradictory_evidence" if analysis.contradicting_evidence_ids
                else "insufficient_evidence"
            )
            return self._escalate(g, trigger, analysis.hypothesis.explanation, AgentRole.INTENT_ANALYST)
        return Stage.REPAIR

    def _repair(self, g: GroupState) -> Stage:
        output = self.agents.repairer(
            self.context, g.detail, g.analysis, llm=self.llm, previous_rejection=g.previous_rejection
        )
        _require(
            isinstance(output, (PatchProposal, EscalationRequest)),
            "repairer must return PatchProposal or EscalationRequest",
        )
        _require(output.group_id == g.group_id, "repairer output is for a different group")
        if isinstance(output, EscalationRequest):
            return self._escalate(g, output.trigger, output.summary, AgentRole.REPAIRER)
        g.proposal = output
        g.patch = PatchRecord(
            patch_id=f"{g.group_id}-r{g.revision}",
            scope=output.scope,
            test_ids=list(g.detail.group.test_ids),
            target_files=list(output.target_files),
            unified_diff=output.unified_diff,
            revision_index=g.revision,
        )
        self.patches.append(g.patch)
        return Stage.PATCH_VALIDATION

    def _patch_validation(self, g: GroupState) -> Stage:
        proposal, patch = g.proposal, g.patch
        assert proposal is not None and patch is not None and g.analysis is not None
        if proposal.decision.value != g.analysis.recommended_decision.value:
            patch.apply_error = (
                f"proposal direction {proposal.decision.value} contradicts the analysis "
                f"decision {g.analysis.recommended_decision.value}"
            )
            return self._rejected(g)
        validation = self.tools.validate(
            self.context.worktree_path, proposal.unified_diff,
            policy=self.context.edit_policy, scope=proposal.scope,
        )
        if validation.ok:
            return Stage.PATCH_APPLICATION
        blocked = [
            p for p in validation.changed_paths
            if not self.context.edit_policy.is_editable(p, proposal.scope)
        ]
        if not validation.allowed and blocked:
            patch.out_of_scope_files = blocked
            patch.application_status = PatchApplicationStatus.BLOCKED_OUT_OF_SCOPE
        else:
            patch.apply_error = "; ".join(validation.rejection_reasons)
            patch.application_status = PatchApplicationStatus.FAILED_TO_APPLY
        return self._rejected(g)

    def _patch_application(self, g: GroupState) -> Stage:
        proposal, patch = g.proposal, g.patch
        assert proposal is not None and patch is not None
        try:
            self.tools.apply(
                self.context.worktree_path, proposal.unified_diff,
                policy=self.context.edit_policy, scope=proposal.scope,
            )
        except ScopeViolationError as exc:
            patch.out_of_scope_files = list(proposal.target_files)
            patch.apply_error = str(exc) or type(exc).__name__
            patch.application_status = PatchApplicationStatus.BLOCKED_OUT_OF_SCOPE
            return self._rejected(g)
        except PatchError as exc:
            patch.apply_error = str(exc) or type(exc).__name__
            patch.application_status = PatchApplicationStatus.FAILED_TO_APPLY
            return self._rejected(g)
        patch.application_status = PatchApplicationStatus.APPLIED
        self._applied_diff = proposal.unified_diff
        return Stage.INTEGRITY_GUARD

    def _integrity_guard(self, g: GroupState) -> Stage:
        assert g.proposal is not None and g.patch is not None and g.analysis is not None
        result = self.agents.integrity_guard(
            self.context, g.detail, g.analysis, g.proposal, llm=self.llm
        )
        _require(isinstance(result, GuardResult), "integrity guard must return GuardResult")
        g.guard = result
        self.integrity = result.integrity
        if result.post_repair_run is not None:
            phase = TestRunPhase.POST_REPAIR if g.revision == 0 else TestRunPhase.POST_REVISION
            self.test_runs.append(result.post_repair_run.to_record(f"{g.patch.patch_id}-post", phase))

        if result.decision is GuardDecision.APPROVE:
            self._applied_diff = None  # the repair stays in the worktree
            g.decision, g.decided_by = g.analysis.recommended_decision, AgentRole.INTENT_ANALYST
            # Integration point: maintainer approval -> authorize_memory_write -> MemoryStore.write.
            return Stage.DECIDED

        self._revert()
        if result.decision is GuardDecision.REJECT:
            g.patch.application_status = PatchApplicationStatus.REJECTED_BY_GUARD
            g.previous_rejection = result
            return self._rejected(g)
        # Explicit Guard ESCALATE: not a rejection, so no revision.  The unverified patch is
        # withdrawn.  EscalationTrigger has no Guard-specific value; the closest is
        # insufficient_evidence (the Guard could not establish the repair is safe).
        g.patch.application_status = PatchApplicationStatus.REVERTED
        return self._escalate(
            g, "insufficient_evidence",
            "integrity guard escalated: " + "; ".join(result.reasons), AgentRole.INTEGRITY_GUARD,
        )

    # -- transitions shared by stages ------------------------------------- #

    def _rejected(self, g: GroupState) -> Stage:
        """A rejected repair gets at most ``revision_limit`` revisions, then escalates."""
        if g.revision < self.revision_limit:
            g.revision += 1
            self.revisions_used += 1
            return Stage.REPAIR
        return self._escalate(
            g, "guard_rejection_after_revision",
            f"repair rejected after {g.revision} revision(s)", AgentRole.ORCHESTRATOR,
        )

    def _escalate(
        self, g: GroupState, trigger: EscalationTrigger, summary: str, by: AgentRole
    ) -> Stage:
        g.decision, g.decided_by, g.escalation = DecisionLabel.ESCALATE, by, (trigger, summary)
        return Stage.ESCALATED

    def _revert(self) -> None:
        if self._applied_diff is not None:
            diff, self._applied_diff = self._applied_diff, None
            self.tools.revert(self.context.worktree_path, diff, policy=self.context.edit_policy)

    def _known_evidence_ids(self) -> set[str]:
        known = {e.evidence_id for e in self.retriever.evidence} if self.retriever else set()
        if self.memory is not None:
            known |= self.memory.usable_ids
        return known

    # -- failure and record ----------------------------------------------- #

    def _fail(self, exc: Exception) -> None:
        group_id, stage = self._current
        operation = f"{stage}:{group_id}" if group_id else str(stage)
        error_id = f"err-{len(self.errors) + 1}"
        if isinstance(exc, LLMStructuredOutputError):
            # The provider answered; the reply failed validation.  Category stays
            # VALIDATION_ERROR (counted against the agent); the run is INCOMPLETE,
            # not API_FAILURE, which is reserved for provider failures.
            record = exc.to_error_record(error_id, operation=operation, fatal=True)
            status = RunStatus.INCOMPLETE
        elif isinstance(exc, TestOrCodeError):
            record = exc.to_error_record(error_id, operation=operation, fatal=True)
            status = exc.run_status
        elif isinstance(exc, NotImplementedError):
            record = ErrorRecord(
                error_id=error_id, category=ErrorCategory.ORCHESTRATOR_ERROR, operation=operation,
                message=f"dependency not implemented: {exc}", fatal=True,
            )
            status = RunStatus.ORCHESTRATOR_ERROR
        elif isinstance(exc, ValidationError):
            record = ErrorRecord(
                error_id=error_id, category=ErrorCategory.VALIDATION_ERROR, operation=operation,
                message=f"invalid {exc.title}: {exc.error_count()} validation error(s)", fatal=True,
            )
            status = RunStatus.ORCHESTRATOR_ERROR
        else:
            record = ErrorRecord(
                error_id=error_id, category=ErrorCategory.ORCHESTRATOR_ERROR, operation=operation,
                message=f"{type(exc).__name__}: {exc}"[:500], fatal=True,
            )
            status = RunStatus.ORCHESTRATOR_ERROR
        self.errors.append(record)
        self.failure_status = status
        logger.error("episode failed at %s: %s", operation, record.category)
        try:
            self._revert()
        except Exception as revert_exc:  # noqa: BLE001 - already failing; record and move on
            self.errors.append(
                ErrorRecord(
                    error_id=f"err-{len(self.errors) + 1}", category=ErrorCategory.PATCH_ERROR,
                    operation="revert_after_failure", message=type(revert_exc).__name__, fatal=True,
                )
            )
        self.trace.append((group_id, Stage.FAILED))

    def _predictions(self) -> list[PredictedDecision]:
        out = []
        for g in self.groups:
            if g.decision is None:
                continue
            a = g.analysis
            for test_id in g.detail.group.test_ids:
                out.append(
                    PredictedDecision(
                        test_id=test_id,
                        group_id=g.group_id,
                        predicted_action=g.decision,
                        hypothesis=a.hypothesis.kind if a else None,
                        predicted_behaviour=a.expected_behaviour if a else None,
                        cited_evidence_ids=list(a.cited_evidence_ids) if a else [],
                        contradicting_evidence_ids=list(a.contradicting_evidence_ids) if a else [],
                        rationale=g.escalation[1] if g.escalation else (a.hypothesis.explanation if a else None),
                        decided_by=g.decided_by or AgentRole.ORCHESTRATOR,
                    )
                )
        return out

    def _escalation_summary(self) -> EscalationSummary | None:
        escalated = [g for g in self.groups if g.escalation is not None]
        if not escalated:
            return None
        return EscalationSummary(
            trigger=escalated[0].escalation[0],
            failing_test_ids=[t for g in escalated for t in g.detail.group.test_ids],
            evidence_found_ids=sorted(
                {e for g in escalated if g.analysis for e in g.analysis.cited_evidence_ids}
            ),
            competing_hypotheses=[
                g.analysis.hypothesis.explanation for g in escalated if g.analysis is not None
            ],
            summary=" | ".join(f"{g.group_id}: {g.escalation[1]}" for g in escalated),
        )

    def _build_record(self, started_at: datetime) -> ExperimentRunRecord:
        predictions = self._predictions()
        if self.failure_status is not None:
            status = self.failure_status
        elif not predictions:
            status = RunStatus.INCOMPLETE  # e.g. every failure was unstable: nothing was decided
        elif all(p.predicted_action is DecisionLabel.ESCALATE for p in predictions):
            status = RunStatus.ESCALATED
        else:
            status = RunStatus.COMPLETED
        final_runs = [r for r in self.test_runs if r.phase in _FINAL_PHASES]
        return ExperimentRunRecord(
            provenance=RunProvenance(
                run_id=f"{self.episode.episode_id}-{uuid.uuid4().hex[:12]}",
                episode_id=self.episode.episode_id,
                condition=self.settings.experiment_condition,
                seed=self.settings.random_seed,
                model_id=self.llm.model_id or self.model_id,
                provider=self.settings.llm_provider,
                started_at=started_at,
                ended_at=datetime.now(UTC),
            ),
            predictions=predictions,
            retrieved_evidence=list(self.retriever.evidence) if self.retriever else [],
            patches=self.patches,
            test_runs=self.test_runs,
            final_test_status=final_runs[-1].status if final_runs else TestRunStatus.NOT_RUN,
            integrity=self.integrity or IntegrityReport(),
            memory_reads=list(self.memory.events) if self.memory else [],
            usage=self.llm.usage,
            errors=self.errors,
            outcome=RunOutcome(
                status=status,
                escalation=self._escalation_summary(),
                revisions_used=self.revisions_used,
            ),
        )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise _ContractViolation(message)


def run_episode(
    episode: EpisodeInput,
    *,
    worktree_path: Path,
    settings: Settings,
    llm: LLMClient,
    memory: MemoryStore | None = None,
    retriever: Retriever | None = None,
    agents: Agents | None = None,
    patch_tools: PatchTools | None = None,
) -> ExperimentRunRecord:
    """Run one episode and return its run record.

    Takes ``EpisodeInput`` (never ``Episode``) so gold labels cannot reach the
    agents.  ``memory=None`` disables memory (condition A2).  Execution
    failures are recorded in the returned record, not raised, wherever possible.

    ``retriever``, ``agents`` and ``patch_tools`` are optional injections; when
    omitted, the retriever is built from ``episode.corpus`` on first use and the
    real agent and tool modules are used.

    Raises:
        ConfigError: no model id is known, so no valid record can be written.
    """
    return EpisodeRun(
        episode,
        worktree_path=worktree_path,
        settings=settings,
        llm=llm,
        memory=memory,
        retriever=retriever,
        agents=agents,
        patch_tools=patch_tools,
    ).execute()
