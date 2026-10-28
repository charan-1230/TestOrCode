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

Conditions: the episode view and memory follow ``conditions.profile_for``
(``settings.experiment_condition``).  A condition without a code path yet
fails the run explicitly (``ORCHESTRATOR_ERROR``); it never silently runs the
multi-agent pipeline under another label.

Tools: every tool call made during the episode (by an agent or by the
orchestrator) is recorded as a ``ToolEvent`` through ``tools.trace`` with its
budget category.  Calls made while an agent works (Intent Analysis, Repair, and
B1's decision loop) count against ``budget.max_tool_calls``; mandatory harness
work (Triage stability runs, patch validation/application, Guard or B1
verification) counts against a finite operational safety cap
(``OPERATIONAL_CAP_FACTOR`` x the agent budget), so fixed verification overhead
cannot starve a multi-group episode's agents.  Reverts and the final suite run
are exempt (never blocked, still logged).

After a Guard APPROVE an optional, harness-supplied ``maintainer`` callback
decides; on approval the Guard's memory record is written through
``authorize_memory_write`` (every attempt is logged in ``memory_writes``).

Per-group Guard verdicts, patches, verification runs and final disposition are
kept in ``ExperimentRunRecord.failure_groups``; ``ExperimentRunRecord.integrity``
still holds the last verdict only (unchanged shared field).
"""

from __future__ import annotations

import subprocess
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

from testorcode.agents import integrity_guard as _integrity_guard
from testorcode.agents import intent_analyst as _intent_analyst
from testorcode.agents import repairer as _repairer
from testorcode.agents import triage as _triage
from testorcode.agents.prompts import B1_VERSION, PROMPT_VERSION, B1Decision, B1Step
from testorcode.agents.repairer import build_diff, editable_files
from testorcode.agents.single_agent import run_single_agent
from testorcode.agents.tools import AgentTools
from testorcode.conditions import episode_for_condition, profile_for
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
from testorcode.llm_client import LLMClient, LLMResponse, describe_validation_error, merge_usage
from testorcode.logging_config import episode_logging, get_logger
from testorcode.memory import MemoryStore
from testorcode.retrieval import Retriever, build_retriever
from testorcode.schemas.contracts import (
    EpisodeContext,
    EscalationRequest,
    EscalationTrigger,
    FailureGroupDetail,
    GuardResult,
    Hypothesis,
    IntentAnalysis,
    MemoryApproval,
    MemoryReadResult,
    PatchProposal,
    PytestExecution,
    TriageResult,
    decision_policy_problem,
    unknown_evidence_ids,
)
from testorcode.schemas.episode_schemas import (
    DecisionLabel,
    EpisodeBudget,
    EpisodeInput,
    FailingTest,
    FailureGroup,
    HypothesisKind,
    MemoryRecord,
    PathClass,
    RepairDirection,
    ToolName,
)
from testorcode.schemas.result_schemas import (
    AgentRole,
    ErrorCategory,
    ErrorRecord,
    EscalationSummary,
    ExperimentCondition,
    ExperimentRunRecord,
    FailureGroupRecord,
    GuardDecision,
    IntegrityReport,
    MemoryReadEvent,
    MemoryWriteAttempt,
    PatchApplicationStatus,
    PatchRecord,
    PredictedDecision,
    RetrievedEvidence,
    RunOutcome,
    RunProvenance,
    RunStatus,
    SupportStatus,
    TestRunPhase,
    TestRunResult,
    TestRunStatus,
    TokenUsage,
)
from testorcode.tools import patch_tools as _patch_tools
from testorcode.tools.patch_tools import scope_for_decision
from testorcode.tools.pytest_runner import run_tests
from testorcode.tools.trace import ToolRecorder, recording, tool_call

logger = get_logger(__name__)


class Stage(StrEnum):
    SETUP = "setup"
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
# Who is acting while a stage runs (``ToolEvent.called_by``).  Patch validation and
# application are done by the orchestrator on the Repairer's behalf.
_STAGE_ROLE = {
    Stage.TRIAGE: AgentRole.TRIAGE,
    Stage.INTENT_ANALYSIS: AgentRole.INTENT_ANALYST,
    Stage.REPAIR: AgentRole.REPAIRER,
    Stage.PATCH_VALIDATION: AgentRole.REPAIRER,
    Stage.PATCH_APPLICATION: AgentRole.REPAIRER,
    Stage.INTEGRITY_GUARD: AgentRole.INTEGRITY_GUARD,
}
# Stages in which an agent is acting: tool calls made then are the agent's own and use the episode tool budget.
# Every other stage is mandatory harness work and uses the operational safety cap.
_AGENT_STAGES = frozenset({Stage.INTENT_ANALYSIS, Stage.REPAIR})
# Operational safety cap = this factor x max_tool_calls (or OPERATIONAL_CAP_DEFAULT when the episode sets no tool
# budget).  Per group the harness needs at most about 8 calls (validate, apply, targeted run, regression run, twice
# with one revision) plus the Triage stability runs once; 3 x 40 = 120 covers well over ten groups and still stops
# a runaway loop.
OPERATIONAL_CAP_FACTOR = 3
OPERATIONAL_CAP_DEFAULT = 200
# Conditions that run on the multi-agent pipeline today; the memory ones need a MemoryStore.
_NO_MEMORY_CONDITIONS = frozenset({ExperimentCondition.A2_NO_MEMORY, ExperimentCondition.B1})
# Which run class executes which condition; any other pairing is refused, so a condition can never run
# under another one's pipeline (e.g. B1 through the multi-agent EpisodeRun).
_MULTI_AGENT_CONDITIONS = frozenset({ExperimentCondition.FULL, ExperimentCondition.A2_NO_MEMORY})
_SINGLE_AGENT_CONDITIONS = frozenset({ExperimentCondition.B1})
# A4 (coarse memory) is not in the submitted proposal and has no implementation of its own, so it is refused
# rather than run as FULL under another label.
_MEMORY_CONDITIONS = frozenset({ExperimentCondition.FULL})
_ALTERNATIVE = {
    "outdated_test": "regression: the change broke behaviour the test rightly checks (FIX_CODE)",
    "regression": "outdated_test: the change was intended and the test encodes the old behaviour (UPDATE_TEST)",
    "unknown": "undecided between outdated_test (UPDATE_TEST) and regression (FIX_CODE)",
}

Maintainer = Callable[[FailureGroupDetail, DecisionLabel, GuardResult], bool]
_FINAL_PHASES = (TestRunPhase.POST_REPAIR, TestRunPhase.POST_REVISION, TestRunPhase.FINAL)


@dataclass(frozen=True)
class Agents:
    """The four agent functions (signatures as in ``docs/interfaces.md``)."""

    triage: Callable[..., TriageResult] = _triage.run_triage
    intent_analyst: Callable[..., IntentAnalysis] = _intent_analyst.analyse_intent
    repairer: Callable[..., PatchProposal | EscalationRequest] = _repairer.propose_repair
    integrity_guard: Callable[..., GuardResult] = _integrity_guard.verify_repair
    memory_record: Callable[..., MemoryRecord] = _integrity_guard.build_memory_record


@dataclass(frozen=True)
class PatchTools:
    """Patch validation/application owned by the sandbox branch."""

    validate: Callable[..., Any] = _patch_tools.validate_patch
    apply: Callable[..., Any] = _patch_tools.apply_patch
    revert: Callable[..., None] = _patch_tools.revert_patch
    # Whole-suite run after every group (the episode's final state).  None = no final run (unit tests that inject
    # fake patch tools on a fake worktree); final_test_status then falls back to the last group verification.
    run_suite: Callable[..., PytestExecution] | None = field(default_factory=lambda: _run_suite)


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
    rejection_reasons: tuple[str, ...] = ()
    unknown_ids: tuple[str, ...] = ()
    revision: int = 0
    decision: DecisionLabel | None = None
    decided_by: AgentRole | None = None
    escalation: tuple[EscalationTrigger, str] | None = None
    accepted: bool = False  # its repair passed verification and stays in the worktree
    guard_results: list[GuardResult] = field(default_factory=list)  # every Guard verdict for this group, in order
    final_failing: list[str] = field(default_factory=list)  # its tests that fail in the final suite run

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
        with tool_call(ToolName.SEARCH_REPO, query_chars=len(query), top_k=top_k):
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
        with tool_call(ToolName.MEMORY_READ, function=function_qualname):
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

    CONDITIONS = _MULTI_AGENT_CONDITIONS  # conditions this class executes
    PROMPTS = PROMPT_VERSION  # recorded as provenance.prompt_template_version

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
        maintainer: Maintainer | None = None,
    ) -> None:
        if llm is None:
            raise ConfigError("run_episode requires an LLMClient")
        self.condition = settings.experiment_condition
        self.condition_problem = _condition_problem(self.condition, memory, self.CONDITIONS)
        if not profile_for(self.condition).memory:
            memory = None
        episode = episode_for_condition(episode, self.condition)
        self.episode = episode
        self.settings = settings
        self.maintainer = maintainer
        self.store = memory
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
        agent_budget = self.context.budget.max_tool_calls
        self.recorder = ToolRecorder(
            max_calls=agent_budget,
            max_operational=OPERATIONAL_CAP_FACTOR * agent_budget if agent_budget else OPERATIONAL_CAP_DEFAULT,
        )
        self.memory_writes: list[MemoryWriteAttempt] = []
        self.maintainer_votes: list[bool] = []
        self._tools: AgentTools | None = None
        self._current: tuple[str | None, Stage] = (None, Stage.SETUP)
        self._applied_diff: str | None = None
        self._started = 0.0

    # -- driver ----------------------------------------------------------- #

    def execute(self) -> ExperimentRunRecord:
        started_at = datetime.now(UTC)
        self._started = time.monotonic()
        with episode_logging(self.episode.episode_id), recording(self.recorder):
            try:
                if self.condition_problem:
                    raise _ConditionNotImplemented(self.condition_problem)
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
                self._final_suite()
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

    def _handlers(self) -> dict[Stage, Callable[[GroupState], Stage]]:
        return {
            Stage.INTENT_ANALYSIS: self._intent_analysis,
            Stage.REPAIR: self._repair,
            Stage.PATCH_VALIDATION: self._patch_validation,
            Stage.PATCH_APPLICATION: self._patch_application,
            Stage.INTEGRITY_GUARD: self._integrity_guard,
        }

    def _run_group(self, g: GroupState) -> None:
        handlers = self._handlers()
        self._enter(g.group_id, g.stage)
        while g.stage not in _GROUP_TERMINAL:
            g.stage = handlers[g.stage](g)
            self._enter(g.group_id, g.stage)

    def _enter(self, group_id: str | None, stage: Stage) -> None:
        self._current = (group_id, stage)
        self.trace.append((group_id, stage))
        self.recorder.role = _STAGE_ROLE.get(stage, AgentRole.ORCHESTRATOR)
        self.recorder.budget = "agent" if stage in _AGENT_STAGES else "operational"
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
        self._ensure_retriever()
        analysis = self.agents.intent_analyst(
            self.context, g.detail, retriever=self.retriever, memory=self.memory, llm=self.llm
        )
        _require(isinstance(analysis, IntentAnalysis), "intent analyst must return IntentAnalysis")
        _require(analysis.group_id == g.group_id, "intent analysis is for a different group")
        g.analysis = analysis

        cited = [*analysis.cited_evidence_ids, *analysis.contradicting_evidence_ids]
        unknown = unknown_evidence_ids(cited, self._known_evidence_ids())
        if unknown:
            g.unknown_ids = tuple(unknown)
            return self._escalate(
                g, "insufficient_evidence",
                f"analysis cited {len(unknown)} evidence id(s) that were never retrieved",
                AgentRole.ORCHESTRATOR,
            )
        problem = decision_policy_problem(analysis)
        if problem:
            trigger = "contradictory_evidence" if "both for and against" in problem else "insufficient_evidence"
            return self._escalate(g, trigger, problem, AgentRole.ORCHESTRATOR)
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
            self.context, g.detail, g.analysis, llm=self.llm,
            previous_rejection=g.previous_rejection, rejection_reasons=g.rejection_reasons,
            tools=self._agent_tools(),
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
        unsupported = unknown_evidence_ids(proposal.evidence_ids, self._known_evidence_ids())
        if unsupported:
            patch.apply_error = (
                "patch cites evidence ids that were never retrieved or are not usable memory: "
                + ", ".join(unsupported)
            )
            return self._rejected(g, (patch.apply_error,))
        if proposal.decision.value != g.analysis.recommended_decision.value:
            patch.apply_error = (
                f"proposal direction {proposal.decision.value} contradicts the analysis "
                f"decision {g.analysis.recommended_decision.value}"
            )
            return self._rejected(g, (patch.apply_error,))
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
        reasons = tuple(validation.rejection_reasons) or (f"files outside the allowed scope: {', '.join(blocked)}",)
        return self._rejected(g, reasons)

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
            return self._rejected(g, (patch.apply_error,))
        except PatchError as exc:
            patch.apply_error = str(exc) or type(exc).__name__
            patch.application_status = PatchApplicationStatus.FAILED_TO_APPLY
            return self._rejected(g, (patch.apply_error,))
        patch.application_status = PatchApplicationStatus.APPLIED
        self._applied_diff = proposal.unified_diff
        return Stage.INTEGRITY_GUARD

    def _integrity_guard(self, g: GroupState) -> Stage:
        assert g.proposal is not None and g.patch is not None and g.analysis is not None
        result = self.agents.integrity_guard(
            self.context, g.detail, g.analysis, g.proposal, llm=self.llm, known_failures=self._known_failures(g),
            # The configured target interpreter and timeout reach the Guard on every path (direct run_episode and the
            # CLI alike); an injected Guard receives the same keywords.
            python_executable=self.settings.test_python, timeout_seconds=self.settings.test_timeout_seconds,
        )
        _require(isinstance(result, GuardResult), "integrity guard must return GuardResult")
        g.guard = result
        g.guard_results.append(result)
        self.integrity = result.integrity
        self._record_verification(g, result.post_repair_run, result.regression_run)

        if result.decision is GuardDecision.APPROVE:
            self._applied_diff = None  # the repair stays in the worktree
            g.accepted = True
            g.decision, g.decided_by = g.analysis.recommended_decision, AgentRole.INTENT_ANALYST
            self._after_approval(g, result)
            return Stage.DECIDED

        self._revert()
        if result.decision is GuardDecision.REJECT:
            g.patch.application_status = PatchApplicationStatus.REJECTED_BY_GUARD
            g.previous_rejection = result
            g.rejection_reasons = ()
            return self._rejected(g)
        # Explicit Guard ESCALATE: not a rejection, so no revision.  The unverified patch is
        # withdrawn.  EscalationTrigger has no Guard-specific value; the closest is
        # insufficient_evidence (the Guard could not establish the repair is safe).
        g.patch.application_status = PatchApplicationStatus.REVERTED
        return self._escalate(
            g, "insufficient_evidence",
            "integrity guard escalated: " + "; ".join(result.reasons), AgentRole.INTEGRITY_GUARD,
        )

    def _record_verification(
        self, g: GroupState, targeted: PytestExecution | None, regression: PytestExecution | None
    ) -> None:
        assert g.patch is not None
        phase = TestRunPhase.POST_REPAIR if g.revision == 0 else TestRunPhase.POST_REVISION
        if targeted is not None:
            self.test_runs.append(targeted.to_record(f"{g.patch.patch_id}-post", phase))
        if regression is not None:
            self.test_runs.append(regression.to_record(f"{g.patch.patch_id}-regression", phase))

    def _known_failures(self, g: GroupState) -> frozenset[str]:
        """CI failures that may still fail while ``g`` is verified: every other group's tests (and unstable tests),
        except those of groups whose repair was already accepted, which must keep passing."""
        accepted = {t for x in self.groups if x.accepted and x is not g for t in x.detail.group.test_ids}
        failing = {t.test_id for t in self.episode.ci_event.failing_tests}
        return frozenset(failing - set(g.detail.group.test_ids) - accepted)

    def _final_suite(self) -> None:
        """One whole-suite run after every group: ``final_test_status`` is the episode's actual end state.

        Harness verification, not an agent action, so like a revert it is never blocked by a tool budget (it runs
        exactly once).  A run that cannot start (NOT_RUN) or cannot execute the tests (ERROR, COLLECTION_ERROR,
        TIMEOUT, kept as the final status) makes the run a ``tool_failure`` with an error record.  An accepted group
        whose tests fail here is marked ``failed_final_verification`` and the run is ``incomplete``.
        """
        if not self.groups or self.tools.run_suite is None:
            return
        selectors = _integrity_guard.suite_selectors(self.context)
        self.recorder.role, self.recorder.budget = AgentRole.ORCHESTRATOR, "exempt"
        try:
            run = self.tools.run_suite(
                self.context.worktree_path, selectors,
                timeout_seconds=self.settings.test_timeout_seconds, python_executable=self.settings.test_python,
            )
        except TestOrCodeError as exc:
            self.test_runs.append(TestRunResult(
                run_id="final-suite", phase=TestRunPhase.FINAL, status=TestRunStatus.NOT_RUN,
                selector=" ".join(selectors),
            ))
            self._final_verification_failed(exc.category, f"final suite could not start: {type(exc).__name__}: {exc}")
            return
        self.test_runs.append(run.to_record("final-suite", TestRunPhase.FINAL))
        if run.status not in (TestRunStatus.PASSED, TestRunStatus.FAILED):
            # The tests were not executed: the end state is unknown, which is an execution failure, not a result.
            category = ErrorCategory.TIMEOUT if run.status is TestRunStatus.TIMEOUT else ErrorCategory.TOOL_ERROR
            detail = f" ({run.error_message[:300]})" if run.error_message else ""
            self._final_verification_failed(category, f"final suite did not execute the tests: {run.status.value}{detail}")
            return
        failing = set(run.failing_test_ids)
        broken = []
        for g in self.groups:
            if g.accepted:
                g.final_failing = sorted(failing & set(g.detail.group.test_ids))
                if g.final_failing:
                    broken.append(f"{g.group_id}: {', '.join(g.final_failing)}")
        if broken:
            # An accepted repair did not survive to the end state.  Its decision and verification history stay as
            # recorded; the run is not reported as completed.
            self.errors.append(ErrorRecord(
                error_id=f"err-{len(self.errors) + 1}", category=ErrorCategory.VALIDATION_ERROR,
                operation="final_suite", fatal=False,
                message="accepted repair(s) fail in the final suite run: " + "; ".join(broken),
            ))
            self.failure_status = self.failure_status or RunStatus.INCOMPLETE

    def _final_verification_failed(self, category: ErrorCategory, message: str) -> None:
        self.errors.append(ErrorRecord(
            error_id=f"err-{len(self.errors) + 1}", category=category, operation="final_suite",
            message=message[:500], fatal=False, counted_against_agent=False,
        ))
        self.failure_status = self.failure_status or RunStatus.TOOL_FAILURE

    # -- transitions shared by stages ------------------------------------- #

    def _rejected(self, g: GroupState, reasons: tuple[str, ...] = ()) -> Stage:
        """A rejected repair gets at most ``revision_limit`` revisions, then escalates.

        ``reasons`` (a rejection before the Guard ran) reach the Repairer on the revision.
        """
        if reasons:
            g.rejection_reasons, g.previous_rejection = reasons, None
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
            budget, self.recorder.budget = self.recorder.budget, "exempt"  # undoing is never blocked
            role, self.recorder.role = self.recorder.role, AgentRole.ORCHESTRATOR
            try:
                self.tools.revert(self.context.worktree_path, diff, policy=self.context.edit_policy)
            finally:
                self.recorder.budget, self.recorder.role = budget, role

    def _after_approval(self, g: GroupState, result: GuardResult) -> None:
        """Simulated maintainer, then the Guard's memory write (proposal workflow step 7)."""
        if self.maintainer is None:
            return  # no reviewer configured: maintainer_approved stays None
        record_id = f"{self.episode.episode_id}-{g.group_id}"
        approved = bool(self.maintainer(g.detail, g.decision, result))
        self.maintainer_votes.append(approved)
        if not approved:
            self.memory_writes.append(
                MemoryWriteAttempt(record_id=record_id, approved=False, blocked_reason="not_approved")
            )
            return
        if self.store is None:
            self.memory_writes.append(
                MemoryWriteAttempt(record_id=record_id, approved=True, blocked_reason="memory_disabled")
            )
            return
        try:
            record = self.agents.memory_record(self.context, g.detail, g.analysis, g.proposal, result)
            approval = _integrity_guard.authorize_memory_write(result, record, maintainer_approved=True)
            with tool_call(ToolName.MEMORY_WRITE, record_id=record.record_id):
                self.store.write(record, approval)
        except (NotImplementedError, TestOrCodeError) as exc:
            self.memory_writes.append(
                MemoryWriteAttempt(record_id=record_id, approved=True, blocked_reason="write_error")
            )
            self.errors.append(
                ErrorRecord(
                    error_id=f"err-{len(self.errors) + 1}", category=ErrorCategory.ORCHESTRATOR_ERROR,
                    operation=f"memory_write:{g.group_id}", message=f"{type(exc).__name__}: {exc}"[:500],
                    fatal=False,
                )
            )
            return
        self.memory_writes.append(MemoryWriteAttempt(record_id=record.record_id, approved=True, written=True))

    def _ensure_retriever(self) -> _RecordingRetriever:
        if self.retriever is None:
            self.retriever = _RecordingRetriever(
                build_retriever(self.episode.corpus, self.context.worktree_path)
            )
        return self.retriever

    def _agent_tools(self) -> AgentTools:
        """The model-callable tools (``search_repo``, ``read_file``, ``run_tests``) for this episode."""
        if self._tools is None:
            self._tools = AgentTools(
                self.context, retriever=self._ensure_retriever(),
                timeout_seconds=self.settings.test_timeout_seconds, python_executable=self.settings.test_python,
            )
        return self._tools

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
        missing: list[str] = []
        hypotheses: list[str] = []
        for g in escalated:
            trigger, a = g.escalation[0], g.analysis
            if g.unknown_ids:
                missing.append(f"{g.group_id}: cited ids that were never retrieved: {', '.join(g.unknown_ids)}")
            if a is None or not a.cited_evidence_ids:
                missing.append(f"{g.group_id}: no cited evidence that the change was intended")
            if trigger == "contradictory_evidence" and a is not None:
                missing.append(
                    f"{g.group_id}: evidence points both ways (contradicting: "
                    f"{', '.join(a.contradicting_evidence_ids)})"
                )
            if g.decided_by is AgentRole.REPAIRER:
                missing.append(f"{g.group_id}: a repair possible within the allowed edit policy (see summary)")
            if g.guard is not None and g.guard.decision is GuardDecision.ESCALATE:
                missing.append(f"{g.group_id}: independent verification: {'; '.join(g.guard.reasons)[:400]}")
            if a is not None:
                hypotheses.append(f"{g.group_id} analyst: {a.hypothesis.kind.value}: {a.hypothesis.explanation}")
                other = a.alternative_explanation or _ALTERNATIVE[a.hypothesis.kind.value]
                hypotheses.append(f"{g.group_id} alternative: {other}")
        return EscalationSummary(
            trigger=escalated[0].escalation[0],
            failing_test_ids=[t for g in escalated for t in g.detail.group.test_ids],
            evidence_found_ids=sorted(
                {e for g in escalated if g.analysis for e in g.analysis.cited_evidence_ids}
            ),
            evidence_missing=missing,
            competing_hypotheses=hypotheses,
            summary=" | ".join(f"{g.group_id}: {g.escalation[1]}" for g in escalated),
        )

    def _failure_groups(self) -> list[FailureGroupRecord]:
        if self.triage_result is None:
            return []
        states = {g.group_id: g for g in self.groups}
        out = []
        for d in self.triage_result.groups:
            g = states.get(d.group.group_id)
            patches = [p for p in self.patches if p.patch_id.startswith(f"{d.group.group_id}-r")]
            last = g.guard_results[-1] if g and g.guard_results else None
            out.append(FailureGroupRecord(
                group_id=d.group.group_id,
                function_qualname=d.group.function_qualname,
                test_ids=list(d.group.test_ids),
                mapped_by=d.mapped_by,
                stability_runs=self.context.stability_runs,
                decision=g.decision if g else None,
                guard_decision=last.decision if last else None,
                guard_reasons=[r for result in (g.guard_results if g else []) for r in result.reasons],
                integrity=last.integrity if last else None,
                patch_ids=[p.patch_id for p in patches],
                verification_run_ids=[
                    r.run_id for r in self.test_runs if any(r.run_id.startswith(f"{p.patch_id}-") for p in patches)
                ],
                disposition=_disposition(g, patches),
                final_failing_test_ids=list(g.final_failing) if g else [],
            ))
        return out

    def _labelled_evidence(self) -> list[RetrievedEvidence]:
        """Retrieved items with the analysts' *claimed* reading: cited = supports, contradicting = contradicts.

        These labels record what the agent said about an item, not verified truth: the only check made is that a
        cited id was really retrieved (``unknown_evidence_ids``).  Whether the item semantically supports the claim
        (citation faithfulness) must be judged by the evaluation harness, never read from this field.  Run-level: an
        item one group cites and another contradicts stays ``unknown``.
        """
        if self.retriever is None:
            return []
        cited = {e for g in self.groups if g.analysis for e in g.analysis.cited_evidence_ids}
        against = {e for g in self.groups if g.analysis for e in g.analysis.contradicting_evidence_ids}
        out = []
        for item in self.retriever.evidence:
            if item.evidence_id in cited and item.evidence_id not in against:
                item = item.model_copy(update={"support": SupportStatus.SUPPORTS})
            elif item.evidence_id in against and item.evidence_id not in cited:
                item = item.model_copy(update={"support": SupportStatus.CONTRADICTS})
            out.append(item)
        return out

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
                code_revision=_code_revision(),
                prompt_template_version=self.PROMPTS,
                started_at=started_at,
                ended_at=datetime.now(UTC),
            ),
            predictions=predictions,
            failure_groups=self._failure_groups(),
            unstable_test_ids=list(self.triage_result.unstable_test_ids) if self.triage_result else [],
            retrieved_evidence=self._labelled_evidence(),
            patches=self.patches,
            test_runs=self.test_runs,
            final_test_status=final_runs[-1].status if final_runs else TestRunStatus.NOT_RUN,
            integrity=self.integrity or IntegrityReport(),
            memory_reads=list(self.memory.events) if self.memory else [],
            memory_writes=self.memory_writes,
            tool_events=list(self.recorder.events),
            usage=self.llm.usage,
            errors=self.errors,
            outcome=RunOutcome(
                status=status,
                escalation=self._escalation_summary(),
                maintainer_approved=all(self.maintainer_votes) if self.maintainer_votes else None,
                revisions_used=self.revisions_used,
            ),
        )


def _disposition(g: GroupState | None, patches: list[PatchRecord]) -> str | None:
    if g is None or g.stage not in _GROUP_TERMINAL:
        return None
    if g.accepted:
        return "failed_final_verification" if g.final_failing else "accepted"
    undone = (PatchApplicationStatus.REVERTED, PatchApplicationStatus.REJECTED_BY_GUARD)
    return "reverted" if any(p.application_status in undone for p in patches) else "not_applied"


def _run_suite(worktree_path: Path, selectors: list[str], **kwargs: Any) -> PytestExecution:
    return run_tests(worktree_path, selectors, **kwargs)  # looked up at call time (patchable in tests)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise _ContractViolation(message)


class _ConditionNotImplemented(TestOrCodeError):
    """The configured condition has no code path yet; recorded as ORCHESTRATOR_ERROR."""


def _condition_problem(
    condition: ExperimentCondition, memory: MemoryStore | None, runnable: frozenset[ExperimentCondition]
) -> str | None:
    if condition not in _NO_MEMORY_CONDITIONS | _MEMORY_CONDITIONS:
        return f"condition {condition.value} is not implemented in run_episode yet"
    if condition not in runnable:
        return f"condition {condition.value} is not executed by this pipeline (use run_episode)"
    if condition in _MEMORY_CONDITIONS and memory is None:
        return f"condition {condition.value} needs a MemoryStore; none was given"
    return None


_REVISION: list[str | None] = []


def _code_revision() -> str | None:
    """Git sha of this repository (``-dirty`` with uncommitted changes); None if git is unavailable."""
    if not _REVISION:
        root = Path(__file__).resolve().parent
        try:
            sha = subprocess.run(
                ["git", "-C", str(root), "rev-parse", "HEAD"], check=False, capture_output=True, text=True, timeout=10
            )
            dirty = subprocess.run(
                ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
                check=False, capture_output=True, text=True, timeout=10,
            )
            value = sha.stdout.strip() if sha.returncode == 0 else None
            if value and dirty.returncode == 0 and dirty.stdout.strip():
                value += "-dirty"
        except (OSError, subprocess.SubprocessError):
            value = None
        _REVISION.append(value)
    return _REVISION[0]


# --------------------------------------------------------------------------- #
# B1: single tool-using agent
# --------------------------------------------------------------------------- #


class SingleAgentRun(EpisodeRun):
    """Condition B1 (proposal section 10): one tool-using agent, standard RAG, no memory, no Integrity Guard.

    Same model, tools, budgets, stability runs, patch validation/application and revision limit as the
    multi-agent run.  Differences: no Triage agent (the code-only ``stable_failures`` step is shared), the
    agent groups the tests itself in its ``B1Decision`` items, and a passing rerun of the item's test files
    decides (there is no independent verification).  Predictions are ``decided_by=SINGLE_AGENT`` and
    ``integrity.guard_ran`` stays False.
    """

    CONDITIONS = _SINGLE_AGENT_CONDITIONS
    PROMPTS = B1_VERSION

    def execute(self) -> ExperimentRunRecord:
        started_at = datetime.now(UTC)
        self._started = time.monotonic()
        self._pending: dict[str, B1Decision | str] = {}
        self._placement_retried: set[tuple[str, int]] = set()
        with episode_logging(self.episode.episode_id), recording(self.recorder):
            try:
                if self.condition_problem:
                    raise _ConditionNotImplemented(self.condition_problem)
                self._enter(None, Stage.TRIAGE)
                baseline, reproduced, unstable = _triage.stable_failures(
                    self.context, self.episode.ci_event, self.settings
                )
                self.test_runs.append(baseline.to_record("baseline", TestRunPhase.PRE_REPAIR))
                details: list[FailureGroupDetail] = []
                if reproduced:
                    self._enter(None, Stage.REPAIR)
                    step = self._ask(reproduced)
                    details = self._plan_groups(step, reproduced)
                self.triage_result = TriageResult(
                    groups=tuple(details), unstable_test_ids=unstable, baseline_run=baseline
                )
                for detail in details:
                    group = GroupState(detail=detail, stage=Stage.REPAIR)
                    self.groups.append(group)
                    self._run_group(group)
                self._final_suite()
                self._enter(None, Stage.DONE)
            except Exception as exc:  # noqa: BLE001 - every failure becomes an ErrorRecord
                self._fail(exc)
            return self._build_record(started_at)

    def _enter(self, group_id: str | None, stage: Stage) -> None:
        super()._enter(group_id, stage)
        self.recorder.role = AgentRole.SINGLE_AGENT

    def _handlers(self) -> dict[Stage, Callable[[GroupState], Stage]]:
        return {
            Stage.REPAIR: self._decide,
            Stage.PATCH_VALIDATION: self._patch_validation,
            Stage.PATCH_APPLICATION: self._patch_application,
            Stage.INTEGRITY_GUARD: self._verify,  # a plain rerun: B1 has no Guard
        }

    def _ask(self, tests: list[FailingTest], reasons: tuple[str, ...] = ()) -> B1Step | None:
        return run_single_agent(
            self.context, self.episode.ci_event, tests, llm=self.llm, tools=self._agent_tools(),
            rejection_reasons=reasons,
        )

    def _plan_groups(self, step: B1Step | None, tests: list[FailingTest]) -> list[FailureGroupDetail]:
        """One group per decision item; uncovered tests form one group that escalates."""
        by_id = {t.test_id: t for t in tests}
        source = [
            c.path for c in self.episode.ci_event.changed_files
            if c.unified_diff and self.context.edit_policy.classify_path(c.path) is PathClass.SRC
        ]
        claimed: set[str] = set()
        details: list[FailureGroupDetail] = []
        for item in step.decisions if step is not None else []:
            ids = [i for i in dict.fromkeys(item.test_ids) if i in by_id and i not in claimed]
            if not ids:
                continue
            claimed.update(ids)
            details.append(self._detail(len(details) + 1, [by_id[i] for i in ids], source))
            self._pending[details[-1].group.group_id] = item
        rest = [t for t in tests if t.test_id not in claimed]
        if rest:
            details.append(self._detail(len(details) + 1, rest, source))
            self._pending[details[-1].group.group_id] = (
                "the agent ran out of steps" if step is None else "the agent gave no decision for these tests"
            )
        return details

    def _detail(self, index: int, tests: list[FailingTest], source: list[str]) -> FailureGroupDetail:
        test_files = sorted({t.test_file for t in tests})
        return FailureGroupDetail(
            group=FailureGroup(group_id=f"g{index}", test_ids=[t.test_id for t in tests]),
            failing_tests=tuple(tests),
            affected_files=(*source, *test_files),
        )

    def _decide(self, g: GroupState) -> Stage:
        """Turn the agent's item for this group into an analysis and a patch proposal."""
        item = self._pending.pop(g.group_id, None)
        if item is None:  # revision: ask again for this group only, with the rejection reasons
            reasons = g.rejection_reasons or (g.previous_rejection.reasons if g.previous_rejection else ())
            item = self._item_for(g, self._ask(list(g.detail.failing_tests), tuple(reasons)), "on revision")
        if isinstance(item, str):
            return self._escalate(g, "insufficient_evidence", item, AgentRole.SINGLE_AGENT)
        try:
            g.analysis = IntentAnalysis(
                group_id=g.group_id,
                hypothesis=Hypothesis(kind=HypothesisKind(item.hypothesis), explanation=item.explanation),
                cited_evidence_ids=tuple(item.cited_evidence_ids),
                confidence=item.confidence,
                recommended_decision=DecisionLabel(item.decision),
            )
        except ValidationError as exc:
            reason = f"invalid decision: {describe_validation_error(exc)}"
            return self._escalate(g, "insufficient_evidence", reason, AgentRole.SINGLE_AGENT)
        unknown = unknown_evidence_ids(g.analysis.cited_evidence_ids, self._known_evidence_ids())
        if unknown:
            g.unknown_ids = tuple(unknown)
            return self._escalate(
                g, "insufficient_evidence", f"cited {len(unknown)} evidence id(s) that were never retrieved",
                AgentRole.SINGLE_AGENT,
            )
        problem = decision_policy_problem(g.analysis)
        if problem:
            trigger = "contradictory_evidence" if "both for and against" in problem else "insufficient_evidence"
            return self._escalate(g, trigger, problem, AgentRole.SINGLE_AGENT)
        if g.analysis.recommended_decision is DecisionLabel.ESCALATE:
            return self._escalate(g, "insufficient_evidence", item.explanation, AgentRole.SINGLE_AGENT)
        scope = scope_for_decision(g.analysis.recommended_decision)
        diff, problem = build_diff(self.context, item.edits, set(editable_files(self.context, g.detail, scope)))
        if not diff:
            # Same as the Repairer: one re-ask when the edits cannot be placed, then an escalation.  Placement is not
            # a rejected repair, so it does not use up the group's revision.
            if (g.group_id, g.revision) not in self._placement_retried:
                self._placement_retried.add((g.group_id, g.revision))
                reason = f"YOUR PREVIOUS EDITS COULD NOT BE APPLIED: {problem}"
                self._pending[g.group_id] = self._item_for(
                    g, self._ask(list(g.detail.failing_tests), (reason,)), "after an unplaceable edit"
                )
                return Stage.REPAIR
            return self._escalate(
                g, "insufficient_evidence", f"the agent could not produce an applicable edit: {problem}",
                AgentRole.SINGLE_AGENT,
            )
        g.proposal = PatchProposal(
            group_id=g.group_id,
            decision=RepairDirection(g.analysis.recommended_decision.value),
            scope=scope,
            target_files=tuple(sorted({e.path.removeprefix("./") for e in item.edits})),
            unified_diff=diff,
            rationale=item.explanation,
            evidence_ids=tuple(item.cited_evidence_ids),
            revision_index=g.revision,
        )
        g.patch = PatchRecord(
            patch_id=f"{g.group_id}-r{g.revision}", scope=scope, test_ids=list(g.detail.group.test_ids),
            target_files=list(g.proposal.target_files), unified_diff=diff, revision_index=g.revision,
        )
        self.patches.append(g.patch)
        return Stage.PATCH_VALIDATION

    def _verify(self, g: GroupState) -> Stage:
        """The same test-level verification as the Guard (``verify_group``), without the Guard's other checks:
        the group's tests pass and no previously passing test fails → decided; otherwise revert and reject."""
        assert g.patch is not None and g.analysis is not None
        targeted, regression, problems = _integrity_guard.verify_group(
            self.context, g.detail, self._known_failures(g),
            timeout_seconds=self.settings.test_timeout_seconds, python_executable=self.settings.test_python,
        )
        self._record_verification(g, targeted, regression)
        if not problems:
            self._applied_diff = None  # the repair stays
            g.accepted = True
            g.decision, g.decided_by = g.analysis.recommended_decision, AgentRole.SINGLE_AGENT
            return Stage.DECIDED
        self._revert()
        g.patch.application_status = PatchApplicationStatus.REVERTED
        g.patch.apply_error = "; ".join(problems)
        return self._rejected(g, tuple(problems))

    @staticmethod
    def _item_for(g: GroupState, step: B1Step | None, when: str) -> B1Decision | str:
        """The agent's decision covering this group's tests, or why there is none."""
        if step is None:
            return f"the agent ran out of steps ({when})"
        wanted = set(g.detail.group.test_ids)
        return next((d for d in step.decisions if wanted & set(d.test_ids)), f"the agent gave no decision ({when})")


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
    maintainer: Maintainer | None = None,
) -> ExperimentRunRecord:
    """Run one episode and return its run record.

    Takes ``EpisodeInput`` (never ``Episode``) so gold labels cannot reach the
    agents.  The condition comes from ``settings.experiment_condition``; memory is
    used only when its profile has memory (``memory`` is ignored otherwise).
    ``maintainer`` is the harness's simulated maintainer, asked after a Guard
    APPROVE; it may see gold labels, so it must never be handed to an agent.
    Execution failures are recorded in the returned record, not raised,
    wherever possible.

    ``retriever``, ``agents`` and ``patch_tools`` are optional injections; when
    omitted, the retriever is built from ``episode.corpus`` on first use and the
    real agent and tool modules are used.

    Raises:
        ConfigError: no model id is known, so no valid record can be written.
    """
    runner = SingleAgentRun if settings.experiment_condition is ExperimentCondition.B1 else EpisodeRun
    return runner(
        episode,
        worktree_path=worktree_path,
        settings=settings,
        llm=llm,
        memory=memory,
        retriever=retriever,
        agents=agents,
        patch_tools=patch_tools,
        maintainer=maintainer,
    ).execute()
