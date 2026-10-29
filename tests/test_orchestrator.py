"""Orchestrator scaffolding: stage wiring and transitions with fake agents, tools and LLM.

Fake agents ask the ``FakeLLMClient`` for their contract object, as the real agents
will, so every test also exercises the shared, metered LLM path.  All data is synthetic.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import BaseModel

from testorcode.agents.prompts import PROMPT_VERSION
from testorcode.config import Settings
from testorcode.errors import ConfigError, LLMRateLimitError, PatchError, ToolError
from testorcode.llm_client import FakeLLMClient, OpenAICompatibleClient
from testorcode.orchestrator import (
    Agents,
    EpisodeRun,
    PatchTools,
    SingleAgentRun,
    Stage,
    run_episode,
)
from testorcode.schemas.contracts import (
    EpisodeContext,
    EscalationRequest,
    FailureGroupDetail,
    GuardResult,
    Hypothesis,
    IntentAnalysis,
    MemoryReadResult,
    MemoryRecordCheck,
    PatchProposal,
    PatchValidation,
    PytestExecution,
    TriageResult,
)
from testorcode.schemas.episode_schemas import (
    DecisionLabel,
    EpisodeBudget,
    EpisodeInput,
    EvidenceSourceType,
    FailingTest,
    HypothesisKind,
    PatchScope,
    RepairDirection,
    ToolName,
)
from testorcode.schemas.result_schemas import (
    AgentRole,
    ErrorCategory,
    ExperimentRunRecord,
    GuardDecision,
    IntegrityReport,
    PatchApplicationStatus,
    ProbeStatus,
    RetrievedEvidence,
    RunStatus,
    SupportStatus,
    TestRunPhase,
    TestRunStatus,
    ToolCallStatus,
)
from testorcode.schemas.tests.conftest import HEAD, make_s1, make_s3, memory_record

WT = Path("/tmp/testorcode-fake-worktree")  # never touched: patch tools are fakes
# A2 (no memory) is the condition the pipeline runs without a MemoryStore; FULL needs one.
SETTINGS = Settings.from_env({"LLM_MODEL": "fake-model-1", "EXPERIMENT_CONDITION": "A2"})
FULL_SETTINGS = Settings.from_env({"LLM_MODEL": "fake-model-1", "EXPERIMENT_CONDITION": "FULL"})
TEST_DIFF = "--- a/tests/test_synthlib.py\n+++ b/tests/test_synthlib.py\n@@\n-x\n+y\n"
SRC_DIFF = "--- a/src/synthlib/core.py\n+++ b/src/synthlib/core.py\n@@\n-x\n+y\n"
# Contracts use str_strip_whitespace, so the diff reaching the tools has lost its final newline.
STORED_TEST_DIFF = TEST_DIFF.rstrip()


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #


class RepairReply(BaseModel):
    """How a Repairer can ask for the ``RepairerOutput`` union as one response model."""

    output: PatchProposal | EscalationRequest


def triage_result(episode: EpisodeInput) -> TriageResult:
    ci = episode.ci_event
    groups = tuple(
        FailureGroupDetail(group=g, failing_tests=tuple(t for t in ci.failing_tests if t.test_id in g.test_ids))
        for g in ci.failure_groups
    )
    baseline = PytestExecution(
        selectors=tuple(t.test_id for t in ci.failing_tests), command=("pytest",),
        status=TestRunStatus.FAILED, exit_code=1, failed=len(ci.failing_tests), failures=tuple(ci.failing_tests),
    )
    return TriageResult(groups=groups, baseline_run=baseline)


def analysis(group_id: str, decision: DecisionLabel = DecisionLabel.UPDATE_TEST, **kw) -> IntentAnalysis:
    kind = {
        DecisionLabel.UPDATE_TEST: HypothesisKind.OUTDATED_TEST,
        DecisionLabel.FIX_CODE: HypothesisKind.REGRESSION,
        DecisionLabel.ESCALATE: HypothesisKind.UNKNOWN,
    }[decision]
    base = {
        "group_id": group_id,
        "hypothesis": Hypothesis(kind=kind, explanation=f"{kind.value} explanation"),
        "cited_evidence_ids": ("ev-msg",) if decision is DecisionLabel.UPDATE_TEST else (),
        "confidence": 0.7,
        "recommended_decision": decision,
    }
    return IntentAnalysis(**{**base, **kw})


def proposal(group_id: str, direction: RepairDirection = RepairDirection.UPDATE_TEST) -> RepairReply:
    test = direction is RepairDirection.UPDATE_TEST
    return RepairReply(output=PatchProposal(
        group_id=group_id, decision=direction, scope=PatchScope.TEST if test else PatchScope.SRC,
        target_files=("tests/test_synthlib.py",) if test else ("src/synthlib/core.py",),
        unified_diff=TEST_DIFF if test else SRC_DIFF, rationale="synthetic",
        evidence_ids=("ev-msg",) if test else (),
    ))


def guard(decision: GuardDecision) -> GuardResult:
    run = PytestExecution(
        selectors=("tests",), command=("pytest",), exit_code=0 if decision is GuardDecision.APPROVE else 1,
        status=TestRunStatus.PASSED if decision is GuardDecision.APPROVE else TestRunStatus.FAILED,
        passed=1, failed=0 if decision is GuardDecision.APPROVE else 1,
    )
    reasons = () if decision is GuardDecision.APPROVE else ("assertion removed",)
    return GuardResult(
        decision=decision, reasons=reasons, post_repair_run=run,
        integrity=IntegrityReport(guard_ran=True, guard_decision=decision, rejection_reasons=list(reasons)),
    )


class FakeAgents:
    """Agent stand-ins: each asks the LLM for its contract, as the real agents will."""

    def __init__(self) -> None:
        self.repair_calls: list[GuardResult | None] = []
        self.repair_reasons: list[tuple[str, ...]] = []
        self.memory_record = None  # None = the real Guard builder (not implemented yet)
        self.known_failures: list[frozenset[str]] = []

    @staticmethod
    def ask(llm, role: AgentRole, model: type[BaseModel]):
        return llm.complete_structured(role=role, system="fake", user="fake", response_model=model).parsed

    def triage(self, context, ci_event, *, llm, settings):
        return self.ask(llm, AgentRole.TRIAGE, TriageResult)

    def intent(self, context, group, *, retriever, memory, llm):
        retriever.search(group.group.function_qualname or "", top_k=3)
        if memory is not None:
            memory.read(context, group.group.function_qualname)
        return self.ask(llm, AgentRole.INTENT_ANALYST, IntentAnalysis)

    def repair(self, context, group, analysis, *, llm, previous_rejection=None, rejection_reasons=(), tools=None):
        self.repair_calls.append(previous_rejection)
        self.repair_reasons.append(rejection_reasons)
        return self.ask(llm, AgentRole.REPAIRER, RepairReply).output

    def guard(self, context, group, analysis, proposal, *, llm=None, known_failures=frozenset(), **_):
        self.known_failures.append(known_failures)
        return self.ask(llm, AgentRole.INTEGRITY_GUARD, GuardResult)

    def agents(self) -> Agents:
        extra = {"memory_record": self.memory_record} if self.memory_record else {}
        return Agents(
            triage=self.triage, intent_analyst=self.intent, repairer=self.repair, integrity_guard=self.guard, **extra
        )


class FakeRetriever:
    def __init__(self, ids=("ev-msg", "ev-changelog")) -> None:
        self.ids = ids

    def search(self, query: str, *, top_k: int) -> list[RetrievedEvidence]:
        return [
            RetrievedEvidence(evidence_id=i, source_type=EvidenceSourceType.COMMIT_MESSAGE, source_ref=HEAD, rank=n)
            for n, i in enumerate(self.ids[:top_k], start=1)
        ]


class FakePatchTools:
    def __init__(self, validations=(), apply_errors=()) -> None:
        self.validations = list(validations)
        self.apply_errors = list(apply_errors)
        self.applied: list[str] = []
        self.reverted: list[str] = []

    def validate(self, worktree_path, diff, *, policy, scope):
        if self.validations:
            return self.validations.pop(0)
        return PatchValidation(valid=True, allowed=True, changed_paths=("tests/test_synthlib.py",))

    def apply(self, worktree_path, diff, *, policy, scope):
        if self.apply_errors:
            raise self.apply_errors.pop(0)
        self.applied.append(diff)
        return PatchValidation(valid=True, allowed=True)

    def revert(self, worktree_path, diff, *, policy=None):
        self.reverted.append(diff)

    def tools(self) -> PatchTools:
        # No final whole-suite run: the worktree here is fictional (tests of it use their own run_suite).
        return PatchTools(validate=self.validate, apply=self.apply, revert=self.revert, run_suite=None)


def run(episode: EpisodeInput, replies: dict, *, tools: FakePatchTools | None = None,
        settings: Settings = SETTINGS, memory=None, retriever=None, fake: FakeAgents | None = None,
        maintainer=None):
    fake = fake or FakeAgents()
    tools = tools or FakePatchTools()
    if memory is not None and settings is SETTINGS:
        settings = FULL_SETTINGS
    replies = {AgentRole.TRIAGE: [triage_result(episode)], **replies}
    episode_run = EpisodeRun(
        episode, worktree_path=WT, settings=settings, llm=FakeLLMClient(replies), memory=memory,
        retriever=retriever or FakeRetriever(), agents=fake.agents(), patch_tools=tools.tools(),
        maintainer=maintainer,
    )
    record = episode_run.execute()
    assert ExperimentRunRecord.model_validate_json(record.to_jsonl()) == record
    return record, episode_run, tools, fake


S1 = make_s1().agent_input
S3 = make_s3().agent_input
G1 = "g-1"


def stages(episode_run: EpisodeRun, group: str | None = G1) -> list[Stage]:
    return [s for g, s in episode_run.trace if g == group]


# --------------------------------------------------------------------------- #
# Happy path and wiring
# --------------------------------------------------------------------------- #


def test_happy_path_walks_every_stage_and_records_the_run() -> None:
    record, er, tools, _ = run(S1, {
        AgentRole.INTENT_ANALYST: [analysis(G1)],
        AgentRole.REPAIRER: [proposal(G1)],
        AgentRole.INTEGRITY_GUARD: [guard(GuardDecision.APPROVE)],
    })
    assert er.trace == [
        (None, Stage.TRIAGE),
        (G1, Stage.INTENT_ANALYSIS), (G1, Stage.REPAIR), (G1, Stage.PATCH_VALIDATION),
        (G1, Stage.PATCH_APPLICATION), (G1, Stage.INTEGRITY_GUARD), (G1, Stage.DECIDED),
        (None, Stage.DONE),
    ]
    assert record.outcome.status is RunStatus.COMPLETED and record.outcome.escalation is None
    [pred] = record.predictions
    assert pred.predicted_action is DecisionLabel.UPDATE_TEST and pred.cited_evidence_ids == ["ev-msg"]
    assert record.patches[0].application_status is PatchApplicationStatus.APPLIED
    assert [r.phase for r in record.test_runs] == [TestRunPhase.PRE_REPAIR, TestRunPhase.POST_REPAIR]
    assert record.final_test_status is TestRunStatus.PASSED
    assert record.integrity.guard_decision is GuardDecision.APPROVE
    assert {e.evidence_id for e in record.retrieved_evidence} == {"ev-msg", "ev-changelog"}
    assert record.provenance.model_id == "fake-model-1" and record.provenance.seed == 42
    assert record.usage.llm_calls == 4 and record.errors == []
    assert tools.applied == [STORED_TEST_DIFF] and tools.reverted == []


def test_context_is_built_from_episode_input_and_shared() -> None:
    seen: list[EpisodeContext] = []
    fake = FakeAgents()
    original = fake.intent

    def intent(context, group, **kw):
        seen.append(context)
        return original(context, group, **kw)

    fake.intent = intent
    run(S1, {AgentRole.INTENT_ANALYST: [analysis(G1, DecisionLabel.ESCALATE)]}, fake=fake)
    [ctx] = seen
    assert ctx == EpisodeContext.from_episode_input(S1, WT)
    assert ctx.commit_sha == S1.ci_event.commit.sha and ctx.worktree_path == WT


def test_mixed_episode_decides_one_group_and_escalates_the_other() -> None:
    record, er, _, _ = run(S3, {
        AgentRole.INTENT_ANALYST: [analysis("g-n"), analysis("g-s", DecisionLabel.ESCALATE)],
        AgentRole.REPAIRER: [proposal("g-n")],
        AgentRole.INTEGRITY_GUARD: [guard(GuardDecision.APPROVE)],
    })
    assert stages(er, "g-n")[-1] is Stage.DECIDED and stages(er, "g-s")[-1] is Stage.ESCALATED
    assert record.outcome.status is RunStatus.COMPLETED
    actions = record.predicted_actions()
    assert set(actions.values()) == {DecisionLabel.UPDATE_TEST, DecisionLabel.ESCALATE}
    assert record.outcome.escalation.trigger == "insufficient_evidence"


# --------------------------------------------------------------------------- #
# Escalation and revision transitions
# --------------------------------------------------------------------------- #


def test_analyst_escalation_skips_repair() -> None:
    record, er, tools, fake = run(S1, {AgentRole.INTENT_ANALYST: [analysis(G1, DecisionLabel.ESCALATE)]})
    assert stages(er) == [Stage.INTENT_ANALYSIS, Stage.ESCALATED]
    assert fake.repair_calls == [] and tools.applied == []
    assert record.outcome.status is RunStatus.ESCALATED
    assert record.predictions[0].decided_by is AgentRole.INTENT_ANALYST


def test_invented_citation_escalates_instead_of_repairing() -> None:
    record, er, _, fake = run(S1, {
        AgentRole.INTENT_ANALYST: [analysis(G1, cited_evidence_ids=("ev-invented",))],
    })
    assert stages(er) == [Stage.INTENT_ANALYSIS, Stage.ESCALATED] and fake.repair_calls == []
    assert record.predictions[0].decided_by is AgentRole.ORCHESTRATOR
    assert record.unretrieved_citations() == {S1.ci_event.failing_tests[0].test_id: ["ev-invented"]}


def test_repairer_escalation_request() -> None:
    req = RepairReply(output=EscalationRequest(group_id=G1, trigger="contradictory_evidence", summary="docs disagree"))
    record, er, _, _ = run(S1, {AgentRole.INTENT_ANALYST: [analysis(G1)], AgentRole.REPAIRER: [req]})
    assert stages(er) == [Stage.INTENT_ANALYSIS, Stage.REPAIR, Stage.ESCALATED]
    assert record.outcome.escalation.trigger == "contradictory_evidence"
    assert record.predictions[0].decided_by is AgentRole.REPAIRER


def test_guard_rejection_gets_one_revision_then_approval() -> None:
    record, er, tools, fake = run(S1, {
        AgentRole.INTENT_ANALYST: [analysis(G1)],
        AgentRole.REPAIRER: [proposal(G1), proposal(G1)],
        AgentRole.INTEGRITY_GUARD: [guard(GuardDecision.REJECT), guard(GuardDecision.APPROVE)],
    })
    assert stages(er).count(Stage.REPAIR) == 2 and stages(er)[-1] is Stage.DECIDED
    assert fake.repair_calls[0] is None and fake.repair_calls[1].decision is GuardDecision.REJECT
    assert [p.application_status for p in record.patches] == [
        PatchApplicationStatus.REJECTED_BY_GUARD, PatchApplicationStatus.APPLIED,
    ]
    assert [p.revision_index for p in record.patches] == [0, 1]
    assert tools.reverted == [STORED_TEST_DIFF]
    assert record.outcome.revisions_used == 1 and record.final_test_status is TestRunStatus.PASSED
    assert record.test_runs[-1].phase is TestRunPhase.POST_REVISION


def test_second_rejection_escalates_and_leaves_worktree_reverted() -> None:
    record, er, tools, _ = run(S1, {
        AgentRole.INTENT_ANALYST: [analysis(G1)],
        AgentRole.REPAIRER: [proposal(G1), proposal(G1)],
        AgentRole.INTEGRITY_GUARD: [guard(GuardDecision.REJECT), guard(GuardDecision.REJECT)],
    })
    assert stages(er)[-1] is Stage.ESCALATED and stages(er).count(Stage.REPAIR) == 2
    assert record.outcome.status is RunStatus.ESCALATED
    assert record.outcome.escalation.trigger == "guard_rejection_after_revision"
    assert len(tools.reverted) == 2 and record.final_test_status is TestRunStatus.FAILED


def test_revision_limit_zero_escalates_on_first_rejection() -> None:
    settings = Settings.from_env({"LLM_MODEL": "fake-model-1", "MAX_REPAIR_REVISIONS": "0", "EXPERIMENT_CONDITION": "A2"})
    record, _, _, fake = run(S1, {
        AgentRole.INTENT_ANALYST: [analysis(G1)],
        AgentRole.REPAIRER: [proposal(G1)],
        AgentRole.INTEGRITY_GUARD: [guard(GuardDecision.REJECT)],
    }, settings=settings)
    assert len(fake.repair_calls) == 1 and record.outcome.status is RunStatus.ESCALATED


def test_out_of_scope_patch_is_blocked_and_revised() -> None:
    bad = PatchValidation(valid=True, allowed=False, changed_paths=("tests/conftest.py",),
                          rejection_reasons=("forbidden path",))
    record, er, tools, _ = run(S1, {
        AgentRole.INTENT_ANALYST: [analysis(G1)],
        AgentRole.REPAIRER: [proposal(G1), proposal(G1)],
        AgentRole.INTEGRITY_GUARD: [guard(GuardDecision.APPROVE)],
    }, tools=FakePatchTools(validations=[bad]))
    assert stages(er)[:4] == [Stage.INTENT_ANALYSIS, Stage.REPAIR, Stage.PATCH_VALIDATION, Stage.REPAIR]
    first = record.patches[0]
    assert first.application_status is PatchApplicationStatus.BLOCKED_OUT_OF_SCOPE
    assert first.out_of_scope_files == ["tests/conftest.py"]
    assert tools.applied == [STORED_TEST_DIFF] and record.outcome.status is RunStatus.COMPLETED


def test_patch_that_does_not_apply_is_a_rejection_not_a_guard_verdict() -> None:
    record, er, _, _ = run(S1, {
        AgentRole.INTENT_ANALYST: [analysis(G1)],
        AgentRole.REPAIRER: [proposal(G1), proposal(G1)],
        AgentRole.INTEGRITY_GUARD: [guard(GuardDecision.APPROVE)],
    }, tools=FakePatchTools(apply_errors=[PatchError("hunk 1 failed")]))
    first = record.patches[0]
    assert first.application_status is PatchApplicationStatus.FAILED_TO_APPLY
    assert first.apply_error == "hunk 1 failed"
    assert stages(er)[-1] is Stage.DECIDED


def test_proposal_contradicting_analysis_direction_is_rejected() -> None:
    record, _, tools, _ = run(S1, {
        AgentRole.INTENT_ANALYST: [analysis(G1)],
        AgentRole.REPAIRER: [proposal(G1, RepairDirection.FIX_CODE), proposal(G1, RepairDirection.FIX_CODE)],
    })
    assert record.outcome.status is RunStatus.ESCALATED and tools.applied == []
    assert "contradicts" in record.patches[0].apply_error


# --------------------------------------------------------------------------- #
# Memory boundary
# --------------------------------------------------------------------------- #


class FakeMemory:
    def __init__(self) -> None:
        self.writes = 0

    def read(self, context, function_qualname):
        rec = memory_record("mem-1")
        return MemoryReadResult(checks=(MemoryRecordCheck(record=rec, probe_status=ProbeStatus.PASSED),))

    def write(self, record, approval):
        self.writes += 1


def test_memory_reads_are_logged_and_cited_records_marked() -> None:
    record, _, _, _ = run(S1, {
        AgentRole.INTENT_ANALYST: [analysis(G1, cited_evidence_ids=("mem-1",))],
        AgentRole.REPAIRER: [proposal(G1)],
        AgentRole.INTEGRITY_GUARD: [guard(GuardDecision.APPROVE)],
    }, memory=FakeMemory())
    [event] = record.memory_reads
    assert event.record_id == "mem-1" and event.used_as_evidence is True
    assert record.outcome.status is RunStatus.COMPLETED


def test_agents_cannot_write_memory() -> None:
    store = FakeMemory()
    fake = FakeAgents()

    def intent(context, group, *, retriever, memory, llm):
        memory.write(memory_record("mem-x"), None)

    fake.intent = intent
    record, er, _, _ = run(S1, {}, memory=store, fake=fake)
    assert store.writes == 0
    assert record.outcome.status is RunStatus.ORCHESTRATOR_ERROR
    assert er.trace[-1] == (G1, Stage.FAILED)


# --------------------------------------------------------------------------- #
# Missing dependencies and execution failures
# --------------------------------------------------------------------------- #


def test_default_agents_tool_failure_is_not_an_escalation() -> None:
    # The real Triage runs pytest; WT does not exist, so the run fails as a tool failure, never an escalation.
    record = run_episode(S1, worktree_path=WT, settings=SETTINGS, llm=FakeLLMClient())
    assert record.outcome.status is RunStatus.TOOL_FAILURE
    [err] = record.errors
    assert err.fatal and err.operation == "triage"
    assert record.predictions == [] and record.outcome.escalation is None


def test_missing_retriever_fails_at_intent_analysis() -> None:
    # A corpus that is not materialised cannot be indexed (building it from a worktree is deferred), so no
    # retriever exists: an explicit failure, never a silent empty search.
    episode = S1.model_copy(update={"corpus": S1.corpus.model_copy(update={"items": None})})
    fake = FakeAgents()
    er = EpisodeRun(
        episode, worktree_path=WT, settings=SETTINGS, agents=fake.agents(),
        llm=FakeLLMClient({AgentRole.TRIAGE: [triage_result(S1)]}), patch_tools=FakePatchTools().tools(),
    )
    record = er.execute()
    assert record.outcome.status is RunStatus.ORCHESTRATOR_ERROR
    assert record.errors[0].operation == f"intent_analysis:{G1}"
    assert "not implemented" in record.errors[0].message.lower()
    assert er.trace[-1] == (G1, Stage.FAILED)


def test_default_retriever_is_built_from_the_episode_corpus() -> None:
    seen: dict[str, list[str]] = {}
    fake = FakeAgents()

    def intent(context, group, *, retriever, memory, llm):
        seen["hits"] = [h.evidence_id for h in retriever.search("normalise capitalised string", top_k=3)]
        raise ToolError("stop after searching")

    agents = Agents(triage=fake.triage, intent_analyst=intent, repairer=fake.repair, integrity_guard=fake.guard)
    er = EpisodeRun(
        S1, worktree_path=WT, settings=SETTINGS, agents=agents,
        llm=FakeLLMClient({AgentRole.TRIAGE: [triage_result(S1)]}), patch_tools=FakePatchTools().tools(),
    )
    record = er.execute()
    corpus_ids = {i.evidence_id for i in S1.corpus.items}
    assert seen["hits"] and set(seen["hits"]) <= corpus_ids
    assert {e.evidence_id for e in record.retrieved_evidence} == set(seen["hits"])  # logged for the run record


def test_provider_failure_is_api_failure_not_counted_against_agent() -> None:
    record, _, _, _ = run(S1, {AgentRole.INTENT_ANALYST: [LLMRateLimitError("RateLimitError (HTTP 429)")]})
    assert record.outcome.status is RunStatus.API_FAILURE
    [err] = record.errors
    assert err.category is ErrorCategory.RATE_LIMIT and err.counted_against_agent is False
    assert record.predictions == []


def test_llm_call_budget_is_enforced() -> None:
    tight = S1.model_copy(update={"budget": EpisodeBudget(max_llm_calls=2)})
    record, _, _, _ = run(tight, {
        AgentRole.INTENT_ANALYST: [analysis(G1)], AgentRole.REPAIRER: [proposal(G1)],
    })
    assert record.outcome.status is RunStatus.BUDGET_EXHAUSTED
    assert record.errors[0].operation == f"repair:{G1}"


def test_tool_failure_after_apply_reverts_patch() -> None:
    fake = FakeAgents()

    def failing_guard(*args, **kw):
        raise ToolError("pytest could not start")

    fake.guard = failing_guard
    record, _, tools, _ = run(S1, {
        AgentRole.INTENT_ANALYST: [analysis(G1)], AgentRole.REPAIRER: [proposal(G1)],
    }, fake=fake)
    assert record.outcome.status is RunStatus.TOOL_FAILURE
    assert tools.reverted == [STORED_TEST_DIFF]


def test_wrong_group_output_is_a_contract_violation() -> None:
    record, _, _, _ = run(S1, {AgentRole.INTENT_ANALYST: [analysis("other-group")]})
    assert record.outcome.status is RunStatus.ORCHESTRATOR_ERROR
    assert "different group" in record.errors[0].message


def test_no_model_id_is_a_config_error() -> None:
    class Anonymous:
        def complete_structured(self, **kw):  # pragma: no cover - never reached
            raise AssertionError

    with pytest.raises(ConfigError, match="LLM_MODEL"):
        run_episode(S1, worktree_path=WT, settings=Settings.from_env({}), llm=Anonymous())


# --------------------------------------------------------------------------- #
# Regressions
# --------------------------------------------------------------------------- #


def test_invalid_structured_output_is_not_reported_as_api_failure() -> None:
    record, _, _, _ = run(S1, {AgentRole.INTENT_ANALYST: [{"group_id": G1}]})  # missing fields
    assert record.outcome.status is RunStatus.INCOMPLETE
    [err] = record.errors
    assert err.category is ErrorCategory.VALIDATION_ERROR and err.counted_against_agent is True
    assert "IntentAnalysis" in err.message and err.operation == f"intent_analysis:{G1}"


def test_usage_of_invalid_replies_reaches_the_run_record() -> None:
    from types import SimpleNamespace as NS

    def reply(text: str):
        return NS(model="served-model-1", usage=NS(prompt_tokens=10, completion_tokens=5, total_tokens=15),
                  choices=[NS(finish_reason="stop", message=NS(content=text))])

    replies = [reply(triage_result(S1).model_dump_json()), reply("{bad"), reply("{still bad")]
    sdk = NS(chat=NS(completions=NS(create=lambda **kw: replies.pop(0))))
    llm = OpenAICompatibleClient(
        Settings.from_env({"LLM_MODEL": "m-1", "LLM_BASE_URL": "http://llm.invalid/v1", "EXPERIMENT_CONDITION": "A2"}),
        sdk_client=sdk, sleep=lambda s: None,
    )
    record = EpisodeRun(
        S1, worktree_path=WT, settings=SETTINGS, llm=llm, retriever=FakeRetriever(),
        agents=FakeAgents().agents(), patch_tools=FakePatchTools().tools(),
    ).execute()
    assert record.outcome.status is RunStatus.INCOMPLETE
    assert record.usage.total_tokens == 45 and record.usage.llm_calls == 3


def test_overlapping_triage_groups_return_a_failure_record() -> None:
    tr = triage_result(S1)
    detail = tr.groups[0]
    twin = detail.model_copy(update={"group": detail.group.model_copy(update={"group_id": "g-2"})})
    overlapping = TriageResult(groups=(detail, twin), baseline_run=tr.baseline_run)
    fake = FakeAgents()
    llm = FakeLLMClient({AgentRole.TRIAGE: [overlapping], AgentRole.INTENT_ANALYST: [analysis(G1)]})
    er = EpisodeRun(S1, worktree_path=WT, settings=SETTINGS, llm=llm, retriever=FakeRetriever(),
                    agents=fake.agents(), patch_tools=FakePatchTools().tools())
    record = er.execute()
    assert record.outcome.status is RunStatus.ORCHESTRATOR_ERROR
    assert detail.group.test_ids[0] in record.errors[0].message
    assert record.predictions == [] and llm.remaining(AgentRole.INTENT_ANALYST) == 1  # no group processed
    assert er.trace[-1] == (None, Stage.FAILED)


def test_triage_without_groups_for_failing_tests_is_an_error() -> None:
    tr = triage_result(S1)
    empty = TriageResult(groups=(), baseline_run=tr.baseline_run)
    record, er, _, _ = run(S1, {AgentRole.TRIAGE: [empty]})
    assert record.outcome.status is RunStatus.ORCHESTRATOR_ERROR
    assert "ungrouped" in record.errors[0].message and record.predictions == []
    assert er.trace[-1] == (None, Stage.FAILED)


def test_all_failures_unstable_is_incomplete_not_completed() -> None:
    tr = triage_result(S1)
    unstable = TriageResult(
        groups=(), baseline_run=tr.baseline_run,
        unstable_test_ids=tuple(t.test_id for t in S1.ci_event.failing_tests),
    )
    record, _, _, _ = run(S1, {AgentRole.TRIAGE: [unstable]})
    assert record.outcome.status is RunStatus.INCOMPLETE and record.errors == []


def test_explicit_guard_escalate_is_not_a_rejection() -> None:
    reasons = ("cannot verify intent from tests alone",)
    escalate = GuardResult(
        decision=GuardDecision.ESCALATE, reasons=reasons,
        integrity=IntegrityReport(guard_ran=True, guard_decision=GuardDecision.ESCALATE),
    )
    record, er, tools, fake = run(S1, {
        AgentRole.INTENT_ANALYST: [analysis(G1)], AgentRole.REPAIRER: [proposal(G1)],
        AgentRole.INTEGRITY_GUARD: [escalate],
    })
    assert stages(er)[-2:] == [Stage.INTEGRITY_GUARD, Stage.ESCALATED]
    assert len(fake.repair_calls) == 1 and record.outcome.revisions_used == 0  # no revision
    assert record.outcome.status is RunStatus.ESCALATED
    assert record.outcome.escalation.trigger != "guard_rejection_after_revision"
    assert reasons[0] in record.outcome.escalation.summary
    [pred] = record.predictions
    assert pred.predicted_action is DecisionLabel.ESCALATE and pred.decided_by is AgentRole.INTEGRITY_GUARD
    assert record.patches[0].application_status is PatchApplicationStatus.REVERTED
    assert tools.reverted == [STORED_TEST_DIFF]
    assert record.integrity.guard_decision is GuardDecision.ESCALATE


# --------------------------------------------------------------------------- #
# Day 2: conditions, revision feedback, tool trajectory, maintainer, summaries
# --------------------------------------------------------------------------- #

APPROVED = {
    AgentRole.INTENT_ANALYST: [analysis(G1)],
    AgentRole.REPAIRER: [proposal(G1)],
    AgentRole.INTEGRITY_GUARD: [guard(GuardDecision.APPROVE)],
}


@pytest.mark.parametrize("condition", ["A3", "A4", "A5", "B0", "B2", "TRIVIAL_UPDATE_TEST"])
def test_condition_without_a_code_path_fails_explicitly(condition: str) -> None:
    settings = Settings.from_env({"LLM_MODEL": "fake-model-1", "EXPERIMENT_CONDITION": condition})
    record, er, _, _ = run(S1, {}, settings=settings)
    assert record.outcome.status is RunStatus.ORCHESTRATOR_ERROR
    [err] = record.errors
    assert "not implemented" in err.message and err.operation == "setup"
    assert record.predictions == [] and er.trace == [(None, Stage.FAILED)]
    assert record.provenance.condition.value == condition


def test_full_without_memory_store_fails_explicitly() -> None:
    record, _, _, _ = run(S1, {}, settings=FULL_SETTINGS)
    assert record.outcome.status is RunStatus.ORCHESTRATOR_ERROR
    assert "needs a MemoryStore" in record.errors[0].message


A1_SETTINGS = Settings.from_env({"LLM_MODEL": "fake-model-1", "EXPERIMENT_CONDITION": "A1"})


def a1_intent(seen: dict):
    """An Intent Analyst stand-in that records what A1 hands it and reads memory, as the real one does."""

    def intent(context, group, *, retriever, memory, llm):
        seen.update(retriever=retriever, allowed_tools=context.allowed_tools)
        if memory is not None:
            memory.read(context, group.group.function_qualname)
        return FakeAgents.ask(llm, AgentRole.INTENT_ANALYST, IntentAnalysis)

    return intent


def test_a1_runs_with_memory_but_without_any_retrieval() -> None:
    seen: dict = {}
    fake = FakeAgents()
    fake.intent = a1_intent(seen)
    record, er, _, _ = run(S1, {
        # Memory is still citable; a patch citing a corpus id would be rejected (nothing was retrieved).
        AgentRole.INTENT_ANALYST: [analysis(G1, DecisionLabel.FIX_CODE, cited_evidence_ids=("mem-1",))],
        AgentRole.REPAIRER: [proposal(G1, RepairDirection.FIX_CODE)],
        AgentRole.INTEGRITY_GUARD: [guard(GuardDecision.APPROVE)],
    }, settings=A1_SETTINGS, memory=FakeMemory(), fake=fake)
    assert record.outcome.status is RunStatus.COMPLETED and record.condition.value == "A1"
    assert seen["retriever"] is None and ToolName.SEARCH_REPO not in seen["allowed_tools"]
    assert er.episode.ci_event.commit_message is None and er.episode.corpus.items == []
    assert record.retrieved_evidence == []
    assert not any(e.tool is ToolName.SEARCH_REPO for e in record.tool_events)
    assert [m.record_id for m in record.memory_reads] == ["mem-1"] and record.memory_reads[0].used_as_evidence
    assert record.provenance.prompt_template_version == PROMPT_VERSION


def test_a1_citing_corpus_evidence_escalates() -> None:
    # Nothing was retrieved, so an id from the (withheld) corpus is an invented citation: escalate, never repair.
    fake = FakeAgents()
    fake.intent = a1_intent({})
    record, er, _, _ = run(S1, {AgentRole.INTENT_ANALYST: [analysis(G1)]}, settings=A1_SETTINGS,
                           memory=FakeMemory(), fake=fake)
    assert record.outcome.status is RunStatus.ESCALATED
    assert record.outcome.escalation.trigger == "insufficient_evidence"
    assert Stage.REPAIR not in stages(er) and record.patches == []


@pytest.mark.parametrize(("condition", "code_only"), [("FULL", False), ("A2", False), ("B1", False), ("A1", True)])
def test_only_a1_limits_read_file_to_code(condition: str, code_only: bool) -> None:
    # The restriction follows the profile (no retrieval of any kind); FULL/A2/B1 keep the unrestricted read_file.
    settings = Settings.from_env({"LLM_MODEL": "fake-model-1", "EXPERIMENT_CONDITION": condition})
    runner = SingleAgentRun if condition == "B1" else EpisodeRun
    er = runner(S1, worktree_path=WT, settings=settings, llm=FakeLLMClient(), memory=FakeMemory(),
                retriever=FakeRetriever(), patch_tools=FakePatchTools().tools())
    assert er._agent_tools().code_only is code_only


def test_a1_without_memory_store_fails_explicitly() -> None:
    record, _, _, _ = run(S1, {}, settings=A1_SETTINGS)
    assert record.outcome.status is RunStatus.ORCHESTRATOR_ERROR
    assert "needs a MemoryStore" in record.errors[0].message


@pytest.mark.parametrize(
    ("condition", "action"),
    [("TRIVIAL_UPDATE_TEST", DecisionLabel.UPDATE_TEST), ("TRIVIAL_FIX_CODE", DecisionLabel.FIX_CODE)],
)
def test_run_episode_answers_trivial_baselines_without_model_or_worktree(condition, action) -> None:
    settings = Settings.from_env({"EXPERIMENT_CONDITION": condition, "RANDOM_SEED": "3"})
    record = run_episode(S1, worktree_path=Path("/nonexistent/worktree"), settings=settings, llm=None)
    assert record.condition.value == condition and record.provenance.seed == 3
    assert {p.test_id for p in record.predictions} == {t.test_id for t in S1.ci_event.failing_tests}
    assert {p.predicted_action for p in record.predictions} == {action}
    assert record.usage.llm_calls == 0 and record.tool_events == [] and record.patches == []
    assert record.final_test_status is TestRunStatus.NOT_RUN and record.outcome.status is RunStatus.COMPLETED


def test_a2_ignores_a_memory_store_and_hides_visible_memory() -> None:
    class CountingMemory(FakeMemory):
        reads = 0

        def read(self, context, function_qualname):
            CountingMemory.reads += 1
            return super().read(context, function_qualname)

    a2 = Settings.from_env({"LLM_MODEL": "fake-model-1", "EXPERIMENT_CONDITION": "A2"})
    _, er, _, _ = run(S1, APPROVED, memory=CountingMemory(), settings=a2)
    assert CountingMemory.reads == 0 and er.episode.visible_memory is None


def test_validation_reasons_reach_the_repairer_on_revision() -> None:
    bad = PatchValidation(valid=False, allowed=True, changed_paths=("tests/test_synthlib.py",),
                          rejection_reasons=("hunk is truncated",))
    _, _, _, fake = run(S1, {
        AgentRole.INTENT_ANALYST: [analysis(G1)],
        AgentRole.REPAIRER: [proposal(G1), proposal(G1)],
        AgentRole.INTEGRITY_GUARD: [guard(GuardDecision.APPROVE)],
    }, tools=FakePatchTools(validations=[bad]))
    assert fake.repair_reasons == [(), ("hunk is truncated",)]
    assert fake.repair_calls == [None, None]


def test_guard_rejection_clears_earlier_validation_reasons() -> None:
    bad = PatchValidation(valid=False, allowed=True, rejection_reasons=("does not apply",))
    settings = Settings.from_env({"LLM_MODEL": "fake-model-1", "EXPERIMENT_CONDITION": "A2"})
    _, _, _, fake = run(S1, {
        AgentRole.INTENT_ANALYST: [analysis(G1)],
        AgentRole.REPAIRER: [proposal(G1), proposal(G1)],
        AgentRole.INTEGRITY_GUARD: [guard(GuardDecision.REJECT)],
    }, tools=FakePatchTools(validations=[bad]), settings=settings)
    assert fake.repair_reasons[1] == ("does not apply",)


def test_tool_calls_are_recorded_with_the_acting_role() -> None:
    record, _, _, _ = run(S1, APPROVED)
    [search] = record.tool_events
    assert search.tool is ToolName.SEARCH_REPO and search.called_by is AgentRole.INTENT_ANALYST
    assert search.status is ToolCallStatus.OK and "query_chars" in search.arguments


def test_tool_call_budget_is_enforced() -> None:
    tight = S1.model_copy(update={"budget": EpisodeBudget(max_tool_calls=1)})
    record, _, _, _ = run(tight, APPROVED, memory=FakeMemory())  # search (1) then memory.read (2)
    assert record.outcome.status is RunStatus.BUDGET_EXHAUSTED
    assert [e.tool for e in record.tool_events] == [ToolName.SEARCH_REPO]


def test_maintainer_approval_without_memory_logs_a_blocked_write() -> None:
    seen = []

    def maintainer(group, decision, result):
        seen.append((group.group.group_id, decision))
        return True

    record, _, _, _ = run(S1, APPROVED, maintainer=maintainer)
    assert seen == [(G1, DecisionLabel.UPDATE_TEST)]
    assert record.outcome.maintainer_approved is True
    [attempt] = record.memory_writes
    assert attempt.approved and not attempt.written and attempt.blocked_reason == "memory_disabled"


def test_maintainer_rejection_blocks_the_write() -> None:
    record, _, _, _ = run(S1, APPROVED, maintainer=lambda *_: False)
    assert record.outcome.maintainer_approved is False
    assert record.memory_writes[0].blocked_reason == "not_approved"


def test_approved_record_is_written_by_the_guard_workflow() -> None:
    store, fake = FakeMemory(), FakeAgents()
    fake.memory_record = lambda *_: memory_record("mem-new")
    record, _, _, _ = run(S1, {**APPROVED, AgentRole.INTENT_ANALYST: [analysis(G1, cited_evidence_ids=("mem-1",))]},
                          memory=store, fake=fake, maintainer=lambda *_: True)
    assert store.writes == 1
    [attempt] = record.memory_writes
    assert attempt.written and attempt.record_id == "mem-new"
    assert record.tool_events[-1].tool is ToolName.MEMORY_WRITE
    assert record.tool_events[-1].called_by is AgentRole.INTEGRITY_GUARD


def test_missing_record_builder_is_a_logged_non_fatal_write_error() -> None:
    store = FakeMemory()
    record, _, _, _ = run(S1, {**APPROVED, AgentRole.INTENT_ANALYST: [analysis(G1, cited_evidence_ids=("mem-1",))]},
                          memory=store, maintainer=lambda *_: True)
    assert store.writes == 0 and record.outcome.status is RunStatus.COMPLETED
    assert record.memory_writes[0].blocked_reason == "write_error"
    [err] = record.errors
    assert not err.fatal and err.operation == f"memory_write:{G1}"


def test_escalation_summary_lists_missing_evidence_and_both_hypotheses() -> None:
    record, _, _, _ = run(S1, {AgentRole.INTENT_ANALYST: [analysis(G1, DecisionLabel.ESCALATE)]})
    summary = record.outcome.escalation
    assert any("no cited evidence" in m for m in summary.evidence_missing)
    assert len(summary.competing_hypotheses) == 2
    assert "undecided" in summary.competing_hypotheses[1]


def test_invented_citation_is_listed_as_missing_evidence() -> None:
    record, _, _, _ = run(S1, {AgentRole.INTENT_ANALYST: [analysis(G1, cited_evidence_ids=("ev-invented",))]})
    assert any("ev-invented" in m for m in record.outcome.escalation.evidence_missing)


def test_provenance_records_prompt_version() -> None:
    record, _, _, _ = run(S1, APPROVED)
    assert record.provenance.prompt_template_version == PROMPT_VERSION


def test_repairer_escalation_explains_what_is_missing() -> None:
    request = EscalationRequest(group_id=G1, trigger="insufficient_evidence", summary="needs a setup change")
    record, _, _, _ = run(S1, {
        AgentRole.INTENT_ANALYST: [analysis(G1)], AgentRole.REPAIRER: [RepairReply(output=request)],
    })
    assert any("within the allowed edit policy" in m for m in record.outcome.escalation.evidence_missing)


def test_patch_citing_unretrieved_evidence_is_rejected_and_fed_back() -> None:
    invented = PatchProposal(
        group_id=G1, decision=RepairDirection.UPDATE_TEST, scope=PatchScope.TEST,
        target_files=("tests/test_synthlib.py",), unified_diff=TEST_DIFF, rationale="r",
        evidence_ids=("ev-msg", "ev-invented"),
    )
    record, _, tools, fake = run(S1, {
        AgentRole.INTENT_ANALYST: [analysis(G1)],
        AgentRole.REPAIRER: [RepairReply(output=invented), proposal(G1)],
        AgentRole.INTEGRITY_GUARD: [guard(GuardDecision.APPROVE)],
    })
    first, second = record.patches
    assert "ev-invented" in first.apply_error and tools.applied == [second.unified_diff]
    assert any("ev-invented" in r for r in fake.repair_reasons[1])
    assert record.outcome.status is RunStatus.COMPLETED and record.outcome.revisions_used == 1


def test_patch_citing_unretrieved_evidence_twice_escalates_without_applying() -> None:
    invented = PatchProposal(
        group_id=G1, decision=RepairDirection.UPDATE_TEST, scope=PatchScope.TEST,
        target_files=("tests/test_synthlib.py",), unified_diff=TEST_DIFF, rationale="r", evidence_ids=("ev-x",),
    )
    record, _, tools, _ = run(S1, {
        AgentRole.INTENT_ANALYST: [analysis(G1)],
        AgentRole.REPAIRER: [RepairReply(output=invented), RepairReply(output=invented)],
    })
    assert tools.applied == [] and record.outcome.status is RunStatus.ESCALATED



def test_record_keeps_triage_groups_and_unstable_tests() -> None:
    tr = triage_result(S3)
    unstable = tr.model_copy(update={"unstable_test_ids": ("tests/flaky.py::test_x",)})
    groups = [d.model_copy(update={"mapped_by": "traceback"}) for d in unstable.groups]
    replies = {AgentRole.TRIAGE: [unstable.model_copy(update={"groups": tuple(groups)})]}
    for d in groups:
        replies.setdefault(AgentRole.INTENT_ANALYST, []).append(analysis(d.group.group_id, DecisionLabel.ESCALATE))
    record, _, _, _ = run(S3, replies)
    assert [g.group_id for g in record.failure_groups] == [d.group.group_id for d in groups]
    assert all(g.mapped_by == "traceback" and g.stability_runs == S3.environment.stability_runs
               for g in record.failure_groups)
    assert record.failure_groups[0].test_ids == list(groups[0].group.test_ids)
    assert record.unstable_test_ids == ["tests/flaky.py::test_x"]


def test_cited_and_contradicting_evidence_are_labelled() -> None:
    record, _, _, _ = run(S1, {
        AgentRole.INTENT_ANALYST: [analysis(G1, cited_evidence_ids=("ev-msg",),
                                            contradicting_evidence_ids=("ev-changelog",))],
        AgentRole.REPAIRER: [proposal(G1)], AgentRole.INTEGRITY_GUARD: [guard(GuardDecision.APPROVE)],
    })
    support = {e.evidence_id: e.support for e in record.retrieved_evidence}
    assert support == {"ev-msg": SupportStatus.SUPPORTS, "ev-changelog": SupportStatus.CONTRADICTS}


def test_analyst_alternative_explanation_reaches_the_escalation_summary() -> None:
    record, _, _, _ = run(S1, {AgentRole.INTENT_ANALYST: [
        analysis(G1, DecisionLabel.ESCALATE, alternative_explanation="maybe the test is right after all")
    ]})
    assert "maybe the test is right after all" in record.outcome.escalation.competing_hypotheses[1]


def test_failed_triage_leaves_no_groups() -> None:
    record, _, _, _ = run(S1, {AgentRole.TRIAGE: [LLMRateLimitError("429")]})
    assert record.failure_groups == [] and record.outcome.status is RunStatus.API_FAILURE



def test_b1_cannot_run_through_the_multi_agent_pipeline() -> None:
    settings = Settings.from_env({"LLM_MODEL": "fake-model-1", "EXPERIMENT_CONDITION": "B1"})
    record, er, _, _ = run(S1, {}, settings=settings)  # EpisodeRun directly, not run_episode
    assert record.outcome.status is RunStatus.ORCHESTRATOR_ERROR
    assert "not executed by this pipeline" in record.errors[0].message and er.trace == [(None, Stage.FAILED)]


def _suite(status: TestRunStatus, error: Exception | None = None, failing: tuple[str, ...] = ()):
    calls = []

    def run_suite(worktree_path, selectors, **_):
        calls.append(list(selectors))
        if error is not None:
            raise error
        extra = {"error_message": "ImportError"} if status in (TestRunStatus.ERROR, TestRunStatus.COLLECTION_ERROR) else {}
        if status is TestRunStatus.TIMEOUT:
            extra = {"timed_out": True, "exit_code": None}
        failures = tuple(FailingTest(test_id=t, test_file=t.split("::")[0], test_name=t.split("::")[-1],
                                     failure_message="assert") for t in failing)
        return PytestExecution(**{"selectors": tuple(selectors), "command": ("pytest",), "status": status,
                                  "failures": failures, "exit_code": 0 if status is TestRunStatus.PASSED else 2,
                                  **extra})
    return run_suite, calls


def run_with_suite(run_suite, episode=S1, replies=None):
    tools = FakePatchTools()
    patch_tools = PatchTools(validate=tools.validate, apply=tools.apply, revert=tools.revert, run_suite=run_suite)
    replies = {AgentRole.TRIAGE: [triage_result(episode)], **(replies or APPROVED)}
    er = EpisodeRun(episode, worktree_path=WT, settings=SETTINGS, llm=FakeLLMClient(replies),
                    retriever=FakeRetriever(), agents=FakeAgents().agents(), patch_tools=patch_tools)
    return er.execute()


@pytest.mark.parametrize("status", [TestRunStatus.PASSED, TestRunStatus.FAILED, TestRunStatus.COLLECTION_ERROR])
def test_final_status_is_the_final_suite_run_not_the_last_group(status) -> None:  # FAILED: tests outside any group
    run_suite, calls = _suite(status)
    record = run_with_suite(run_suite)
    assert len(calls) == 1 and record.test_runs[-1].phase is TestRunPhase.FINAL
    assert record.test_runs[-2].status is TestRunStatus.PASSED  # the group's own verification still recorded
    assert record.final_test_status is status  # an infrastructure status stays itself, never PASSED/FAILED


def test_final_suite_that_cannot_start_is_not_run_with_an_error() -> None:
    run_suite, _ = _suite(TestRunStatus.PASSED, error=ToolError("sandbox unavailable"))
    record = run_with_suite(run_suite)
    assert record.final_test_status is TestRunStatus.NOT_RUN
    [err] = record.errors
    assert err.operation == "final_suite" and not err.fatal and err.category is ErrorCategory.TOOL_ERROR
    assert record.outcome.status is RunStatus.TOOL_FAILURE  # the end state is unknown: never reported as completed
    assert record.predictions[0].predicted_action is DecisionLabel.UPDATE_TEST  # the decision history is kept


@pytest.mark.parametrize(
    ("status", "category"),
    [(TestRunStatus.ERROR, ErrorCategory.TOOL_ERROR), (TestRunStatus.COLLECTION_ERROR, ErrorCategory.TOOL_ERROR),
     (TestRunStatus.TIMEOUT, ErrorCategory.TIMEOUT)],
)
def test_final_suite_that_does_not_execute_the_tests_is_a_tool_failure(status, category) -> None:
    run_suite, _ = _suite(status)
    record = run_with_suite(run_suite)
    assert record.final_test_status is status  # keeps its own status
    [err] = record.errors
    assert err.operation == "final_suite" and err.category is category and status.value in err.message
    assert record.outcome.status is RunStatus.TOOL_FAILURE and not err.counted_against_agent
    [group] = record.failure_groups
    assert group.disposition == "accepted" and group.guard_decision is GuardDecision.APPROVE


def test_accepted_group_failing_in_the_final_suite_is_flagged_not_completed() -> None:
    # Two groups: g-n accepted, g-s escalated.  The final suite shows g-n's test failing again (and g-s's, which is
    # expected: it was never repaired).
    g_n, g_s = (d.group for d in triage_result(S3).groups)
    run_suite, _ = _suite(TestRunStatus.FAILED, failing=(g_n.test_ids[0], *g_s.test_ids))
    record = run_with_suite(run_suite, episode=S3, replies={
        AgentRole.INTENT_ANALYST: [analysis("g-n"), analysis("g-s", DecisionLabel.ESCALATE)],
        AgentRole.REPAIRER: [proposal("g-n")],
        AgentRole.INTEGRITY_GUARD: [guard(GuardDecision.APPROVE)],
    })
    groups = {g.group_id: g for g in record.failure_groups}
    assert groups["g-n"].disposition == "failed_final_verification"
    assert groups["g-n"].final_failing_test_ids == [g_n.test_ids[0]]
    assert groups["g-n"].guard_decision is GuardDecision.APPROVE  # the verification history is kept
    assert groups["g-s"].disposition == "not_applied" and groups["g-s"].final_failing_test_ids == []
    assert record.outcome.status is RunStatus.INCOMPLETE
    [err] = record.errors
    assert err.operation == "final_suite" and "g-n" in err.message and g_n.test_ids[0] in err.message
    assert {p.test_id: p.predicted_action for p in record.predictions}[g_n.test_ids[0]] is DecisionLabel.UPDATE_TEST


def test_failure_group_records_keep_every_guard_verdict() -> None:
    record, _, _, _ = run(S1, {
        AgentRole.INTENT_ANALYST: [analysis(G1)],
        AgentRole.REPAIRER: [proposal(G1), proposal(G1)],
        AgentRole.INTEGRITY_GUARD: [guard(GuardDecision.REJECT), guard(GuardDecision.APPROVE)],
    })
    [group] = record.failure_groups
    assert group.decision is DecisionLabel.UPDATE_TEST and group.guard_decision is GuardDecision.APPROVE
    assert "assertion removed" in " ".join(group.guard_reasons)
    assert group.patch_ids == [f"{G1}-r0", f"{G1}-r1"] and group.disposition == "accepted"
    assert group.integrity is not None


def test_final_suite_is_not_blocked_by_the_tool_budget() -> None:
    from testorcode.tools.trace import tool_call
    run_suite_inner, _ = _suite(TestRunStatus.PASSED)

    def recorded(worktree_path, selectors, **kw):  # behaves like the real runner: a recorded tool call
        with tool_call(ToolName.RUN_TESTS, selectors=list(selectors)):
            return run_suite_inner(worktree_path, selectors, **kw)
    tight = S1.model_copy(update={"budget": EpisodeBudget(max_tool_calls=1)})  # the one search uses it up
    record = run_with_suite(recorded, episode=tight)
    assert record.final_test_status is TestRunStatus.PASSED
    assert record.tool_events[-1].called_by is AgentRole.ORCHESTRATOR
