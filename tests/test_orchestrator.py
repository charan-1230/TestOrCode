"""Orchestrator scaffolding: stage wiring and transitions with fake agents, tools and LLM.

Fake agents ask the ``FakeLLMClient`` for their contract object, as the real agents
will, so every test also exercises the shared, metered LLM path.  All data is synthetic.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import BaseModel

from testorcode.config import Settings
from testorcode.errors import ConfigError, LLMRateLimitError, PatchError, ToolError
from testorcode.llm_client import FakeLLMClient, OpenAICompatibleClient
from testorcode.orchestrator import Agents, EpisodeRun, PatchTools, Stage, run_episode
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
    HypothesisKind,
    PatchScope,
    RepairDirection,
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
    TestRunPhase,
    TestRunStatus,
)
from testorcode.schemas.tests.conftest import HEAD, make_s1, make_s3, memory_record

WT = Path("/tmp/testorcode-fake-worktree")  # never touched: patch tools are fakes
SETTINGS = Settings.from_env({"LLM_MODEL": "fake-model-1"})
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

    def repair(self, context, group, analysis, *, llm, previous_rejection=None):
        self.repair_calls.append(previous_rejection)
        return self.ask(llm, AgentRole.REPAIRER, RepairReply).output

    def guard(self, context, group, analysis, proposal, *, llm=None):
        return self.ask(llm, AgentRole.INTEGRITY_GUARD, GuardResult)

    def agents(self) -> Agents:
        return Agents(triage=self.triage, intent_analyst=self.intent, repairer=self.repair, integrity_guard=self.guard)


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

    def revert(self, worktree_path, diff):
        self.reverted.append(diff)

    def tools(self) -> PatchTools:
        return PatchTools(validate=self.validate, apply=self.apply, revert=self.revert)


def run(episode: EpisodeInput, replies: dict, *, tools: FakePatchTools | None = None,
        settings: Settings = SETTINGS, memory=None, retriever=None, fake: FakeAgents | None = None):
    fake = fake or FakeAgents()
    tools = tools or FakePatchTools()
    replies = {AgentRole.TRIAGE: [triage_result(episode)], **replies}
    episode_run = EpisodeRun(
        episode, worktree_path=WT, settings=settings, llm=FakeLLMClient(replies), memory=memory,
        retriever=retriever or FakeRetriever(), agents=fake.agents(), patch_tools=tools.tools(),
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
    settings = Settings.from_env({"LLM_MODEL": "fake-model-1", "MAX_REPAIR_REVISIONS": "0"})
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


def test_default_stub_agents_fail_explicitly_not_as_escalation() -> None:
    record = run_episode(S1, worktree_path=WT, settings=SETTINGS, llm=FakeLLMClient())
    assert record.outcome.status is RunStatus.ORCHESTRATOR_ERROR
    [err] = record.errors
    assert err.fatal and "not implemented" in err.message and err.operation == "triage"
    assert record.predictions == [] and record.outcome.escalation is None


def test_missing_retriever_fails_at_intent_analysis() -> None:
    fake = FakeAgents()
    er = EpisodeRun(
        S1, worktree_path=WT, settings=SETTINGS, agents=fake.agents(),
        llm=FakeLLMClient({AgentRole.TRIAGE: [triage_result(S1)]}), patch_tools=FakePatchTools().tools(),
    )
    record = er.execute()
    assert record.outcome.status is RunStatus.ORCHESTRATOR_ERROR
    assert record.errors[0].operation == f"intent_analysis:{G1}"
    assert er.trace[-1] == (G1, Stage.FAILED)


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
        Settings.from_env({"LLM_MODEL": "m-1", "LLM_BASE_URL": "http://llm.invalid/v1"}),
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
