"""Scenario T: injected tool faults (``tools.faults``) at the agent tools and at patch application."""

from __future__ import annotations

from pathlib import Path

from test_orchestrator import G1, S1, FakePatchTools, analysis, guard, proposal, run

from testorcode.agents.tools import AgentTools
from testorcode.schemas.contracts import EpisodeContext
from testorcode.schemas.episode_schemas import (
    ToolFaultInjection,
    ToolFaultKind,
    ToolFaultPlan,
    ToolName,
)
from testorcode.schemas.result_schemas import (
    AgentRole,
    GuardDecision,
    PatchApplicationStatus,
    ToolCallStatus,
)
from testorcode.schemas.tests.conftest import make_s1
from testorcode.tools.faults import FaultInjector, agent_message
from testorcode.tools.trace import ToolRecorder, recording, tool_call


def plan(*injections: tuple[ToolName, ToolFaultKind, int], rate: float | None = None, seed: int | None = 42) -> ToolFaultPlan:
    return ToolFaultPlan(enabled=True, seed=seed, failure_rate=rate, injections=[
        ToolFaultInjection(tool=t, fault_kind=k, call_index=i) for t, k, i in injections])


def test_injector_fires_on_the_planned_call_index_only() -> None:
    injector = FaultInjector(plan((ToolName.RUN_TESTS, ToolFaultKind.INVALID_TEST_SELECTOR, 1)))
    assert [injector.fault_for(ToolName.RUN_TESTS) for _ in range(3)] == [None, ToolFaultKind.INVALID_TEST_SELECTOR, None]
    assert injector.fault_for(ToolName.READ_FILE) is None
    assert injector.fired == [(ToolName.RUN_TESTS, 1, ToolFaultKind.INVALID_TEST_SELECTOR)]


def test_rate_based_faults_are_deterministic_and_realistic_per_tool() -> None:
    def decisions():
        injector = FaultInjector(plan(rate=0.5, seed=7))
        return [injector.fault_for(t) for t in [ToolName.APPLY_PATCH] * 10 + [ToolName.READ_FILE] * 10]

    first = decisions()
    assert first == decisions() and any(first) and not all(first)
    assert {k for k in first[:10] if k} == {ToolFaultKind.PATCH_DOES_NOT_APPLY}
    assert {k for k in first[10:] if k} == {ToolFaultKind.TIMEOUT}


def test_disabled_or_missing_plan_injects_nothing() -> None:
    assert FaultInjector.for_plan(None) is None and FaultInjector.for_plan(ToolFaultPlan()) is None


def test_agent_message_never_mentions_the_injection() -> None:
    for kind in ToolFaultKind:
        for tool in (ToolName.RUN_TESTS, ToolName.APPLY_PATCH, ToolName.READ_FILE):
            assert "inject" not in agent_message(kind, tool).lower()


def test_agent_tool_fault_is_recorded_and_the_tool_is_not_run(tmp_path: Path) -> None:
    (tmp_path / "src/synthlib").mkdir(parents=True)
    (tmp_path / "src/synthlib/core.py").write_text("x = 1\n")
    context = EpisodeContext.from_episode_input(make_s1().agent_input, tmp_path)
    tools = AgentTools(context, retriever=None, timeout_seconds=30,  # type: ignore[arg-type]
                       faults=FaultInjector(plan((ToolName.READ_FILE, ToolFaultKind.TIMEOUT, 0))))
    recorder = ToolRecorder(max_calls=5)
    with recording(recorder):
        failed = tools.call("read_file", path="src/synthlib/core.py")
        ok = tools.call("read_file", path="src/synthlib/core.py")
    assert failed == agent_message(ToolFaultKind.TIMEOUT, ToolName.READ_FILE) and "x = 1" in ok
    first, second = recorder.events
    assert first.status is ToolCallStatus.INJECTED_FAULT and first.fault_kind is ToolFaultKind.TIMEOUT
    assert second.status is ToolCallStatus.OK and recorder.used["agent"] == 2  # a fault costs budget like a call


class _RecordingPatchTools(FakePatchTools):
    """Fake patch tools that log their application like the real ``apply_patch`` does."""

    def apply(self, worktree_path, diff, *, policy, scope):
        with tool_call(ToolName.APPLY_PATCH, operation="apply", scope=str(scope)):
            return super().apply(worktree_path, diff, policy=policy, scope=scope)


def test_patch_application_fault_uses_the_revision_and_is_marked_recovered(monkeypatch) -> None:
    from testorcode import orchestrator

    original = orchestrator.EpisodeRun.__init__

    def with_plan(self, *args, **kwargs):
        kwargs["fault_plan"] = plan((ToolName.APPLY_PATCH, ToolFaultKind.PATCH_DOES_NOT_APPLY, 0))
        original(self, *args, **kwargs)

    monkeypatch.setattr(orchestrator.EpisodeRun, "__init__", with_plan)
    record, _, tools, fake = run(S1, {
        AgentRole.INTENT_ANALYST: [analysis(G1)],
        AgentRole.REPAIRER: [proposal(G1), proposal(G1)],
        AgentRole.INTEGRITY_GUARD: [guard(GuardDecision.APPROVE)],
    }, tools=_RecordingPatchTools())
    first, second = record.patches
    assert first.application_status is PatchApplicationStatus.FAILED_TO_APPLY and "inject" not in first.apply_error
    assert second.application_status is PatchApplicationStatus.APPLIED and len(tools.applied) == 1
    assert fake.repair_reasons[1] == (first.apply_error,)  # the revision is told what a real failure would say
    [fault] = [e for e in record.tool_events if e.status is ToolCallStatus.INJECTED_FAULT]
    assert fault.tool is ToolName.APPLY_PATCH and fault.fault_kind is ToolFaultKind.PATCH_DOES_NOT_APPLY
    assert fault.recovered is True
