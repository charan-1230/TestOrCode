"""Decision policy (proposal section 7): UPDATE_TEST needs positive intent evidence, FIX_CODE needs a cited contradiction,
otherwise ESCALATE.  Runs the real Intent Analyst (real BM25 over the real pilot corpora) inside the real orchestrator;
only the model reply and the test run are scripted, and the repair stage is a recorder."""

from __future__ import annotations

from pathlib import Path

import pytest

from testorcode.config import Settings
from testorcode.dataset import agent_view_path, load_agent_inputs
from testorcode.llm_client import FakeLLMClient
from testorcode.orchestrator import Agents, EpisodeRun, PatchTools
from testorcode.schemas.contracts import (
    EscalationRequest,
    FailureGroupDetail,
    Hypothesis,
    IntentAnalysis,
    PytestExecution,
    TriageResult,
    decision_policy_problem,
)
from testorcode.schemas.episode_schemas import DecisionLabel, FailureGroup, HypothesisKind
from testorcode.schemas.result_schemas import AgentRole, RunStatus, TestRunStatus

SETTINGS = Settings.from_env({"LLM_MODEL": "fake-model-1", "EXPERIMENT_CONDITION": "A2"})
DEV = {e.episode_id: e for e in load_agent_inputs(agent_view_path("dev"))}


def analysis(decision: str, kind: str, cited=(), against=()) -> IntentAnalysis:
    return IntentAnalysis(
        group_id="g1", hypothesis=Hypothesis(kind=HypothesisKind(kind), explanation="scripted"),
        cited_evidence_ids=tuple(cited), contradicting_evidence_ids=tuple(against), confidence=0.8,
        recommended_decision=DecisionLabel(decision),
    )


@pytest.mark.parametrize(
    ("decision", "kind", "cited", "against", "problem"),
    [
        ("FIX_CODE", "regression", (), (), "absence of intent evidence"),
        ("FIX_CODE", "regression", ("docstring:x",), (), None),
        ("FIX_CODE", "regression", ("msg:a",), ("msg:a",), "both for and against"),
        ("UPDATE_TEST", "outdated_test", ("msg:a",), ("msg:a",), "both for and against"),
        ("UPDATE_TEST", "outdated_test", ("msg:a",), ("doc:b",), None),
        ("ESCALATE", "unknown", ("msg:a",), ("msg:a",), None),
    ],
)
def test_policy_check(decision, kind, cited, against, problem) -> None:
    found = decision_policy_problem(analysis(decision, kind, cited, against))
    assert (found is None) if problem is None else (problem in found)


def run_dev(episode_id: str, reply: IntentAnalysis, tmp_path: Path):
    """The real Intent Analyst on a pilot episode; Triage is a fixed group and the Repairer only records the call."""
    episode = DEV[episode_id]
    tests = tuple(episode.ci_event.failing_tests)
    baseline = PytestExecution(selectors=tuple(t.test_id for t in tests), command=("pytest",),
                               status=TestRunStatus.FAILED, exit_code=1, failed=len(tests), failures=tests)
    group = FailureGroupDetail(
        group=FailureGroup(group_id="g1", test_ids=[t.test_id for t in tests]), failing_tests=tests,
        affected_files=tuple(c.path for c in episode.ci_event.changed_files),
        diff_context="\n".join(c.unified_diff or "" for c in episode.ci_event.changed_files),
    )
    repaired = []

    def repairer(context, g, a, **_):
        repaired.append(a.recommended_decision)
        return EscalationRequest(group_id=g.group.group_id, trigger="insufficient_evidence", summary="stop here")

    agents = Agents(triage=lambda *a, **k: TriageResult(groups=(group,), baseline_run=baseline), repairer=repairer)
    llm = FakeLLMClient({AgentRole.INTENT_ANALYST: [reply]})
    run = EpisodeRun(episode, worktree_path=tmp_path, settings=SETTINGS, llm=llm, agents=agents,
                     patch_tools=PatchTools(run_suite=None))  # no repository here: no final suite run
    return run.execute(), repaired, llm


def test_dev03_regression_citing_the_unrelated_ci_commit_message_reaches_the_repairer(tmp_path: Path) -> None:
    # dev-03 (S2): the CI commit is about MultiReplace; strip_ansi changing is not what it says.  The retrieved commit
    # message is the contradiction the FIX_CODE decision cites.
    record, repaired, llm = run_dev("dev-03", analysis("FIX_CODE", "regression", ("msg:bba729bf4539",)), tmp_path)
    assert repaired == [DecisionLabel.FIX_CODE]  # policy satisfied: the repair stage was reached
    assert "msg:bba729bf4539" in {e.evidence_id for e in record.retrieved_evidence}
    assert "[msg:bba729bf4539]" in llm.calls[0]["user"]


def test_dev08_intent_stripped_fix_code_without_evidence_escalates(tmp_path: Path) -> None:
    # dev-08 (S5): intent evidence was removed on purpose; "nothing says it was intended" must not become FIX_CODE.
    record, repaired, llm = run_dev("dev-08", analysis("FIX_CODE", "regression"), tmp_path)
    assert repaired == [] and record.outcome.status is RunStatus.ESCALATED
    assert record.outcome.escalation.trigger == "insufficient_evidence"
    assert "absence of intent evidence" in record.outcome.escalation.summary
    [pred] = record.predictions
    assert pred.predicted_action is DecisionLabel.ESCALATE and pred.decided_by is AgentRole.ORCHESTRATOR
    prompt = llm.calls[0]["user"].lower()
    assert "gold" not in prompt and "expected_action" not in prompt and "reference_repair" not in prompt


def test_unsupported_citation_escalates_before_repair(tmp_path: Path) -> None:
    record, repaired, _ = run_dev("dev-08", analysis("FIX_CODE", "regression", ("docstring:never.retrieved",)), tmp_path)
    assert repaired == [] and "never retrieved" in record.outcome.escalation.summary


def test_evidence_cited_both_ways_escalates_as_contradictory(tmp_path: Path) -> None:
    reply = analysis("FIX_CODE", "regression", ("msg:bba729bf4539",), ("msg:bba729bf4539",))
    record, repaired, _ = run_dev("dev-03", reply, tmp_path)
    assert repaired == [] and record.outcome.escalation.trigger == "contradictory_evidence"
