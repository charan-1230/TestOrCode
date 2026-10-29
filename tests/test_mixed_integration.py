"""Offline integration for mixed failures, final status and B1/A2 budgets.

Real orchestrator, real agents, real git worktree, real pytest (``trusted-host``: this fixture repository is a
trusted test fixture), real patch tools and probe; only the model replies are scripted.

The repository has two CI failures from one commit: ``normalise`` regressed (capitalize instead of lower: FIX_CODE)
and ``twice`` was deliberately changed to triple (the test is outdated: UPDATE_TEST).  A third test, ``test_shout``,
passes before any repair and catches collateral damage.
"""

from __future__ import annotations

import subprocess
from functools import partial
from pathlib import Path

import pytest

from testorcode import orchestrator
from testorcode.agents import integrity_guard, triage
from testorcode.agents import tools as agent_tools
from testorcode.config import Settings
from testorcode.llm_client import FakeLLMClient
from testorcode.orchestrator import Agents, run_episode
from testorcode.schemas.episode_schemas import (
    DecisionLabel,
    EpisodeBudget,
    EvidenceItem,
    EvidenceSourceType,
    FailingTest,
    ToolName,
)
from testorcode.schemas.result_schemas import (
    AgentRole,
    ErrorCategory,
    GuardDecision,
    PatchApplicationStatus,
    RunStatus,
    TestRunPhase,
    TestRunStatus,
)
from testorcode.schemas.tests.conftest import make_s1
from testorcode.tools.mutation import MutationCheck
from testorcode.tools.pytest_runner import run_tests
from testorcode.tools.verification import ProbeCase, TrustedProbe, verify_behaviour

CORE = (
    "def normalise(s):\n    return s.capitalize()\n\n\n"
    "def twice(x):\n    return x * 3\n\n\n"
    "def shout(s):\n    return s.upper() + '!'\n"
)
NORMALISE_TEST = "def test_normalise():\n    assert normalise('a') == 'a'\n"
TWICE_TEST = "def test_twice():\n    assert twice(2) == 4\n"
SHOUT_TEST = "def test_shout():\n    assert shout('a') == 'A!'\n"
IMPORTS = "from synthlib.core import normalise, shout, twice\n\n\n"
DOC = "docstring:synthlib.core.normalise"
MSG = "msg:twice"
NO_TOOL = {"query": "", "path": "", "start_line": 1, "end_line": 0, "selectors": []}
PROBE = TrustedProbe(probe_id="p-normalise", module="synthlib.core", function="normalise",
                     cases=(ProbeCase(args=("AbC",), expected="abc"),), provenance="dataset")


def make_episode(twice_file: str):
    base = make_s1().agent_input
    norm_id, twice_id = "tests/test_core.py::test_normalise", f"{twice_file}::test_twice"
    tests = [  # twice first: groups keep first-seen order, so the passing group is processed last
        FailingTest(test_id=twice_id, test_file=twice_file, test_name="test_twice",
                    target_function="synthlib.core.twice", failure_message="assert 6 == 4"),
        FailingTest(test_id=norm_id, test_file="tests/test_core.py", test_name="test_normalise",
                    target_function="synthlib.core.normalise", failure_message="assert 'A' == 'a'"),
    ]
    corpus = base.corpus.model_copy(update={"items": [
        *base.corpus.items,
        EvidenceItem(evidence_id=DOC, source_type=EvidenceSourceType.DOCSTRING, source_ref="src/synthlib/core.py",
                     text="normalise(s): return s stripped and lower-cased."),
        EvidenceItem(evidence_id=MSG, source_type=EvidenceSourceType.CHANGELOG, source_ref="CHANGELOG",
                     text="twice() now triples its argument on purpose (test_twice must follow)."),
    ]})
    event = base.ci_event.model_copy(update={"failing_tests": tests})
    return base.model_copy(update={"ci_event": event, "corpus": corpus}), norm_id, twice_id


@pytest.fixture(params=["tests/test_core.py", "tests/test_twice.py"], ids=["same-file", "different-files"])
def mixed(request, tmp_path: Path, monkeypatch):
    twice_file = request.param
    root = tmp_path / "repo"
    files = {"pyproject.toml": '[tool.pytest.ini_options]\npythonpath = ["src"]\n', "src/synthlib/__init__.py": "",
             "src/synthlib/core.py": CORE}
    if twice_file == "tests/test_core.py":
        files["tests/test_core.py"] = IMPORTS + NORMALISE_TEST + "\n\n" + TWICE_TEST + "\n\n" + SHOUT_TEST
    else:
        files["tests/test_core.py"] = IMPORTS + NORMALISE_TEST + "\n\n" + SHOUT_TEST
        files["tests/test_twice.py"] = IMPORTS + TWICE_TEST
    for rel, text in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)
    for args in (["init", "-q"], ["add", "."], ["-c", "user.name=t", "-c", "user.email=t@x", "commit", "-qm", "ci"]):
        subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)
    trusted = partial(run_tests, isolation="trusted-host")
    for module in (triage, integrity_guard, agent_tools, orchestrator):
        monkeypatch.setattr(module, "run_tests", trusted)
    monkeypatch.setattr(integrity_guard, "verify_behaviour", partial(verify_behaviour, isolation="trusted-host"))
    episode, norm_id, twice_id = make_episode(twice_file)
    return root, episode, norm_id, twice_id, twice_file


FIX = ("src/synthlib/core.py", "return s.capitalize()", "return s.lower()")


def b1_item(test_id: str, decision: str, edits, cited) -> dict:
    kind = {"UPDATE_TEST": "outdated_test", "FIX_CODE": "regression"}[decision]
    return {"test_ids": [test_id], "decision": decision, "hypothesis": kind, "explanation": "scripted",
            "cited_evidence_ids": cited, "confidence": 0.8,
            "edits": [{"path": p, "old": o, "new": n} for p, o, n in edits]}


def b1_finish(*items) -> dict:
    return {"action": "finish", "decisions": list(items), "rationale": "scripted", **NO_TOOL}


B1 = Settings.from_env({"LLM_MODEL": "fake-model-1", "EXPERIMENT_CONDITION": "B1", "TEST_TIMEOUT_SECONDS": "60"})
A2 = Settings.from_env({"LLM_MODEL": "fake-model-1", "EXPERIMENT_CONDITION": "A2", "TEST_TIMEOUT_SECONDS": "60"})


def twice_update(twice_file: str):
    return (twice_file, "twice(2) == 4", "twice(2) == 6")


# --------------------------------------------------------------------------- #
# Mixed failures: two groups, two directions, same file and different files
# --------------------------------------------------------------------------- #


def test_b1_repairs_both_directions_and_records_each_test(mixed) -> None:
    root, episode, norm_id, twice_id, twice_file = mixed
    llm = FakeLLMClient({AgentRole.SINGLE_AGENT: [b1_finish(
        b1_item(twice_id, "UPDATE_TEST", [twice_update(twice_file)], [MSG]),
        b1_item(norm_id, "FIX_CODE", [FIX], [DOC]),
    )]})
    record = run_episode(episode, worktree_path=root, settings=B1, llm=llm)

    assert record.outcome.status is RunStatus.COMPLETED and record.outcome.revisions_used == 0
    assert {p.test_id: p.predicted_action for p in record.predictions} == {
        twice_id: DecisionLabel.UPDATE_TEST, norm_id: DecisionLabel.FIX_CODE}
    assert [p.application_status for p in record.patches] == [PatchApplicationStatus.APPLIED] * 2
    # Each group was verified while the other was still failing (same file: before this fix, both were rejected).
    assert record.final_test_status is TestRunStatus.PASSED
    assert record.test_runs[-1].phase is TestRunPhase.FINAL and record.test_runs[-1].selector == "tests"
    assert "return s.lower()" in (root / "src/synthlib/core.py").read_text()
    assert "twice(2) == 6" in (root / twice_file).read_text()
    assert record.usage.llm_calls == len(llm.calls) == 1
    # Same budget categories as A2: its RAG searches are the agent's (like the Analyst's), the harness's stability
    # runs, patch handling and reruns are operational, the final suite is exempt.
    budgets = {(e.called_by, e.arguments["budget"]) for e in record.tool_events}
    assert budgets == {(AgentRole.SINGLE_AGENT, "agent"), (AgentRole.SINGLE_AGENT, "operational"),
                       (AgentRole.ORCHESTRATOR, "exempt")}
    assert {e.tool for e in record.tool_events if e.arguments["budget"] == "agent"} == {ToolName.SEARCH_REPO}


def test_a2_approves_the_probed_fix_while_the_other_group_stays_failing(mixed, monkeypatch) -> None:
    root, episode, norm_id, twice_id, twice_file = mixed
    # The test repair's strength cannot be measured here (no executable mutant), so it must not be approved; the
    # source fix keeps its (unstubbed-in-effect) clean mutation evidence.
    unmeasured = MutationCheck(gaps=("no executable cached mutant (test harness)",))
    monkeypatch.setattr(integrity_guard, "mutation_check", lambda context, group, proposal, *a, **k: (
        unmeasured if proposal.decision.value == "UPDATE_TEST" else MutationCheck()))

    def guard(context, group, analysis, proposal, **kw):  # the harness probe covers normalise only
        probe = PROBE if group.group.function_qualname == "synthlib.core.normalise" else None
        return integrity_guard.verify_repair(context, group, analysis, proposal, probe=probe, **kw)

    def intent(decision, cited):
        kind = {"UPDATE_TEST": "outdated_test", "FIX_CODE": "regression"}[decision]
        return lambda gid: {"group_id": gid, "hypothesis": {"kind": kind, "explanation": "s"}, "expected_behaviour": "s",
                            "cited_evidence_ids": cited, "contradicting_evidence_ids": [], "confidence": 0.9,
                            "recommended_decision": decision}

    def repair(edit, cited):
        return {"action": "patch", "edits": [dict(zip(("path", "old", "new"), edit, strict=True))],
                "rationale": "s", "evidence_ids": cited, **NO_TOOL}

    consistent = {"verdict": "consistent", "test_meaning_preserved": True, "reasons": []}
    llm = FakeLLMClient({
        AgentRole.INTENT_ANALYST: [intent("UPDATE_TEST", [MSG])("g1"), intent("FIX_CODE", [DOC])("g2")],
        AgentRole.REPAIRER: [repair(twice_update(twice_file), [MSG]), repair(FIX, [])],
        AgentRole.INTEGRITY_GUARD: [consistent, consistent],
    })
    record = run_episode(episode, worktree_path=root, settings=A2, llm=llm, agents=Agents(integrity_guard=guard))

    preds = {p.test_id: p for p in record.predictions}
    assert preds[norm_id].predicted_action is DecisionLabel.FIX_CODE  # approved despite twice still failing
    assert preds[twice_id].predicted_action is DecisionLabel.ESCALATE  # unmeasured strength: never approved on green
    assert preds[twice_id].hypothesis.value == "outdated_test"  # the direction survives for scoring
    assert [p.application_status for p in record.patches] == [PatchApplicationStatus.REVERTED,
                                                              PatchApplicationStatus.APPLIED]
    # The last group (normalise) passed, but the earlier unresolved failure remains: the episode ends red.
    runs = {r.run_id: r for r in record.test_runs}
    assert runs["g2-r0-post"].status is TestRunStatus.PASSED  # the last group's own test passes
    # Its regression run fails only on the other group's known, unresolved failure, which does not block it.
    assert runs["g2-r0-regression"].failing_test_ids == [twice_id]
    assert record.final_test_status is TestRunStatus.FAILED
    assert record.test_runs[-1].failing_test_ids == [twice_id]
    assert "twice(2) == 4" in (root / twice_file).read_text()  # escalated test repair rolled back
    assert record.integrity.guard_decision is GuardDecision.APPROVE  # the last verdict (normalise)
    assert record.usage.llm_calls == len(llm.calls) == 6


def test_a_repair_that_breaks_a_passing_test_is_rejected_and_revised(mixed) -> None:
    root, episode, norm_id, twice_id, twice_file = mixed
    breaking = [FIX, ("src/synthlib/core.py", "return s.upper() + '!'", "return s.upper()")]
    llm = FakeLLMClient({AgentRole.SINGLE_AGENT: [
        b1_finish(b1_item(twice_id, "UPDATE_TEST", [twice_update(twice_file)], [MSG]),
                  b1_item(norm_id, "FIX_CODE", breaking, [DOC])),
        b1_finish(b1_item(norm_id, "FIX_CODE", [FIX], [DOC])),  # the revision, after the rejection
    ]})
    record = run_episode(episode, worktree_path=root, settings=B1, llm=llm)

    first, second, third = record.patches
    assert first.application_status is PatchApplicationStatus.APPLIED  # twice
    assert second.application_status is PatchApplicationStatus.REVERTED
    assert "previously passing tests now fail: tests/test_core.py::test_shout" in second.apply_error
    assert "previously passing tests now fail" in llm.calls[1]["user"]  # the reason reached the revision
    assert third.application_status is PatchApplicationStatus.APPLIED
    assert record.outcome.status is RunStatus.COMPLETED and record.final_test_status is TestRunStatus.PASSED
    assert "return s.upper() + '!'" in (root / "src/synthlib/core.py").read_text()  # the breakage was rolled back


def test_a_later_repair_may_not_break_an_already_accepted_group(mixed) -> None:
    root, episode, norm_id, twice_id, twice_file = mixed
    # twice (accepted first) is broken by the normalise "fix", which also changes twice back: rejected.
    sneaky = [FIX, ("src/synthlib/core.py", "return x * 3", "return x * 2")]
    llm = FakeLLMClient({AgentRole.SINGLE_AGENT: [
        b1_finish(b1_item(twice_id, "UPDATE_TEST", [twice_update(twice_file)], [MSG]),
                  b1_item(norm_id, "FIX_CODE", sneaky, [DOC])),
        b1_finish(),  # no new decision on revision: escalate
    ]})
    record = run_episode(episode, worktree_path=root, settings=B1, llm=llm)

    preds = {p.test_id: p.predicted_action for p in record.predictions}
    assert preds == {twice_id: DecisionLabel.UPDATE_TEST, norm_id: DecisionLabel.ESCALATE}
    assert f"previously passing tests now fail: {twice_id}" in record.patches[1].apply_error
    assert "return x * 3" in (root / "src/synthlib/core.py").read_text()  # accepted state kept
    assert record.final_test_status is TestRunStatus.FAILED  # normalise is still red


# --------------------------------------------------------------------------- #
# B1 budgets and placement retry
# --------------------------------------------------------------------------- #


def search(query="twice"):
    return {"action": "search_repo", "decisions": [], "rationale": "look", **NO_TOOL, "query": query}


def test_b1_step_cap_scales_with_the_failing_tests(mixed) -> None:
    root, episode, norm_id, twice_id, twice_file = mixed
    # Two failing tests -> 12 steps: 7 tool calls then finish.  (A flat 6-step cap would have ended at step 6.)
    replies = [search(f"q{i}") for i in range(7)] + [b1_finish(
        b1_item(twice_id, "UPDATE_TEST", [twice_update(twice_file)], [MSG]), b1_item(norm_id, "FIX_CODE", [FIX], [DOC]))]
    llm = FakeLLMClient({AgentRole.SINGLE_AGENT: replies})
    record = run_episode(episode, worktree_path=root, settings=B1, llm=llm)
    assert record.outcome.status is RunStatus.COMPLETED
    assert record.usage.llm_calls == len(llm.calls) == 8
    assert "STEP 8 of 12" in llm.calls[-1]["user"]


def test_b1_unplaceable_edit_gets_one_reask_without_using_the_revision(mixed) -> None:
    root, episode, norm_id, twice_id, twice_file = mixed
    llm = FakeLLMClient({AgentRole.SINGLE_AGENT: [
        b1_finish(b1_item(twice_id, "UPDATE_TEST", [twice_update(twice_file)], [MSG]),
                  b1_item(norm_id, "FIX_CODE", [("src/synthlib/core.py", "not in the file", "x")], [DOC])),
        b1_finish(b1_item(norm_id, "FIX_CODE", [FIX], [DOC])),
    ]})
    record = run_episode(episode, worktree_path=root, settings=B1, llm=llm)
    assert "COULD NOT BE APPLIED" in llm.calls[1]["user"]
    assert record.outcome.status is RunStatus.COMPLETED and record.outcome.revisions_used == 0


def test_b1_fix_code_without_cited_evidence_escalates(mixed) -> None:
    root, episode, norm_id, twice_id, twice_file = mixed
    llm = FakeLLMClient({AgentRole.SINGLE_AGENT: [b1_finish(
        b1_item(twice_id, "UPDATE_TEST", [twice_update(twice_file)], [MSG]), b1_item(norm_id, "FIX_CODE", [FIX], []))]})
    record = run_episode(episode, worktree_path=root, settings=B1, llm=llm)
    preds = {p.test_id: p.predicted_action for p in record.predictions}
    assert preds[norm_id] is DecisionLabel.ESCALATE
    assert "absence of intent evidence" in record.outcome.escalation.summary


@pytest.mark.parametrize("condition", ["A2", "B1"])
def test_episode_llm_budget_is_shared_and_exhaustion_is_not_a_decision(mixed, condition) -> None:
    root, episode, *_ = mixed
    tight = episode.model_copy(update={"budget": EpisodeBudget(max_llm_calls=2)})
    settings = A2 if condition == "A2" else B1
    llm = FakeLLMClient({role: [search()] * 5 for role in (AgentRole.SINGLE_AGENT,)} | {
        AgentRole.INTENT_ANALYST: [{"group_id": "g1", "hypothesis": {"kind": "outdated_test", "explanation": "s"},
                                    "cited_evidence_ids": [MSG], "confidence": 0.9,
                                    "recommended_decision": "UPDATE_TEST"}],
        AgentRole.REPAIRER: [{"action": "search_repo", "edits": [], "rationale": "s", "evidence_ids": [],
                              **NO_TOOL, "query": "twice"}] * 5,
    })
    record = run_episode(tight, worktree_path=root, settings=settings, llm=llm)
    assert record.outcome.status is RunStatus.BUDGET_EXHAUSTED and record.predictions == []
    assert record.usage.llm_calls == 2  # the third call was refused before reaching the model
    status = subprocess.run(["git", "-C", str(root), "status", "--porcelain"], capture_output=True, text=True, check=True)
    assert status.stdout == ""


def test_malformed_tool_loop_reply_is_a_validation_error_not_an_escalation(mixed) -> None:
    root, episode, *_ = mixed
    llm = FakeLLMClient({AgentRole.SINGLE_AGENT: [{"action": "finish"}]})  # missing required fields
    record = run_episode(episode, worktree_path=root, settings=B1, llm=llm)
    assert record.outcome.status is RunStatus.INCOMPLETE and record.outcome.escalation is None
    assert record.errors[0].category is ErrorCategory.VALIDATION_ERROR and record.patches == []
