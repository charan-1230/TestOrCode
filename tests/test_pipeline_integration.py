"""Offline integration: the real orchestrator and the real four agents on a real git worktree.

Only the model is scripted (``FakeLLMClient``).  Everything else runs for real: ``run_tests`` executes pytest,
``validate_patch``/``apply_patch``/``revert_patch`` change the worktree with ``git apply``, and the trusted probe is
executed by ``verify_behaviour``.  pytest and the probe run with ``isolation="trusted-host"``, which the interfaces
reserve for trusted fixtures (this repository is one), so the test does not depend on bubblewrap or Docker.
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
from testorcode.schemas.contracts import MemoryReadResult, MemoryRecordCheck
from testorcode.schemas.episode_schemas import (
    DecisionLabel,
    EvidenceItem,
    EvidenceSourceType,
    RepairDirection,
    ToolName,
)
from testorcode.schemas.result_schemas import (
    AgentRole,
    ExperimentCondition,
    GuardDecision,
    PatchApplicationStatus,
    ProbeStatus,
    RunStatus,
    TestRunPhase,
    TestRunStatus,
    ToolCallStatus,
)
from testorcode.schemas.tests.conftest import make_s1, memory_record
from testorcode.tools.mutation import MutationCheck, mutation_check
from testorcode.tools.pytest_runner import run_tests
from testorcode.tools.verification import ProbeCase, TrustedProbe, verify_behaviour

_S1 = make_s1().agent_input  # failing test tests/test_synthlib.py::test_normalise, corpus ev-msg / ev-changelog
# A FIX_CODE decision must cite the expected behaviour the change breaks, so the fixture corpus also carries the
# function's docstring (as the pilot corpora carry ``docstring:<module>`` items).
DOC = "docstring:synthlib.core.normalise"
S1 = _S1.model_copy(update={"corpus": _S1.corpus.model_copy(update={"items": [
    *_S1.corpus.items,
    EvidenceItem(evidence_id=DOC, source_type=EvidenceSourceType.DOCSTRING, source_ref="src/synthlib/core.py",
                 text="normalise(s): return s stripped and lower-cased."),
]})})
SETTINGS = Settings.from_env({"LLM_MODEL": "fake-model-1", "EXPERIMENT_CONDITION": "A2", "TEST_TIMEOUT_SECONDS": "60"})
CORE = "def normalise(s):\n    return s.capitalize()\n"  # the CI state: the regression or intended change
TESTS = (
    "from synthlib.core import normalise\n\n\n"
    "def test_normalise():\n    assert normalise('a') == 'a'\n\n\n"
    "def test_other():\n    assert len('x') == 1\n"
)
PROBE = TrustedProbe(
    probe_id="probe-normalise", module="synthlib.core", function="normalise",
    cases=(ProbeCase(args=("AbC",), expected="abc"),), provenance="dataset",
)


@pytest.fixture
def worktree(tmp_path: Path, monkeypatch) -> Path:
    root = tmp_path / "repo"
    files = {
        "pyproject.toml": "[tool.pytest.ini_options]\npythonpath = [\"src\"]\n",
        "src/synthlib/__init__.py": "",
        "src/synthlib/core.py": CORE,
        "tests/test_synthlib.py": TESTS,
    }
    for rel, text in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)
    for args in (["init", "-q"], ["add", "."], ["-c", "user.name=t", "-c", "user.email=t@example.invalid",
                                                "commit", "-qm", "ci state"]):
        subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)
    trusted = partial(run_tests, isolation="trusted-host")
    monkeypatch.setattr(triage, "run_tests", trusted)
    monkeypatch.setattr(integrity_guard, "run_tests", trusted)
    monkeypatch.setattr(agent_tools, "run_tests", trusted)
    monkeypatch.setattr(orchestrator, "run_tests", trusted)  # B1's post-repair rerun
    monkeypatch.setattr(integrity_guard, "verify_behaviour", partial(verify_behaviour, isolation="trusted-host"))
    monkeypatch.setattr(integrity_guard, "mutation_check", partial(mutation_check, isolation="trusted-host"))
    return root


def analysis(decision: DecisionLabel) -> dict:
    kind = {"UPDATE_TEST": "outdated_test", "FIX_CODE": "regression"}[decision.value]
    return {
        "group_id": "g1", "hypothesis": {"kind": kind, "explanation": "scripted"},
        "expected_behaviour": "scripted", "cited_evidence_ids": ["ev-msg"] if kind == "outdated_test" else [DOC],
        "contradicting_evidence_ids": [], "confidence": 0.9, "recommended_decision": decision.value,
    }


NO_TOOL = {"query": "", "path": "", "start_line": 1, "end_line": 0, "selectors": []}


def edit(path: str, old: str, new: str, evidence=()) -> dict:
    """A final Repairer step (the Repairer runs as a tool loop inside the orchestrator)."""
    return {"action": "patch", "edits": [{"path": path, "old": old, "new": new}], "rationale": "scripted",
            "evidence_ids": list(evidence), **NO_TOOL}


def tool_step(action: str, **arguments) -> dict:
    return {"action": action, "edits": [], "rationale": "look first", "evidence_ids": [], **NO_TOOL, **arguments}


CONSISTENT = {"verdict": "consistent", "test_meaning_preserved": True, "reasons": []}


def test_fix_code_repair_is_verified_by_a_trusted_probe_and_approved(worktree: Path) -> None:
    llm = FakeLLMClient({
        AgentRole.INTENT_ANALYST: [analysis(DecisionLabel.FIX_CODE)],
        AgentRole.REPAIRER: [edit("src/synthlib/core.py", "s.capitalize()", "s.lower()")],
        AgentRole.INTEGRITY_GUARD: [CONSISTENT],
    })
    approvals = []
    record = run_episode(
        S1, worktree_path=worktree, settings=SETTINGS, llm=llm,
        agents=Agents(integrity_guard=partial(integrity_guard.verify_repair, probe=PROBE)),
        maintainer=lambda group, decision, result: approvals.append(decision) or True,
    )

    assert record.outcome.status is RunStatus.COMPLETED and record.errors == []
    [pred] = record.predictions
    assert pred.predicted_action is DecisionLabel.FIX_CODE and pred.test_id == S1.ci_event.failing_tests[0].test_id
    assert [(r.run_id.split("-", 1)[-1], r.phase, r.status) for r in record.test_runs] == [
        ("baseline", TestRunPhase.PRE_REPAIR, TestRunStatus.FAILED),
        ("r0-post", TestRunPhase.POST_REPAIR, TestRunStatus.PASSED),  # the group's own test
        ("r0-regression", TestRunPhase.POST_REPAIR, TestRunStatus.PASSED),  # the whole suite after the repair
        ("suite", TestRunPhase.FINAL, TestRunStatus.PASSED),  # the episode's end state
    ]
    assert record.final_test_status is TestRunStatus.PASSED
    assert record.integrity.guard_decision is GuardDecision.APPROVE
    assert record.integrity.tests_untouched_for_fix_code is True
    assert record.patches[0].application_status is PatchApplicationStatus.APPLIED
    assert (worktree / "src/synthlib/core.py").read_text() == "def normalise(s):\n    return s.lower()\n"
    # Maintainer asked once; with A2 there is no store, so the write is logged as blocked, not faked.
    assert approvals == [DecisionLabel.FIX_CODE] and record.outcome.maintainer_approved is True
    [write] = record.memory_writes
    assert write.approved and not write.written and write.blocked_reason == "memory_disabled"
    # Triage mapped the test without the model; the model was asked by the other three roles only.
    assert [c["role"] for c in llm.calls] == [AgentRole.INTENT_ANALYST, AgentRole.REPAIRER, AgentRole.INTEGRITY_GUARD]
    tools = [(e.tool, e.called_by, e.arguments.get("operation")) for e in record.tool_events]
    assert tools[0] == (ToolName.RUN_TESTS, AgentRole.TRIAGE, None)
    assert (ToolName.APPLY_PATCH, AgentRole.REPAIRER, "validate") in tools
    assert (ToolName.APPLY_PATCH, AgentRole.REPAIRER, "apply") in tools
    assert tools[-4:] == [(ToolName.RUN_TESTS, AgentRole.INTEGRITY_GUARD, None)] * 2 + [
        (ToolName.RUN_MUTANTS, AgentRole.INTEGRITY_GUARD, None),  # FIX_CODE oracle-sensitivity check
        (ToolName.RUN_TESTS, AgentRole.ORCHESTRATOR, None)]  # targeted, regression, mutants, final suite
    assert record.integrity.mutation.killed_by_repaired_tests >= 1  # the untouched test detects broken variants
    assert all(e.status is ToolCallStatus.OK for e in record.tool_events)


def test_guard_rejection_feeds_the_revision_and_an_unprobed_repair_escalates(worktree: Path, monkeypatch) -> None:
    # The corrected test repair passes, but its strength cannot be measured (no executable cached mutant): escalate.
    monkeypatch.setattr(integrity_guard, "mutation_check",
                        lambda *a, **k: MutationCheck(gaps=("no executable cached mutant (test harness)",)))
    llm = FakeLLMClient({
        AgentRole.INTENT_ANALYST: [analysis(DecisionLabel.UPDATE_TEST)],
        AgentRole.REPAIRER: [
            edit("tests/test_synthlib.py", "== 'a'", "== 'X'", ["ev-msg"]),  # wrong value: real rerun fails
            edit("tests/test_synthlib.py", "== 'a'", "== 'A'", ["ev-msg"]),  # right value
        ],
        AgentRole.INTEGRITY_GUARD: [CONSISTENT],  # asked only once: after the first rerun failed it is skipped
    })
    record = run_episode(S1, worktree_path=worktree, settings=SETTINGS, llm=llm)

    assert record.outcome.status is RunStatus.ESCALATED and record.outcome.revisions_used == 1
    assert [p.application_status for p in record.patches] == [
        PatchApplicationStatus.REJECTED_BY_GUARD, PatchApplicationStatus.REVERTED,
    ]
    assert [(r.phase, r.status) for r in record.test_runs] == [
        (TestRunPhase.PRE_REPAIR, TestRunStatus.FAILED),
        (TestRunPhase.POST_REPAIR, TestRunStatus.FAILED),  # wrong literal: targeted run fails, no regression run
        (TestRunPhase.POST_REVISION, TestRunStatus.PASSED),  # targeted
        (TestRunPhase.POST_REVISION, TestRunStatus.PASSED),  # regression
        (TestRunPhase.FINAL, TestRunStatus.FAILED),  # the unverified repair was reverted: the episode ends red
    ]
    assert record.final_test_status is TestRunStatus.FAILED
    repair_prompts = [c["user"] for c in llm.calls if c["role"] is AgentRole.REPAIRER]
    assert "YOUR PREVIOUS REPAIR WAS REJECTED" in repair_prompts[1] and "post-repair run" in repair_prompts[1]
    assert record.integrity.guard_decision is GuardDecision.ESCALATE  # passing tests, strength not measured
    assert any("independent verification" in m for m in record.outcome.escalation.evidence_missing)
    assert (worktree / "tests/test_synthlib.py").read_text() == TESTS  # both patches undone
    assert record.outcome.maintainer_approved is None and record.memory_writes == []


def test_weakening_patch_is_blocked_by_the_real_patch_tools_and_escalates(worktree: Path) -> None:
    weaken = edit("tests/test_synthlib.py", "    assert normalise('a') == 'a'\n", "    pass\n", ["ev-msg"])
    llm = FakeLLMClient({
        AgentRole.INTENT_ANALYST: [analysis(DecisionLabel.UPDATE_TEST)],
        AgentRole.REPAIRER: [weaken, weaken],
    })
    record = run_episode(S1, worktree_path=worktree, settings=SETTINGS, llm=llm)

    assert record.outcome.status is RunStatus.ESCALATED
    assert record.outcome.escalation.trigger == "guard_rejection_after_revision"
    assert all(p.application_status is PatchApplicationStatus.FAILED_TO_APPLY for p in record.patches)
    assert all("not limited to expected literal values" in p.apply_error for p in record.patches)
    repair_prompts = [c["user"] for c in llm.calls if c["role"] is AgentRole.REPAIRER]
    assert "not limited to expected literal values" in repair_prompts[1]  # the reason reached the revision
    validations = [e for e in record.tool_events if e.arguments.get("operation") == "validate"]
    assert len(validations) == 2 and all(e.status is not ToolCallStatus.OK for e in validations)
    assert not any(e.arguments.get("operation") == "apply" for e in record.tool_events)
    assert (worktree / "tests/test_synthlib.py").read_text() == TESTS
    assert record.integrity.guard_ran is False


def test_repairer_uses_real_tools_before_patching(worktree: Path) -> None:
    llm = FakeLLMClient({
        AgentRole.INTENT_ANALYST: [analysis(DecisionLabel.FIX_CODE)],
        AgentRole.REPAIRER: [
            tool_step("read_file", path="src/synthlib/core.py"),
            tool_step("run_tests", selectors=["tests/test_synthlib.py"]),
            edit("src/synthlib/core.py", "s.capitalize()", "s.lower()"),
        ],
        AgentRole.INTEGRITY_GUARD: [CONSISTENT],
    })
    record = run_episode(S1, worktree_path=worktree, settings=SETTINGS, llm=llm,
                         agents=Agents(integrity_guard=partial(integrity_guard.verify_repair, probe=PROBE)))

    assert record.outcome.status is RunStatus.COMPLETED
    repairer_tools = [(e.tool, e.status) for e in record.tool_events if e.called_by is AgentRole.REPAIRER]
    assert repairer_tools[:2] == [(ToolName.READ_FILE, ToolCallStatus.OK), (ToolName.RUN_TESTS, ToolCallStatus.OK)]
    third_prompt = [c["user"] for c in llm.calls if c["role"] is AgentRole.REPAIRER][2]
    assert "return s.capitalize()" in third_prompt  # the real file content came back
    assert "status=failed" in third_prompt and "test_normalise" in third_prompt  # the real pytest result came back
    assert record.provenance.prompt_template_version.endswith("+repair-v5+guard-v1")


def test_tool_outside_allowed_tools_is_refused_and_the_model_continues(worktree: Path) -> None:
    episode = S1.model_copy(update={"allowed_tools": [ToolName.RUN_TESTS, ToolName.SEARCH_REPO]})
    llm = FakeLLMClient({
        AgentRole.INTENT_ANALYST: [analysis(DecisionLabel.FIX_CODE)],
        AgentRole.REPAIRER: [tool_step("read_file", path="src/synthlib/core.py"),
                             edit("src/synthlib/core.py", "s.capitalize()", "s.lower()")],
        AgentRole.INTEGRITY_GUARD: [CONSISTENT],
    })
    record = run_episode(episode, worktree_path=worktree, settings=SETTINGS, llm=llm)
    refused = [e for e in record.tool_events if e.tool is ToolName.READ_FILE]
    assert [e.status for e in refused] == [ToolCallStatus.NOT_PERMITTED]
    assert "REFUSED: read_file" in [c["user"] for c in llm.calls if c["role"] is AgentRole.REPAIRER][1]
    # FIX_CODE with untouched tests, a passing group run and no new suite failure is verified by the original tests.
    assert record.integrity.guard_decision is GuardDecision.APPROVE



# --------------------------------------------------------------------------- #
# B1: single tool-using agent on the same fixture
# --------------------------------------------------------------------------- #

B1_SETTINGS = Settings.from_env({"LLM_MODEL": "fake-model-1", "EXPERIMENT_CONDITION": "B1", "TEST_TIMEOUT_SECONDS": "60"})
TEST_ID = S1.ci_event.failing_tests[0].test_id


def b1_item(decision: str, edits=(), cited=None, hypothesis=None) -> dict:
    kind = hypothesis or {"UPDATE_TEST": "outdated_test", "FIX_CODE": "regression", "ESCALATE": "unknown"}[decision]
    cited = cited if cited is not None else ([DOC] if decision == "FIX_CODE" else [])
    return {"test_ids": [TEST_ID], "decision": decision, "hypothesis": kind, "explanation": "scripted",
            "cited_evidence_ids": list(cited), "confidence": 0.8,
            "edits": [{"path": p, "old": o, "new": n} for p, o, n in edits]}


def b1_finish(*items: dict) -> dict:
    return {"action": "finish", "decisions": list(items), "rationale": "scripted", **NO_TOOL}


LOWER = ("src/synthlib/core.py", "s.capitalize()", "s.lower()")


def test_b1_repairs_with_a_passing_rerun_and_no_guard(worktree: Path) -> None:
    llm = FakeLLMClient({AgentRole.SINGLE_AGENT: [
        {"action": "read_file", "decisions": [], "rationale": "look", **NO_TOOL, "path": "src/synthlib/core.py"},
        b1_finish(b1_item("FIX_CODE", [LOWER])),
    ]})
    record = run_episode(S1, worktree_path=worktree, settings=B1_SETTINGS, llm=llm)

    assert record.provenance.condition.value == "B1" and record.outcome.status is RunStatus.COMPLETED
    [pred] = record.predictions
    assert pred.predicted_action is DecisionLabel.FIX_CODE and pred.decided_by is AgentRole.SINGLE_AGENT
    assert record.integrity.guard_ran is False  # no independent verification in B1
    assert [(r.phase, r.status) for r in record.test_runs][-3:] == [
        (TestRunPhase.POST_REPAIR, TestRunStatus.PASSED), (TestRunPhase.POST_REPAIR, TestRunStatus.PASSED),
        (TestRunPhase.FINAL, TestRunStatus.PASSED)]
    assert (worktree / "src/synthlib/core.py").read_text() == "def normalise(s):\n    return s.lower()\n"
    assert {c["role"] for c in llm.calls} == {AgentRole.SINGLE_AGENT}
    tools = [(e.tool, e.called_by) for e in record.tool_events]
    assert (ToolName.SEARCH_REPO, AgentRole.SINGLE_AGENT) in tools  # standard RAG on the failing test's text
    assert (ToolName.READ_FILE, AgentRole.SINGLE_AGENT) in tools
    assert record.failure_groups[0].test_ids == [TEST_ID]


def test_b1_failed_rerun_is_reverted_and_revised_with_the_reason(worktree: Path) -> None:
    wrong = ("src/synthlib/core.py", "s.capitalize()", "s.title()")
    llm = FakeLLMClient({AgentRole.SINGLE_AGENT: [
        b1_finish(b1_item("FIX_CODE", [wrong])), b1_finish(b1_item("FIX_CODE", [LOWER])),
    ]})
    record = run_episode(S1, worktree_path=worktree, settings=B1_SETTINGS, llm=llm)

    assert record.outcome.status is RunStatus.COMPLETED and record.outcome.revisions_used == 1
    assert [p.application_status for p in record.patches] == [PatchApplicationStatus.REVERTED,
                                                              PatchApplicationStatus.APPLIED]
    assert "YOUR PREVIOUS REPAIR WAS REJECTED" in llm.calls[1]["user"] and "post-repair run" in llm.calls[1]["user"]


def test_b1_update_test_without_citation_is_an_invalid_decision(worktree: Path) -> None:
    llm = FakeLLMClient({AgentRole.SINGLE_AGENT: [
        b1_finish(b1_item("UPDATE_TEST", [("tests/test_synthlib.py", "== 'a'", "== 'A'")])),
    ]})
    record = run_episode(S1, worktree_path=worktree, settings=B1_SETTINGS, llm=llm)
    assert record.outcome.status is RunStatus.ESCALATED and record.patches == []
    assert "invalid decision" in record.outcome.escalation.summary
    assert (worktree / "tests/test_synthlib.py").read_text() == TESTS


def test_b1_tests_without_a_decision_escalate(worktree: Path) -> None:
    llm = FakeLLMClient({AgentRole.SINGLE_AGENT: [b1_finish()]})
    record = run_episode(S1, worktree_path=worktree, settings=B1_SETTINGS, llm=llm)
    assert record.outcome.status is RunStatus.ESCALATED
    assert "gave no decision" in record.outcome.escalation.summary


def test_a2_and_b1_records_on_the_same_input_are_comparable(worktree: Path) -> None:
    a2 = run_episode(S1, worktree_path=worktree, settings=SETTINGS, llm=FakeLLMClient({
        AgentRole.INTENT_ANALYST: [analysis(DecisionLabel.FIX_CODE)],
        AgentRole.REPAIRER: [edit(*LOWER)], AgentRole.INTEGRITY_GUARD: [CONSISTENT],
    }))
    subprocess.run(["git", "-C", str(worktree), "checkout", "-q", "--", "."], check=True)  # back to the CI state
    b1 = run_episode(S1, worktree_path=worktree, settings=B1_SETTINGS,
                     llm=FakeLLMClient({AgentRole.SINGLE_AGENT: [b1_finish(b1_item("FIX_CODE", [LOWER]))]}))

    assert (a2.provenance.condition.value, b1.provenance.condition.value) == ("A2", "B1")
    assert [p.test_id for p in a2.predictions] == [p.test_id for p in b1.predictions] == [TEST_ID]
    assert a2.predictions[0].predicted_action is DecisionLabel.FIX_CODE  # Guard: verified by the untouched tests
    assert a2.integrity.guard_ran and a2.integrity.tests_untouched_for_fix_code is True
    assert b1.predictions[0].predicted_action is DecisionLabel.FIX_CODE  # B1 trusts the passing rerun
    assert not b1.integrity.guard_ran
    assert a2.provenance.prompt_template_version != b1.provenance.prompt_template_version
    assert set(a2.model_dump()) == set(b1.model_dump())  # one result format for every condition


class _ProbedMemory:
    """A store whose one record passed its probe on the current code (what ``JsonlMemoryStore.read`` reports)."""

    RECORD = memory_record("mem-lower").model_copy(update={
        "direction": RepairDirection.FIX_CODE, "behaviour_signature": "normalise() returns the lower-cased string",
    })

    def read(self, context, function_qualname):
        return MemoryReadResult(checks=(MemoryRecordCheck(record=self.RECORD, probe_status=ProbeStatus.PASSED),))

    def write(self, record, approval):  # pragma: no cover - the A1 run below has no maintainer
        raise AssertionError("not expected")


def test_a1_real_pipeline_uses_memory_but_no_retrieval_commit_message_or_search(worktree: Path) -> None:
    a1 = SETTINGS.model_copy(update={"experiment_condition": ExperimentCondition.A1_NO_INTENT_RETRIEVAL})
    (worktree / "CHANGELOG.md").write_text("INTENT-MARKER: normalise now capitalises on purpose\n")
    fix = analysis(DecisionLabel.FIX_CODE) | {"cited_evidence_ids": ["mem-lower"]}  # memory is still evidence
    llm = FakeLLMClient({
        AgentRole.INTENT_ANALYST: [fix],
        # The Repairer is not told about search_repo or change notes; if it asks anyway, the calls are refused at
        # runtime, while reading source code still works.
        AgentRole.REPAIRER: [
            tool_step("search_repo", query="normalise intent"),
            tool_step("read_file", path="CHANGELOG.md"),
            tool_step("read_file", path="src/synthlib/core.py"),
            edit(*LOWER),
        ],
        AgentRole.INTEGRITY_GUARD: [CONSISTENT],
    })
    record = run_episode(
        S1, worktree_path=worktree, settings=a1, llm=llm, memory=_ProbedMemory(),
        agents=Agents(integrity_guard=partial(integrity_guard.verify_repair, probe=PROBE)),
    )

    assert record.outcome.status is RunStatus.COMPLETED
    assert record.condition is ExperimentCondition.A1_NO_INTENT_RETRIEVAL
    assert record.predictions[0].predicted_action is DecisionLabel.FIX_CODE
    assert record.integrity.guard_decision is GuardDecision.APPROVE and record.retrieved_evidence == []
    assert [m.record_id for m in record.memory_reads] == ["mem-lower"] and record.memory_reads[0].used_as_evidence
    [search] = [e for e in record.tool_events if e.tool is ToolName.SEARCH_REPO]
    assert search.status is ToolCallStatus.NOT_PERMITTED and search.called_by is AgentRole.REPAIRER

    analyst = next(c for c in llm.calls if c["role"] is AgentRole.INTENT_ANALYST)
    assert "evidence retrieval is disabled" in analyst["user"] and "[mem-lower]" in analyst["user"]
    assert S1.ci_event.commit_message not in analyst["user"] and "[ev-msg]" not in analyst["user"]
    repairer_calls = [c for c in llm.calls if c["role"] is AgentRole.REPAIRER]
    assert all("search_repo(" not in c["system"] for c in repairer_calls)
    assert all("changelog files are not readable" in c["system"] for c in repairer_calls)
    assert "REFUSED: search_repo" in repairer_calls[1]["user"]
    reads = [e for e in record.tool_events if e.tool is ToolName.READ_FILE]
    assert [(e.arguments["path"], e.status) for e in reads] == [
        ("CHANGELOG.md", ToolCallStatus.NOT_PERMITTED), ("src/synthlib/core.py", ToolCallStatus.OK)]
    assert all("INTENT-MARKER" not in c["user"] for c in llm.calls)  # never reached any prompt
    # The withheld commit message reaches no role's prompt (Analyst, Repairer, Guard review).
    assert {c["role"] for c in llm.calls} == {AgentRole.INTENT_ANALYST, AgentRole.REPAIRER, AgentRole.INTEGRITY_GUARD}
    assert all(S1.ci_event.commit_message not in c["user"] + c["system"] for c in llm.calls)
    # The source read did reach the Repairer (its tool-result header; the prompt's FILES section alone has no such line).
    assert "src/synthlib/core.py lines 1-2 of 2:" in repairer_calls[3]["user"]


def test_full_repairer_may_still_read_change_notes(worktree: Path) -> None:
    # Positive control: outside A1 the same read is permitted (FULL/A2/B1 behaviour unchanged).
    (worktree / "CHANGELOG.md").write_text("INTENT-MARKER: normalise now capitalises on purpose\n")
    llm = FakeLLMClient({
        AgentRole.INTENT_ANALYST: [analysis(DecisionLabel.FIX_CODE)],
        AgentRole.REPAIRER: [tool_step("read_file", path="CHANGELOG.md"), edit(*LOWER)],
        AgentRole.INTEGRITY_GUARD: [CONSISTENT],
    })
    record = run_episode(S1, worktree_path=worktree, settings=SETTINGS, llm=llm)
    [read] = [e for e in record.tool_events if e.tool is ToolName.READ_FILE]
    assert read.status is ToolCallStatus.OK
    assert "INTENT-MARKER" in [c for c in llm.calls if c["role"] is AgentRole.REPAIRER][1]["user"]
