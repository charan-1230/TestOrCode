"""FULL condition end to end, offline: real orchestrator, real agents, real git worktree, real pytest, real patch tools,
real ``JsonlMemoryStore`` with sandboxed probes, real cached-mutant runs.  Only the model replies are scripted.

Covered: an approved FIX_CODE repair reaches the maintainer and the memory write; the written record is usable while
its probe passes and blocked once the behaviour regresses; an UPDATE_TEST repair is approved on measured mutant-kill
strength and stored with a probe derived from the updated test; a loosened test edit that pins nothing is escalated
and writes nothing; a maintainer refusal writes nothing.
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
from testorcode.memory import JsonlMemoryStore
from testorcode.orchestrator import Agents, run_episode
from testorcode.schemas.contracts import EpisodeContext
from testorcode.schemas.episode_schemas import (
    DecisionLabel,
    EvidenceItem,
    EvidenceSourceType,
    RepairDirection,
    ToolName,
)
from testorcode.schemas.result_schemas import (
    AgentRole,
    GuardDecision,
    PatchApplicationStatus,
    ProbeStatus,
    RunStatus,
    TestRunStatus,
)
from testorcode.schemas.tests.conftest import NORM, failing_test, make_s1
from testorcode.tools import sandbox
from testorcode.tools.pytest_runner import run_tests

pytestmark = pytest.mark.skipif(not sandbox.bwrap_available(), reason="bubblewrap unavailable: probes/mutants not run")

DOC = "docstring:synthlib.core.normalise"
SETTINGS = Settings.from_env({"LLM_MODEL": "fake-model-1", "EXPERIMENT_CONDITION": "FULL", "TEST_TIMEOUT_SECONDS": "60"})
CORE = "def normalise(s):\n    return s.capitalize()\n"
TESTS = (
    "from synthlib.core import normalise\n\n\n"
    "def test_normalise():\n    assert normalise('a') == 'a'\n\n\n"
    "def test_other():\n    assert len('x') == 1\n"
)
NO_TOOL = {"query": "", "path": "", "start_line": 1, "end_line": 0, "selectors": []}
CONSISTENT = {"verdict": "consistent", "test_meaning_preserved": True, "reasons": []}


def git(root: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True, text=True).stdout.strip()


def build(tmp_path: Path, monkeypatch, tests: str = TESTS, core: str = CORE):
    root = tmp_path / "repo"
    files = {"pyproject.toml": "[tool.pytest.ini_options]\npythonpath = [\"src\"]\n", "src/synthlib/__init__.py": "",
             "src/synthlib/core.py": core, "tests/test_synthlib.py": tests}
    for rel, text in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)
    git(root, "init", "-q")
    git(root, "add", ".")
    git(root, "-c", "user.name=t", "-c", "user.email=t@x", "commit", "-qm", "ci state")
    trusted = partial(run_tests, isolation="trusted-host")  # the fixture's own tests, as in the other integration tests
    for module in (triage, integrity_guard, agent_tools, orchestrator):
        monkeypatch.setattr(module, "run_tests", trusted)
    # Probes and mutants are NOT patched: they run in the bwrap sandbox, as in production.
    base = make_s1().agent_input
    sha = git(root, "rev-parse", "HEAD")  # memory checks history, so the episode names the fixture's real commit
    items = [*base.corpus.items, EvidenceItem(evidence_id=DOC, source_type=EvidenceSourceType.DOCSTRING,
                                              source_ref="src/synthlib/core.py",
                                              text="normalise(s): return s stripped and lower-cased.")]
    episode = base.model_copy(update={
        "ci_event": base.ci_event.model_copy(update={"commit": base.ci_event.commit.model_copy(update={"sha": sha})}),
        "corpus": base.corpus.model_copy(update={"items": items, "history_cutoff_commit": sha}),
    })
    store = JsonlMemoryStore(tmp_path / "memory.jsonl", timeout_seconds=60, isolation="bwrap")
    return root, episode, store


def analysis(decision: DecisionLabel, cited: list[str]) -> dict:
    kind = {"UPDATE_TEST": "outdated_test", "FIX_CODE": "regression"}[decision.value]
    return {"group_id": "g1", "hypothesis": {"kind": kind, "explanation": "scripted"},
            "expected_behaviour": "normalise lower-cases" if kind == "regression" else "normalise capitalises",
            "cited_evidence_ids": cited, "contradicting_evidence_ids": [], "confidence": 0.9,
            "recommended_decision": decision.value}


def edit(path: str, old: str, new: str, evidence=()) -> dict:
    return {"action": "patch", "edits": [{"path": path, "old": old, "new": new}], "rationale": "scripted",
            "evidence_ids": list(evidence), **NO_TOOL}


def fix_code_llm(cited=(DOC,)) -> FakeLLMClient:
    return FakeLLMClient({
        AgentRole.INTENT_ANALYST: [analysis(DecisionLabel.FIX_CODE, list(cited))],
        AgentRole.REPAIRER: [edit("src/synthlib/core.py", "s.capitalize()", "s.lower()")],
        AgentRole.INTEGRITY_GUARD: [CONSISTENT],
    })


def test_full_fix_code_is_approved_written_reused_and_later_invalidated(tmp_path, monkeypatch) -> None:
    root, episode, store = build(tmp_path, monkeypatch)
    asked = []
    record = run_episode(episode, worktree_path=root, settings=SETTINGS, llm=fix_code_llm(), memory=store,
                         maintainer=lambda group, decision, result: asked.append(decision) or True)

    assert record.provenance.condition.value == "FULL" and record.outcome.status is RunStatus.COMPLETED
    assert record.integrity.guard_decision is GuardDecision.APPROVE and record.final_test_status is TestRunStatus.PASSED
    assert asked == [DecisionLabel.FIX_CODE] and record.outcome.maintainer_approved is True
    [write] = record.memory_writes
    assert write.written and write.approved and write.record_id == f"{episode.episode_id}-g1"
    [stored] = store.records()
    assert stored.direction is RepairDirection.FIX_CODE and stored.function_qualname == NORM
    assert "def test_normalise" in stored.probe.probe_code and "test_other" not in stored.probe.probe_code
    assert DOC in stored.evidence_ids
    events = [(e.tool, e.called_by) for e in record.tool_events]
    assert (ToolName.MEMORY_READ, AgentRole.INTENT_ANALYST) in events  # read (empty) before deciding
    assert (ToolName.MEMORY_WRITE, AgentRole.INTEGRITY_GUARD) in events

    ctx = EpisodeContext.from_episode_input(episode, root)
    assert store.read(ctx, NORM).usable == [stored]  # the approved behaviour holds on the repaired code
    # A later regression of the same function: the record's probe fails, so it is not evidence any more.
    (root / "src/synthlib/core.py").write_text(CORE)
    second = run_episode(episode.model_copy(update={"episode_id": "ep-0002"}), worktree_path=root, settings=SETTINGS,
                         llm=fix_code_llm(cited=(DOC, stored.record_id)), memory=store)
    [read] = second.memory_reads
    assert read.record_id == stored.record_id and read.probe_status is ProbeStatus.FAILED
    assert read.blocked_reason == "probe_failed" and read.used_as_evidence is False
    # Citing the blocked record is citing an unknown id: the analysis is not accepted.
    assert second.predictions[0].predicted_action is DecisionLabel.ESCALATE


def test_full_update_test_is_approved_on_measured_strength_and_stored(tmp_path, monkeypatch) -> None:
    root, episode, store = build(tmp_path, monkeypatch)
    llm = FakeLLMClient({
        AgentRole.INTENT_ANALYST: [analysis(DecisionLabel.UPDATE_TEST, ["ev-msg"])],
        AgentRole.REPAIRER: [edit("tests/test_synthlib.py", "== 'a'", "== 'A'", ["ev-msg"])],
        AgentRole.INTEGRITY_GUARD: [CONSISTENT],
    })
    record = run_episode(episode, worktree_path=root, settings=SETTINGS, llm=llm, memory=store,
                         maintainer=lambda *_: True)

    assert record.integrity.guard_decision is GuardDecision.APPROVE
    mutation = record.integrity.mutation
    assert mutation.status is TestRunStatus.PASSED and mutation.mutants_total >= 1
    assert mutation.killed_by_repaired_tests >= 1 and mutation.killed_by_reference_tests is None
    assert record.integrity.weakening.checks_run and record.integrity.weakening.weakening_detected is False
    assert any(e.tool is ToolName.RUN_MUTANTS and e.called_by is AgentRole.INTEGRITY_GUARD for e in record.tool_events)
    [stored] = store.records()
    assert stored.direction is RepairDirection.UPDATE_TEST and "== 'A'" in stored.probe.probe_code
    assert store.read(EpisodeContext.from_episode_input(episode, root), NORM).usable == [stored]


def test_a_loosened_inequality_edit_is_rejected_revised_and_not_stored(tmp_path, monkeypatch) -> None:
    # The CI test asserts an inequality; changing its literal is allowed by the tool-level patch rule, but a new
    # excluded value pins no new behaviour: the Guard's weakening analysis rejects it (and the revision again).
    tests = TESTS.replace("normalise('a') == 'a'", "normalise('a') != 'A'")
    root, episode, store = build(tmp_path, monkeypatch, tests)
    loosen = edit("tests/test_synthlib.py", "!= 'A'", "!= 'zz'", ["ev-msg"])
    llm = FakeLLMClient({
        AgentRole.INTENT_ANALYST: [analysis(DecisionLabel.UPDATE_TEST, ["ev-msg"])],
        AgentRole.REPAIRER: [loosen, loosen],
    })
    asked = []
    record = run_episode(episode, worktree_path=root, settings=SETTINGS, llm=llm, memory=store,
                         maintainer=lambda *a: asked.append(a) or True)

    assert record.integrity.guard_decision is GuardDecision.REJECT
    assert any("comparison(s) loosened" in r for r in record.integrity.rejection_reasons)
    assert record.outcome.escalation.trigger == "guard_rejection_after_revision"
    assert [p.application_status for p in record.patches] == [PatchApplicationStatus.REJECTED_BY_GUARD] * 2
    assert (root / "tests/test_synthlib.py").read_text() == tests  # rolled back
    assert asked == [] and record.memory_writes == [] and store.records() == []


def test_a_vacuous_setup_change_is_caught_only_by_cached_mutants(tmp_path, monkeypatch) -> None:
    """Weakening no syntax check can see: a literal setup keyword (allowed) routes the test around the logic."""
    from testorcode.agents import integrity_guard as guard_module
    from testorcode.tools.mutation import generate_mutants

    core = "def normalise(s, mode='x'):\n    if mode == 'raw':\n        return s\n    return s.capitalize()\n"
    tests = "from synthlib.core import normalise\n\n\ndef test_normalise():\n    r = normalise('a', mode='x')\n    assert r == 'a'\n"
    root, episode, store = build(tmp_path, monkeypatch, tests, core)
    # A selected/cached mutant of the function's core logic (validated and executed now, never trusted as a result).
    cached = [m for m in generate_mutants(core, "src/synthlib/core.py", "normalise", module="synthlib.core", limit=20)
              if "line 4" in m.operator]
    assert cached
    llm = FakeLLMClient({
        AgentRole.INTENT_ANALYST: [analysis(DecisionLabel.UPDATE_TEST, ["ev-msg"])],
        AgentRole.REPAIRER: [edit("tests/test_synthlib.py", "mode='x'", "mode='raw'", ["ev-msg"])],
        AgentRole.INTEGRITY_GUARD: [CONSISTENT],
    })
    guard = partial(guard_module.verify_repair, cached_mutants={NORM: cached})
    record = run_episode(episode, worktree_path=root, settings=SETTINGS, llm=llm, memory=store,
                         agents=Agents(integrity_guard=guard), maintainer=lambda *_: True)

    assert record.integrity.weakening.weakening_detected is False  # syntactically a legitimate setup change
    assert record.integrity.guard_decision is GuardDecision.ESCALATE
    assert "updated tests detect none" in record.outcome.escalation.summary
    assert record.integrity.mutation.killed_by_repaired_tests == 0 and record.integrity.mutation.mutants_total >= 1
    assert record.memory_writes == [] and store.records() == []
    assert "mode='x'" in (root / "tests/test_synthlib.py").read_text()  # reverted


def test_an_unsafe_fix_code_patch_is_never_approved(tmp_path, monkeypatch) -> None:
    root, episode, store = build(tmp_path, monkeypatch)
    gamed = {"action": "patch", "edits": [
        {"path": "src/synthlib/core.py", "old": "def normalise(s):", "new": "import sys\n\n\ndef normalise(s):"},
        {"path": "src/synthlib/core.py", "old": "return s.capitalize()",
         "new": "return s.lower() if 'pytest' in sys.modules else s.capitalize()"},
    ], "rationale": "s", "evidence_ids": [], **NO_TOOL}
    llm = FakeLLMClient({AgentRole.INTENT_ANALYST: [analysis(DecisionLabel.FIX_CODE, [DOC])],
                         AgentRole.REPAIRER: [gamed, gamed]})
    asked = []
    record = run_episode(episode, worktree_path=root, settings=SETTINGS, llm=llm, memory=store,
                         maintainer=lambda *a: asked.append(a) or True)

    assert record.outcome.status is RunStatus.ESCALATED and record.integrity.guard_ran is False
    assert all(p.application_status is PatchApplicationStatus.FAILED_TO_APPLY for p in record.patches)
    assert all("test-run detection" in p.apply_error for p in record.patches)
    assert (root / "src/synthlib/core.py").read_text() == CORE and asked == [] and store.records() == []


def test_a_stored_record_is_read_cited_and_reused_in_a_later_episode(tmp_path, monkeypatch) -> None:
    root, episode, store = build(tmp_path, monkeypatch)
    first = run_episode(episode, worktree_path=root, settings=SETTINGS, memory=store, maintainer=lambda *_: True,
                        llm=FakeLLMClient({
                            AgentRole.INTENT_ANALYST: [analysis(DecisionLabel.UPDATE_TEST, ["ev-msg"])],
                            AgentRole.REPAIRER: [edit("tests/test_synthlib.py", "== 'a'", "== 'A'", ["ev-msg"])],
                            AgentRole.INTEGRITY_GUARD: [CONSISTENT]}))
    [stored] = store.records()
    assert first.integrity.guard_decision is GuardDecision.APPROVE and stored.record_id == f"{episode.episode_id}-g1"

    # A later episode in the same stream: a later commit adds another test of the same function that still encodes
    # the old behaviour, so it fails for the same, already approved change.
    test_file = root / "tests/test_synthlib.py"
    test_file.write_text(test_file.read_text() + "\n\ndef test_two():\n    assert normalise('b') == 'b'\n")
    later = episode.model_copy(update={"episode_id": "ep-0002", "ci_event": episode.ci_event.model_copy(update={
        "failing_tests": [failing_test("tests/test_synthlib.py::test_two", NORM)]})})
    second = run_episode(later, worktree_path=root, settings=SETTINGS, memory=store, maintainer=lambda *_: True,
                         llm=FakeLLMClient({
                             AgentRole.INTENT_ANALYST: [analysis(DecisionLabel.UPDATE_TEST, [stored.record_id])],
                             AgentRole.REPAIRER: [edit("tests/test_synthlib.py", "== 'b'", "== 'B'", [stored.record_id])],
                             AgentRole.INTEGRITY_GUARD: [CONSISTENT]}))

    [read] = second.memory_reads
    assert read.record_id == stored.record_id and read.probe_status is ProbeStatus.PASSED and read.used_as_evidence
    assert second.predictions[0].predicted_action is DecisionLabel.UPDATE_TEST
    assert second.predictions[0].cited_evidence_ids == [stored.record_id]
    assert second.integrity.guard_decision is GuardDecision.APPROVE and second.outcome.status is RunStatus.COMPLETED
    newest = {r.record_id: r for r in store.records()}["ep-0002-g1"]
    assert newest.supersedes == stored.record_id  # a newer approved behaviour change of the same function


def test_maintainer_refusal_writes_nothing(tmp_path, monkeypatch) -> None:
    root, episode, store = build(tmp_path, monkeypatch)
    record = run_episode(episode, worktree_path=root, settings=SETTINGS, llm=fix_code_llm(), memory=store,
                         maintainer=lambda *_: False)
    assert record.integrity.guard_decision is GuardDecision.APPROVE
    [write] = record.memory_writes
    assert not write.written and write.blocked_reason == "not_approved" and store.records() == []
