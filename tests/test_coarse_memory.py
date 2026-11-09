"""Coarse memory (ablation A4): module-level notes, no probes, same approval and history guarantees."""

from __future__ import annotations

import dataclasses
import subprocess
from pathlib import Path

import pytest

from testorcode.conditions import profile_for
from testorcode.memory import JsonlMemoryStore, module_of
from testorcode.orchestrator import _MULTI_AGENT_CONDITIONS, _condition_problem
from testorcode.schemas.contracts import EpisodeContext, MemoryApproval
from testorcode.schemas.episode_schemas import (
    EditPolicy,
    ExecutableProbe,
    MemoryRecord,
    RepairDirection,
    RepoRef,
)
from testorcode.schemas.result_schemas import ExperimentCondition as C
from testorcode.schemas.result_schemas import GuardDecision, MemoryReadEvent, ProbeStatus

POLICY = EditPolicy(allowed_source_roots=("pkg/",), allowed_test_roots=("tests/",))
FAILING_PROBE = "def test_probe():\n    assert False\n"  # would block the record in a scoped store


def git(root: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    files = {"pkg/__init__.py": "", "pkg/core.py": "def shout(s):\n    return s.upper()\n\n\ndef whisper(s):\n    return s\n",
             "pkg/other.py": "def f():\n    return 1\n", "tests/test_core.py": "def test_x():\n    pass\n"}
    for rel, text in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)
    git(root, "init", "-q")
    git(root, "add", ".")
    git(root, "-c", "user.name=t", "-c", "user.email=t@x", "commit", "-qm", "c1")
    return root


def context(root: Path) -> EpisodeContext:
    return EpisodeContext(episode_id="ep-1", repo=RepoRef(repo_id="r"), worktree_path=root,
                          commit_sha=git(root, "rev-parse", "HEAD"), edit_policy=POLICY)


def record(root: Path, record_id: str, function: str, **update) -> MemoryRecord:
    return MemoryRecord(
        record_id=record_id, repo_id="r", function_qualname=function, behaviour_signature=f"{function} changed",
        direction=RepairDirection.UPDATE_TEST,
        probe=ExecutableProbe(probe_id=f"probe-{record_id}", probe_code=FAILING_PROBE),
        **{"approved_at_commit": git(root, "rev-parse", "HEAD"), **update},
    )


def approval(rec: MemoryRecord) -> MemoryApproval:
    return MemoryApproval(record_id=rec.record_id, guard_decision=GuardDecision.APPROVE, maintainer_approved=True,
                          approved_at_commit=rec.approved_at_commit)


def coarse(tmp_path: Path) -> JsonlMemoryStore:
    return JsonlMemoryStore(tmp_path / "mem.jsonl", scope="module")


def test_module_of_uses_the_source_file_and_falls_back_without_a_worktree(repo: Path) -> None:
    assert module_of("pkg.core.shout", repo) == "pkg.core"
    assert module_of("pkg.core.Klass.method", repo) == "pkg.core"
    assert module_of("boltons.cacheutils.CachedFunction.__call__") == "boltons.cacheutils"  # first class component
    assert module_of("boltons.strutils.bytes2human") == "boltons.strutils"


def test_a_coarse_read_returns_every_note_of_the_module_unchecked(tmp_path: Path, repo: Path) -> None:
    store = coarse(tmp_path)
    for rec in (record(repo, "m-shout", "pkg.core.shout"), record(repo, "m-other", "pkg.other.f")):
        store.write(rec, approval(rec))
    result = store.read(context(repo), "pkg.core.whisper")  # another function of the same module
    assert [c.record.record_id for c in result.checks] == ["m-shout"]  # the other module's note is not returned
    check = result.checks[0]
    # No probe ran (its probe would fail), yet the note is usable: that is what coarse memory removes.
    assert check.probe_status is ProbeStatus.NOT_RUN and not check.probe_gated and check.valid_as_evidence
    assert store.records()[0].scope == "module" and not store.records()[0].flagged_stale


def test_a_scoped_store_blocks_the_same_note(tmp_path: Path, repo: Path) -> None:
    store = JsonlMemoryStore(tmp_path / "scoped.jsonl", isolation="bwrap", timeout_seconds=30)
    rec = record(repo, "m-shout", "pkg.core.shout")
    store.write(rec, approval(rec))
    assert store.read(context(repo), "pkg.core.whisper").checks == ()  # another function: nothing
    assert store.records()[0].scope == "function"


def test_coarse_memory_keeps_supersession_and_the_history_check(tmp_path: Path, repo: Path) -> None:
    store = coarse(tmp_path)
    old, new = record(repo, "m-1", "pkg.core.shout"), record(repo, "m-2", "pkg.core.shout")
    store.write(old, approval(old))
    store.write(new, approval(new))
    future = record(repo, "m-3", "pkg.core.whisper", approved_at_commit="0" * 40)
    store.seed([future], worktree_path=repo)
    blocked = {c.record.record_id: c.blocked_reason for c in store.read(context(repo), "pkg.core.shout").checks}
    assert blocked == {"m-1": "superseded", "m-2": None, "m-3": "out_of_scope"}


def test_unknown_scope_is_refused(tmp_path: Path) -> None:
    with pytest.raises(Exception, match="scope"):
        JsonlMemoryStore(tmp_path / "m.jsonl", scope="file")  # type: ignore[arg-type]


def test_a4_is_full_with_module_scope_and_runs_only_on_a_coarse_store(tmp_path: Path) -> None:
    assert dataclasses.replace(profile_for(C.A4_COARSE_MEMORY), memory_scope="function") == profile_for(C.FULL)
    assert C.A4_COARSE_MEMORY in _MULTI_AGENT_CONDITIONS
    scoped, module = JsonlMemoryStore(tmp_path / "a.jsonl"), JsonlMemoryStore(tmp_path / "b.jsonl", scope="module")
    assert _condition_problem(C.A4_COARSE_MEMORY, module, _MULTI_AGENT_CONDITIONS) is None
    assert "module-scope" in _condition_problem(C.A4_COARSE_MEMORY, scoped, _MULTI_AGENT_CONDITIONS)
    assert "function-scope" in _condition_problem(C.FULL, module, _MULTI_AGENT_CONDITIONS)
    assert _condition_problem(C.FULL, scoped, _MULTI_AGENT_CONDITIONS) is None


def test_read_events_allow_unchecked_evidence_only_when_not_probe_gated() -> None:
    event = MemoryReadEvent(record_id="m-1", probe_gated=False, used_as_evidence=True)
    assert event.usable
    assert not MemoryReadEvent(record_id="m-1", probe_gated=False, blocked_reason="superseded").usable
    with pytest.raises(ValueError, match="probe passed"):
        MemoryReadEvent(record_id="m-1", used_as_evidence=True)


def test_the_analyst_is_told_coarse_notes_were_not_rechecked(repo: Path) -> None:
    from testorcode.agents.prompts import build_intent_user
    from testorcode.schemas.contracts import FailureGroupDetail
    from testorcode.schemas.episode_schemas import FailingTest, FailureGroup

    test = FailingTest(test_id="tests/test_core.py::test_x", test_file="tests/test_core.py", test_name="test_x")
    group = FailureGroupDetail(group=FailureGroup(group_id="g1", test_ids=[test.test_id]), failing_tests=(test,))
    checked = build_intent_user(context(repo), group, [], ["[m-1] note"])
    unchecked = build_intent_user(context(repo), group, [], ["[m-1] note"], probe_checked=False)
    assert "probe passed on the current code" in checked and "NOT re-checked" not in checked
    assert "NOT re-checked" in unchecked and "probe passed" not in unchecked
