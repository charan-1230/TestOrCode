"""FULL through the experiment runner: one memory store per condition, seed and stream, reused chronologically.

Real: ``experiment.run_experiment``/``default_run_one``, the orchestrator and agents, git, sandboxed pytest (bwrap),
patch tools, the Integrity Guard (weakening, mutants), ``JsonlMemoryStore`` with sandboxed probes, scoring.  Replaced:
the hosted model (scripted ``FakeLLMClient``), the episode materialiser (clones of a fixture repository instead of
the pinned boltons clone) and the split loader (two fixture episodes of one stream).
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from testorcode import experiment
from testorcode.config import Settings
from testorcode.experiment import (
    ExperimentConfig,
    default_run_one,
    memory_dir,
    read_records,
    run_experiment,
)
from testorcode.llm_client import FakeLLMClient
from testorcode.memory import memory_path
from testorcode.schemas.episode_schemas import RepairDirection
from testorcode.schemas.result_schemas import (
    AgentRole,
    ExperimentCondition,
    GuardDecision,
    RunStatus,
)
from testorcode.schemas.tests.conftest import NORM, failing_test, label, make_s1
from testorcode.tools import sandbox

pytestmark = pytest.mark.skipif(not sandbox.bwrap_available(), reason="bubblewrap unavailable")

FULL, A2, B1 = ExperimentCondition.FULL, ExperimentCondition.A2_NO_MEMORY, ExperimentCondition.B1
T1, T2 = "tests/test_synthlib.py::test_normalise", "tests/test_synthlib.py::test_two"
NO_TOOL = {"query": "", "path": "", "start_line": 1, "end_line": 0, "selectors": []}
CONSISTENT = {"verdict": "consistent", "test_meaning_preserved": True, "reasons": []}
SETTINGS = Settings.from_env({"LLM_MODEL": "fake-model-1", "TEST_TIMEOUT_SECONDS": "60"})


def git(root: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def stream(tmp_path: Path):
    """A fixture repository with two chronological CI commits and the two episodes of their stream."""
    base = tmp_path / "base"
    files = {"pyproject.toml": '[tool.pytest.ini_options]\npythonpath = ["src"]\n', "src/synthlib/__init__.py": "",
             "src/synthlib/core.py": "def normalise(s):\n    return s.capitalize()\n",
             "tests/test_synthlib.py": "from synthlib.core import normalise\n\n\ndef test_normalise():\n"
                                       "    assert normalise('a') == 'a'\n"}
    for rel, text in files.items():
        (base / rel).parent.mkdir(parents=True, exist_ok=True)
        (base / rel).write_text(text)
    git(base, "init", "-q")
    git(base, "add", ".")
    git(base, "-c", "user.name=t", "-c", "user.email=t@x", "commit", "-qm", "ci 1")
    first = git(base, "rev-parse", "HEAD")
    # A later commit: the first test was updated by the developer, a new test still encodes the old behaviour.
    (base / "tests/test_synthlib.py").write_text(files["tests/test_synthlib.py"].replace("== 'a'", "== 'A'")
                                                 + "\n\ndef test_two():\n    assert normalise('b') == 'b'\n")
    git(base, "-c", "user.name=t", "-c", "user.email=t@x", "commit", "-qam", "ci 2")
    second = git(base, "rev-parse", "HEAD")

    s1 = make_s1()

    def episode(eid: str, sha: str, test: str, position: int):
        ai = s1.agent_input
        agent_input = ai.model_copy(update={
            "episode_id": eid,
            "ci_event": ai.ci_event.model_copy(update={
                "commit": ai.ci_event.commit.model_copy(update={"sha": sha}),
                "failing_tests": [failing_test(test, NORM)]}),
            "corpus": ai.corpus.model_copy(update={"history_cutoff_commit": sha}),
        })
        gold = s1.model_copy(update={
            "episode_id": eid, "stream_position": position, "agent_input": agent_input,
            "ground_truth": s1.ground_truth.model_copy(update={"episode_id": eid, "per_test_labels": [
                label(test, RepairDirection.UPDATE_TEST, function=NORM, evidence=["ev-msg"])]}),
        })
        return agent_input, gold

    pairs = [episode("ep-0002", second, T2, 2), episode("ep-0001", first, T1, 1)]  # given out of order on purpose

    def materialize(repo, dest, episode_json):
        subprocess.run(["git", "clone", "-q", str(base), str(dest)], check=True, capture_output=True)
        git(dest, "checkout", "-q", episode_json["ci_event"]["commit"]["sha"])
        return dest

    return pairs, materialize


def update_test_llm(old: str, new: str, cited: str) -> FakeLLMClient:
    return FakeLLMClient({
        AgentRole.INTENT_ANALYST: [{"group_id": "g1", "hypothesis": {"kind": "outdated_test", "explanation": "s"},
                                    "expected_behaviour": "normalise capitalises", "cited_evidence_ids": [cited],
                                    "contradicting_evidence_ids": [], "confidence": 0.9,
                                    "recommended_decision": "UPDATE_TEST"}],
        AgentRole.REPAIRER: [{"action": "patch", "edits": [{"path": "tests/test_synthlib.py", "old": old, "new": new}],
                              "rationale": "s", "evidence_ids": [cited], **NO_TOOL}],
        AgentRole.INTEGRITY_GUARD: [CONSISTENT],
    })


def scripted(clients: dict) -> callable:
    """llm_factory: the next scripted client for the run's (condition, seed)."""
    def factory(settings: Settings) -> FakeLLMClient:
        return clients[(settings.experiment_condition, settings.random_seed)].pop(0)
    return factory


def run(tmp_path: Path, stream, monkeypatch, conditions, seeds, clients):
    pairs, materialize = stream
    config = ExperimentConfig(conditions=tuple(conditions), seeds=tuple(seeds), out=tmp_path / "out",
                              worktrees=tmp_path / "wt", repo=tmp_path / "unused")
    ordered = sorted(pairs, key=lambda p: p[1].stream_position)  # what load_split does with the real split
    monkeypatch.setattr(experiment, "load_split", lambda cfg: ordered)
    run_one = default_run_one(config, settings=SETTINGS, llm_factory=scripted(clients), materializer=materialize)
    result = run_experiment(config, run_one=run_one)
    return config, result, read_records(config.out / "runs.jsonl")


def test_full_runs_through_the_runner_and_reuses_memory_within_its_stream(tmp_path, stream, monkeypatch) -> None:
    clients = {(FULL, 0): [update_test_llm("== 'a'", "== 'A'", "ev-msg"),
                           update_test_llm("== 'b'", "== 'B'", "ep-0001-g1")]}  # the second cites the stored record
    config, result, records = run(tmp_path, stream, monkeypatch, [FULL], [0], clients)

    assert result.ran == 2 and result.skipped_unavailable == []
    first, second = records
    assert [r.episode_id for r in records] == ["ep-0001", "ep-0002"]  # stream order
    assert first.condition is FULL and first.outcome.status is RunStatus.COMPLETED
    assert first.integrity.guard_decision is GuardDecision.APPROVE
    assert [w.written for w in first.memory_writes] == [True]
    [read] = second.memory_reads  # the record written by the earlier episode, probe re-run on the later code
    assert read.record_id == "ep-0001-g1" and read.probe_status.value == "passed" and read.used_as_evidence
    assert second.integrity.guard_decision is GuardDecision.APPROVE
    assert all(r.evaluation is not None and r.provenance.dataset_version for r in records)  # scored by the runner
    path = memory_path(memory_dir(config), "FULL", 0, "dev")
    assert {json.loads(line)["record"]["record_id"] for line in path.read_text().splitlines()} == {
        "ep-0001-g1", "ep-0002-g1"}
    assert path.with_name(path.name + ".namespace").read_text().strip() == "FULL/seed=0/dev"


def test_seeds_have_separate_memory_and_memory_free_conditions_have_none(tmp_path, stream, monkeypatch) -> None:
    clients = {
        # Each seed's stream writes its own ep-0001-g1 in its own store and reuses only that one.
        (FULL, 0): [update_test_llm("== 'a'", "== 'A'", "ev-msg"), update_test_llm("== 'b'", "== 'B'", "ep-0001-g1")],
        (FULL, 1): [update_test_llm("== 'a'", "== 'A'", "ev-msg"), update_test_llm("== 'b'", "== 'B'", "ep-0001-g1")],
        (A2, 0): [update_test_llm("== 'a'", "== 'A'", "ev-msg"), update_test_llm("== 'b'", "== 'B'", "ev-msg")],
        (A2, 1): [update_test_llm("== 'a'", "== 'A'", "ev-msg"), update_test_llm("== 'b'", "== 'B'", "ev-msg")],
    }
    config, result, records = run(tmp_path, stream, monkeypatch, [FULL, A2], [0, 1], clients)
    assert result.ran == 8
    by = {(r.episode_id, r.condition, r.provenance.seed): r for r in records}
    for seed in (0, 1):  # each seed's stream has its own store and reuses only its own earlier record
        later = by[("ep-0002", FULL, seed)]
        assert [m.record_id for m in later.memory_reads] == ["ep-0001-g1"]
        assert memory_path(memory_dir(config), "FULL", seed, "dev").is_file()
    for seed in (0, 1):  # A2 never reads or writes memory
        for eid in ("ep-0001", "ep-0002"):
            run_ = by[(eid, A2, seed)]
            assert run_.memory_reads == [] and all(not w.written for w in run_.memory_writes)
    assert not (memory_dir(config) / "A2").exists()


def test_episodes_are_run_in_stream_order_whatever_order_they_are_given_in(tmp_path) -> None:
    config = ExperimentConfig(conditions=(FULL,), out=tmp_path, episode_ids=("dev-07", "dev-04", "dev-06"))
    assert [e.episode_id for e, _ in experiment.load_split(config)] == ["dev-04", "dev-06", "dev-07"]


def test_b1_through_the_runner_has_no_memory(tmp_path, stream, monkeypatch) -> None:
    finish = {"action": "finish", "rationale": "scripted", **NO_TOOL, "decisions": [
        {"test_ids": [T1], "decision": "ESCALATE", "hypothesis": "unknown", "explanation": "s",
         "cited_evidence_ids": [], "confidence": 0.5, "edits": []}]}
    clients = {(B1, 0): [FakeLLMClient({AgentRole.SINGLE_AGENT: [finish]}),
                         FakeLLMClient({AgentRole.SINGLE_AGENT: [{**finish, "decisions": [
                             {**finish["decisions"][0], "test_ids": [T2]}]}]})]}
    config, result, records = run(tmp_path, stream, monkeypatch, [B1], [0], clients)
    assert result.ran == 2 and all(r.condition is B1 and r.memory_reads == [] for r in records)
    assert not memory_dir(config).exists()


def test_a3_through_the_runner_reads_but_never_writes_its_own_store(tmp_path, stream, monkeypatch) -> None:
    """A3: its own namespaced store (seeded like FULL's), green reruns accepted without a Guard, no writes, and a
    resumed experiment does not re-run finished episodes."""
    a3 = ExperimentCondition.A3_NO_INTEGRITY_GUARD
    llms = [update_test_llm("== 'a'", "== 'A'", "ev-msg"), update_test_llm("== 'b'", "== 'B'", "ev-msg")]
    clients = {(a3, 0): llms}
    config, result, records = run(tmp_path, stream, monkeypatch, [a3], [0], clients)

    assert result.ran == 2 and result.skipped_unavailable == []
    for r in records:
        assert r.condition is a3 and r.outcome.status is RunStatus.COMPLETED
        assert r.integrity.guard_ran is False and r.memory_writes == []
        assert [p.decided_by for p in r.predictions] == [AgentRole.INTENT_ANALYST]
        assert r.evaluation is not None
    assert not any(c["role"] is AgentRole.INTEGRITY_GUARD for llm in llms for c in llm.calls)
    path = memory_path(memory_dir(config), "A3", 0, "dev")
    assert path.with_name(path.name + ".namespace").read_text().strip() == "A3/seed=0/dev"
    assert not path.exists() or path.read_text().strip() == ""  # nothing seeded here, nothing written

    again = run_experiment(config, run_one=lambda *a, **k: pytest.fail("finished runs must not be re-run"))
    assert again.ran == 0 and len(read_records(config.out / "runs.jsonl")) == 2
