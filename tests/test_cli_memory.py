"""``scripts/run_episode.py --memory``: the CLI builds a namespaced ``JsonlMemoryStore``, seeds the episode's declared
memory before any agent acts, runs FULL through the real orchestrator and writes the approved record.

Real: orchestrator, agents, git worktree, sandboxed pytest (bwrap), patch tools, probes, mutants, store.  Replaced:
the hosted model (scripted ``FakeLLMClient``), the dataset loaders and the episode materialiser (a fixture repository
instead of the pinned boltons clone).
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from testorcode.llm_client import FakeLLMClient
from testorcode.schemas.episode_schemas import ExecutableProbe, MemoryRecord, RepairDirection
from testorcode.schemas.result_schemas import AgentRole
from testorcode.schemas.tests.conftest import NORM, make_s1
from testorcode.tools import sandbox

pytestmark = pytest.mark.skipif(not sandbox.bwrap_available(), reason="bubblewrap unavailable")

ROOT = Path(__file__).resolve().parents[1]
CORE = "def normalise(s):\n    return s.capitalize()\n"
TESTS = "from synthlib.core import normalise\n\n\ndef test_normalise():\n    assert normalise('a') == 'a'\n"
NO_TOOL = {"query": "", "path": "", "start_line": 1, "end_line": 0, "selectors": []}


def git(root: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True, text=True).stdout.strip()


def load_cli():
    spec = importlib.util.spec_from_file_location("run_episode_cli", ROOT / "scripts" / "run_episode.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture_repo(root: Path) -> str:
    for rel, text in {"pyproject.toml": '[tool.pytest.ini_options]\npythonpath = ["src"]\n',
                      "src/synthlib/__init__.py": "", "src/synthlib/core.py": CORE,
                      "tests/test_synthlib.py": TESTS}.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)
    git(root, "init", "-q")
    git(root, "add", ".")
    git(root, "-c", "user.name=t", "-c", "user.email=t@x", "commit", "-qm", "ci")
    return git(root, "rev-parse", "HEAD")


def run_cli(tmp_path: Path, monkeypatch, condition: str, memory: Path) -> tuple[int, Path]:
    worktree = tmp_path / f"wt-{condition}"
    worktree.mkdir()
    sha = fixture_repo(worktree)
    full = make_s1()
    seeded = MemoryRecord(
        record_id="mem-declared", repo_id=full.agent_input.repo.repo_id, function_qualname=NORM,
        behaviour_signature="normalise capitalises (declared prior memory)", direction=RepairDirection.UPDATE_TEST,
        probe=ExecutableProbe(probe_id="p-declared", probe_code="def test_declared():\n    assert normalise('x') == 'X'\n",
                              probe_selector="tests/test_synthlib.py::test_normalise"),
        evidence_ids=["ev-msg"], approved_at_commit=sha,
    )
    ai = full.agent_input
    agent_input = ai.model_copy(update={
        "ci_event": ai.ci_event.model_copy(update={"commit": ai.ci_event.commit.model_copy(update={"sha": sha})}),
        "corpus": ai.corpus.model_copy(update={"history_cutoff_commit": sha}),
        "visible_memory": [seeded],
    })
    full = full.model_copy(update={"agent_input": agent_input})
    cli = load_cli()
    monkeypatch.setattr(cli, "_materializer", lambda: (lambda repo, dest, episode: dest))  # already built above
    monkeypatch.setattr(cli, "load_agent_inputs", lambda path: [agent_input])
    monkeypatch.setattr(cli, "load_episodes", lambda path: [full])
    monkeypatch.setattr(cli, "_dataset_version", lambda split: "fixture")
    monkeypatch.setattr(cli, "configure_logging", lambda *a, **k: None)  # global logger state must not leak
    monkeypatch.setattr(cli, "create_llm_client", lambda settings: FakeLLMClient({
        AgentRole.INTENT_ANALYST: [{"group_id": "g1", "hypothesis": {"kind": "outdated_test", "explanation": "s"},
                                    "expected_behaviour": "normalise capitalises", "cited_evidence_ids": ["ev-msg"],
                                    "contradicting_evidence_ids": [], "confidence": 0.9,
                                    "recommended_decision": "UPDATE_TEST"}],
        AgentRole.REPAIRER: [{"action": "patch", "edits": [{"path": "tests/test_synthlib.py", "old": "== 'a'",
                                                            "new": "== 'A'"}], "rationale": "s",
                              "evidence_ids": ["ev-msg"], **NO_TOOL}],
        AgentRole.INTEGRITY_GUARD: [{"verdict": "consistent", "test_meaning_preserved": True, "reasons": []}],
    }))
    for key in ("EXPERIMENT_CONDITION", "RANDOM_SEED", "TEST_PYTHON", "LLM_MODEL"):
        monkeypatch.delenv(key, raising=False)
    out = tmp_path / f"out-{condition}"
    monkeypatch.setattr(sys, "argv", ["run_episode.py", "--episode", full.episode_id, "--condition", condition,
                                      "--worktree", str(worktree), "--out", str(out), "--memory", str(memory)])
    return cli.main(), out


def test_cli_full_seeds_namespaces_and_writes_memory(tmp_path, monkeypatch, capsys) -> None:
    memory = tmp_path / "memory" / "store.jsonl"
    status, out = run_cli(tmp_path, monkeypatch, "FULL", memory)
    assert status == 0
    assert "memory: seeded ['mem-declared']" in capsys.readouterr().out
    [summary_file] = out.glob("*.summary.json")
    summary = json.loads(summary_file.read_text())
    assert summary["condition"] == "FULL" and summary["guard_decision"] == "approve"
    assert summary["maintainer_approved"] is True
    assert [w["written"] for w in summary["memory_writes"]] == [True]
    [record_file] = [p for p in out.glob("*.json") if not p.name.endswith(".summary.json")]
    reads = json.loads(record_file.read_text())["memory_reads"]
    assert [(r["record_id"], r["probe_status"]) for r in reads] == [("mem-declared", "passed")]  # seeded, probed
    lines = [json.loads(line) for line in memory.read_text().splitlines()]
    assert {entry["record"]["record_id"]: entry["origin"] for entry in lines} == {
        "mem-declared": "seeded", "ep-0001-g1": "approved"}
    assert (memory.parent / "store.jsonl.namespace").read_text().strip() == "FULL/seed=42/dev"


def test_cli_ignores_memory_for_a_condition_without_memory(tmp_path, monkeypatch) -> None:
    memory = tmp_path / "memory" / "a2.jsonl"
    status, _ = run_cli(tmp_path, monkeypatch, "A2", memory)
    assert status == 0 and not memory.exists() and not memory.parent.exists()


def test_cli_refuses_a_store_of_another_condition(tmp_path, monkeypatch) -> None:
    from testorcode.errors import ConfigError

    memory = tmp_path / "memory" / "store.jsonl"
    memory.parent.mkdir()
    (memory.parent / "store.jsonl.namespace").write_text("A3/seed=42/dev\n")
    with pytest.raises(ConfigError, match="needs its own store"):
        run_cli(tmp_path, monkeypatch, "FULL", memory)
