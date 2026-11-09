"""Contamination probe: real change from the clone, no-context prompt, deterministic recall."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from testorcode.contamination import (
    MIN_LINE_CHARS,
    added_line_recall,
    build_user,
    real_changes,
    run_probe,
)
from testorcode.dataset import episode_path, load_episodes
from testorcode.llm_client import FakeLLMClient
from testorcode.schemas.result_schemas import AgentRole
from testorcode.schemas.tests.conftest import make_s1

BEFORE = "def normalise(s):\n    return s.lower()\n\n\ndef other():\n    return 1\n"
AFTER = "def normalise(s):\n    value = s.strip()\n    return value.capitalize()\n\n\ndef other():\n    return 1\n"


def git(root: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def repo_and_episode(tmp_path: Path):
    root = tmp_path / "repo"
    (root / "src/synthlib").mkdir(parents=True)
    git(root.parent, "init", "-q", str(root))
    shas = []
    for text, message in ((BEFORE, "c1"), (AFTER, "secret intent message")):
        (root / "src/synthlib/core.py").write_text(text)
        git(root, "add", ".")
        git(root, "-c", "user.name=t", "-c", "user.email=t@x", "commit", "-qm", message)
        shas.append(git(root, "rev-parse", "HEAD"))
    episode = make_s1()
    event = episode.agent_input.ci_event
    event = event.model_copy(update={
        "commit": event.commit.model_copy(update={"sha": shas[1]}),
        "parent_commit": event.commit.model_copy(update={"sha": shas[0]}),
    })
    agent_input = episode.agent_input.model_copy(update={"ci_event": event})
    return root, episode.model_copy(update={"agent_input": agent_input})


def test_the_real_change_comes_from_the_clone_and_the_prompt_has_no_context(repo_and_episode) -> None:
    root, episode = repo_and_episode
    [change] = real_changes(root, episode)
    assert change.path == "src/synthlib/core.py" and "return s.lower()" in change.excerpt
    assert change.added_lines == ("value = s.strip()", "return value.capitalize()")
    user = build_user(episode, change)
    assert "secret intent message" not in user and "capitalize" not in user  # no message, no answer
    assert episode.agent_input.ci_event.commit.sha in user


def test_recall_separates_a_memorised_answer_from_a_guess(repo_and_episode) -> None:
    root, episode = repo_and_episode
    llm = FakeLLMClient({AgentRole.SINGLE_AGENT: [{"code": AFTER}, {"code": BEFORE}]})
    remembered = run_probe([episode], root, llm, model_id="fake")
    guessed = run_probe([episode], root, llm, model_id="fake")
    assert remembered["episodes"][0]["added_line_recall"] == 1.0 and remembered["episodes"][0]["memorised"]
    assert guessed["episodes"][0]["added_line_recall"] == 0.0 and not guessed["episodes"][0]["memorised"]
    assert remembered["summary"] == {"episodes_scored": 1, "pooled_added_line_recall": 1.0, "episodes_memorised": 1}


def test_recall_ignores_short_lines_and_is_none_without_countable_lines() -> None:
    assert added_line_recall([], "x") is None
    assert added_line_recall(["a = compute(1)", "b = compute(2)"], "  a = compute(1)\nzzz") == 0.5
    assert MIN_LINE_CHARS >= 8


BOLTONS = Path(__file__).resolve().parents[2] / "boltons"


@pytest.mark.skipif(not (BOLTONS / ".git").exists(), reason="pinned boltons clone not present")
def test_an_s2_episode_probes_the_real_commit_not_the_injected_mutant() -> None:
    dev = {e.episode_id: e for e in load_episodes(episode_path("dev"))}
    episode = dev["dev-02"]  # S2: the agent-view diff holds the real change plus the mutant
    changes = real_changes(BOLTONS, episode)
    real = {line for c in changes for line in c.added_lines}
    agent_diff = "\n".join(c.unified_diff or "" for c in episode.agent_input.ci_event.changed_files)
    agent_added = {line[1:].strip() for line in agent_diff.splitlines() if line.startswith("+") and not line.startswith("+++")}
    assert real and real <= agent_added  # the developer's lines are in the episode diff ...
    assert agent_added - real  # ... and so is the mutant, which the probe never asks the model to reproduce
