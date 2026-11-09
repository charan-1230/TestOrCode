"""History tools (``tools.git_tools``) and their agent wrappers: nothing newer than the cutoff is ever returned.

The episode worktree is the CI commit's parent plus the CI diff (uncommitted), in a clone whose object database also
holds every later commit.  The synthetic repository reproduces that; the pilot test does it on the real boltons clone for
the S5 episode (dev-01), whose withheld commit message must stay invisible.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from testorcode.agents.tools import AgentTools
from testorcode.errors import HistoryCutoffError, ToolError
from testorcode.schemas.contracts import EpisodeContext
from testorcode.schemas.episode_schemas import ToolName
from testorcode.schemas.result_schemas import ToolCallStatus
from testorcode.schemas.tests.conftest import make_s1
from testorcode.tools import git_tools
from testorcode.tools.trace import ToolRecorder, recording

ROOT = Path(__file__).resolve().parents[1]
BOLTONS = ROOT.parent / "boltons"
SECRET = "SECRET future message: the old test is outdated"


def git(root: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def repo(tmp_path: Path) -> dict[str, object]:
    """old -> parent -> future (the CI commit); the worktree is checked out at parent with an uncommitted change."""
    root = tmp_path / "repo"
    root.mkdir()
    git(root, "init", "-q")
    git(root, "config", "user.email", "t@example.com")
    git(root, "config", "user.name", "T")
    (root / "lib.py").write_text("def f():\n    return 1\n\n\ndef g():\n    return 2\n")
    git(root, "add", ".")
    git(root, "commit", "-qm", "old: add f and g")
    old = git(root, "rev-parse", "HEAD")
    (root / "lib.py").write_text("def f():\n    return 1\n\n\ndef g():\n    return 3\n")
    git(root, "commit", "-qam", "parent: g returns 3")
    parent = git(root, "rev-parse", "HEAD")
    (root / "lib.py").write_text("def f():\n    return 10\n\n\ndef g():\n    return 3\n")
    (root / "test_lib.py").write_text("assert True\n")
    git(root, "add", ".")
    git(root, "commit", "-qm", SECRET)
    future = git(root, "rev-parse", "HEAD")
    git(root, "checkout", "-q", "--detach", parent)
    (root / "lib.py").write_text("def f():\n    return 10\n\n\ndef g():\n    return 3\n")  # the CI diff, uncommitted
    return {"root": root, "old": old, "parent": parent, "future": future}


def test_git_log_walks_only_the_cutoffs_history(repo) -> None:
    commits = git_tools.git_log(repo["root"], cutoff_commit=repo["parent"])
    assert [c.sha for c in commits] == [repo["parent"], repo["old"]]
    assert all(SECRET not in c.message for c in commits)
    assert [c.sha for c in git_tools.git_log(repo["root"], cutoff_commit=repo["parent"], path="lib.py", max_commits=1)] == [
        repo["parent"]]


def test_git_show_refuses_a_commit_after_the_cutoff(repo) -> None:
    text = git_tools.git_show(repo["root"], repo["old"][:10], cutoff_commit=repo["parent"])
    assert "old: add f and g" in text and "+def g():" in text
    with pytest.raises(HistoryCutoffError):
        git_tools.git_show(repo["root"], repo["future"], cutoff_commit=repo["parent"])


@pytest.mark.parametrize("sha", ["HEAD", "--all", "main~1", "abc", "a" * 41, "../x"])
def test_git_show_accepts_only_hex_shas(repo, sha: str) -> None:
    with pytest.raises(ToolError):
        git_tools.git_show(repo["root"], sha, cutoff_commit=repo["parent"])


def test_git_blame_names_only_visible_commits_and_skips_the_ci_diff(repo) -> None:
    blamed = git_tools.git_blame(repo["root"], "lib.py", 1, 6, cutoff_commit=repo["parent"])
    assert {c.sha for c in blamed} == {repo["old"], repo["parent"]}  # line 2 (return 10) is the uncommitted CI diff
    assert git_tools.git_blame(repo["root"], "lib.py", 2, 2, cutoff_commit=repo["parent"]) == []
    with pytest.raises(ToolError):
        git_tools.git_blame(repo["root"], "../lib.py", 1, 1, cutoff_commit=repo["parent"])


def _tools(repo, *, cutoff: str | None, allowed=None) -> AgentTools:
    context = EpisodeContext.from_episode_input(make_s1().agent_input, repo["root"]).model_copy(
        update={"history_cutoff_sha": cutoff, **({"allowed_tools": tuple(allowed)} if allowed is not None else {})})
    return AgentTools(context, retriever=None, timeout_seconds=30)  # type: ignore[arg-type]


def test_agent_git_tools_record_events_and_refuse_future_commits(repo) -> None:
    tools = _tools(repo, cutoff=repo["parent"])
    recorder = ToolRecorder()
    with recording(recorder):
        log = tools.call("git_log", path="lib.py")
        refused = tools.call("git_show", sha=repo["future"])
        shown = tools.call("git_show", sha=repo["old"])
        blame = tools.call("git_blame", path="lib.py", start_line=4, end_line=6)
    assert repo["parent"][:12] in log and SECRET not in log
    assert refused.startswith("REFUSED") and SECRET not in refused
    assert "old: add f and g" in shown and repo["parent"][:12] in blame
    assert [e.tool for e in recorder.events] == [ToolName.GIT_LOG, ToolName.GIT_SHOW, ToolName.GIT_SHOW, ToolName.GIT_BLAME]
    assert recorder.events[1].status is ToolCallStatus.NOT_PERMITTED


def test_agent_git_tools_need_a_cutoff_and_permission(repo) -> None:
    assert _tools(repo, cutoff=None).call("git_log").startswith("INVALID")
    no_git = [ToolName.READ_FILE, ToolName.RUN_TESTS]
    assert _tools(repo, cutoff=repo["parent"], allowed=no_git).call("git_log").startswith("REFUSED")


def test_episode_context_cuts_history_at_the_parent_commit() -> None:
    episode = make_s1().agent_input
    context = EpisodeContext.from_episode_input(episode, Path("/tmp"))
    assert context.history_cutoff_sha == episode.ci_event.parent_commit.sha != context.commit_sha


@pytest.mark.skipif(not (BOLTONS / ".git").exists(), reason="needs the pinned boltons clone at ../boltons")
def test_withheld_s5_message_is_invisible_through_history_tools() -> None:
    """dev-01 (S5): with the cutoff at the CI commit's parent, no history tool returns the withheld message."""
    verification = json.loads((ROOT / "datasets" / "pilot_verification.json").read_text())
    episodes = {json.loads(line)["episode_id"]: json.loads(line) for line in
                (ROOT / "datasets" / "pilot_dev_agent_view.jsonl").read_text().splitlines() if line.strip()}
    ci = episodes["dev-01"]["ci_event"]
    ci_sha, parent = ci["commit"]["sha"], ci["parent_commit"]["sha"]
    notes = json.dumps(verification)
    withheld_hash = "d6a334fb701d85ebdf318f16784bf9ac135adff457268d9948b02b06ea94d4c1"  # datasets/README.md
    assert withheld_hash in notes

    def digest(message: str) -> str:
        return hashlib.sha256(message.strip().encode()).hexdigest()

    commits = git_tools.git_log(BOLTONS, cutoff_commit=parent, max_commits=50)
    assert commits and all(c.sha != ci_sha and digest(c.message) != withheld_hash for c in commits)
    for path in {c["path"] for c in ci["changed_files"]}:
        assert all(digest(c.message) != withheld_hash
                   for c in git_tools.git_log(BOLTONS, cutoff_commit=parent, path=path, max_commits=50))
    with pytest.raises(HistoryCutoffError):
        git_tools.git_show(BOLTONS, ci_sha, cutoff_commit=parent)
