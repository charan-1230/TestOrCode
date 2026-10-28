"""Safety of the pilot scripts' destructive worktree resets (F-5).

``build_dataset.py`` and ``materialize_episode.py`` reset their worktree with ``git checkout -f`` and ``git clean -fdx``.  These
tests use throw-away git repositories and check that the main clone, standalone repositories, foreign worktrees and unmarked
worktrees are refused WITHOUT losing the uncommitted file placed in them.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts" / "pilot"
GIT_ID = ["-c", "user.name=t", "-c", "user.email=t@example.invalid"]


@pytest.fixture(scope="module")
def mat():
    sys.path.insert(0, str(SCRIPTS))
    try:
        import materialize_episode

        yield materialize_episode
    finally:
        sys.path.remove(str(SCRIPTS))


@pytest.fixture(scope="module")
def build():
    sys.path.insert(0, str(SCRIPTS))
    try:
        import build_dataset

        yield build_dataset
    finally:
        sys.path.remove(str(SCRIPTS))


def git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *GIT_ID, "-C", str(cwd), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def make_repo(path: Path) -> tuple[Path, str]:
    path.mkdir(parents=True)
    git(path, "init", "-q")
    (path / "a.txt").write_text("one\n")
    git(path, "add", ".")
    git(path, "commit", "-qm", "init")
    return path, git(path, "rev-parse", "HEAD")


def precious(where: Path) -> Path:
    """An uncommitted, untracked file: exactly what ``git clean -fdx`` would destroy."""
    file = where / "precious.txt"
    file.write_text("uncommitted work\n")
    return file


def marker_of(mat, worktree: Path) -> Path:
    return Path(git(worktree, "rev-parse", "--absolute-git-dir")) / mat.MARKER


DIFF = "diff --git a/a.txt b/a.txt\n--- a/a.txt\n+++ b/a.txt\n@@ -1 +1 @@\n-one\n+two\n"


def agent_input(sha: str) -> dict:
    return {"ci_event": {"parent_commit": {"sha": sha}, "changed_files": [{"unified_diff": DIFF}]}}


# --------------------------------------------------------------------------- #
# Always refused (even with --force), and nothing is lost
# --------------------------------------------------------------------------- #


def test_the_main_clone_is_refused_even_with_force(mat, tmp_path: Path) -> None:
    repo, sha = make_repo(tmp_path / "clone")
    keep = precious(repo)
    for force in (False, True):
        with pytest.raises(mat.WorktreeSafetyError, match="never allowed"):
            mat.prepare_worktree(repo, repo, sha, force=force)
        with pytest.raises(mat.WorktreeSafetyError):
            mat.materialize(repo, repo, agent_input(sha), force=force)
    assert keep.read_text() == "uncommitted work\n"


def test_a_standalone_repository_is_refused_even_with_force(mat, tmp_path: Path) -> None:
    repo, sha = make_repo(tmp_path / "clone")
    other, _ = make_repo(tmp_path / "somebody-elses-checkout")
    keep = precious(other)
    with pytest.raises(mat.WorktreeSafetyError, match="standalone repository"):
        mat.prepare_worktree(repo, other, sha, force=True)
    assert keep.exists()


def test_paths_containing_or_inside_the_clone_or_this_project_are_refused(mat, tmp_path: Path) -> None:
    repo, sha = make_repo(tmp_path / "clone")
    inside_project = mat.PROJECT_ROOT / "scratch-must-not-be-created"
    for dest in (repo / "sub", tmp_path, repo / ".git" / "x", inside_project, mat.PROJECT_ROOT, mat.PROJECT_ROOT.parent):
        with pytest.raises(mat.WorktreeSafetyError, match="never allowed"):
            mat.prepare_worktree(repo, dest, sha, force=True)
    assert not inside_project.exists() and not (repo / "sub").exists()


def test_a_worktree_of_another_repository_is_refused_even_with_force(mat, tmp_path: Path) -> None:
    repo, sha = make_repo(tmp_path / "clone")
    other, other_sha = make_repo(tmp_path / "other")
    foreign = tmp_path / "foreign-worktree"
    git(other, "worktree", "add", "-q", "--detach", str(foreign), other_sha)
    keep = precious(foreign)
    with pytest.raises(mat.WorktreeSafetyError, match="not a worktree of"):
        mat.prepare_worktree(repo, foreign, sha, force=True)
    assert keep.exists()


def test_a_non_empty_plain_directory_is_refused(mat, tmp_path: Path) -> None:
    repo, sha = make_repo(tmp_path / "clone")
    plain = tmp_path / "plain"
    plain.mkdir()
    keep = precious(plain)
    with pytest.raises(mat.WorktreeSafetyError, match="non-empty directory"):
        mat.prepare_worktree(repo, plain, sha, force=True)
    assert keep.exists()


def test_repo_must_be_a_main_clone(mat, tmp_path: Path) -> None:
    repo, sha = make_repo(tmp_path / "clone")
    linked = tmp_path / "linked"
    git(repo, "worktree", "add", "-q", "--detach", str(linked), sha)
    with pytest.raises(mat.WorktreeSafetyError, match="main clone"):
        mat.prepare_worktree(linked, tmp_path / "dest", sha)


# --------------------------------------------------------------------------- #
# Allowed paths
# --------------------------------------------------------------------------- #


def test_a_fresh_worktree_is_created_marked_and_reusable_without_force(mat, tmp_path: Path) -> None:
    repo, sha = make_repo(tmp_path / "clone")
    wt = mat.prepare_worktree(repo, tmp_path / "wt", sha)
    assert (wt / ".git").is_file() and marker_of(mat, wt).exists()  # linked worktree, marked disposable
    assert mat.prepare_worktree(repo, wt, sha) == wt  # no flag needed the second time
    assert not (repo / mat.MARKER).exists()  # the marker never lands in a work tree


def test_an_unmarked_linked_worktree_needs_force_and_is_preserved_when_refused(mat, tmp_path: Path) -> None:
    repo, sha = make_repo(tmp_path / "clone")
    legacy = tmp_path / "legacy-worktree"
    git(repo, "worktree", "add", "-q", "--detach", str(legacy), sha)
    keep = precious(legacy)
    with pytest.raises(mat.WorktreeSafetyError, match="--force"):
        mat.prepare_worktree(repo, legacy, sha)
    assert keep.exists() and not marker_of(mat, legacy).exists()  # refused: nothing changed
    assert mat.prepare_worktree(repo, legacy, sha, force=True) == legacy
    assert marker_of(mat, legacy).exists()
    assert mat.prepare_worktree(repo, legacy, sha) == legacy  # adopted: no flag any more
    assert keep.exists()  # prepare_worktree itself never deletes anything


def test_materialize_builds_the_ci_state_and_only_resets_a_disposable_worktree(mat, tmp_path: Path) -> None:
    repo, sha = make_repo(tmp_path / "clone")
    wt = mat.materialize(repo, tmp_path / "ci", agent_input(sha))
    assert (wt / "a.txt").read_text() == "two\n"
    stray = precious(wt)
    mat.materialize(repo, wt, agent_input(sha))  # disposable (marked): reset is allowed
    assert not stray.exists() and (wt / "a.txt").read_text() == "two\n"


# --------------------------------------------------------------------------- #
# The scripts themselves
# --------------------------------------------------------------------------- #


def test_materialize_cli_refuses_the_clone_with_exit_status_2(mat, tmp_path: Path) -> None:
    repo, sha = make_repo(tmp_path / "clone")
    keep = precious(repo)
    view = tmp_path / "view.jsonl"
    view.write_text(json.dumps({"episode_id": "e1", **agent_input(sha)}) + "\n")
    done = subprocess.run(
        [sys.executable, str(SCRIPTS / "materialize_episode.py"), "--repo", str(repo), "--dataset", str(view),
         "--episode", "e1", "--dest", str(repo), "--force"],
        capture_output=True, text=True, check=False,
    )  # fmt: skip
    assert done.returncode == 2 and "refusing" in done.stderr
    assert keep.read_text() == "uncommitted work\n"


def test_builder_context_uses_the_same_rules(build, mat, tmp_path: Path) -> None:
    repo, _ = make_repo(tmp_path / "clone")
    python = Path(sys.executable)
    keep = precious(repo)
    with pytest.raises(mat.WorktreeSafetyError):
        build.Ctx(repo, python, repo, force=True)
    assert keep.exists()
    ctx = build.Ctx(repo, python, tmp_path / "work")  # fresh: created and marked
    assert marker_of(mat, ctx.work).exists()
    legacy = tmp_path / "legacy"
    git(repo, "worktree", "add", "-q", "--detach", str(legacy), "HEAD")
    with pytest.raises(mat.WorktreeSafetyError):
        build.Ctx(repo, python, legacy)
    assert build.Ctx(repo, python, legacy, force=True).work == legacy.resolve()


def test_builder_main_refuses_before_doing_any_work(build, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo, _ = make_repo(tmp_path / "clone")
    keep = precious(repo)
    out = tmp_path / "out"
    monkeypatch.setattr(
        sys, "argv",
        ["build_dataset.py", "--repo", str(repo), "--python", sys.executable, "--work", str(repo), "--out", str(out)],
    )  # fmt: skip
    assert build.main() == 2
    assert keep.exists() and not out.exists()  # refused before building or writing anything
