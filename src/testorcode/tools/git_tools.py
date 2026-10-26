"""Narrow git history operations.  Every call takes ``cutoff_commit`` and must not
return anything newer: this is how future commits are kept out of an episode."""

from __future__ import annotations

from pathlib import Path

from testorcode.schemas.contracts import CommitInfo


def git_log(
    worktree_path: Path, *, cutoff_commit: str, path: str | None = None, max_commits: int = 20
) -> list[CommitInfo]:
    """Commits reachable from ``cutoff_commit``, newest first, optionally limited to ``path``.

    Raises:
        ToolError: git fails.
    """
    raise NotImplementedError("git tools are not implemented yet")


def git_show(worktree_path: Path, sha: str, *, cutoff_commit: str) -> str:
    """Commit message and diff for ``sha``.

    Raises:
        HistoryCutoffError: ``sha`` is not an ancestor of (or equal to) ``cutoff_commit``.
        ToolError: git fails.
    """
    raise NotImplementedError("git tools are not implemented yet")


def git_blame(
    worktree_path: Path, path: str, line_start: int, line_end: int, *, cutoff_commit: str
) -> list[CommitInfo]:
    """Commits that last touched the given lines.

    Raises:
        HistoryCutoffError, ToolError.
    """
    raise NotImplementedError("git tools are not implemented yet")
