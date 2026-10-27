"""Narrow git history operations.  Every call takes ``cutoff_commit`` and must not
return anything newer: this is how future commits are kept out of an episode.

``run_git`` and ``require_worktree_root`` are the minimal helpers the patch
tools use: git runs with an argument vector (never a shell), a cleaned
environment and a timeout.
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Final

from testorcode.errors import ToolError, ToolTimeoutError
from testorcode.schemas.contracts import CommitInfo

GIT_TIMEOUT_SECONDS: Final[int] = 60

# Inherited variables that could point git at another repository, index or config.
_GIT_ENV_BLOCKLIST_PREFIX: Final[str] = "GIT_"
# Repository config could otherwise make git run commands or use an external diff.
_GIT_SAFE_CONFIG: Final[tuple[str, ...]] = (
    "-c",
    "core.fsmonitor=false",
    "-c",
    "core.hooksPath=/dev/null",
)


def _git_env() -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith(_GIT_ENV_BLOCKLIST_PREFIX)}
    env.update(
        GIT_CONFIG_NOSYSTEM="1",
        GIT_TERMINAL_PROMPT="0",
        GIT_OPTIONAL_LOCKS="0",
        LC_ALL="C",
    )
    return env


def run_git(
    worktree_path: Path,
    args: Sequence[str],
    *,
    input_text: str | None = None,
    timeout_seconds: int = GIT_TIMEOUT_SECONDS,
) -> subprocess.CompletedProcess[str]:
    """Run ``git -C worktree_path <args>`` and return the completed process.

    A non-zero exit code is returned, not raised: callers decide what it means.

    Raises:
        ToolError: git is not installed or cannot be started.
        ToolTimeoutError: git did not finish within ``timeout_seconds``.
    """
    command = ["git", *_GIT_SAFE_CONFIG, "-C", str(worktree_path), *args]
    try:
        return subprocess.run(
            command,
            input=input_text,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
            env=_git_env(),
            check=False,
        )
    except FileNotFoundError:
        raise ToolError("git executable not found") from None
    except subprocess.TimeoutExpired:
        raise ToolTimeoutError(f"git {args[0] if args else ''} timed out") from None
    except OSError as exc:
        raise ToolError(f"git could not be started: {type(exc).__name__}") from None


def require_worktree_root(worktree_path: Path) -> Path:
    """Resolve ``worktree_path`` and check it is the top level of a git work tree.

    Patch paths are relative to the work-tree root; running ``git apply`` from a
    subdirectory would silently skip paths outside it.

    Raises:
        ToolError: the path is not a directory or not a git work-tree root.
    """
    root = Path(worktree_path).resolve()
    if not root.is_dir():
        raise ToolError(f"worktree does not exist or is not a directory: {root}")
    result = run_git(root, ["rev-parse", "--show-toplevel"])
    if result.returncode != 0:
        raise ToolError(f"not a git work tree: {root}")
    top = Path(result.stdout.strip()).resolve()
    if top != root:
        raise ToolError(f"worktree path is not the root of its git work tree: {root}")
    return root


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
