"""Narrow git history operations.  Every call takes ``cutoff_commit`` and must not
return anything newer: this is how future commits are kept out of an episode.

``run_git`` and ``require_worktree_root`` are the minimal helpers the patch
tools use: git runs with an argument vector (never a shell), a cleaned
environment and a timeout.
"""

from __future__ import annotations

import os
import re
import subprocess
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Final

from testorcode.errors import HistoryCutoffError, ToolError, ToolTimeoutError
from testorcode.schemas.contracts import CommitInfo

GIT_TIMEOUT_SECONDS: Final[int] = 60
MAX_LOG_COMMITS: Final[int] = 50
MAX_BLAME_LINES: Final[int] = 200
MAX_MESSAGE_CHARS: Final[int] = 4000
MAX_SHOW_CHARS: Final[int] = 20000
_SHA: Final[re.Pattern[str]] = re.compile(r"[0-9a-fA-F]{7,40}")
_FULL_SHA: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{40}")
# Separators that cannot occur in commit messages git prints (ASCII unit/record separators).
_FIELD_SEP: Final[str] = "\x1f"
_RECORD_SEP: Final[str] = "\x1e"

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


def _resolve_commit(root: Path, sha: str) -> str:
    """Full sha of the commit ``sha`` names.  Only plain hex abbreviations are accepted (no refs, ranges or options).

    Raises:
        ToolError: not a hex sha, or no such commit.
    """
    if not _SHA.fullmatch(sha or ""):
        raise ToolError(f"not a commit sha (7-40 hex characters): {sha!r}")
    result = run_git(root, ["rev-parse", "--verify", "--quiet", "--end-of-options", f"{sha}^{{commit}}"])
    if result.returncode != 0 or not result.stdout.strip():
        raise ToolError(f"unknown commit {sha}")
    return result.stdout.strip()


def _require_visible(root: Path, sha: str, cutoff_commit: str) -> tuple[str, str]:
    """``(full sha, full cutoff)``; the commit must be the cutoff or one of its ancestors.

    Raises:
        HistoryCutoffError: ``sha`` is newer than (or unrelated to) the cutoff.
        ToolError: either sha is unknown.
    """
    cutoff = _resolve_commit(root, cutoff_commit)
    full = _resolve_commit(root, sha)
    check = run_git(root, ["merge-base", "--is-ancestor", full, cutoff])
    if check.returncode == 1:
        raise HistoryCutoffError(f"commit {sha} is not visible at the history cutoff")
    if check.returncode != 0:
        raise ToolError(f"git merge-base failed: {check.stderr.strip()[:200]}")
    return full, cutoff


def _repo_path(path: str) -> str:
    """A plain repository-relative path (it is passed after ``--``, so it can never become an option).

    Raises:
        ToolError: absolute, ``..``, ``.git`` or empty.
    """
    norm = (path or "").replace("\\", "/").strip()
    parts = PurePosixPath(norm).parts
    if not norm or norm.startswith("/") or ".." in parts or ".git" in parts:
        raise ToolError(f"path must be a plain repository-relative path: {path!r}")
    return norm


def _parse_log(stdout: str) -> list[CommitInfo]:
    commits = []
    for record in stdout.split(_RECORD_SEP):
        record = record.strip("\n")
        if not record:
            continue
        sha, date, message = (record.split(_FIELD_SEP, 2) + ["", ""])[:3]
        commits.append(CommitInfo(
            sha=sha.strip(), message=message.strip()[:MAX_MESSAGE_CHARS],
            committed_at=datetime.fromisoformat(date.strip()) if date.strip() else None,
        ))
    return commits


def git_log(
    worktree_path: Path, *, cutoff_commit: str, path: str | None = None, max_commits: int = 20
) -> list[CommitInfo]:
    """Commits reachable from ``cutoff_commit``, newest first, optionally limited to ``path``.

    Only the cutoff's own history is walked (never ``--all``, refs or reflogs), so a commit made after the cutoff
    cannot appear even though the clone's object database holds it.

    Raises:
        ToolError: git fails, a bad path or sha.
    """
    root = require_worktree_root(worktree_path)
    cutoff = _resolve_commit(root, cutoff_commit)
    limit = max(1, min(int(max_commits), MAX_LOG_COMMITS))
    args = ["log", f"--max-count={limit}", "--no-merges", "--no-decorate",
            f"--format=%H{_FIELD_SEP}%cI{_FIELD_SEP}%B{_RECORD_SEP}", cutoff, "--"]
    if path:
        args.append(_repo_path(path))
    result = run_git(root, args)
    if result.returncode != 0:
        raise ToolError(f"git log failed: {result.stderr.strip()[:200]}")
    return _parse_log(result.stdout)


def git_show(worktree_path: Path, sha: str, *, cutoff_commit: str) -> str:
    """Commit message and diff for ``sha`` (bounded to ``MAX_SHOW_CHARS``).

    Raises:
        HistoryCutoffError: ``sha`` is not an ancestor of (or equal to) ``cutoff_commit``.
        ToolError: git fails.
    """
    root = require_worktree_root(worktree_path)
    full, _ = _require_visible(root, sha, cutoff_commit)
    result = run_git(root, ["show", "--no-color", "--no-ext-diff", "--no-textconv", "--format=commit %H%nDate: %cI%n%n%B",
                            "--stat", "--patch", full, "--"])
    if result.returncode != 0:
        raise ToolError(f"git show failed: {result.stderr.strip()[:200]}")
    text = result.stdout
    if len(text) > MAX_SHOW_CHARS:
        text = text[:MAX_SHOW_CHARS] + f"\n... (truncated, {len(result.stdout) - MAX_SHOW_CHARS} more characters)"
    return text


def git_blame(
    worktree_path: Path, path: str, line_start: int, line_end: int, *, cutoff_commit: str
) -> list[CommitInfo]:
    """Commits that last touched the given lines, newest first (one entry per commit).

    When the worktree's HEAD is the cutoff (how episode worktrees are built), the lines are those of the **working file**
    (what ``read_file`` shows): lines that differ from the cutoff (the CI diff) belong to no commit and are left out.
    Otherwise the cutoff's own version of the file is blamed (its line numbers).  Either way only the cutoff's history is
    walked, so nothing newer than the cutoff is named.

    Raises:
        HistoryCutoffError, ToolError.
    """
    root = require_worktree_root(worktree_path)
    rel = _repo_path(path)
    cutoff = _resolve_commit(root, cutoff_commit)
    target = (root / rel).resolve()
    if root not in target.parents or not target.is_file():
        raise ToolError(f"{path} is not a file in the worktree")
    if line_start < 1 or line_end < line_start:
        raise ToolError("line_start must be >= 1 and line_end >= line_start")
    line_end = min(line_end, line_start + MAX_BLAME_LINES - 1)
    head = run_git(root, ["rev-parse", "--verify", "--quiet", "HEAD"]).stdout.strip()
    # ``--contents`` blames the working file against HEAD (older git refuses an explicit revision with it).
    source = ["--contents", str(target)] if head == cutoff else [cutoff]
    result = run_git(root, ["blame", "--porcelain", "-L", f"{line_start},{line_end}", *source, "--", rel])
    if result.returncode != 0:
        raise ToolError(f"git blame failed: {result.stderr.strip()[:200]}")
    seen: dict[str, dict[str, str]] = {}
    current: str | None = None
    for line in result.stdout.splitlines():
        head = line.split(" ", 1)
        if _FULL_SHA.fullmatch(head[0]) and len(line.split()) >= 3:
            current = head[0]
            seen.setdefault(current, {})
        elif current and line.startswith(("summary ", "committer-time ")):
            key, _, value = line.partition(" ")
            seen[current].setdefault(key, value)
    commits = []
    for sha, fields in seen.items():
        if set(sha) == {"0"}:
            continue  # "Not Committed Yet": the line differs from the cutoff
        _require_visible(root, sha, cutoff)  # defence in depth: blame walks only the cutoff's history
        when = fields.get("committer-time")
        commits.append(CommitInfo(sha=sha, message=fields.get("summary", ""),
                                  committed_at=datetime.fromtimestamp(int(when), UTC) if when else None))
    return sorted(commits, key=lambda c: c.committed_at or datetime.min.replace(tzinfo=UTC), reverse=True)
