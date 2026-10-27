"""Patch validation and application.  Scope rules live in ``EditPolicy``.

Every patch goes through the same checks before it touches the worktree:

1. the diff parses (strict unified/git format, ``a/`` and ``b/`` prefixes);
2. no unsupported operation (binary, symlink, submodule, mode change);
3. every old and new path is a plain relative path inside the worktree that
   does not pass through ``..``, ``.git`` or an existing symlink;
4. no path is a protected execution-controlling file in any directory
   (``is_protected_path``: pytest/tox/setup/project config, conftest, CI,
   dependency files, ``.env``, compiled/generated Python files, backups), on
   top of the episode's ``EditPolicy``;
5. every path is editable under ``scope`` according to ``EditPolicy.classify_path``,
   and one patch never mixes source and test files;
6. ``git apply --check`` accepts it;
7. the content checks in ``patch_content`` pass: a test-only patch may change
   nothing but expected literal values in existing ``assert <expr> ==/!= <literal>``
   statements (AST comparison); source files must parse and may not add
   pytest-tampering constructs.  A passing patch is still not *verified*:
   see ``tools.verification``.

Only ``apply_patch`` and ``revert_patch`` write, through ``git apply`` in the
worktree.  ``git apply`` is all-or-nothing; the touched files are also
snapshotted and restored if anything goes wrong.  The index is never touched.

Test-weakening checks (assertion removal, skip/xfail, mutation) are the
Integrity Guard's job and are not done here.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import posixpath
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from testorcode.errors import PatchError, ScopeViolationError, ToolError
from testorcode.logging_config import get_logger
from testorcode.schemas.contracts import PatchValidation
from testorcode.schemas.episode_schemas import DecisionLabel, EditPolicy, PatchScope, PathClass
from testorcode.tools.git_tools import require_worktree_root, run_git
from testorcode.tools.patch_content import content_problems, forward_states

logger = get_logger(__name__)

MAX_PATCH_BYTES: Final[int] = 1_000_000
MAX_REASON_CHARS: Final[int] = 500
_ALLOWED_FILE_MODES: Final[frozenset[str]] = frozenset({"100644"})
_HUNK_HEADER: Final[re.Pattern[str]] = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
# Only unambiguous names (no spaces); git itself writes the ---/+++ names.
_GIT_HEADER: Final[re.Pattern[str]] = re.compile(r"^diff --git a/(\S+) b/(\S+)$")
_PATH_HEADERS: Final[tuple[str, ...]] = ("rename from ", "rename to ", "copy from ", "copy to ")
_MODE_HEADERS: Final[tuple[str, ...]] = (
    "old mode ",
    "new mode ",
    "new file mode ",
    "deleted file mode ",
)
# Files that control how tests are collected and run, or what gets installed.
# Matched on the file name in *any* directory (``tests/pytest.ini`` configures
# pytest just like a root one); ``EditPolicy.forbidden_globs`` covers only
# the paths it lists, and an episode may supply its own globs.
PROTECTED_FILE_NAMES: Final[tuple[str, ...]] = (
    "conftest.py",
    "pytest.ini",
    ".pytest.ini",
    "pytest.toml",
    ".pytest.toml",
    "tox.ini",
    "setup.cfg",
    "setup.py",
    "pyproject.toml",
    "noxfile.py",
    "sitecustomize.py",
    "usercustomize.py",
    ".coveragerc",
    "requirements*.txt",
    "requirements*.in",
    "constraints*.txt",
    "pipfile",
    "*.lock",
    "manifest.in",
    "makefile",
    "dockerfile*",
    ".pre-commit-config.yaml",
    ".gitlab-ci.yml",
    ".travis.yml",
    "azure-pipelines.yml",
    "jenkinsfile",
    ".gitattributes",
    ".gitmodules",
    ".pdbrc",
    ".env",
    ".env.*",
    ".envrc",
    ".python-version",
    "runtime.txt",
    "environment.yml",
    "environment.yaml",
    # Generated or compiled Python artefacts (can be imported instead of the source).
    "*.pyc",
    "*.pyo",
    "*.pyd",
    "*.so",
    "*.pth",
    "*.egg-link",
    # Editor/merge backups.
    "*.bak",
    "*.orig",
    "*.rej",
    "*.swp",
    "*.swo",
    "*~",
)
PROTECTED_DIR_NAMES: Final[tuple[str, ...]] = (
    ".github",
    ".circleci",
    ".gitlab",
    ".tox",
    ".nox",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    "*.egg-info",
)

_SCOPE_CLASS: Final[dict[PatchScope, PathClass]] = {
    PatchScope.SRC: PathClass.SRC,
    PatchScope.TEST: PathClass.TEST,
}


def scope_for_decision(decision: DecisionLabel) -> PatchScope:
    """Patch scope a decision permits: FIX_CODE -> SRC, UPDATE_TEST -> TEST.

    Raises:
        ScopeViolationError: ESCALATE (an escalation never applies a patch).
    """
    if decision is DecisionLabel.FIX_CODE:
        return PatchScope.SRC
    if decision is DecisionLabel.UPDATE_TEST:
        return PatchScope.TEST
    raise ScopeViolationError("ESCALATE does not permit applying a patch")


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #


@dataclass
class _ParsedDiff:
    paths: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)

    def add_path(self, raw: str) -> None:
        if raw not in self.paths:
            self.paths.append(raw)


def _header_path(value: str, prefix: str, parsed: _ParsedDiff) -> None:
    """Record the path from a ``---``/``+++`` header (``/dev/null`` is not a path)."""
    value = value.split("\t", 1)[0].rstrip("\r")
    if value == "/dev/null":
        return
    if value.startswith('"'):
        parsed.problems.append("quoted (escaped) paths are not supported")
        return
    if not value.startswith(prefix):
        parsed.problems.append(f"path header without the {prefix!r} prefix: {value[:200]!r}")
        return
    parsed.add_path(value[len(prefix) :])


def _parse_diff(diff: str) -> _ParsedDiff:
    """Collect every old/new path and reject unsupported or malformed content.

    Hunk bodies are consumed by their line counts, so a removed line that starts
    with ``--`` is never mistaken for a file header.
    """
    parsed = _ParsedDiff()
    lines = diff.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    hunks = 0
    i = 0
    while i < len(lines):
        line = lines[i].rstrip("\r")
        i += 1
        if line.startswith("@@"):
            match = _HUNK_HEADER.match(line)
            if not match:
                parsed.problems.append(f"malformed hunk header at line {i}")
                return parsed
            old = int(match.group(2)) if match.group(2) is not None else 1
            new = int(match.group(4)) if match.group(4) is not None else 1
            hunks += 1
            while old > 0 or new > 0:
                if i >= len(lines):
                    parsed.problems.append("hunk is truncated (fewer lines than its header says)")
                    return parsed
                body = lines[i].rstrip("\r")
                i += 1
                tag = body[:1]
                if tag == " " or body == "":
                    old, new = old - 1, new - 1
                elif tag == "-":
                    old -= 1
                elif tag == "+":
                    new -= 1
                elif tag == "\\":
                    continue
                else:
                    parsed.problems.append(f"unexpected line inside hunk at line {i}")
                    return parsed
                if old < 0 or new < 0:
                    parsed.problems.append(f"hunk line counts do not match its header at line {i}")
                    return parsed
            while i < len(lines) and lines[i].startswith("\\"):
                i += 1
        elif match := _GIT_HEADER.match(line):
            parsed.add_path(match.group(1))
            parsed.add_path(match.group(2))
        elif line.startswith("--- "):
            _header_path(line[4:], "a/", parsed)
        elif line.startswith("+++ "):
            _header_path(line[4:], "b/", parsed)
        elif line.startswith(_PATH_HEADERS):
            value = line.split(" ", 2)[2]
            if value.startswith('"'):
                parsed.problems.append("quoted (escaped) paths are not supported")
            else:
                parsed.add_path(value)
        elif line.startswith(_MODE_HEADERS):
            mode = line.rsplit(" ", 1)[-1]
            if mode not in _ALLOWED_FILE_MODES:
                parsed.problems.append(
                    f"unsupported file mode {mode} (symlinks, submodules, executables "
                    "and mode changes are not allowed)"
                )
            elif line.startswith(("old mode ", "new mode ")):
                parsed.problems.append("mode changes are not allowed")
        elif line.startswith(("GIT binary patch", "Binary files ")):
            parsed.problems.append("binary patches are not allowed")
    if hunks == 0 and not parsed.paths and not parsed.problems:
        parsed.problems.append("no file changes found in the diff")
    return parsed


def _numstat_paths(root: Path, diff: str) -> tuple[list[str], str | None]:
    """Paths ``git apply`` itself reads from the diff, or git's parse error."""
    result = run_git(root, ["apply", "--numstat", "-z", "-"], input_text=diff)
    if result.returncode != 0:
        return [], _short(result.stderr) or "git could not parse the diff"
    paths = []
    for entry in result.stdout.split("\0"):
        # "added<TAB>deleted<TAB>path"; renames may put the path in the next entry.
        path = entry.split("\t", 2)[2] if entry.count("\t") >= 2 else entry
        if path:
            paths.append(path)
    return paths, None


# --------------------------------------------------------------------------- #
# Path checks
# --------------------------------------------------------------------------- #


def _path_problem(root: Path, path: str) -> str | None:
    """Why ``path`` is not a safe in-worktree relative path, or None if it is."""
    if not path or any(ch in path for ch in "\0\n\r"):
        return f"invalid path {path[:200]!r}"
    if "\\" in path:
        return f"backslashes are not allowed in paths: {path[:200]!r}"
    if path.startswith("/") or re.match(r"^[A-Za-z]:", path):
        return f"absolute path not allowed: {path[:200]!r}"
    parts = path.split("/")
    if any(part in ("", ".", "..") for part in parts) or posixpath.normpath(path) != path:
        return f"path traversal or non-normalised path not allowed: {path[:200]!r}"
    if any(part.lower() == ".git" for part in parts):
        return f"paths inside .git are not allowed: {path[:200]!r}"
    current = root
    for part in parts:
        current = current / part
        if current.is_symlink():
            return f"path goes through a symlink: {path[:200]!r}"
    if not current.resolve().is_relative_to(root):
        return f"path escapes the worktree: {path[:200]!r}"
    return None


def is_protected_path(path: str) -> bool:
    """True for execution-controlling files and CI directories, at any depth.

    Case-insensitive.  Applied in addition to ``EditPolicy`` (it only ever
    rejects more), so the Integrity Guard can reuse it.
    """
    parts = path.replace("\\", "/").lower().split("/")
    if any(fnmatch.fnmatchcase(part, p) for part in parts[:-1] for p in PROTECTED_DIR_NAMES):
        return True
    return any(fnmatch.fnmatchcase(parts[-1], pattern) for pattern in PROTECTED_FILE_NAMES)


def _scope_problem(policy: EditPolicy, path: str, scope: PatchScope | None) -> str | None:
    """Why ``path`` may not be edited, or None.

    ``scope=None`` (a revert without a scope) still requires a source or test
    path: forbidden, protected and outside paths are never allowed.
    """
    cls = PathClass.FORBIDDEN if is_protected_path(path) else policy.classify_path(path)
    if scope is None and cls in (PathClass.SRC, PathClass.TEST):
        return None
    if scope is not None and cls is _SCOPE_CLASS[scope]:
        return None
    if cls is PathClass.FORBIDDEN:
        return f"forbidden file: {path}"
    if cls is PathClass.OUTSIDE:
        return f"outside the approved source/test locations: {path}"
    side = "source" if cls is PathClass.SRC else "test"
    allowed = "source" if scope is PatchScope.SRC else "test"
    return f"{side} file not editable under a {allowed}-only patch: {path}"


# --------------------------------------------------------------------------- #
# Inspection shared by validate / apply / revert
# --------------------------------------------------------------------------- #


@dataclass
class _Inspection:
    paths: list[str] = field(default_factory=list)
    format_problems: list[str] = field(default_factory=list)
    scope_problems: list[str] = field(default_factory=list)
    apply_problem: str | None = None

    @property
    def valid(self) -> bool:
        return not self.format_problems and self.apply_problem is None

    @property
    def allowed(self) -> bool:
        return bool(self.paths) and not self.scope_problems

    def reasons(self) -> tuple[str, ...]:
        out = [*self.scope_problems, *self.format_problems]
        if self.apply_problem:
            out.append(self.apply_problem)
        if not self.paths and not out:
            out.append("the diff names no files")
        return tuple(out)

    def to_validation(self) -> PatchValidation:
        return PatchValidation(
            valid=self.valid,
            allowed=self.allowed,
            changed_paths=tuple(self.paths),
            rejection_reasons=self.reasons(),
        )


def _coerce_scope(scope: PatchScope | str) -> PatchScope:
    try:
        return PatchScope(scope)
    except ValueError:
        raise ScopeViolationError(
            f"unknown patch scope {scope!r}; ESCALATE never applies a patch"
        ) from None


def _inspect(
    root: Path,
    unified_diff: str,
    *,
    policy: EditPolicy | None,
    scope: PatchScope | None,
    reverse: bool = False,
) -> _Inspection:
    """Run every static check.  Never writes to the worktree."""
    result = _Inspection()
    if not isinstance(unified_diff, str) or not unified_diff.strip():
        result.format_problems.append("empty patch")
        return result
    if len(unified_diff.encode("utf-8", errors="replace")) > MAX_PATCH_BYTES:
        result.format_problems.append(f"patch is larger than {MAX_PATCH_BYTES} bytes")
        return result
    if "\0" in unified_diff:
        result.format_problems.append("patch contains NUL bytes")
        return result

    parsed = _parse_diff(unified_diff)
    result.format_problems.extend(parsed.problems)
    git_paths, git_error = _numstat_paths(root, unified_diff)
    if git_error:
        result.format_problems.append(f"malformed diff: {git_error}")
    for path in [*parsed.paths, *git_paths]:
        if path not in result.paths:
            result.paths.append(path)

    policy = policy if policy is not None else EditPolicy()
    for path in result.paths:
        problem = _path_problem(root, path) or _scope_problem(policy, path, scope)
        if problem:
            result.scope_problems.append(problem)
    if scope is None and not result.scope_problems:
        classes = {policy.classify_path(path) for path in result.paths}
        if len(classes) > 1:
            result.scope_problems.append(
                "patch touches both source and test files; a single repair direction "
                "(scope) is required"
            )

    if not result.format_problems:
        args = ["apply", "--check", "--whitespace=nowarn"]
        if reverse:
            args.append("-R")
        check = run_git(root, [*args, "-"], input_text=unified_diff)
        if check.returncode != 0:
            what = "reverted" if reverse else "applied"
            result.apply_problem = f"patch cannot be {what} cleanly: {_short(check.stderr)}"

    # Content checks need safe paths and a patch that applies; they judge the
    # patch in its forward direction, also when it is being reverted.
    if result.paths and not result.scope_problems and result.valid:
        try:
            before, after = forward_states(root, unified_diff, result.paths, reverse=reverse)
        except ToolError as exc:  # fail closed: unchecked content is not allowed
            result.scope_problems.append(f"patch content could not be checked: {exc}")
        else:
            result.scope_problems.extend(content_problems(before, after, policy))
    return result


def _short(text: str) -> str:
    text = " ".join(text.split())
    return text[:MAX_REASON_CHARS] + ("..." if len(text) > MAX_REASON_CHARS else "")


# --------------------------------------------------------------------------- #
# Writing (snapshot + git apply + restore on failure)
# --------------------------------------------------------------------------- #


def _snapshot(root: Path, paths: list[str]) -> dict[str, bytes | None]:
    return {p: (root / p).read_bytes() if (root / p).is_file() else None for p in paths}


def _restore(root: Path, snapshot: dict[str, bytes | None]) -> None:
    for path, content in snapshot.items():
        target = root / path
        if content is None:
            if target.is_file() and not target.is_symlink():
                target.unlink()
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)


def _git_apply(root: Path, unified_diff: str, paths: list[str], *, reverse: bool) -> None:
    """Apply (or reverse) the diff; on any failure put the touched files back."""
    snapshot = _snapshot(root, paths)
    args = ["apply", "--whitespace=nowarn", *(["-R"] if reverse else []), "-"]
    try:
        result = run_git(root, args, input_text=unified_diff)
    except ToolError:
        _restore(root, snapshot)
        raise
    if result.returncode != 0:
        _restore(root, snapshot)
        what = "revert" if reverse else "apply"
        raise PatchError(f"git apply failed to {what} the patch: {_short(result.stderr)}")


# --------------------------------------------------------------------------- #
# Applied-patch records (inside .git, which no patch can touch)
# --------------------------------------------------------------------------- #

_RECORD_DIR: Final[str] = "testorcode-applied"


def _record_path(root: Path, unified_diff: str) -> Path:
    result = run_git(root, ["rev-parse", "--absolute-git-dir"])
    if result.returncode != 0:
        raise ToolError("cannot locate the git directory of the worktree")
    digest = hashlib.sha256(unified_diff.encode("utf-8", errors="surrogateescape")).hexdigest()
    return Path(result.stdout.strip()) / _RECORD_DIR / f"{digest}.json"


def _write_record(root: Path, unified_diff: str, scope: PatchScope, paths: list[str]) -> None:
    path = _record_path(root, unified_diff)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"scope": scope.value, "paths": paths}))


def _read_record(root: Path, unified_diff: str) -> PatchScope | None:
    """Scope the patch was applied with by ``apply_patch``, or None if it never was."""
    path = _record_path(root, unified_diff)
    try:
        return PatchScope(json.loads(path.read_text())["scope"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def _with_final_newline(unified_diff: str) -> str:
    """Contracts strip edge whitespace (``PatchProposal.unified_diff`` loses its final
    newline), and ``git apply`` calls such a diff corrupt.  Non-strings pass through
    so ``_inspect`` reports them."""
    if isinstance(unified_diff, str) and unified_diff and not unified_diff.endswith("\n"):
        return unified_diff + "\n"
    return unified_diff


def validate_patch(
    worktree_path: Path, unified_diff: str, *, policy: EditPolicy, scope: PatchScope
) -> PatchValidation:
    """Static check: diff is well-formed, applies cleanly, touches only allowed paths.

    Never modifies the worktree.  A bad patch is reported in the result, not raised.

    Raises:
        ScopeViolationError: ``scope`` is not a ``PatchScope`` (e.g. an ESCALATE).
        ToolError: the worktree is not a git work-tree root, or git cannot run.
    """
    scope = _coerce_scope(scope)
    root = require_worktree_root(worktree_path)
    unified_diff = _with_final_newline(unified_diff)
    validation = _inspect(root, unified_diff, policy=policy, scope=scope).to_validation()
    logger.info(
        "validate_patch: %d paths, valid=%s, allowed=%s, %d reasons",
        len(validation.changed_paths),
        validation.valid,
        validation.allowed,
        len(validation.rejection_reasons),
    )
    return validation


def apply_patch(
    worktree_path: Path, unified_diff: str, *, policy: EditPolicy, scope: PatchScope
) -> PatchValidation:
    """Apply a validated patch to the episode worktree (never the user's checkout).

    The patch is re-validated here, immediately before writing.  Only the
    working-tree files named in the diff change; the index is untouched.

    Raises:
        ScopeViolationError: a path is outside ``policy`` for ``scope``, or unsafe.
        PatchError: the patch is malformed or does not apply.
        ToolError: the worktree is not a git work-tree root, or git cannot run.
    """
    scope = _coerce_scope(scope)
    root = require_worktree_root(worktree_path)
    unified_diff = _with_final_newline(unified_diff)
    inspection = _inspect(root, unified_diff, policy=policy, scope=scope)
    if inspection.scope_problems:
        raise ScopeViolationError("patch rejected: " + "; ".join(inspection.reasons()))
    if not inspection.valid or not inspection.paths:
        raise PatchError("patch rejected: " + "; ".join(inspection.reasons()))
    _git_apply(root, unified_diff, inspection.paths, reverse=False)
    try:
        _write_record(root, unified_diff, scope, inspection.paths)
    except OSError:
        _git_apply(root, unified_diff, inspection.paths, reverse=True)
        raise ToolError("could not record the applied patch; it was undone") from None
    logger.info("apply_patch: applied %d paths (scope=%s)", len(inspection.paths), scope)
    return inspection.to_validation()


def revert_patch(
    worktree_path: Path,
    unified_diff: str,
    *,
    policy: EditPolicy | None = None,
    scope: PatchScope | None = None,
) -> None:
    """Undo a previously applied patch with ``git apply -R``.

    Only the hunks of this patch are reversed: other changes in the worktree,
    including other edits to the same files, are kept.  Nothing is reset.
    Only a patch that ``apply_patch`` applied in this worktree (the exact same
    diff text) can be reverted: ``apply_patch`` records it, with its scope, in
    the git directory, and the record is removed after a successful revert.
    The repair direction is therefore the recorded one; a ``scope`` that
    disagrees with it is rejected.  Path safety, protected files, the
    single-direction rule and the content checks are all enforced again.
    Without ``policy`` the default ``EditPolicy`` is used.

    Raises:
        ScopeViolationError: a path is unsafe, forbidden or outside the scope,
            the patch mixes source and test files, or ``scope`` disagrees with
            the recorded scope.
        PatchError: the patch was not applied by ``apply_patch`` here, or it
            cannot be reversed cleanly (e.g. the files changed since).
        ToolError: the worktree is not a git work-tree root, or git cannot run.
    """
    if scope is not None:
        scope = _coerce_scope(scope)
    root = require_worktree_root(worktree_path)
    unified_diff = _with_final_newline(unified_diff)
    recorded = _read_record(root, unified_diff) if isinstance(unified_diff, str) else None
    if scope is not None and recorded is not None and scope is not recorded:
        raise ScopeViolationError(
            f"revert rejected: patch was applied with scope {recorded}, not {scope}"
        )
    effective = scope or recorded
    inspection = _inspect(root, unified_diff, policy=policy, scope=effective, reverse=True)
    if inspection.scope_problems:
        raise ScopeViolationError("revert rejected: " + "; ".join(inspection.reasons()))
    if not inspection.valid or not inspection.paths:
        raise PatchError("revert rejected: " + "; ".join(inspection.reasons()))
    if recorded is None:
        raise PatchError(
            "revert rejected: this patch was not applied by apply_patch in this worktree"
        )
    _git_apply(root, unified_diff, inspection.paths, reverse=True)
    _record_path(root, unified_diff).unlink(missing_ok=True)
    logger.info("revert_patch: reverted %d paths (scope=%s)", len(inspection.paths), recorded)
