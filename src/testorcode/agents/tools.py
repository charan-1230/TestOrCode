"""Tools a model may call itself during an episode: ``search_repo``, ``read_file``, ``run_tests``.

The orchestrator builds one ``AgentTools`` per episode and hands it to the
agents that use a tool loop (``agents.tool_loop``).  Every call:

* is refused (``NOT_PERMITTED``) unless the tool is in ``context.allowed_tools``;
* has its arguments checked (``INVALID_ARGUMENTS`` on a bad path or selector);
* is recorded as a ``ToolEvent`` and counted against the episode's tool budget
  (``tools.trace``); ``search_repo`` and ``run_tests`` record through the
  retriever wrapper and ``pytest_runner`` respectively.

A refused or invalid call is reported back to the model as text, not raised:
the model can correct itself.  Budget exhaustion and tool infrastructure
failures are raised and end the run as execution failures.

Nothing here writes to the worktree: patches stay proposals that the
orchestrator validates and applies.  ``read_file`` sees the episode worktree only
(no ``.git`` metadata, so no history past the CI commit, and no ``.env`` files).

``code_only=True`` (conditions without any retrieval, i.e. A1) narrows ``read_file`` to code: Python source files
under the policy's source roots and files under its test roots (test modules and fixtures).  Documentation and
change notes (README, CHANGELOG, ``docs/``, ``.md``/``.rst`` files, anything outside those roots) are refused as
``NOT_PERMITTED``, by the requested path and by the symlink target, because they carry the change-intent evidence the
condition withholds.  ``search_repo`` is withheld from those episodes separately (``allowed_tools``).
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Protocol

from testorcode.agents._common import clip, numbered, read_worktree_file
from testorcode.errors import ToolError
from testorcode.schemas.contracts import EpisodeContext
from testorcode.schemas.episode_schemas import EditPolicy, ToolName
from testorcode.schemas.result_schemas import RetrievedEvidence, ToolCallStatus
from testorcode.tools.pytest_runner import run_tests
from testorcode.tools.trace import tool_call

MAX_READ_LINES = 200
MAX_RESULT_CHARS = 6000
TOOL_NAMES = ("search_repo", "read_file", "run_tests")
# code_only: what counts as a documentation/change-note file wherever it lives, and the source-file suffixes.
_DOC_SUFFIXES = frozenset({".md", ".rst", ".adoc"})
_DOC_STEMS = ("readme", "changelog", "changes", "history", "news", "releasenotes", "release_notes", "todo",
              "authors", "contributing")
_SOURCE_SUFFIXES = frozenset({".py", ".pyi"})


class _Searcher(Protocol):
    def search(self, query: str, *, top_k: int) -> list[RetrievedEvidence]: ...


class AgentTools:
    """The three model-callable tools for one episode."""

    def __init__(
        self,
        context: EpisodeContext,
        *,
        retriever: _Searcher,
        timeout_seconds: int,
        python_executable: str | Path | None = None,
        top_k: int = 8,
        code_only: bool = False,
    ) -> None:
        self._context = context
        self._retriever = retriever
        self._timeout = timeout_seconds
        self._python = python_executable
        self._top_k = top_k
        self.code_only = code_only  # read_file limited to source and test files (A1)

    def call(self, name: str, *, query: str = "", path: str = "", start_line: int = 1, end_line: int = 0,
             selectors: list[str] | tuple[str, ...] = ()) -> str:
        """Run tool ``name`` and return its result as bounded text for the model."""
        if name == "search_repo":
            return self.search_repo(query)
        if name == "read_file":
            return self.read_file(path, start_line, end_line)
        if name == "run_tests":
            return self.run_tests(list(selectors))
        raise ValueError(f"unknown tool {name!r}")  # the step models only allow TOOL_NAMES

    # ----------------------------------------------------------------- #

    def search_repo(self, query: str) -> str:
        if not self._allowed(ToolName.SEARCH_REPO, query_chars=len(query)):
            return "REFUSED: search_repo is not permitted in this episode."
        if not query.strip():
            return self._invalid(ToolName.SEARCH_REPO, "search_repo needs a non-empty query")
        hits = self._retriever.search(query, top_k=self._top_k)  # recorded by the retriever wrapper
        if not hits:
            return "No results."
        return clip("\n".join(f"[{h.evidence_id}] ({h.source_type}) {clip(h.snippet, 500)}" for h in hits),
                    MAX_RESULT_CHARS)

    def read_file(self, path: str, start_line: int = 1, end_line: int = 0) -> str:
        if not self._allowed(ToolName.READ_FILE, path=path):
            return "REFUSED: read_file is not permitted in this episode."
        problem = _path_problem(path) or _resolved_problem(self._context.worktree_path, path)
        if problem is None and (start_line < 1 or (end_line and end_line < start_line)):
            problem = "start_line must be >= 1 and end_line >= start_line (0 = to the end)"
        if problem:
            return self._invalid(ToolName.READ_FILE, problem, path=path)
        if self.code_only:
            target = _resolved_relative(self._context.worktree_path, path)
            refusal = _code_only_problem(self._context.edit_policy, path) or (
                _code_only_problem(self._context.edit_policy, target) if target and target != path else None
            )
            if refusal:
                with tool_call(ToolName.READ_FILE, path=path) as call:
                    call.fail(ToolCallStatus.NOT_PERMITTED, refusal)
                return f"REFUSED: {refusal}"
        with tool_call(ToolName.READ_FILE, path=path, start_line=start_line, end_line=end_line) as call:
            try:
                text = read_worktree_file(self._context.worktree_path, path)
            except ToolError as exc:
                call.fail(ToolCallStatus.INVALID_ARGUMENTS, str(exc))
                return f"INVALID: {exc}"
            if text is None:
                call.fail(ToolCallStatus.INVALID_ARGUMENTS, f"{path} does not exist")
                return f"INVALID: {path} does not exist in the worktree."
            lines = text.splitlines()
            last = min(len(lines), end_line or len(lines), start_line + MAX_READ_LINES - 1)
            shown = numbered(lines[start_line - 1 : last], start_line)
            more = f"\n... ({len(lines) - last} more lines)" if last < len(lines) else ""
            return clip(f"{path} lines {start_line}-{last} of {len(lines)}:\n{shown}{more}", MAX_RESULT_CHARS)

    def run_tests(self, selectors: list[str]) -> str:
        if not self._allowed(ToolName.RUN_TESTS, selectors=selectors[:20]):
            return "REFUSED: run_tests is not permitted in this episode."
        roots = self._context.edit_policy.allowed_test_roots
        bad = [s for s in selectors if _path_problem(s.split("::", 1)[0]) or not s.startswith(roots)]
        if not selectors or bad:
            return self._invalid(
                ToolName.RUN_TESTS, f"selectors must be test ids or paths under {', '.join(roots)}: {bad or 'none given'}",
                selectors=selectors[:20],
            )
        result = run_tests(  # sandboxed; records its own ToolEvent
            self._context.worktree_path, selectors, timeout_seconds=self._timeout, python_executable=self._python
        )
        lines = [f"status={result.status.value} passed={result.passed} failed={result.failed} errors={result.errors}"]
        lines += [f"FAILED {f.test_id}: {clip(f.failure_message, 400)}" for f in result.failures[:10]]
        if result.error_message:
            lines.append("error: " + clip(result.error_message, 600))
        return clip("\n".join(lines), MAX_RESULT_CHARS)

    # ----------------------------------------------------------------- #

    def _allowed(self, tool: ToolName, **arguments) -> bool:
        if tool in self._context.allowed_tools:
            return True
        with tool_call(tool, **arguments) as call:
            call.fail(ToolCallStatus.NOT_PERMITTED, f"{tool.value} is not in this episode's allowed_tools")
        return False

    @staticmethod
    def _invalid(tool: ToolName, problem: str, **arguments) -> str:
        with tool_call(tool, **arguments) as call:
            call.fail(ToolCallStatus.INVALID_ARGUMENTS, problem)
        return f"INVALID: {problem}"


def _resolved_problem(worktree: Path, path: str) -> str | None:
    """The same rule applied to where ``path`` really points (a symlink such as ``notes -> .env`` stays inside
    the worktree but must not be readable); escaping the worktree is refused by ``read_worktree_file``."""
    root = worktree.resolve()
    target = (root / path).resolve()
    if root == target or root not in target.parents:
        return None
    return _path_problem(target.relative_to(root).as_posix())


def _resolved_relative(worktree: Path, path: str) -> str | None:
    """Where ``path`` really points, relative to the worktree (follows symlinks); None if it leaves the worktree."""
    root = worktree.resolve()
    target = (root / path).resolve()
    if root != target and root not in target.parents:
        return None
    return target.relative_to(root).as_posix()


def _code_only_problem(policy: EditPolicy, path: str) -> str | None:
    """Why ``path`` is not a code file a ``code_only`` episode may read (None = readable).

    Readable: Python source under ``policy.allowed_source_roots`` and any file under ``policy.allowed_test_roots``
    (test modules and their fixture data), unless it is documentation or a change note.
    """
    norm = path.replace("\\", "/")
    while norm.startswith("./"):
        norm = norm[2:]
    pure = PurePosixPath(norm)
    name = pure.name.lower()
    if pure.suffix.lower() in _DOC_SUFFIXES or name.startswith(_DOC_STEMS):
        return f"{path} is documentation or a change note; this run may read only source and test files"
    if any(norm.startswith(root) for root in policy.allowed_test_roots):
        return None
    if any(norm.startswith(root) for root in policy.allowed_source_roots):
        if pure.suffix.lower() in _SOURCE_SUFFIXES:
            return None
        return f"{path} is not a Python source file; this run may read only source and test files"
    return f"{path} is outside the source and test roots; this run may read only source and test files"


def _path_problem(path: str) -> str | None:
    """Reasons a model-supplied path may not be read (it must stay a plain repository path)."""
    if not path or path.startswith(("/", "\\")) or "\\" in path:
        return "path must be a relative repository path"
    parts = PurePosixPath(path).parts
    if ".." in parts:
        return "path may not contain '..'"
    if any(p == ".git" or p.startswith(".env") for p in parts):
        return "git metadata and .env files cannot be read"
    return None
