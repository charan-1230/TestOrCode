"""Executable probes for scoped memory: derive them from tests, run them against the current code.

A memory record counts as evidence only while its ``ExecutableProbe`` passes on the current code (proposal section 7).
The probe is pytest code held by the harness (the memory store), never read from the episode worktree:

* ``probe_from_tests`` builds a *self-contained* probe from a test module's text: the module's support code (imports,
  helper classes and functions, constants) plus the selected test functions.  The memory store calls it when a record
  is written (from the approved, Guard-verified test file) or seeded (from the episode's pristine CI state, before any
  agent acts).  Afterwards the probe never depends on a file an agent can edit.
* ``run_executable_probe`` copies only the repository's **source roots** of the current worktree into a scratch
  directory, writes the probe next to them as the only test module (no repository ``conftest.py``, no config), and runs
  it with ``pytest_runner.run_pytest`` in the sandbox (Docker / bubblewrap; ``trusted-host`` only when asked).  The
  source roots are put on ``sys.path`` with the same layout rule as ``verification.import_base`` (``src/`` modules or a
  package root such as ``boltons/``).

Outcome vocabulary is ``ProbeStatus``: ``PASSED`` (every probe test passed), ``FAILED`` (a probe test failed: the
behaviour the record approved no longer holds), ``ERROR`` (malformed probe, collection/import error, timeout, nothing
executed: *not verified*, never counted as a pass or as a failure of the behaviour).
"""

from __future__ import annotations

import ast
import shutil
import tempfile
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Final

from testorcode.errors import ToolError
from testorcode.logging_config import get_logger
from testorcode.schemas.episode_schemas import EditPolicy, ExecutableProbe
from testorcode.schemas.result_schemas import ProbeStatus, TestRunStatus
from testorcode.tools.pytest_runner import run_pytest
from testorcode.tools.sandbox import Isolation, resolve_isolation
from testorcode.tools.verification import TrustedVerification

logger = get_logger(__name__)

PROBE_FILE: Final[str] = "test_testorcode_probe.py"
_SKIPPED_NAMES: Final[tuple[str, ...]] = ("__pycache__", ".pytest_cache", ".git", "conftest.py")
_SKIPPED_SUFFIXES: Final[tuple[str, ...]] = (".pyc", ".pyo", ".so", ".pyd", ".pth")
_BROKEN_PROBE_ERRORS: Final[tuple[str, ...]] = ("NameError", "ImportError", "ModuleNotFoundError")


# --------------------------------------------------------------------------- #
# Deriving a self-contained probe
# --------------------------------------------------------------------------- #


def qualname_of_test(test_id: str) -> str:
    """``tests/x.py::TestC::test_m[p]`` -> ``TestC.test_m`` (``test_f`` for a module-level test)."""
    return ".".join(test_id.split("::")[1:]).split("[", 1)[0]


def probe_from_tests(module_text: str, qualnames: Iterable[str]) -> str:
    """Self-contained probe code: ``module_text``'s support code plus the tests named in ``qualnames``.

    Kept: imports, helper functions/classes, assignments (except ``pytestmark``/``__test__``) and the selected
    ``test*`` functions (or the selected ``test*`` methods of a ``Test*`` class, with its non-test members).  Dropped:
    every other test, module-level skip markers and ``if __name__ == ...`` blocks.

    Raises:
        ValueError: the module does not parse or none of the tests was found.
    """
    tree = ast.parse(module_text)
    wanted = set(qualnames)
    found: set[str] = set()
    body: list[ast.stmt] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test"):
            if node.name in wanted:
                found.add(node.name)
                body.append(node)
        elif isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
            keep = [f"{node.name}.{m.name}" for m in node.body if _is_test_def(m) and f"{node.name}.{m.name}" in wanted]
            if keep:
                found.update(keep)
                node.body = [m for m in node.body if not _is_test_def(m) or f"{node.name}.{m.name}" in wanted]
                body.append(node)
        elif _is_marker_assignment(node) or _is_main_guard(node):
            continue
        else:
            body.append(node)
    if not found:
        raise ValueError(f"none of the tests {sorted(wanted)} is defined in the module")
    return ast.unparse(ast.Module(body=body, type_ignores=[])) + "\n"


def support_code(module_text: str) -> str:
    """``module_text`` without any test, skip marker or main guard: the imports and helpers a probe needs.

    Raises:
        ValueError: the module does not parse.
    """
    tree = ast.parse(module_text)
    body = [
        n for n in tree.body
        if not (_is_test_def(n) or (isinstance(n, ast.ClassDef) and n.name.startswith("Test"))
                or _is_marker_assignment(n) or _is_main_guard(n))
    ]
    return ast.unparse(ast.Module(body=body, type_ignores=[])) + "\n" if body else ""


def probe_problem(probe_code: str) -> str | None:
    """Why ``probe_code`` cannot be a probe (does not parse, defines no test), or None."""
    try:
        tree = ast.parse(probe_code)
    except (SyntaxError, ValueError):
        return "probe code does not parse"
    has_test = any(
        (isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name.startswith("test"))
        or (isinstance(n, ast.ClassDef) and n.name.startswith("Test") and any(_is_test_def(m) for m in n.body))
        for n in tree.body
    )
    return None if has_test else "probe code defines no test function"


def _is_test_def(node: ast.stmt) -> bool:
    return isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test")


def _is_marker_assignment(node: ast.stmt) -> bool:
    targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
    return any(isinstance(t, ast.Name) and t.id in ("pytestmark", "__test__") for t in targets)


def _is_main_guard(node: ast.stmt) -> bool:
    return isinstance(node, ast.If) and "__name__" in ast.dump(node.test)


# --------------------------------------------------------------------------- #
# Running a probe
# --------------------------------------------------------------------------- #


def run_executable_probe(
    worktree_path: Path,
    probe: ExecutableProbe,
    *,
    policy: EditPolicy,
    timeout_seconds: int,
    python_executable: str | Path | None = None,
    isolation: Isolation | str = Isolation.AUTO,
    docker_image: str | None = None,
) -> TrustedVerification:
    """Run the harness-held ``probe`` against the worktree's current source code, sandboxed.

    Only ``policy.allowed_source_roots`` are copied (no tests, conftest or config), so the result cannot depend on a
    file an agent may edit apart from the source under judgement.  Never raises for probe problems: they are
    ``ERROR`` (not verified).

    Raises:
        ToolError: no sandbox is available (fail closed) or the worktree is missing.
    """
    root = Path(worktree_path).resolve()
    if not root.is_dir():
        raise ToolError(f"worktree does not exist or is not a directory: {root}")

    def unverified(status: ProbeStatus, reason: str, mode: str | None = None) -> TrustedVerification:
        return TrustedVerification(status=status, probe_id=probe.probe_id, reasons=(reason,), isolation=mode)

    problem = probe_problem(probe.probe_code)
    if problem:
        return unverified(ProbeStatus.ERROR, problem)
    roots = [r.strip("/") for r in policy.allowed_source_roots if r.strip("/")]
    present = [r for r in roots if (root / r).is_dir() and not (root / r).is_symlink()]
    if not present:
        return unverified(ProbeStatus.ERROR, "no source root of the edit policy exists in the worktree")
    mode_used = resolve_isolation(isolation, docker_image)  # no usable sandbox: ToolError here (fail closed)

    with tempfile.TemporaryDirectory(prefix="testorcode-probe-") as tmp:
        scratch = Path(tmp) / "work"
        scratch.mkdir()
        for rel in present:
            _copy_tree(root / rel, scratch / rel)
        (scratch / PROBE_FILE).write_text(_probe_module(probe.probe_code, _import_dirs(scratch, present)))
        try:
            run = run_pytest(
                scratch, [PROBE_FILE], timeout_seconds=timeout_seconds, python_executable=python_executable,
                isolation=mode_used, docker_image=docker_image,
            )
        except ToolError as exc:  # the sandbox exists but this probe could not be started: not verified
            return unverified(ProbeStatus.ERROR, f"probe could not be run: {exc}", mode_used.value)
    execution, mode = run.execution, run.isolation.value
    logger.info("probe %s: %s (passed=%s failed=%s)", probe.probe_id, execution.status, execution.passed, execution.failed)
    if execution.status is TestRunStatus.PASSED:
        return TrustedVerification(status=ProbeStatus.PASSED, probe_id=probe.probe_id, isolation=mode)
    if execution.status is TestRunStatus.FAILED and all(
        (t.failure_message or "").startswith(_BROKEN_PROBE_ERRORS) for t in execution.failures
    ):
        # The probe could not even reach the behaviour (a missing import or name): a broken probe, not a regression.
        return unverified(ProbeStatus.ERROR, "probe is broken (missing name or import), not a behaviour failure", mode)
    if execution.status is TestRunStatus.FAILED:
        failed = ", ".join(t.split("::", 1)[-1] for t in execution.failing_test_ids) or "a probe test"
        return unverified(ProbeStatus.FAILED, f"probe test failed on the current code: {failed}", mode)
    detail = (execution.error_message or execution.status.value)[:300]
    return unverified(ProbeStatus.ERROR, f"probe did not execute cleanly ({execution.status.value}): {detail}", mode)


def _import_dirs(scratch: Path, roots: Sequence[str]) -> list[str]:
    """Per source root: its parent for a package root (``boltons/``), else the root itself (``src/``)."""
    dirs = []
    for rel in roots:
        base = scratch / rel
        target = base.parent if (base / "__init__.py").is_file() else base
        dirs.append(target.relative_to(scratch).as_posix())
    return list(dict.fromkeys(dirs))


def _probe_module(probe_code: str, import_dirs: Sequence[str]) -> str:
    """The probe as a test module: ``from __future__`` imports first, then the source roots on ``sys.path``."""
    tree = ast.parse(probe_code)
    future = [n for n in tree.body if isinstance(n, ast.ImportFrom) and n.module == "__future__"]
    rest = [n for n in tree.body if n not in future]
    header = (
        "import os as _testorcode_os, sys as _testorcode_sys\n"
        "_testorcode_here = _testorcode_os.path.dirname(_testorcode_os.path.abspath(__file__))\n"
        f"_testorcode_sys.path[:0] = [_testorcode_os.path.normpath(_testorcode_os.path.join(_testorcode_here, d)) "
        f"for d in {list(import_dirs)!r}]\n"
    )
    parts = [ast.unparse(ast.Module(body=future, type_ignores=[]))] if future else []
    parts += [header, ast.unparse(ast.Module(body=rest, type_ignores=[]))]
    return "\n".join(parts) + "\n"


def _copy_tree(src: Path, dst: Path) -> None:
    """Copy regular files only (no symlinks, caches, compiled files or conftest)."""
    dst.mkdir(parents=True, exist_ok=True)
    for entry in src.iterdir():
        if entry.is_symlink() or entry.name in _SKIPPED_NAMES or entry.name.endswith(_SKIPPED_SUFFIXES):
            continue
        if entry.is_dir():
            _copy_tree(entry, dst / entry.name)
        elif entry.is_file():
            shutil.copyfile(entry, dst / entry.name)
