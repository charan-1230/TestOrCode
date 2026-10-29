"""Run selected tests and return a structured result, isolated in a sandbox.

pytest runs with an argument vector (never a shell), inside the episode
worktree, under one of the isolation modes in ``tools.sandbox``:

* ``auto`` (default): Docker if an image is given and the daemon is reachable,
  else bubblewrap, else ``ToolError`` - repository code is never run on the
  host unless ``isolation="trusted-host"`` is passed explicitly (trusted
  local fixtures only);
* in both sandboxes: no network, worktree mounted read-only, private ``/tmp``,
  no access to the home directory, host secrets or the Docker socket, and the
  whole sandbox is killed on timeout.

In every mode the child gets a minimal environment (no inherited
``PYTHONPATH``, API keys, ``PYTEST_ADDOPTS``/``PYTEST_PLUGINS``), no cache or
bytecode is written, and ``--confcutdir`` stops conftest loading at the worktree.

Results come from pytest's JUnit XML report plus a small probe plugin
(``_testorcode_probe``) that records how the session ended and whether a
``conftest.py`` failed to import.  The two are cross-checked against each other
and the exit code; any inconsistency is an ``ERROR``, never a pass.

**Limitation:** test code runs in the same process as pytest, so malicious
tests can still forge *both* the report and the probe record consistently.
These checks catch crude tampering (e.g. a forged report followed by
``os._exit``); they do not prove a result honest.  That remains the job of the
independent Integrity Guard (and mutation testing).
"""

from __future__ import annotations

import contextlib
import json
import os
import posixpath
import signal
import subprocess
import sys
import tempfile
import time
import uuid
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from testorcode.errors import ToolError
from testorcode.logging_config import get_logger
from testorcode.schemas.contracts import PytestExecution
from testorcode.schemas.episode_schemas import FailingTest, ToolName
from testorcode.schemas.result_schemas import TestRunStatus, ToolCallStatus
from testorcode.tools.sandbox import (
    Isolation,
    Layout,
    layout_for,
    resolve_isolation,
    stop_container,
    wrap_command,
)
from testorcode.tools.trace import tool_call

logger = get_logger(__name__)

MAX_OUTPUT_CHARS: Final[int] = 20_000
MAX_ERROR_MESSAGE_CHARS: Final[int] = 4_000
MAX_FAILURE_TEXT_CHARS: Final[int] = 4_000
MAX_REPORT_BYTES: Final[int] = 20_000_000
KILL_GRACE_SECONDS: Final[float] = 3.0

# pytest exit codes (pytest.ExitCode).
_OK, _TESTS_FAILED, _INTERRUPTED, _INTERNAL_ERROR, _USAGE_ERROR, _NO_TESTS = 0, 1, 2, 3, 4, 5

# Inherited variables kept for a trusted-host child; everything else (PYTHONPATH,
# PYTEST_*, API keys, tokens, GIT_*) is dropped.
_ENV_ALLOWLIST: Final[tuple[str, ...]] = (
    "PATH",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TZ",
    "USER",
    "LOGNAME",
    "SYSTEMROOT",
)
_SANDBOX_PATH: Final[str] = "/usr/local/bin:/usr/bin:/bin"

PROBE_MODULE: Final[str] = "_testorcode_probe"
PROBE_ENV_VAR: Final[str] = "TESTORCODE_PROBE_OUT"
# Loaded with ``-p``; records facts pytest only exposes through hooks.  It runs
# in the same (untrusted) process as the tests, so its record is a cross-check,
# not proof.
_PROBE_SOURCE: Final[str] = """\
import json, os
import pytest

_OUT = os.environ.get("TESTORCODE_PROBE_OUT")


def _write(key, value):
    if not _OUT:
        return
    try:
        with open(_OUT) as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        data = {}
    data[key] = value
    with open(_OUT, "w") as fh:
        json.dump(data, fh)


@pytest.hookimpl(wrapper=True)
def pytest_load_initial_conftests(early_config, parser, args):
    try:
        return (yield)
    except BaseException as exc:
        _write("conftest_error", {"type": type(exc).__name__, "path": str(getattr(exc, "path", ""))})
        raise


@pytest.hookimpl(wrapper=True)
def pytest_sessionfinish(session, exitstatus):
    try:
        return (yield)
    finally:
        _write("session", {
            "exitstatus": int(session.exitstatus),
            "testscollected": int(session.testscollected),
            "testsfailed": int(session.testsfailed),
        })
"""


@dataclass(frozen=True)
class PytestRun:
    """A ``PytestExecution`` plus the bounded console output.

    ``PytestExecution`` has no stdout/stderr fields, so callers that need the
    raw output (debugging, the Guard) use ``run_pytest`` instead of ``run_tests``.
    """

    execution: PytestExecution
    stdout: str
    stderr: str
    isolation: Isolation


def run_tests(
    worktree_path: Path,
    selectors: Sequence[str],
    *,
    timeout_seconds: int,
    python_executable: str | Path | None = None,
    isolation: Isolation | str = Isolation.AUTO,
    docker_image: str | None = None,
) -> PytestExecution:
    """Run pytest on ``selectors`` inside ``worktree_path``, sandboxed.

    Test failures are returned (``status=FAILED``); a missing selector or
    collection problem is ``ERROR`` / ``COLLECTION_ERROR``; a kill is ``TIMEOUT``.
    None of these raise.  ``python_executable`` defaults to the current
    interpreter (``python`` inside a Docker image).  See the module docstring
    for ``isolation``.

    Raises:
        ToolError: pytest cannot be started at all, or no sandbox is available.
        BudgetExceededError: the episode's tool-call budget is used up.
    """
    logged = [selectors] if isinstance(selectors, str) else [str(x) for x in list(selectors)[:50]]
    with tool_call(ToolName.RUN_TESTS, selectors=logged) as call:
        execution = run_pytest(
            worktree_path,
            selectors,
            timeout_seconds=timeout_seconds,
            python_executable=python_executable,
            isolation=isolation,
            docker_image=docker_image,
        ).execution
        if execution.status is TestRunStatus.TIMEOUT:
            call.fail(ToolCallStatus.TIMEOUT, f"pytest killed after {timeout_seconds}s")
        elif execution.status in (TestRunStatus.ERROR, TestRunStatus.COLLECTION_ERROR):
            invalid = (execution.error_message or "").startswith("invalid test selector")
            call.fail(
                ToolCallStatus.INVALID_ARGUMENTS if invalid else ToolCallStatus.ERROR,
                execution.error_message or execution.status.value,
            )
    return execution


def run_pytest(
    worktree_path: Path,
    selectors: Sequence[str],
    *,
    timeout_seconds: int,
    python_executable: str | Path | None = None,
    isolation: Isolation | str = Isolation.AUTO,
    docker_image: str | None = None,
) -> PytestRun:
    """``run_tests`` that also returns the bounded stdout/stderr and the isolation used."""
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, int)
        or timeout_seconds <= 0
    ):
        raise ValueError("timeout_seconds must be a positive integer")
    if isinstance(selectors, str):
        raise TypeError("selectors must be a sequence of strings, not a single string")
    root = Path(worktree_path).resolve()
    if not root.is_dir():
        raise ToolError(f"worktree does not exist or is not a directory: {root}")
    selectors = tuple(selectors)
    mode = resolve_isolation(isolation, docker_image)

    bad = [problem for s in selectors if (problem := _selector_problem(root, s))]
    if bad:
        execution = PytestExecution(
            selectors=tuple(str(s) for s in selectors),
            command=(),
            status=TestRunStatus.ERROR,
            timeout_seconds=timeout_seconds,
            error_message=_clip(
                "invalid test selector: " + "; ".join(bad), MAX_ERROR_MESSAGE_CHARS
            ),
        )
        logger.info("run_tests: rejected %d selectors before execution", len(bad))
        return PytestRun(execution, "", "", mode)

    if mode is Isolation.DOCKER:
        python = str(python_executable or "python")
    else:
        python = str(python_executable or sys.executable)
        if os.sep in python and not os.path.isabs(python):  # relative: unresolvable from the worktree/sandbox
            python = os.path.abspath(python)  # abspath, not resolve: a venv's python symlink must stay the venv's
        if not Path(python).is_file():
            raise ToolError(f"pytest could not be started: python executable not found: {python}")

    with tempfile.TemporaryDirectory(prefix="testorcode-pytest-") as tmp:
        tmp_dir = Path(tmp)
        outdir, probedir = tmp_dir / "out", tmp_dir / "probe"
        outdir.mkdir()
        probedir.mkdir()
        (probedir / f"{PROBE_MODULE}.py").write_text(_PROBE_SOURCE)
        layout = layout_for(mode, root, outdir, probedir, python)
        inner = build_command(
            layout.python, Path(layout.worktree), selectors, Path(layout.outdir) / "report.xml"
        )
        container = f"testorcode-{uuid.uuid4().hex[:12]}" if mode is Isolation.DOCKER else None
        command, popen_env = wrap_command(
            mode,
            inner,
            worktree=root,
            outdir=outdir,
            probedir=probedir,
            python=python,
            env=_child_env(mode, tmp_dir, layout),
            docker_image=docker_image,
            container_name=container,
        )
        outcome = _execute(command, root, tmp_dir, timeout_seconds, popen_env, container)
        junit = _read_junit(outdir / "report.xml")
        probe = _read_probe(outdir / "probe.json")
    execution = _to_execution(selectors, command, timeout_seconds, outcome, junit, probe)
    logger.info(
        "run_tests[%s]: %d selectors -> %s (exit=%s, %.1fs, passed=%s failed=%s errors=%s)",
        mode,
        len(selectors),
        execution.status,
        execution.exit_code,
        outcome.duration,
        execution.passed,
        execution.failed,
        execution.errors,
    )
    return PytestRun(execution, outcome.stdout, outcome.stderr, mode)


def build_command(python: str, root: Path, selectors: Sequence[str], report: Path) -> list[str]:
    """pytest argument vector, using paths as seen from inside the sandbox."""
    return [
        python,
        "-m",
        "pytest",
        "-p",
        PROBE_MODULE,
        "-p",
        "no:cacheprovider",
        f"--rootdir={root}",
        f"--confcutdir={root}",
        f"--junitxml={report}",
        "-o",
        "junit_family=xunit1",
        "--color=no",
        "-q",
        "--",
        *selectors,
    ]


# --------------------------------------------------------------------------- #
# Selectors and environment
# --------------------------------------------------------------------------- #


def _selector_problem(root: Path, selector: object) -> str | None:
    """Why ``selector`` must not reach pytest, or None.

    Selectors are node ids or paths relative to the worktree.  Options (``-x``),
    argument files (``@args``), absolute paths and paths escaping the worktree are
    rejected: selectors may come from an agent.
    """
    if not isinstance(selector, str) or not selector.strip():
        return f"empty or non-string selector {selector!r}"
    shown = repr(selector[:200])
    if any(ch in selector for ch in "\0\n\r"):
        return f"control characters in selector {shown}"
    if selector != selector.strip():
        return f"leading/trailing whitespace in selector {shown}"
    if selector.startswith(("-", "@")):
        return f"selector looks like an option or argument file: {shown}"
    path = selector.split("::", 1)[0]
    if "\\" in path or path.startswith("/") or (len(path) > 1 and path[1] == ":"):
        return f"selector must be a relative path inside the worktree: {shown}"
    norm = posixpath.normpath(path)
    if norm == ".." or norm.startswith("../"):
        return f"selector escapes the worktree: {shown}"
    if not (root / norm).resolve().is_relative_to(root):
        return f"selector resolves outside the worktree (symlink?): {shown}"
    return None


def _child_env(mode: Isolation, tmp_dir: Path, layout: Layout) -> dict[str, str]:
    """Environment for pytest itself (inside the sandbox when there is one)."""
    if mode is Isolation.TRUSTED_HOST:
        env = {k: os.environ[k] for k in _ENV_ALLOWLIST if k in os.environ}
        env.setdefault("PATH", os.defpath)
        env.update(HOME=str(tmp_dir), TMPDIR=str(tmp_dir))
    elif mode is Isolation.BWRAP:
        env = {
            "PATH": _SANDBOX_PATH,
            "HOME": "/tmp/home",
            "TMPDIR": "/tmp",
            "USER": "sandbox",
            "LANG": "C.UTF-8",
        }
    else:  # docker: keep the image's PATH
        env = {"HOME": "/tmp", "TMPDIR": "/tmp", "USER": "sandbox", "LANG": "C.UTF-8"}
    env.update(
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONNOUSERSITE="1",
        PYTHONHASHSEED="0",
        PYTHONIOENCODING="utf-8",
        NO_COLOR="1",
        PYTHONPATH=layout.probedir,
        **{PROBE_ENV_VAR: f"{layout.outdir}/probe.json"},
    )
    return env


# --------------------------------------------------------------------------- #
# Execution
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class _Outcome:
    exit_code: int | None
    timed_out: bool
    duration: float
    stdout: str
    stderr: str


def _execute(
    command: list[str],
    root: Path,
    tmp_dir: Path,
    timeout_seconds: int,
    env: dict[str, str] | None,
    container: str | None,
) -> _Outcome:
    """Run ``command``; on timeout kill the sandbox (or the whole process group).

    Output goes to host files, not pipes, so nothing left running can block us,
    and the sandbox cannot see or rewrite them.
    """
    out_path, err_path = tmp_dir / "stdout.txt", tmp_dir / "stderr.txt"
    posix = os.name == "posix"
    start = time.monotonic()
    with open(out_path, "wb") as out, open(err_path, "wb") as err:
        try:
            proc = subprocess.Popen(
                command,
                cwd=root,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=out,
                stderr=err,
                start_new_session=posix,
            )
        except OSError as exc:
            raise ToolError(
                f"pytest could not be started: {type(exc).__name__}: {exc.strerror}"
            ) from None
        timed_out = False
        try:
            exit_code: int | None = proc.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            exit_code = None
        finally:
            if container is not None and proc.poll() is None:
                stop_container(container)  # killing the docker CLI alone leaves it running
            # bwrap: killing the launcher kills its pid namespace (--die-with-parent).
            # Normal exit: remove stray background processes.  Interrupted
            # (e.g. KeyboardInterrupt): never leak the run.
            _kill_group(proc, posix, graceful=proc.poll() is None)
    duration = time.monotonic() - start
    return _Outcome(
        exit_code, timed_out, duration, _read_bounded(out_path), _read_bounded(err_path)
    )


def _kill_group(proc: subprocess.Popen[bytes], posix: bool, *, graceful: bool = True) -> None:
    """Stop the process and everything it started (SIGTERM, then SIGKILL)."""
    if posix:
        if graceful and proc.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGTERM)
            with contextlib.suppress(subprocess.TimeoutExpired):
                proc.wait(timeout=KILL_GRACE_SECONDS)
        # Also reaches grandchildren left behind after the leader exited.
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(proc.pid, signal.SIGKILL)
    elif proc.poll() is None:
        proc.kill()
    try:
        proc.wait(timeout=KILL_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        logger.warning("pytest process %d did not exit after SIGKILL", proc.pid)


def _read_bounded(path: Path, limit: int = MAX_OUTPUT_CHARS) -> str:
    """Head and tail of a possibly huge file, without reading it all."""
    size = path.stat().st_size
    with open(path, "rb") as fh:
        if size <= limit:
            return fh.read().decode("utf-8", errors="replace")
        half = limit // 2
        head = fh.read(half)
        fh.seek(size - half)
        tail = fh.read(half)
    dropped = size - 2 * half
    return (
        head.decode("utf-8", errors="replace")
        + f"\n... [{dropped} bytes truncated] ...\n"
        + tail.decode("utf-8", errors="replace")
    )


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 15] + "... [truncated]"


def _tail(text: str, limit: int) -> str:
    return text if len(text) <= limit else "[truncated] ..." + text[-(limit - 15) :]


# --------------------------------------------------------------------------- #
# JUnit report -> counts and failures
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class _Junit:
    tests: int
    failed: int
    errors: int
    skipped: int
    failures: tuple[FailingTest, ...]
    collection_errors: tuple[str, ...]
    files: tuple[str, ...]  # "file" attribute of every test case (may repeat)
    suite_failures: int
    suite_skipped: int

    @property
    def passed(self) -> int:
        return max(self.tests - self.failed - self.errors - self.skipped, 0)


def _read_junit(report: Path) -> _Junit | None:
    """Parse pytest's JUnit XML, or None when it is missing or unusable."""
    if not report.is_file() or report.stat().st_size > MAX_REPORT_BYTES:
        return None
    try:
        tree = ET.parse(report)
    except ET.ParseError:
        return None
    tests = failed = errors = skipped = 0
    failures: list[FailingTest] = []
    collection: list[str] = []
    files: list[str] = []
    suite_failures = suite_skipped = 0
    for suite in tree.iter("testsuite"):
        try:
            suite_failures += int(suite.get("failures") or 0)
            suite_skipped += int(suite.get("skipped") or 0)
        except ValueError:
            return None
    for case in tree.iter("testcase"):
        if case.get("file"):
            files.append(case.get("file") or "")
        failure, error = case.find("failure"), case.find("error")
        if case.find("skipped") is not None and failure is None and error is None:
            skipped += 1
            tests += 1
            continue
        problem = failure if failure is not None else error
        if problem is None:
            tests += 1
            continue
        test_id, test_file = _node_id(case)
        # pytest's junitxml writes collection failures as an <error> with this
        # exact message on an entry with no classname (it has no test function).
        if (
            not case.get("classname")
            and error is not None
            and (error.get("message") == "collection failure")
        ):
            collection.append(f"{case.get('name') or '?'}: {_tail(error.text or '', 1000)}")
            continue
        tests += 1
        if failure is not None:
            failed += 1
        else:
            errors += 1
        failures.append(
            FailingTest(
                test_id=test_id[:256],
                test_file=test_file,
                test_name=(case.get("name") or "?")[:256],
                failure_message=_clip(problem.get("message") or "", MAX_FAILURE_TEXT_CHARS) or None,
                traceback=_tail(problem.text or "", MAX_FAILURE_TEXT_CHARS) or None,
            )
        )
    return _Junit(
        tests,
        failed,
        errors,
        skipped,
        tuple(failures),
        tuple(collection),
        tuple(files),
        suite_failures,
        suite_skipped,
    )


def _read_probe(path: Path) -> dict[str, object]:
    """The probe plugin's record ({} when missing or unreadable)."""
    try:
        if path.stat().st_size > 100_000:
            return {}
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _inconsistencies(
    code: int | None, junit: _Junit, probe: dict[str, object], selectors: tuple[str, ...]
) -> list[str]:
    """Disagreements between the exit code, the JUnit report and the probe record.

    Applied to runs that claim to have completed (exit 0 or 1).  Each check is
    cheap for honest pytest runs to satisfy and catches crude tampering.
    """
    problems = []
    bad = junit.failed + junit.errors
    if code == _OK and bad:
        problems.append(f"exit code 0 but the report lists {bad} failing tests")
    if code == _TESTS_FAILED and not bad:
        problems.append("exit code 1 but the report lists no failing test")
    if junit.suite_failures != junit.failed or junit.suite_skipped != junit.skipped:
        problems.append("report totals do not match its test cases")
    session = probe.get("session")
    if not isinstance(session, dict):
        problems.append("pytest did not finish its session normally (no session record)")
    else:
        if session.get("exitstatus") != code:
            problems.append(f"session exit status {session.get('exitstatus')} != exit code {code}")
        collected = session.get("testscollected")
        if isinstance(collected, int) and junit.tests > collected:
            problems.append(f"report has {junit.tests} tests but only {collected} were collected")
        failed = session.get("testsfailed")
        if isinstance(failed, int) and (failed > 0) != (bad > 0):
            problems.append("session failure count disagrees with the report")
    roots = [posixpath.normpath(s.split("::", 1)[0]) for s in selectors]
    if roots and "." not in roots:
        outside = sorted(
            {f for f in junit.files if not any(f == r or f.startswith(r + "/") for r in roots)}
        )
        if outside:
            problems.append(f"report lists tests outside the selection: {outside[:5]}")
    return problems


def _node_id(case: ET.Element) -> tuple[str, str]:
    """Rebuild the pytest node id from an xunit1 ``testcase`` element."""
    name = case.get("name") or "?"
    classname = case.get("classname") or ""
    test_file = case.get("file") or ""
    if test_file.endswith(".py"):
        module = test_file[:-3].replace("/", ".")
        rest = classname[len(module) :].lstrip(".") if classname.startswith(module) else ""
        parts = [test_file, *([p for p in rest.split(".") if p]), name]
        return "::".join(parts), test_file
    return f"{classname}::{name}" if classname else name, test_file or classname or "?"


# --------------------------------------------------------------------------- #
# Status
# --------------------------------------------------------------------------- #


def _to_execution(
    selectors: tuple[str, ...],
    command: list[str],
    timeout_seconds: int,
    outcome: _Outcome,
    junit: _Junit | None,
    probe: dict[str, object],
) -> PytestExecution:
    common = {
        "selectors": selectors,
        "command": tuple(command),
        "timeout_seconds": timeout_seconds,
        "duration_seconds": round(outcome.duration, 3),
    }
    console = (outcome.stdout + "\n" + outcome.stderr).strip()

    def error(status: TestRunStatus, message: str, with_counts: bool = True) -> PytestExecution:
        return PytestExecution(
            **common,
            status=status,
            exit_code=outcome.exit_code,
            error_message=_clip(message, MAX_ERROR_MESSAGE_CHARS),
            **(_counts(junit) if with_counts else {}),
        )

    if outcome.timed_out:
        return PytestExecution(
            **common,
            status=TestRunStatus.TIMEOUT,
            exit_code=None,
            timed_out=True,
            error_message=f"pytest timed out after {timeout_seconds}s; process group killed",
        )

    code = outcome.exit_code
    if junit is None and "No module named pytest" in outcome.stderr:
        raise ToolError("pytest is not installed in the selected Python environment")

    # Collection errors come only from structured signals (pytest's own
    # collect-error report entries, or the probe seeing an initial conftest fail
    # to import), never from console text a test could print.  They override
    # any exit code, so an inconsistent report is never a pass.
    conftest_error = probe.get("conftest_error")
    if isinstance(conftest_error, dict) and junit is None and code not in (_OK, _TESTS_FAILED):
        return error(
            TestRunStatus.COLLECTION_ERROR,
            f"collection error (exit code {code}): conftest could not be loaded "
            f"({conftest_error.get('type')}: {conftest_error.get('path')}): "
            + _tail(outcome.stderr, MAX_ERROR_MESSAGE_CHARS - 300),
            with_counts=False,
        )
    if junit is not None and junit.collection_errors:
        return error(
            TestRunStatus.COLLECTION_ERROR,
            f"collection error (exit code {code}): " + "; ".join(junit.collection_errors),
            with_counts=False,
        )
    if junit is None:
        return error(
            TestRunStatus.ERROR,
            f"pytest exited with code {code} and wrote no test report: "
            + _tail(console, MAX_ERROR_MESSAGE_CHARS - 80),
            with_counts=False,
        )
    if code in (_OK, _TESTS_FAILED):
        problems = _inconsistencies(code, junit, probe, selectors)
        if problems:
            return error(
                TestRunStatus.ERROR,
                "inconsistent pytest results (possible tampering; not trusted): "
                + "; ".join(problems),
                with_counts=False,
            )
    if code == _OK:
        if junit.passed == 0:
            return error(
                TestRunStatus.ERROR,
                f"no test passed: {junit.tests} selected, {junit.skipped} skipped; "
                "an unexecuted test is not a pass",
            )
        return PytestExecution(
            **common,
            status=TestRunStatus.PASSED,
            exit_code=code,
            **_counts(junit),
        )
    if code == _TESTS_FAILED and junit.failures:
        return PytestExecution(
            **common,
            status=TestRunStatus.FAILED,
            exit_code=code,
            failures=junit.failures,
            **_counts(junit),
        )
    if code == _NO_TESTS:
        message = "no tests were collected for the given selectors: "
    elif code == _USAGE_ERROR:
        message = "pytest usage error (missing file or test, or bad option): "
    elif code == _INTERNAL_ERROR:
        message = "pytest internal error: "
    elif code == _INTERRUPTED:
        message = "pytest run was interrupted: "
    elif code is not None and code < 0:
        message = f"pytest was killed by signal {-code}: "
    else:
        message = f"unexpected pytest exit code {code}: "
    return PytestExecution(
        **common,
        status=TestRunStatus.ERROR,
        exit_code=code,
        failures=junit.failures,
        error_message=_clip(
            message + _tail(console, MAX_ERROR_MESSAGE_CHARS - 80), MAX_ERROR_MESSAGE_CHARS
        ),
        **_counts(junit),
    )


def _counts(junit: _Junit | None) -> dict[str, int | None]:
    if junit is None:
        return {}
    return {
        "passed": junit.passed,
        "failed": junit.failed,
        "errors": junit.errors,
        "skipped": junit.skipped,
    }
