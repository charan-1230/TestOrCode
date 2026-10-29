"""D3: a relative interpreter path must reach the sandbox as an absolute one.

Before the fix, ``run_pytest`` checked a relative ``python_executable`` against the host's working directory and then
ran it from the worktree / bind-mounted it into bwrap as a relative path ("bwrap: Can't find source path"), so every
run was an ERROR and mutant baselines were unusable.  Real: pytest, the sandbox (trusted-host, and bwrap when
available), ``verify_behaviour`` and ``run_mutants``; the repository is a fixture.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from testorcode.errors import ToolError
from testorcode.schemas.result_schemas import ProbeStatus, TestRunStatus
from testorcode.tools import pytest_runner, sandbox, verification
from testorcode.tools.mutation import generate_mutants, run_mutants
from testorcode.tools.pytest_runner import run_pytest
from testorcode.tools.verification import ProbeCase, TrustedProbe, verify_behaviour

CORE = "def normalise(s):\n    return s.lower()\n"
TESTS = "from synthlib.core import normalise\n\n\ndef test_normalise():\n    assert normalise('AB') == 'ab'\n"
PROBE = TrustedProbe(probe_id="p", module="synthlib.core", function="normalise",
                     cases=(ProbeCase(args=("AB",), expected="ab"),), provenance="dataset")
ISOLATIONS = ["trusted-host", pytest.param("bwrap", marks=pytest.mark.skipif(
    not sandbox.bwrap_available(), reason="bubblewrap unavailable"))]


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    for rel, text in {"pyproject.toml": '[tool.pytest.ini_options]\npythonpath = ["src"]\n',
                      "src/synthlib/__init__.py": "", "src/synthlib/core.py": CORE,
                      "tests/test_synthlib.py": TESTS}.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)
    return root


@pytest.fixture
def relative_python(tmp_path: Path, monkeypatch) -> str:
    """The harness interpreter as a path relative to a working directory that is neither the repo nor its parent."""
    cwd = tmp_path / "elsewhere" / "deeper"
    cwd.mkdir(parents=True)
    monkeypatch.chdir(cwd)
    rel = os.path.relpath(sys.executable, cwd)
    assert not os.path.isabs(rel) and os.sep in rel and Path(rel).is_file()
    return rel


def spy(monkeypatch, module) -> list[str]:
    """Record the interpreter path each call hands to the sandbox wrapper (and still run it)."""
    seen: list[str] = []
    real = module.wrap_command

    def wrapped(mode, inner, **kwargs):
        seen.append(kwargs["python"])
        return real(mode, inner, **kwargs)

    monkeypatch.setattr(module, "wrap_command", wrapped)
    return seen


@pytest.mark.parametrize("isolation", ISOLATIONS)
def test_relative_interpreter_runs_tests_from_another_working_directory(repo, relative_python, monkeypatch,
                                                                         isolation) -> None:
    seen = spy(monkeypatch, pytest_runner)
    run = run_pytest(repo, ["tests"], timeout_seconds=60, python_executable=relative_python, isolation=isolation)
    assert run.execution.status is TestRunStatus.PASSED, run.execution.error_message
    assert run.execution.passed == 1
    [given] = seen
    assert os.path.isabs(given) and given == os.path.abspath(relative_python)  # absolute, symlink kept (no resolve)
    assert Path(given).is_file()


@pytest.mark.parametrize("isolation", ISOLATIONS)
def test_relative_interpreter_runs_a_trusted_probe(repo, relative_python, monkeypatch, isolation) -> None:
    seen = spy(monkeypatch, verification)
    result = verify_behaviour(repo, PROBE, timeout_seconds=60, python_executable=relative_python,
                              isolation=isolation)
    assert result.status is ProbeStatus.PASSED, result.reasons
    [given] = seen
    assert os.path.isabs(given) and given == os.path.abspath(relative_python)


@pytest.mark.parametrize("isolation", ISOLATIONS)
def test_relative_interpreter_gives_a_usable_mutant_baseline(repo, relative_python, isolation) -> None:
    mutants = generate_mutants(CORE, "src/synthlib/core.py", "normalise", module="synthlib.core")
    run = run_mutants(repo, mutants, ["tests/test_synthlib.py"], timeout_seconds=60,
                      python_executable=relative_python, isolation=isolation)
    assert run.status is TestRunStatus.PASSED, run.reason  # was ERROR: no usable baseline
    assert any(o.outcome == "killed" for o in run.outcomes)


def test_missing_relative_interpreter_is_an_error_naming_the_absolute_path(repo, relative_python) -> None:
    missing = relative_python + "-does-not-exist"
    with pytest.raises(ToolError, match="python executable not found") as caught:
        run_pytest(repo, ["tests"], timeout_seconds=60, python_executable=missing, isolation="trusted-host")
    assert os.path.abspath(missing) in str(caught.value)


@pytest.mark.parametrize("isolation", ISOLATIONS)
def test_an_interpreter_that_fails_is_an_error_not_a_measurement(repo, tmp_path, monkeypatch, isolation) -> None:
    """A real executable that is not a working Python: reported as ERROR (pytest) / ERROR (probe), never PASSED."""
    fake = tmp_path / "elsewhere" / "fake-venv" / "bin" / "python"
    fake.parent.mkdir(parents=True)
    fake.write_text("#!/bin/sh\nexit 3\n")
    fake.chmod(0o755)
    monkeypatch.chdir(tmp_path / "elsewhere")
    rel = os.path.join("fake-venv", "bin", "python")
    run = run_pytest(repo, ["tests"], timeout_seconds=60, python_executable=rel, isolation=isolation)
    assert run.execution.status is TestRunStatus.ERROR and run.execution.passed is None
    probe = verify_behaviour(repo, PROBE, timeout_seconds=60, python_executable=rel, isolation=isolation)
    assert probe.status is ProbeStatus.ERROR and not probe.verified
    baseline = run_mutants(repo, generate_mutants(CORE, "src/synthlib/core.py", "normalise", module="synthlib.core"),
                           ["tests/test_synthlib.py"], timeout_seconds=60, python_executable=rel,
                           isolation=isolation)
    assert baseline.status is TestRunStatus.ERROR and baseline.outcomes == ()
