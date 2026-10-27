"""Execution isolation for the pytest runner.

Adversarial fixtures (tests that try to escape or forge results) run only
inside bubblewrap; they are skipped, never run on the host, when it is missing.
Docker cannot be exercised without daemon access, so its command and timeout
handling are checked with a fake ``docker`` executable.
"""

from __future__ import annotations

import shutil
import socket
import textwrap
import time
import uuid
from pathlib import Path

import pytest

from testorcode.errors import ToolError
from testorcode.schemas.result_schemas import TestRunStatus
from testorcode.tools import sandbox
from testorcode.tools.pytest_runner import run_pytest, run_tests
from testorcode.tools.sandbox import DOCKER_WORKDIR, Isolation, wrap_command

needs_bwrap = pytest.mark.skipif(
    not sandbox.bwrap_available(), reason="bubblewrap unavailable: adversarial fixtures not run"
)


def project(tmp_path: Path, files: dict[str, str]) -> Path:
    root = tmp_path / "proj"
    for rel, text in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(textwrap.dedent(text))
    return root


def processes_with(marker: str) -> list[int]:
    found = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            if marker.encode() in (proc / "cmdline").read_bytes() and (
                (proc / "stat").read_text().split(")")[-1].split()[0] != "Z"
            ):
                found.append(int(proc.name))
        except OSError:
            continue
    return found


# --------------------------------------------------------------------------- #
# Mode selection: fail closed
# --------------------------------------------------------------------------- #


@pytest.fixture
def no_sandboxes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sandbox, "docker_available", lambda: False)
    monkeypatch.setattr(sandbox, "bwrap_available", lambda: False)


def test_auto_fails_closed_without_any_sandbox(tmp_path: Path, no_sandboxes: None) -> None:
    marker = tmp_path / "ran"
    root = project(
        tmp_path,
        {
            "conftest.py": f"open({str(marker)!r}, 'w').close()\n",
            "tests/test_ok.py": "def test_ok():\n    pass\n",
        },
    )
    with pytest.raises(ToolError, match="no sandbox available"):
        run_tests(root, ["tests"], timeout_seconds=30)
    assert not marker.exists(), "repository code must not run on the host"


def test_explicit_sandboxes_fail_closed_when_unavailable(
    tmp_path: Path, no_sandboxes: None
) -> None:
    root = project(tmp_path, {"tests/test_ok.py": "def test_ok():\n    pass\n"})
    with pytest.raises(ToolError, match="needs a docker image"):
        run_tests(root, ["tests"], timeout_seconds=30, isolation="docker")
    with pytest.raises(ToolError, match="daemon is not reachable"):
        run_tests(root, ["tests"], timeout_seconds=30, isolation="docker", docker_image="img")
    with pytest.raises(ToolError, match="bubblewrap is unavailable"):
        run_tests(root, ["tests"], timeout_seconds=30, isolation="bwrap")
    with pytest.raises(ValueError):
        run_tests(root, ["tests"], timeout_seconds=30, isolation="none")


def test_trusted_host_mode_must_be_explicit_and_still_works(
    tmp_path: Path, no_sandboxes: None
) -> None:
    root = project(tmp_path, {"tests/test_ok.py": "def test_ok():\n    assert True\n"})
    run = run_pytest(root, ["tests"], timeout_seconds=60, isolation="trusted-host")
    assert run.isolation is Isolation.TRUSTED_HOST
    assert run.execution.status is TestRunStatus.PASSED


@needs_bwrap
def test_auto_uses_bwrap_when_docker_is_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sandbox, "docker_available", lambda: False)
    root = project(tmp_path, {"tests/test_ok.py": "def test_ok():\n    assert True\n"})
    run = run_pytest(root, ["tests"], timeout_seconds=60, docker_image="img")
    assert run.isolation is Isolation.BWRAP
    command = run.execution.command
    assert "bwrap" in command and "--unshare-all" in command
    if "prlimit" in command:  # resource limits wrap the sandbox launcher
        assert command.index("prlimit") < command.index("bwrap") and "--core=0" in command
    assert run.execution.status is TestRunStatus.PASSED


# --------------------------------------------------------------------------- #
# Docker: command shape and timeout handling (no daemon needed)
# --------------------------------------------------------------------------- #


def test_docker_command_is_locked_down(tmp_path: Path) -> None:
    command, launcher_env = wrap_command(
        Isolation.DOCKER,
        ["python", "-m", "pytest"],
        worktree=tmp_path / "wt",
        outdir=tmp_path / "out",
        probedir=tmp_path / "probe",
        python="python",
        env={"SECRET_LOOKING": "only-what-we-pass"},
        docker_image="testorcode/py312:pinned",
        container_name="testorcode-abc",
    )
    joined = " ".join(command)
    for flag in (
        "--network none",
        "--read-only",
        "--cap-drop ALL",
        "no-new-privileges",
        "--pids-limit",
        "--memory",
        "--rm",
        "--init",
    ):
        assert flag in joined, flag
    assert f"dst={DOCKER_WORKDIR},readonly" in joined
    assert "docker.sock" not in joined and "--privileged" not in joined
    user = command[command.index("--user") + 1]
    assert not user.startswith("0:")
    assert command[-4:] == ["testorcode/py312:pinned", "python", "-m", "pytest"]
    assert set(launcher_env or {}) <= {"PATH", "LC_ALL"}


def test_docker_timeout_kills_the_container(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    log = tmp_path / "docker.log"
    (fake_bin / "docker").write_text(
        f'#!/bin/sh\necho "$1 $2" >> {log}\nif [ "$1" = run ]; then exec sleep 60; fi\n'
    )
    (fake_bin / "docker").chmod(0o755)
    monkeypatch.setenv("PATH", f"{fake_bin}:/usr/bin:/bin")
    monkeypatch.setattr(sandbox, "docker_available", lambda: True)
    root = project(tmp_path, {"tests/test_ok.py": "def test_ok():\n    pass\n"})
    start = time.monotonic()
    result = run_tests(root, ["tests"], timeout_seconds=2, isolation="docker", docker_image="img")
    assert result.status is TestRunStatus.TIMEOUT and result.timed_out
    assert time.monotonic() - start < 30
    calls = log.read_text().split("\n")
    assert calls[0].startswith("run ")
    assert any(c.startswith("kill testorcode-") for c in calls), calls


# --------------------------------------------------------------------------- #
# bubblewrap: what malicious tests can no longer do
# --------------------------------------------------------------------------- #


@needs_bwrap
def test_sandbox_blocks_network_host_files_secrets_and_worktree_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LLM_API_KEY", "sk-should-not-leak-0123456789")
    secret = tmp_path / "host_secret.txt"
    secret.write_text("top secret\n")
    escaped = tmp_path / "escaped.txt"
    home = Path.home()
    root = project(
        tmp_path,
        {
            "src/mod.py": "VALUE = 1\n",
            "tests/test_escape.py": f"""
            import os, socket
            from pathlib import Path

            def test_isolation():
                try:
                    socket.create_connection(("1.1.1.1", 53), timeout=2)
                    network = True
                except OSError:
                    network = False
                assert not network, "network reachable"
                assert not Path({str(secret)!r}).exists(), "host file visible"
                assert not Path({str(home)!r}).exists() or not any(Path({str(home)!r}).iterdir())
                assert not Path("/var/run/docker.sock").exists()
                assert not Path("/run/docker.sock").exists()
                assert "LLM_API_KEY" not in os.environ
                if {shutil.which("prlimit") is not None!r}:
                    import resource
                    assert resource.getrlimit(resource.RLIMIT_CORE) == (0, 0)
                    assert resource.getrlimit(resource.RLIMIT_FSIZE)[0] == 1 << 30
                try:
                    Path("src/mod.py").write_text("VALUE = 2\\n")
                    wrote = True
                except OSError:
                    wrote = False
                assert not wrote, "worktree writable"
                try:
                    Path({str(escaped)!r}).write_text("x")
                except OSError:
                    pass
        """,
        },
    )
    result = run_tests(root, ["tests"], timeout_seconds=60, isolation="bwrap")
    assert result.status is TestRunStatus.PASSED, result.failures or result.error_message
    assert not escaped.exists(), "a file was written outside the worktree on the host"
    assert (root / "src/mod.py").read_text() == "VALUE = 1\n"


@needs_bwrap
def test_detached_processes_do_not_survive_a_finished_run(tmp_path: Path) -> None:
    marker = f"testorcode-marker-{uuid.uuid4().hex}"
    root = project(
        tmp_path,
        {
            "tests/test_bg.py": f"""
        import subprocess, sys

        def test_spawn():
            subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)", {marker!r}],
                             start_new_session=True)
    """
        },
    )
    assert run_tests(root, ["tests"], timeout_seconds=60, isolation="bwrap").status is (
        TestRunStatus.PASSED
    )
    deadline = time.monotonic() + 5
    while processes_with(marker) and time.monotonic() < deadline:
        time.sleep(0.1)
    assert processes_with(marker) == []


FORGED_REPORT = (
    '<testsuites><testsuite name="pytest" errors="0" failures="0" skipped="0" tests="1">'
    '<testcase classname="tests.test_f" name="test_bad" file="tests/test_f.py"/>'
    "</testsuite></testsuites>"
)


@needs_bwrap
def test_report_forged_before_os_exit_is_not_trusted(tmp_path: Path) -> None:
    root = project(
        tmp_path,
        {
            "tests/test_f.py": f"""
        import os, sys

        def test_bad():
            report = [a.split("=", 1)[1] for a in sys.argv if a.startswith("--junitxml=")][0]
            with open(report, "w") as fh:
                fh.write({FORGED_REPORT!r})
            os._exit(0)
    """
        },
    )
    result = run_tests(root, ["tests"], timeout_seconds=60, isolation="bwrap")
    assert result.status is TestRunStatus.ERROR
    assert "inconsistent pytest results" in (result.error_message or "")


@needs_bwrap
def test_report_overwritten_at_exit_is_not_trusted(tmp_path: Path) -> None:
    root = project(
        tmp_path,
        {
            "tests/test_f.py": f"""
        import atexit, sys

        REPORT = [a.split("=", 1)[1] for a in sys.argv if a.startswith("--junitxml=")][0]

        def _forge():
            with open(REPORT, "w") as fh:
                fh.write({FORGED_REPORT!r})

        atexit.register(_forge)

        def test_bad():
            assert False
    """
        },
    )
    result = run_tests(root, ["tests"], timeout_seconds=60, isolation="bwrap")
    assert result.status is TestRunStatus.ERROR
    assert "inconsistent pytest results" in (result.error_message or "")


@needs_bwrap
def test_known_limitation_a_fully_consistent_forgery_is_not_detected(tmp_path: Path) -> None:
    """Documents a limitation, not desired behaviour: in-process test code that forges
    the report *and* the probe record consistently is reported as PASSED.  Only the
    independent Integrity Guard / mutation testing can catch this."""
    root = project(
        tmp_path,
        {
            "tests/test_f.py": f"""
        import json, os, sys

        def test_bad():
            report = [a.split("=", 1)[1] for a in sys.argv if a.startswith("--junitxml=")][0]
            with open(report, "w") as fh:
                fh.write({FORGED_REPORT!r})
            with open(os.environ["TESTORCODE_PROBE_OUT"], "w") as fh:
                json.dump({{"session": {{"exitstatus": 0, "testscollected": 1, "testsfailed": 0}}}}, fh)
            os._exit(0)
    """
        },
    )
    assert run_tests(root, ["tests"], timeout_seconds=60, isolation="bwrap").status is (
        TestRunStatus.PASSED
    )


def test_network_is_reachable_check_is_meaningful() -> None:
    """Guard against a vacuous isolation test: report whether the host itself has network."""
    try:
        socket.create_connection(("1.1.1.1", 53), timeout=2).close()
    except OSError:
        pytest.skip("host has no network; the sandbox network assertion is not discriminating")
