"""The pytest runner against small, trusted fixture projects in a temporary directory."""

from __future__ import annotations

import sys
import textwrap
import time
import uuid
from pathlib import Path

import pytest

from testorcode.errors import ToolError
from testorcode.schemas.result_schemas import TestRunPhase, TestRunStatus
from testorcode.tools.pytest_runner import MAX_OUTPUT_CHARS, run_pytest, run_tests

PASSING = """
    def test_one():
        assert 1 + 1 == 2

    def test_two():
        assert "a".upper() == "A"
"""

MIXED = """
    import pytest

    def test_ok():
        assert True

    def test_bad():
        assert 2 + 2 == 5, "math is broken"

    @pytest.mark.skip(reason="not today")
    def test_skipped():
        pass

    class TestGroup:
        def test_method(self):
            assert [1, 2] == [1, 3]
"""


def project(tmp_path: Path, files: dict[str, str]) -> Path:
    root = tmp_path / "proj"
    for rel, text in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(textwrap.dedent(text))
    return root


def test_passing_tests(tmp_path: Path) -> None:
    root = project(tmp_path, {"tests/test_ok.py": PASSING})
    result = run_tests(root, ["tests/test_ok.py"], timeout_seconds=60)
    assert result.status is TestRunStatus.PASSED
    assert result.exit_code == 0 and not result.timed_out
    assert (result.passed, result.failed, result.errors, result.skipped) == (2, 0, 0, 0)
    assert result.failures == ()
    assert result.selectors == ("tests/test_ok.py",)
    # The recorded command is the full argv, sandbox launcher included.
    start = result.command.index(sys.executable)
    assert result.command[start : start + 3] == (sys.executable, "-m", "pytest")
    assert result.command[-2:] == ("--", "tests/test_ok.py")
    assert result.timeout_seconds == 60 and result.duration_seconds is not None


def test_assertion_failures_are_failed_with_details(tmp_path: Path) -> None:
    root = project(tmp_path, {"tests/test_mixed.py": MIXED})
    result = run_tests(root, ["tests"], timeout_seconds=60)
    assert result.status is TestRunStatus.FAILED
    assert result.exit_code == 1
    assert (result.passed, result.failed, result.errors, result.skipped) == (1, 2, 0, 1)
    assert sorted(result.failing_test_ids) == [
        "tests/test_mixed.py::TestGroup::test_method",
        "tests/test_mixed.py::test_bad",
    ]
    bad = next(f for f in result.failures if f.test_name == "test_bad")
    assert bad.test_file == "tests/test_mixed.py"
    assert "math is broken" in (bad.failure_message or "")
    assert "assert" in (bad.traceback or "")
    record = result.to_record("run-1", TestRunPhase.PRE_REPAIR)
    assert record.status is TestRunStatus.FAILED and record.failed == 2


def test_node_id_selector_runs_only_that_test(tmp_path: Path) -> None:
    root = project(tmp_path, {"tests/test_mixed.py": MIXED})
    result = run_tests(root, ["tests/test_mixed.py::test_ok"], timeout_seconds=60)
    assert result.status is TestRunStatus.PASSED
    assert result.passed == 1


def test_fixture_setup_error_is_reported_not_passed(tmp_path: Path) -> None:
    root = project(
        tmp_path,
        {
            "tests/test_fx.py": """
        import pytest

        @pytest.fixture
        def broken():
            raise RuntimeError("setup exploded")

        def test_uses_broken(broken):
            pass
    """
        },
    )
    result = run_tests(root, ["tests"], timeout_seconds=60)
    assert result.status is TestRunStatus.FAILED
    assert result.errors == 1 and result.passed == 0
    assert result.failing_test_ids == ["tests/test_fx.py::test_uses_broken"]


@pytest.mark.parametrize(
    "source",
    ["import module_that_does_not_exist\n\ndef test_x():\n    pass\n", "def test_x(:\n    pass\n"],
    ids=["import-error", "syntax-error"],
)
def test_collection_errors(tmp_path: Path, source: str) -> None:
    root = project(tmp_path, {"tests/test_ok.py": PASSING, "tests/test_broken.py": source})
    result = run_tests(root, ["tests"], timeout_seconds=60)
    assert result.status is TestRunStatus.COLLECTION_ERROR
    assert result.exit_code == 2
    assert result.passed is None and result.failed is None  # unknown, never 0
    assert "test_broken" in (result.error_message or "")


def test_missing_selector_is_an_error(tmp_path: Path) -> None:
    root = project(tmp_path, {"tests/test_ok.py": PASSING})
    for selector in ("tests/test_missing.py", "tests/test_ok.py::test_missing"):
        result = run_tests(root, [selector], timeout_seconds=60)
        assert result.status is TestRunStatus.ERROR, selector
        assert result.exit_code == 4
        assert result.error_message


def test_no_tests_collected_is_an_error(tmp_path: Path) -> None:
    root = project(tmp_path, {"tests/test_empty.py": "x = 1\n"})
    result = run_tests(root, ["tests"], timeout_seconds=60)
    assert result.status is TestRunStatus.ERROR
    assert result.exit_code == 5
    assert "no tests were collected" in (result.error_message or "")


def test_all_skipped_is_not_a_pass(tmp_path: Path) -> None:
    root = project(
        tmp_path,
        {
            "tests/test_skip.py": """
        import pytest

        @pytest.mark.skip(reason="no")
        def test_a():
            pass
    """
        },
    )
    result = run_tests(root, ["tests"], timeout_seconds=60)
    assert result.status is TestRunStatus.ERROR
    assert result.skipped == 1 and result.passed == 0
    assert "not a pass" in (result.error_message or "")


@pytest.mark.parametrize(
    "selector",
    [
        "-p",
        "--rootdir=/",
        "-x",
        "@args.txt",
        "/etc/passwd",
        "../outside.py",
        "tests/../../x.py",
        "",
        " tests",
        "tests\nother",
        "C:\\tests",
    ],
)
def test_unsafe_selectors_are_rejected_without_running(tmp_path: Path, selector: str) -> None:
    root = project(tmp_path, {"tests/test_ok.py": PASSING})
    marker = tmp_path / "ran"
    (root / "conftest.py").write_text(f"open({str(marker)!r}, 'w').close()\n")
    result = run_tests(root, [selector], timeout_seconds=60)
    assert result.status is TestRunStatus.ERROR
    assert result.exit_code is None and result.command == ()
    assert "invalid test selector" in (result.error_message or "")
    assert not marker.exists(), "pytest must not have been started"


def test_symlinked_selector_outside_the_worktree_is_rejected(tmp_path: Path) -> None:
    root = project(tmp_path, {"tests/test_ok.py": PASSING})
    outside = project(tmp_path / "other", {"tests/test_evil.py": PASSING})
    (root / "evil").symlink_to(outside / "tests", target_is_directory=True)
    result = run_tests(root, ["evil/test_evil.py"], timeout_seconds=60)
    assert result.status is TestRunStatus.ERROR and result.command == ()


def test_timeout_kills_the_whole_process_group(tmp_path: Path) -> None:
    marker = f"testorcode-marker-{uuid.uuid4().hex}"
    root = project(
        tmp_path,
        {
            "tests/test_slow.py": f"""
        import subprocess, sys, time

        def test_hangs():
            # One child in the same session, one that detaches into its own session.
            for detach in (False, True):
                subprocess.Popen(
                    [sys.executable, "-c", "import time; time.sleep(120)", {marker!r}],
                    start_new_session=detach,
                )
            time.sleep(120)
    """
        },
    )
    start = time.monotonic()
    result = run_tests(root, ["tests"], timeout_seconds=3)
    elapsed = time.monotonic() - start
    assert result.status is TestRunStatus.TIMEOUT
    assert result.timed_out is True and result.exit_code is None
    assert result.passed is None
    assert "timed out after 3s" in (result.error_message or "")
    assert elapsed < 30
    deadline = time.monotonic() + 5
    while _processes_with(marker) and time.monotonic() < deadline:
        time.sleep(0.1)
    assert _processes_with(marker) == [], "a test process survived the timeout"


def _processes_with(marker: str) -> list[int]:
    """Live (non-zombie) host processes whose command line contains ``marker``."""
    found = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            if marker.encode() not in (proc / "cmdline").read_bytes():
                continue
            if (proc / "stat").read_text().split(")")[-1].split()[0] != "Z":
                found.append(int(proc.name))
        except OSError:
            continue
    return found


def test_output_is_captured_and_bounded(tmp_path: Path) -> None:
    root = project(
        tmp_path,
        {
            "tests/test_noisy.py": f"""
        import sys

        def test_noisy():
            print("x" * {MAX_OUTPUT_CHARS * 5})
            sys.stderr.write("to-stderr\\n")
            assert False, "noisy failure"
    """
        },
    )
    run = run_pytest(root, ["tests/test_noisy.py"], timeout_seconds=60)
    assert run.execution.status is TestRunStatus.FAILED
    assert len(run.stdout) <= MAX_OUTPUT_CHARS + 100
    assert "bytes truncated" in run.stdout
    assert "noisy failure" in run.stdout  # the summary at the end survives truncation
    assert all(len(f.traceback or "") <= 4_000 for f in run.execution.failures)


def test_environment_is_sanitised(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_API_KEY", "sk-should-not-leak-0123456789")
    monkeypatch.setenv("PYTHONPATH", "/opt/ros/humble/lib/python3.10/site-packages")
    monkeypatch.setenv("PYTEST_ADDOPTS", "--this-option-does-not-exist")
    monkeypatch.setenv("PYTEST_PLUGINS", "plugin_that_does_not_exist")
    root = project(
        tmp_path,
        {
            "tests/test_env.py": """
        import os

        def test_env():
            for name in ("LLM_API_KEY", "PYTEST_ADDOPTS", "PYTEST_PLUGINS"):
                assert name not in os.environ, name
            # PYTHONPATH holds only the runner's probe directory, never the inherited value.
            assert "/opt/ros" not in os.environ.get("PYTHONPATH", "")
            assert os.environ["PYTHONDONTWRITEBYTECODE"] == "1"
    """
        },
    )
    result = run_tests(root, ["tests"], timeout_seconds=60)
    assert result.status is TestRunStatus.PASSED, result.error_message


def test_run_leaves_no_cache_or_bytecode_in_the_worktree(tmp_path: Path) -> None:
    root = project(tmp_path, {"tests/test_ok.py": PASSING, "tests/helper.py": "X = 1\n"})
    before = sorted(p.relative_to(root) for p in root.rglob("*"))
    run_tests(root, ["tests"], timeout_seconds=60)
    assert sorted(p.relative_to(root) for p in root.rglob("*")) == before


def test_conftest_above_the_worktree_is_not_loaded(tmp_path: Path) -> None:
    (tmp_path / "conftest.py").write_text("raise RuntimeError('outer conftest loaded')\n")
    root = project(tmp_path, {"tests/test_ok.py": PASSING})
    assert run_tests(root, ["tests"], timeout_seconds=60).status is TestRunStatus.PASSED


def test_conftest_exiting_zero_is_not_a_pass(tmp_path: Path) -> None:
    root = project(
        tmp_path, {"tests/test_ok.py": PASSING, "conftest.py": "import os\nos._exit(0)\n"}
    )
    result = run_tests(root, ["tests"], timeout_seconds=60)
    assert result.status is TestRunStatus.ERROR
    assert "wrote no test report" in (result.error_message or "")


def test_python_without_pytest_raises_tool_error(tmp_path: Path) -> None:
    root = project(tmp_path, {"tests/test_ok.py": PASSING})
    fake = tmp_path / "fakepython"
    fake.write_text("#!/bin/sh\necho 'No module named pytest' >&2\nexit 1\n")
    fake.chmod(0o755)
    with pytest.raises(ToolError, match="not installed"):
        run_tests(root, ["tests"], timeout_seconds=60, python_executable=fake)


def test_missing_python_or_worktree_raises_tool_error(tmp_path: Path) -> None:
    root = project(tmp_path, {"tests/test_ok.py": PASSING})
    with pytest.raises(ToolError, match="could not be started"):
        run_tests(root, ["tests"], timeout_seconds=60, python_executable=tmp_path / "nope")
    with pytest.raises(ToolError, match="does not exist"):
        run_tests(tmp_path / "missing", ["tests"], timeout_seconds=60)


def test_unexpected_exit_code_is_an_error(tmp_path: Path) -> None:
    root = project(
        tmp_path,
        {
            "tests/test_ok.py": PASSING,
            "conftest.py": """
        def pytest_sessionfinish(session, exitstatus):
            session.exitstatus = 7
    """,
        },
    )
    result = run_tests(root, ["tests"], timeout_seconds=60)
    assert result.status is TestRunStatus.ERROR
    assert result.exit_code == 7
    assert "unexpected pytest exit code 7" in (result.error_message or "")


@pytest.mark.parametrize("timeout", [0, -1, 1.5, True])
def test_bad_timeout_is_rejected(tmp_path: Path, timeout: object) -> None:
    with pytest.raises(ValueError):
        run_tests(tmp_path, ["tests"], timeout_seconds=timeout)  # type: ignore[arg-type]


def test_selectors_must_be_a_sequence_not_a_string(tmp_path: Path) -> None:
    with pytest.raises(TypeError):
        run_tests(tmp_path, "tests", timeout_seconds=5)


# --------------------------------------------------------------------------- #
# Regression: COLLECTION_ERROR comes from pytest's report, not console text
# --------------------------------------------------------------------------- #

MISLEADING = "print('ERROR collecting tests/test_x.py'); print('1 errors during collection')"


def test_failing_test_printing_collection_text_is_failed(tmp_path: Path) -> None:
    root = project(
        tmp_path,
        {
            "tests/test_p.py": f"""
        def test_ok():
            pass

        def test_lies():
            {MISLEADING}
            assert False, "real assertion failure"
    """
        },
    )
    run = run_pytest(root, ["tests"], timeout_seconds=60)
    assert "errors during collection" in run.stdout  # the text really is in the console
    result = run.execution
    assert result.status is TestRunStatus.FAILED
    assert (result.passed, result.failed) == (1, 1)
    assert result.failing_test_ids == ["tests/test_p.py::test_lies"]


def test_passing_test_writing_collection_text_to_the_console_is_passed(tmp_path: Path) -> None:
    root = project(
        tmp_path,
        {
            "tests/test_p.py": """
        import os

        def test_ok():
            os.write(1, b"ERROR collecting tests/x.py\\n1 errors during collection\\n")
    """
        },
    )
    result = run_tests(root, ["tests"], timeout_seconds=60)
    assert result.status is TestRunStatus.PASSED and result.passed == 1


def test_interrupted_run_with_collection_text_is_error_not_collection_error(
    tmp_path: Path,
) -> None:
    root = project(
        tmp_path,
        {
            "tests/test_p.py": f"""
        def test_interrupt():
            {MISLEADING}
            raise KeyboardInterrupt
    """
        },
    )
    result = run_tests(root, ["tests"], timeout_seconds=60)
    assert result.exit_code == 2
    assert result.status is TestRunStatus.ERROR
    assert "interrupted" in (result.error_message or "")


def test_genuine_collection_error_is_reported_even_if_exit_code_says_ok(tmp_path: Path) -> None:
    """An inconsistent run (collection failed, exit status forced to 0) is never PASSED."""
    root = project(
        tmp_path,
        {
            "tests/test_ok.py": PASSING,
            "tests/test_broken.py": "import module_that_does_not_exist\n",
            "conftest.py": """
            def pytest_sessionfinish(session, exitstatus):
                session.exitstatus = 0
        """,
        },
    )
    result = run_tests(root, ["tests"], timeout_seconds=60)
    assert result.exit_code == 0
    assert result.status is TestRunStatus.COLLECTION_ERROR
    assert "tests.test_broken" in (result.error_message or "")


def test_genuine_collection_error_with_passing_tests_elsewhere(tmp_path: Path) -> None:
    root = project(
        tmp_path,
        {
            "tests/test_ok.py": PASSING,
            "tests/test_broken.py": "def test_x(:\n    pass\n",
        },
    )
    result = run_tests(root, ["tests"], timeout_seconds=60)
    assert result.status is TestRunStatus.COLLECTION_ERROR
    assert result.exit_code == 2 and not result.timed_out


# --------------------------------------------------------------------------- #
# Regression: conftest failures and report/exit-code consistency
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("rel", "source"),
    [
        ("tests/conftest.py", "import nope_missing_module\n"),
        ("conftest.py", "raise RuntimeError('broken conftest')\n"),
        ("tests/conftest.py", "def broken(:\n    pass\n"),
    ],
    ids=["import-error", "runtime-error", "syntax-error"],
)
def test_broken_conftest_is_a_collection_error(tmp_path: Path, rel: str, source: str) -> None:
    root = project(tmp_path, {"tests/test_ok.py": PASSING, rel: source})
    result = run_tests(root, ["tests"], timeout_seconds=60)
    assert result.status is TestRunStatus.COLLECTION_ERROR
    assert "conftest could not be loaded" in (result.error_message or "")
    assert result.passed is None


def test_console_text_about_conftest_does_not_change_a_pass(tmp_path: Path) -> None:
    root = project(
        tmp_path,
        {
            "tests/test_p.py": """
        import os, sys

        def test_ok():
            os.write(2, b"ImportError while loading conftest 'tests/conftest.py'.\\n")
    """
        },
    )
    assert run_tests(root, ["tests"], timeout_seconds=60).status is TestRunStatus.PASSED


def test_missing_selector_is_still_an_error_not_a_collection_error(tmp_path: Path) -> None:
    root = project(tmp_path, {"tests/test_ok.py": PASSING})
    result = run_tests(root, ["tests/test_missing.py"], timeout_seconds=60)
    assert result.status is TestRunStatus.ERROR and result.exit_code == 4


def test_exit_code_forced_to_zero_with_failures_is_not_a_pass(tmp_path: Path) -> None:
    root = project(
        tmp_path,
        {
            "tests/test_mixed.py": MIXED,
            "conftest.py": """
        def pytest_sessionfinish(session, exitstatus):
            session.exitstatus = 0
    """,
        },
    )
    result = run_tests(root, ["tests"], timeout_seconds=60)
    assert result.exit_code == 0
    assert result.status is TestRunStatus.ERROR
    assert "exit code 0 but the report lists 2 failing tests" in (result.error_message or "")


def test_report_listing_unselected_tests_is_not_trusted() -> None:
    from testorcode.tools.pytest_runner import _inconsistencies, _Junit

    session = {"session": {"exitstatus": 0, "testscollected": 2, "testsfailed": 0}}
    report = _Junit(
        tests=2,
        failed=0,
        errors=0,
        skipped=0,
        failures=(),
        collection_errors=(),
        files=("tests/test_b.py", "tests/test_other.py"),
        suite_failures=0,
        suite_skipped=0,
    )
    problems = _inconsistencies(0, report, session, ("tests/test_b.py",))
    assert any("outside the selection" in p for p in problems)
    assert _inconsistencies(0, report, session, ("tests",)) == []
