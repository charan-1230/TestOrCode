"""Executable probes (``tools.probes``) and trusted probes on both repository layouts, executed in the bwrap sandbox.

The package layout mirrors the pilot repository (``boltons/``: a source root that is itself a package and uses relative
imports); the ``src/`` layout mirrors the synthetic fixtures.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from testorcode.schemas.episode_schemas import EditPolicy, ExecutableProbe
from testorcode.schemas.result_schemas import ProbeStatus
from testorcode.tools import sandbox
from testorcode.tools.probes import (
    probe_from_tests,
    probe_problem,
    qualname_of_test,
    run_executable_probe,
    support_code,
)
from testorcode.tools.verification import ProbeCase, TrustedProbe, verify_behaviour

needs_bwrap = pytest.mark.skipif(not sandbox.bwrap_available(), reason="bubblewrap unavailable: probes not executed")

PKG_POLICY = EditPolicy(allowed_source_roots=("pkg/",), allowed_test_roots=("tests/",))
SRC_POLICY = EditPolicy(allowed_source_roots=("src/",), allowed_test_roots=("tests/",))
TESTS = """\
from pkg.core import shout


class Box:
    def __init__(self, value):
        self.value = value


def test_shout():
    assert shout(Box('a').value) == 'A!'


def test_other():
    assert shout('b') == 'B!'
"""


def make_repo(tmp_path: Path, files: dict[str, str]) -> Path:
    root = tmp_path / "repo"
    for rel, text in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)
    for args in (["init", "-q"], ["add", "."], ["-c", "user.name=t", "-c", "user.email=t@x", "commit", "-qm", "ci"]):
        subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)
    return root


@pytest.fixture
def pkg_repo(tmp_path: Path) -> Path:
    return make_repo(tmp_path, {
        "pkg/__init__.py": "",
        "pkg/helpers.py": "SUFFIX = '!'\n",
        "pkg/core.py": "from .helpers import SUFFIX\n\n\ndef shout(s):\n    return s.upper() + SUFFIX\n",
        "tests/test_core.py": TESTS,
    })


def probe(code: str, probe_id: str = "p-1") -> ExecutableProbe:
    return ExecutableProbe(probe_id=probe_id, probe_code=code)


def run(root: Path, p: ExecutableProbe, policy: EditPolicy = PKG_POLICY, timeout: int = 30):
    return run_executable_probe(root, p, policy=policy, timeout_seconds=timeout, isolation="bwrap")


# --------------------------------------------------------------------------- #
# Derivation
# --------------------------------------------------------------------------- #


def test_derived_probe_is_self_contained_and_keeps_only_the_selected_tests() -> None:
    code = probe_from_tests(TESTS, ["test_shout"])
    assert "from pkg.core import shout" in code and "class Box" in code and "def test_shout" in code
    assert "test_other" not in code
    assert probe_problem(code) is None


def test_derivation_keeps_selected_methods_and_drops_skip_markers() -> None:
    module = (
        "import pytest\npytestmark = pytest.mark.skip\n\n\nclass TestA:\n    def helper(self):\n        return 1\n\n"
        "    def test_one(self):\n        assert self.helper() == 1\n\n    def test_two(self):\n        assert 0\n"
    )
    code = probe_from_tests(module, [qualname_of_test("tests/t.py::TestA::test_one[x]")])
    assert "test_one" in code and "helper" in code and "test_two" not in code and "pytestmark" not in code
    assert "pytestmark" not in support_code(module) and "TestA" not in support_code(module)
    with pytest.raises(ValueError):
        probe_from_tests(module, ["test_missing"])


@pytest.mark.parametrize(("code", "reason"), [("def test_x(:\n", "does not parse"),
                                              ("x = 1\n", "defines no test"), ("def helper():\n    pass\n", "no test")])
def test_malformed_probes_are_recognised(code: str, reason: str) -> None:
    assert reason in probe_problem(code)


# --------------------------------------------------------------------------- #
# Execution (bwrap)
# --------------------------------------------------------------------------- #


@needs_bwrap
def test_probe_passes_on_the_package_layout_with_relative_imports(pkg_repo: Path) -> None:
    result = run(pkg_repo, probe(probe_from_tests(TESTS, ["test_shout"])))
    assert result.status is ProbeStatus.PASSED and result.verified and result.isolation == "bwrap"


@needs_bwrap
def test_probe_fails_when_the_behaviour_changed(pkg_repo: Path) -> None:
    (pkg_repo / "pkg/helpers.py").write_text("SUFFIX = '?'\n")
    result = run(pkg_repo, probe(probe_from_tests(TESTS, ["test_shout"])))
    assert result.status is ProbeStatus.FAILED and "test_shout" in result.reasons[0]


@needs_bwrap
def test_probe_ignores_agent_editable_tests_and_conftest(pkg_repo: Path) -> None:
    """The worktree's tests are not read: editing them (or adding a conftest) cannot make a probe pass."""
    (pkg_repo / "pkg/helpers.py").write_text("SUFFIX = '?'\n")  # the behaviour is broken ...
    (pkg_repo / "tests/test_core.py").write_text(TESTS.replace("'A!'", "'A?'"))  # ... and the test was bent to it
    (pkg_repo / "tests/conftest.py").write_text("import pytest\n\n@pytest.fixture(autouse=True)\ndef _s():\n    pytest.skip('x')\n")
    (pkg_repo / "pkg/conftest.py").write_text("import pytest\n\n@pytest.fixture(autouse=True)\ndef _s():\n    pytest.skip('x')\n")
    result = run(pkg_repo, probe(probe_from_tests(TESTS, ["test_shout"])))
    assert result.status is ProbeStatus.FAILED


@needs_bwrap
@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("def test_x():\n    assert undefined_name('a') == 'A!'\n", "broken"),  # missing import: not a regression
        ("import not_a_module_anywhere\n\ndef test_x():\n    pass\n", "did not execute cleanly"),
        ("import pytest\n\ndef test_x():\n    pytest.skip('x')\n", "did not execute cleanly"),  # skipped is no pass
    ],
)
def test_broken_probes_are_errors_never_passes_or_failures(pkg_repo: Path, code: str, expected: str) -> None:
    result = run(pkg_repo, probe(code))
    assert result.status is ProbeStatus.ERROR and not result.verified and expected in result.reasons[0]


@needs_bwrap
def test_malformed_probe_is_not_executed(pkg_repo: Path) -> None:
    result = run(pkg_repo, probe("x = 1\n"))
    assert result.status is ProbeStatus.ERROR and result.isolation is None  # rejected before any sandbox ran


@needs_bwrap
def test_probe_timeout_is_an_error(pkg_repo: Path) -> None:
    result = run(pkg_repo, probe("def test_spin():\n    while True:\n        pass\n"), timeout=2)
    assert result.status is ProbeStatus.ERROR and "timeout" in result.reasons[0]


@needs_bwrap
def test_probe_on_the_src_layout(tmp_path: Path) -> None:
    root = make_repo(tmp_path, {"src/synthlib/__init__.py": "", "src/synthlib/core.py": "def twice(x):\n    return x * 2\n",
                                "tests/test_core.py": "from synthlib.core import twice\n\n\ndef test_twice():\n    assert twice(2) == 4\n"})
    code = probe_from_tests((root / "tests/test_core.py").read_text(), ["test_twice"])
    assert run(root, probe(code), SRC_POLICY).status is ProbeStatus.PASSED
    (root / "src/synthlib/core.py").write_text("def twice(x):\n    return x * 3\n")
    assert run(root, probe(code), SRC_POLICY).status is ProbeStatus.FAILED


def test_missing_source_root_is_an_error(tmp_path: Path) -> None:
    root = make_repo(tmp_path, {"tests/test_x.py": "def test_x():\n    pass\n"})
    result = run_executable_probe(root, probe("def test_x():\n    pass\n"), policy=PKG_POLICY, timeout_seconds=5,
                                  isolation="bwrap")
    assert result.status is ProbeStatus.ERROR and "source root" in result.reasons[0]


# --------------------------------------------------------------------------- #
# Trusted probes on the package layout (the import-root fix)
# --------------------------------------------------------------------------- #


@needs_bwrap
def test_trusted_probe_imports_a_package_root_by_its_package_name(pkg_repo: Path) -> None:
    good = TrustedProbe(probe_id="t", module="pkg.core", function="shout", provenance="dataset",
                        cases=(ProbeCase(args=("a",), expected="A!"),))
    result = verify_behaviour(pkg_repo, good, timeout_seconds=30, policy=PKG_POLICY, isolation="bwrap")
    assert result.status is ProbeStatus.PASSED  # relative import inside the package works
    outside = good.model_copy(update={"module": "core"})  # not importable as a top-level module any more
    assert verify_behaviour(pkg_repo, outside, timeout_seconds=30, policy=PKG_POLICY,
                            isolation="bwrap").status is ProbeStatus.ERROR
    tests_module = good.model_copy(update={"module": "tests.test_core"})  # tests are never a probe target
    assert verify_behaviour(pkg_repo, tests_module, timeout_seconds=30, policy=PKG_POLICY,
                            isolation="bwrap").status is ProbeStatus.ERROR
