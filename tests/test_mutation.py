"""Cached-mutant runner (``tools.mutation``) and the Guard's approval rule (``verification.repair_outcome``).

Mutants really execute in the bwrap sandbox against a scratch copy of a fixture repository; every outcome below is
measured, not simulated.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from testorcode.schemas.contracts import (
    EpisodeContext,
    FailureGroupDetail,
    PatchProposal,
    PytestExecution,
)
from testorcode.schemas.episode_schemas import (
    EditPolicy,
    FailingTest,
    FailureGroup,
    MutantSpec,
    MutationFixture,
    PatchScope,
    RepairDirection,
    RepoRef,
)
from testorcode.schemas.result_schemas import (
    GuardDecision,
    MutationResult,
    ProbeStatus,
    TestRunStatus,
)
from testorcode.tools import sandbox
from testorcode.tools.mutation import (
    generate_mutants,
    load_cached_mutants,
    locate_function,
    mutation_check,
    run_mutants,
)
from testorcode.tools.verification import TrustedVerification, repair_outcome

needs_bwrap = pytest.mark.skipif(not sandbox.bwrap_available(), reason="bubblewrap unavailable: mutants not executed")

POLICY = EditPolicy(allowed_source_roots=("pkg/",), allowed_test_roots=("tests/",))
CORE = "def grade(score):\n    if score >= 50:\n        return 'pass'\n    return 'fail'\n"
STRONG = ("from pkg.core import grade\n\n\ndef test_pass():\n    assert grade(70) == 'pass'\n\n\n"
          "def test_fail():\n    assert grade(10) == 'fail'\n\n\ndef test_edge():\n    assert grade(50) == 'pass'\n")


def git(root: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True, text=True).stdout


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    for rel, text in {"pkg/__init__.py": "", "pkg/core.py": CORE, "tests/test_core.py": STRONG}.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)
    git(root, "init", "-q")
    git(root, "add", ".")
    git(root, "-c", "user.name=t", "-c", "user.email=t@x", "commit", "-qm", "c")
    return root


def mutants() -> list[MutantSpec]:
    return generate_mutants(CORE, "pkg/core.py", "grade", module="pkg.core", limit=20)


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #


def test_generation_is_deterministic_and_every_mutant_parses() -> None:
    first, second = mutants(), mutants()
    assert first == second and len(first) >= 4
    ops = " ".join(m.operator for m in first)
    assert "GtE -> Gt" in ops and "return value -> None" in ops and "constant 50 -> 51" in ops
    for m in first:
        assert m.function_qualname == "pkg.core.grade" and m.file_path == "pkg/core.py"
        assert m.unified_diff.startswith("--- a/pkg/core.py\n+++ b/pkg/core.py\n")


def test_generation_samples_with_a_fixed_seed_and_skips_unknown_functions() -> None:
    limited = generate_mutants(CORE, "pkg/core.py", "grade", limit=2, seed=3)
    assert len(limited) == 2 and limited == generate_mutants(CORE, "pkg/core.py", "grade", limit=2, seed=3)
    assert generate_mutants(CORE, "pkg/core.py", "missing") == []
    assert generate_mutants("def broken(:\n", "pkg/core.py", "broken") == []
    method = "class A:\n    def f(self, x):\n        return x + 1\n"
    assert {m.operator.split(": ")[1] for m in generate_mutants(method, "pkg/a.py", "A.f")} >= {"Add -> Sub"}


def test_locate_function_on_the_package_layout(repo: Path) -> None:
    ctx = EpisodeContext(episode_id="e", repo=RepoRef(repo_id="r"), worktree_path=repo, commit_sha="abcdef1",
                         edit_policy=POLICY)
    assert locate_function(ctx, "pkg.core.grade") == ("pkg/core.py", "pkg.core", "grade")
    assert locate_function(ctx, "nowhere.f") is None


# --------------------------------------------------------------------------- #
# Execution: killed / survived / timeout / error / invalid
# --------------------------------------------------------------------------- #


def run(root: Path, specs, selectors=("tests/test_core.py",), **kw):
    return run_mutants(root, specs, list(selectors), timeout_seconds=kw.pop("timeout", 30), isolation="bwrap", **kw)


@needs_bwrap
def test_strong_tests_kill_and_weak_tests_let_mutants_survive(repo: Path) -> None:
    specs = mutants()
    strong = run(repo, specs)
    assert strong.status is TestRunStatus.PASSED and strong.baseline_failing == frozenset()
    outcomes = {o.mutant_id: o for o in strong.outcomes}
    assert {o.outcome for o in outcomes.values()} <= {"killed", "survived"}
    assert sum(o.outcome == "killed" for o in outcomes.values()) >= 4
    killed = next(o for o in outcomes.values() if o.outcome == "killed")
    assert killed.killed_by and all(t.startswith("tests/test_core.py::") for t in killed.killed_by)

    (repo / "tests/test_core.py").write_text("from pkg.core import grade\n\n\ndef test_any():\n    assert grade(70) != 'x'\n")
    weak = run(repo, specs)
    assert sum(o.outcome == "killed" for o in weak.outcomes) < sum(o.outcome == "killed" for o in strong.outcomes)


@needs_bwrap
def test_timeouts_errors_and_invalid_mutants_are_never_kills(repo: Path) -> None:
    hang = MutantSpec(mutant_id="hang", function_qualname="pkg.core.grade", file_path="pkg/core.py", unified_diff=(
        "--- a/pkg/core.py\n+++ b/pkg/core.py\n@@ -1,2 +1,3 @@\n def grade(score):\n+    while True: pass\n"
        "     if score >= 50:\n"))
    crash = MutantSpec(mutant_id="crash", function_qualname="pkg.core.grade", file_path="pkg/core.py", unified_diff=(
        "--- a/pkg/core.py\n+++ b/pkg/core.py\n@@ -1,2 +1,2 @@\n-def grade(score):\n+def grade(score:\n"
        "     if score >= 50:\n"))
    stale = MutantSpec(mutant_id="stale", function_qualname="pkg.core.grade", file_path="pkg/core.py", unified_diff=(
        "--- a/pkg/core.py\n+++ b/pkg/core.py\n@@ -1,1 +1,1 @@\n-def other(x):\n+def other(y):\n"))
    escape = MutantSpec(mutant_id="escape", function_qualname="f", file_path="../outside.py", unified_diff="x")
    result = run(repo, [hang, crash, stale, escape], timeout=3)
    assert {o.mutant_id: o.outcome for o in result.outcomes} == {
        "hang": "timeout", "crash": "error", "stale": "invalid", "escape": "invalid"}
    assert git(repo, "status", "--porcelain") == ""  # the episode worktree was never written
    assert not (repo.parent / "outside.py").exists()


@needs_bwrap
def test_a_baseline_that_cannot_run_measures_nothing(repo: Path) -> None:
    result = run(repo, mutants(), selectors=("tests/missing_test.py",))
    assert result.status is TestRunStatus.ERROR and result.outcomes == () and "baseline" in result.reason
    assert run(repo, mutants(), selectors=()).status is TestRunStatus.ERROR


@needs_bwrap
def test_overrides_change_only_the_scratch_copy(repo: Path) -> None:
    weak = "from pkg.core import grade\n\n\ndef test_any():\n    assert grade(70) != 'x'\n"
    result = run(repo, mutants(), overrides={"tests/test_core.py": weak})
    assert all(o.outcome == "survived" or o.killed_by == ("tests/test_core.py::test_any",) for o in result.outcomes)
    assert (repo / "tests/test_core.py").read_text() == STRONG
    assert run(repo, mutants(), overrides={"../x.py": "y"}).status is TestRunStatus.ERROR


# --------------------------------------------------------------------------- #
# The Guard's mutation check
# --------------------------------------------------------------------------- #


OUTDATED = "from pkg.core import grade\n\n\ndef test_pass():\n    assert grade(70) == 'ok'\n"  # 'ok' was renamed 'pass'


def check(root: Path, repaired: str, pre: str = OUTDATED, decision=RepairDirection.UPDATE_TEST):
    (root / "tests/test_core.py").write_text(repaired)
    test = FailingTest(test_id="tests/test_core.py::test_pass", test_file="tests/test_core.py", test_name="test_pass")
    group = FailureGroupDetail(group=FailureGroup(group_id="g1", function_qualname="pkg.core.grade",
                                                  test_ids=[test.test_id]), failing_tests=(test,))
    scope = PatchScope.TEST if decision is RepairDirection.UPDATE_TEST else PatchScope.SRC
    proposal = PatchProposal(group_id="g1", decision=decision, scope=scope, target_files=("tests/test_core.py",),
                             unified_diff="--- a/x\n+++ b/x\n", rationale="r", evidence_ids=("ev",))
    ctx = EpisodeContext(episode_id="e", repo=RepoRef(repo_id="r"), worktree_path=root, commit_sha="abcdef1",
                         edit_policy=POLICY)
    return mutation_check(ctx, group, proposal, {"tests/test_core.py": pre}, timeout_seconds=30, isolation="bwrap")


@needs_bwrap
def test_a_legitimate_update_has_measured_strength(repo: Path) -> None:
    result = check(repo, OUTDATED.replace("'ok'", "'pass'"))
    assert result.problems == () and result.gaps == ()
    assert result.result.status is TestRunStatus.PASSED and result.result.mutants_total >= 4
    assert result.result.killed_by_repaired_tests >= 1 and result.result.killed_by_reference_tests is None


@needs_bwrap
def test_a_loosened_update_that_detects_nothing_is_not_verified(repo: Path) -> None:
    # Only the expected literal changed (allowed by the patch rule), but a != assertion pins nothing.
    result = check(repo, OUTDATED.replace("== 'ok'", "!= 'nope'"), pre=OUTDATED.replace("== 'ok'", "!= 'pass'"))
    assert result.problems == () and "detect none" in result.gaps[0]


@needs_bwrap
def test_mutants_killed_before_the_repair_but_not_after_are_measured_weakening(repo: Path) -> None:
    weak = STRONG.replace("    assert grade(10) == 'fail'\n", "    pass\n")
    result = check(repo, weak, pre=STRONG)
    assert result.problems and "measured test weakening" in result.problems[0]


@needs_bwrap
@needs_bwrap
def test_fix_code_oracle_sensitivity_is_measured_on_the_untouched_tests(repo: Path) -> None:
    result = check(repo, STRONG, decision=RepairDirection.FIX_CODE)
    assert result.problems == result.gaps == () and result.outcomes_before == ()  # one state: the repaired code
    assert result.result.status is TestRunStatus.PASSED and result.result.killed_by_repaired_tests >= 1
    assert result.killed_by_file_before is None


@needs_bwrap
def test_fix_code_whose_tests_detect_no_mutant_is_not_verified(repo: Path) -> None:
    vacuous = "from pkg.core import grade\n\n\ndef test_pass():\n    assert grade(70) != 'x'\n"
    result = check(repo, vacuous, decision=RepairDirection.FIX_CODE)
    assert result.problems == () and "tests that verify this fix detect none" in result.gaps[0]


@needs_bwrap
def test_group_kills_and_file_kills_are_reported_separately(repo: Path) -> None:
    # test_pass is the repaired group test; test_fail/test_edge are other tests of the file that kill more mutants.
    result = check(repo, STRONG, pre=STRONG.replace("grade(70) == 'pass'", "grade(70) == 'ok'"))
    assert result.result.killed_by_repaired_tests < result.killed_by_file_after  # group-only vs whole file
    group_killers = {o.mutant_id for o in result.outcomes_after if "tests/test_core.py::test_pass" in o.killed_by}
    assert result.result.killed_by_repaired_tests == len(group_killers)


@needs_bwrap
def test_the_overall_deadline_stops_execution_and_is_never_a_kill(repo: Path) -> None:
    import time
    result = run_mutants(repo, mutants(), ["tests/test_core.py"], timeout_seconds=30, isolation="bwrap",
                         deadline=time.monotonic() + 1.5)
    assert result.status is TestRunStatus.PASSED or "deadline" in (result.reason or "")
    assert any(o.outcome == "not_run" for o in result.outcomes) or result.status is TestRunStatus.ERROR
    assert all(o.outcome != "killed" or o.killed_by for o in result.outcomes)
    expired = run_mutants(repo, mutants(), ["tests/test_core.py"], timeout_seconds=30, isolation="bwrap",
                          deadline=time.monotonic())
    assert expired.status is TestRunStatus.ERROR and "deadline" in expired.reason
    gate = check_with_deadline(repo, 0)
    assert gate.result.status is TestRunStatus.ERROR and "could not run" in gate.gaps[0]


def check_with_deadline(root: Path, seconds: int):
    (root / "tests/test_core.py").write_text(OUTDATED.replace("'ok'", "'pass'"))
    test = FailingTest(test_id="tests/test_core.py::test_pass", test_file="tests/test_core.py", test_name="test_pass")
    group = FailureGroupDetail(group=FailureGroup(group_id="g1", function_qualname="pkg.core.grade",
                                                  test_ids=[test.test_id]), failing_tests=(test,))
    proposal = PatchProposal(group_id="g1", decision=RepairDirection.UPDATE_TEST, scope=PatchScope.TEST,
                             target_files=("tests/test_core.py",), unified_diff="--- a/x\n+++ b/x\n", rationale="r",
                             evidence_ids=("ev",))
    ctx = EpisodeContext(episode_id="e", repo=RepoRef(repo_id="r"), worktree_path=root, commit_sha="abcdef1",
                         edit_policy=POLICY)
    return mutation_check(ctx, group, proposal, {"tests/test_core.py": OUTDATED}, timeout_seconds=30,
                          isolation="bwrap", deadline_seconds=seconds)


# --------------------------------------------------------------------------- #
# Cached catalogues
# --------------------------------------------------------------------------- #


def write_catalogue(path: Path, fixture: dict) -> Path:
    path.write_text(json.dumps({"entries": [{"mutation_fixture": fixture,
                                             "outcomes": [{"mutant_id": "mut-01", "outcome": "killed"}]}]}))
    return path


def test_cached_catalogue_loads_only_validated_specifications(tmp_path: Path) -> None:
    specs = mutants()
    fixture = MutationFixture(function_qualname="pkg.core.grade", mutants=specs).model_dump(mode="json")
    loaded = load_cached_mutants(write_catalogue(tmp_path / "c.json", fixture))
    assert loaded == {"pkg.core.grade": specs}  # stored outcomes are not part of what is loaded
    wrong = {**fixture, "mutants": [{**fixture["mutants"][0], "function_qualname": "pkg.core.other"}]}
    with pytest.raises(ValueError, match="names"):
        load_cached_mutants(write_catalogue(tmp_path / "w.json", wrong))
    unsafe = {**fixture, "mutants": [{**fixture["mutants"][0], "file_path": "../outside.py"}]}
    with pytest.raises(ValueError, match="unsafe path"):
        load_cached_mutants(write_catalogue(tmp_path / "u.json", unsafe))


@needs_bwrap
def test_cached_mutants_are_executed_not_trusted(repo: Path, tmp_path: Path) -> None:
    """A catalogue written for other code: its mutants no longer apply and are measured as invalid, never kills."""
    other = "def grade(score):\n    if score > 90:\n        return 'A'\n    return 'B'\n"
    stale = generate_mutants(other, "pkg/core.py", "grade", module="pkg.core")
    fixture = MutationFixture(function_qualname="pkg.core.grade", mutants=stale).model_dump(mode="json")
    loaded = load_cached_mutants(write_catalogue(tmp_path / "c.json", fixture))
    result = run(repo, loaded["pkg.core.grade"])
    assert {o.outcome for o in result.outcomes} == {"invalid"}


def test_no_function_or_no_mutants_leaves_the_strength_unmeasured(repo: Path) -> None:
    ctx = EpisodeContext(episode_id="e", repo=RepoRef(repo_id="r"), worktree_path=repo, commit_sha="abcdef1",
                         edit_policy=POLICY)
    test = FailingTest(test_id="tests/test_core.py::test_pass", test_file="tests/test_core.py", test_name="test_pass")
    proposal = PatchProposal(group_id="g1", decision=RepairDirection.UPDATE_TEST, scope=PatchScope.TEST,
                             target_files=("tests/test_core.py",), unified_diff="--- a/x\n+++ b/x\n", rationale="r",
                             evidence_ids=("ev",))
    for function in (None, "pkg.core.missing", "nowhere.f"):
        group = FailureGroupDetail(group=FailureGroup(group_id="g1", function_qualname=function, test_ids=[test.test_id]),
                                   failing_tests=(test,))
        result = mutation_check(ctx, group, proposal, {}, timeout_seconds=5, isolation="bwrap")
        assert result.gaps and result.problems == () and result.result.status is TestRunStatus.NOT_RUN


# --------------------------------------------------------------------------- #
# The approval rule
# --------------------------------------------------------------------------- #

PASSED = PytestExecution(selectors=("t",), command=(), status=TestRunStatus.PASSED, passed=1, failed=0)
FAILED = PytestExecution(selectors=("t",), command=(), status=TestRunStatus.FAILED, passed=0, failed=1)
MEASURED = MutationResult(status=TestRunStatus.PASSED, mutants_total=5, killed_by_repaired_tests=4)
FIX, UPDATE = RepairDirection.FIX_CODE, RepairDirection.UPDATE_TEST


@pytest.mark.parametrize(
    ("kwargs", "decision", "basis"),
    [
        ({"direction": FIX, "post_repair_run": PASSED}, GuardDecision.APPROVE, "original_tests"),
        ({"direction": FIX, "post_repair_run": FAILED}, GuardDecision.REJECT, None),
        ({"direction": FIX, "post_repair_run": None}, GuardDecision.REJECT, None),
        ({"direction": FIX, "post_repair_run": PASSED, "probe_verification": TrustedVerification(
            status=ProbeStatus.PASSED, probe_id="p")}, GuardDecision.APPROVE, "probe"),
        ({"direction": FIX, "post_repair_run": PASSED, "probe_verification": TrustedVerification(
            status=ProbeStatus.FAILED, probe_id="p", reasons=("case 1",))}, GuardDecision.REJECT, None),
        ({"direction": FIX, "post_repair_run": PASSED, "probe_verification": TrustedVerification(
            status=ProbeStatus.ERROR, probe_id="p", reasons=("timeout",))}, GuardDecision.ESCALATE, None),
        # A green run alone never approves a test edit.
        ({"direction": UPDATE, "post_repair_run": PASSED}, GuardDecision.ESCALATE, None),
        ({"direction": UPDATE, "post_repair_run": PASSED, "probe_verification": TrustedVerification(
            status=ProbeStatus.PASSED, probe_id="p")}, GuardDecision.ESCALATE, None),
        ({"direction": UPDATE, "post_repair_run": PASSED, "mutation": MutationResult()}, GuardDecision.ESCALATE, None),
        ({"direction": UPDATE, "post_repair_run": PASSED, "mutation": MutationResult(status=TestRunStatus.ERROR)},
         GuardDecision.ESCALATE, None),
        ({"direction": UPDATE, "post_repair_run": PASSED, "mutation": MEASURED, "mutation_gaps": ("detect none",)},
         GuardDecision.ESCALATE, None),
        ({"direction": UPDATE, "post_repair_run": PASSED, "mutation": MEASURED, "mutation_problems": ("weakened",)},
         GuardDecision.REJECT, None),
        ({"direction": UPDATE, "post_repair_run": FAILED, "mutation": MEASURED}, GuardDecision.REJECT, None),
        ({"direction": UPDATE, "post_repair_run": PASSED, "mutation": MEASURED}, GuardDecision.APPROVE, "mutation"),
    ],
)
def test_repair_outcome(kwargs: dict, decision: GuardDecision, basis: str | None) -> None:
    direction = kwargs.pop("direction")
    run_ = kwargs.pop("post_repair_run")
    outcome = repair_outcome(direction, run_, **kwargs)
    assert outcome.recommended_decision is decision
    assert outcome.basis == basis
    assert outcome.verified == (decision is GuardDecision.APPROVE)
