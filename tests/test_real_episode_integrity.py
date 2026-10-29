"""Patch scope and memory probes on real pilot episodes (Day 3, Human 2).

Real: the pinned boltons clone, each dev episode's CI state (``scripts/pilot/materialize_episode.py``), the patch
tools, the bubblewrap sandbox with the pilot interpreter, ``JsonlMemoryStore`` and executable probes.  Nothing is
scripted and no model is called.  Dev split only: the test split stays frozen.

Skipped unless ``../boltons`` (at the dataset pin) and ``../boltons-venv`` exist next to this project (README §8a).
"""

from __future__ import annotations

import difflib
import importlib.util
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from testorcode.dataset import episode_path, load_episodes
from testorcode.memory import JsonlMemoryStore
from testorcode.schemas.contracts import EpisodeContext
from testorcode.schemas.episode_schemas import (
    ExecutableProbe,
    MemoryRecord,
    PatchScope,
    RepairDirection,
)
from testorcode.schemas.result_schemas import ProbeStatus
from testorcode.tools import sandbox
from testorcode.tools.patch_tools import apply_patch, revert_patch, validate_patch
from testorcode.tools.probes import probe_from_tests, qualname_of_test, run_executable_probe

ROOT = Path(__file__).resolve().parents[1]
REPO = (ROOT / ".." / "boltons").resolve()
PYTHON = (ROOT / ".." / "boltons-venv" / "bin" / "python").absolute()  # the venv path, not its resolved symlink
PIN = "d31d821d56c8f7756af9c30a864077f2e5855321"


def _pinned() -> bool:
    if not (REPO / ".git").exists() or not PYTHON.exists():
        return False
    found = subprocess.run(["git", "-C", str(REPO), "cat-file", "-e", f"{PIN}^{{commit}}"], capture_output=True, check=False)
    return found.returncode == 0


pytestmark = [
    pytest.mark.skipif(not _pinned(), reason="needs ../boltons at the dataset pin and ../boltons-venv (README §8a)"),
    pytest.mark.skipif(not sandbox.bwrap_available(), reason="bubblewrap unavailable"),
]

_spec = importlib.util.spec_from_file_location("materialize_episode", ROOT / "scripts/pilot/materialize_episode.py")
_materialize = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_materialize)


@pytest.fixture(scope="module")
def dev_episodes():
    return {e.episode_id: e for e in load_episodes(episode_path("dev"))}


@pytest.fixture
def ci_state(tmp_path: Path, dev_episodes):
    """``ci_state("dev-04")`` -> (full episode, worktree at its CI state); worktrees are removed afterwards."""
    made: list[Path] = []

    def build(episode_id: str):
        episode = dev_episodes[episode_id]
        dest = tmp_path / episode_id
        _materialize.materialize(REPO, dest, episode.agent_input.model_dump(mode="json"))
        made.append(dest)
        return episode, dest

    yield build
    for dest in made:
        subprocess.run(["git", "-C", str(REPO), "worktree", "remove", "--force", str(dest)], capture_output=True, check=False)
        shutil.rmtree(dest, ignore_errors=True)
    subprocess.run(["git", "-C", str(REPO), "worktree", "prune"], capture_output=True, check=False)


def file_diff(path: str, old: str, new: str) -> str:
    before = old.splitlines(keepends=True)
    return "".join(difflib.unified_diff(before, new.splitlines(keepends=True),
                                        f"a/{path}" if old else "/dev/null", f"b/{path}"))


def edit(worktree: Path, path: str, change) -> str:
    old = (worktree / path).read_text() if (worktree / path).exists() else ""
    return file_diff(path, old, change(old))


def inverse(diff: str) -> str:
    """The reverse of a unified diff (the fix for an injected mutant)."""
    out = []
    for line in diff.rstrip("\n").splitlines():
        if line.startswith("--- "):
            out.append("+++ " + line[4:].replace("a/", "b/", 1))
        elif line.startswith("+++ "):
            out.append("--- " + line[4:].replace("b/", "a/", 1))
        elif line.startswith("@@"):
            m = re.match(r"@@ -(\d+(?:,\d+)?) \+(\d+(?:,\d+)?) @@(.*)", line)
            out.append(f"@@ -{m.group(2)} +{m.group(1)} @@{m.group(3)}")
        elif line.startswith("+"):
            out.append("-" + line[1:])
        elif line.startswith("-"):
            out.append("+" + line[1:])
        else:
            out.append(line)
    # Header lines must precede each other in the --- / +++ order.
    for i in range(len(out) - 1):
        if out[i].startswith("+++ ") and out[i + 1].startswith("--- "):
            out[i], out[i + 1] = out[i + 1], out[i]
    return "\n".join(out) + "\n"


def changed(line: str) -> bool:
    return line[:1] in "+-" and not line.startswith(("+++ ", "--- "))


def refused(worktree: Path, diff: str, policy, scope: PatchScope) -> tuple[str, ...]:
    result = validate_patch(worktree, diff, policy=policy, scope=scope)
    assert not (result.valid and result.allowed), f"accepted: {diff[:200]}"
    return result.rejection_reasons


# --------------------------------------------------------------------------- #
# Patch scope on a real CI state (dev-04, S2: strutils.strip_ansi)
# --------------------------------------------------------------------------- #

SKIP = "@pytest.mark.skip(reason='flaky')\n"
# The first of test_strip_ansi's five assertions (two physical lines), removed whole so the file still parses.
FIRST_ASSERT = "    assert strutils.strip_ansi(\n        '\\x1b[0m\\x1b[1;36mart\\x1b[46;34m\\xdc') == 'art\\xdc'\n"


def test_patches_outside_the_allowed_scope_are_refused_on_a_real_episode(ci_state) -> None:
    episode, wt = ci_state("dev-04")
    policy = episode.agent_input.edit_policy
    test_file = "tests/test_strutils.py"
    failing = qualname_of_test(episode.agent_input.ci_event.failing_tests[0].test_id)

    blocked = {
        "new conftest": (file_diff("tests/conftest.py", "", "collect_ignore = ['test_strutils.py']\n"), PatchScope.TEST),
        "pyproject": (edit(wt, "pyproject.toml", lambda t: t + "\n# x\n"), PatchScope.SRC),
        "setup.cfg": (edit(wt, "setup.cfg", lambda t: t + "\n# x\n"), PatchScope.SRC),
        "tox.ini": (edit(wt, "tox.ini", lambda t: t + "\n# x\n"), PatchScope.TEST),
        "CI workflow": (edit(wt, ".github/workflows/tests.yaml", lambda t: t + "\n# x\n"), PatchScope.SRC),
        "docs outside both roots": (edit(wt, "docs/strutils.rst", lambda t: t + "\nx\n"), PatchScope.SRC),
        "test file under SRC scope": (edit(wt, test_file, lambda t: t + "\n# x\n"), PatchScope.SRC),
        "source file under TEST scope": (edit(wt, "boltons/strutils.py", lambda t: t + "\n# x\n"), PatchScope.TEST),
        "skip marker": (edit(wt, test_file, lambda t: t.replace(f"def {failing}(", SKIP + f"def {failing}(", 1)),
                        PatchScope.TEST),
        "deleted assertion": (edit(wt, test_file, lambda t: t.replace(FIRST_ASSERT, "", 1)), PatchScope.TEST),
    }
    mixed = edit(wt, "boltons/strutils.py", lambda t: t + "\n# x\n") + edit(wt, test_file, lambda t: t + "\n# x\n")
    blocked["mixed source and test"] = (mixed, PatchScope.SRC)

    empty = [n for n, (d, _) in blocked.items() if not any(changed(line) for line in d.splitlines())]
    assert empty == []  # every attack is a real edit, so no refusal is vacuous
    reasons = {name: refused(wt, diff, policy, scope) for name, (diff, scope) in blocked.items()}
    assert all(reasons.values())
    assert any("not limited to expected literal values" in r for r in reasons["deleted assertion"])
    assert any("not limited to expected literal values" in r for r in reasons["skip marker"])
    assert any("forbidden file" in r for r in reasons["new conftest"] + reasons["CI workflow"])
    assert any("outside the approved" in r for r in reasons["docs outside both roots"])
    assert subprocess.run(["git", "-C", str(wt), "status", "--porcelain", "--untracked-files=all"],
                          capture_output=True, text=True, check=False).stdout.strip() == "M boltons/strutils.py"  # CI diff only


def test_in_scope_repairs_are_accepted_on_real_episodes(ci_state) -> None:
    for episode_id in ("dev-02", "dev-03", "dev-04"):  # S2: the fix is the inverse of the injected mutant
        episode, wt = ci_state(episode_id)
        [mutant] = episode.ground_truth.injected_mutants
        fix = inverse(mutant.unified_diff)
        result = validate_patch(wt, fix, policy=episode.agent_input.edit_policy, scope=PatchScope.SRC)
        assert result.valid and result.allowed, (episode_id, result.rejection_reasons)


def test_developer_test_repairs_fall_outside_the_agent_edit_class(ci_state) -> None:
    """Documented limitation: the reference repairs of dev-06 add tests, so the literal-only test-patch rule refuses
    them (the Repairer must find the literal or setup-keyword part); the rule is a tool-level safety boundary."""
    episode, wt = ci_state("dev-06")
    [reference] = episode.ground_truth.reference_repairs
    reasons = refused(wt, reference.unified_diff, episode.agent_input.edit_policy, PatchScope.TEST)
    assert any("not limited to expected literal values" in r for r in reasons)


# --------------------------------------------------------------------------- #
# Memory probes on real code
# --------------------------------------------------------------------------- #


def store(tmp_path: Path, name: str) -> JsonlMemoryStore:
    return JsonlMemoryStore(tmp_path / f"{name}.jsonl", python_executable=PYTHON, timeout_seconds=120,
                            namespace=f"test/{name}")


def test_seeded_s4_decoy_record_passes_on_the_mutated_code_as_designed(ci_state, tmp_path) -> None:
    episode, wt = ci_state("dev-07")
    [record] = episode.agent_input.visible_memory
    memory = store(tmp_path, "dev-07")
    assert memory.seed([record], worktree_path=wt) == [record.record_id]
    context = EpisodeContext.from_episode_input(episode.agent_input, wt)
    [check] = memory.read(context, record.function_qualname).checks
    # A passing probe shows the approved behaviour still holds, not that the new failure is that change (S4).
    assert check.probe_status is ProbeStatus.PASSED and check.valid_as_evidence


def test_probe_from_a_fix_passes_after_it_fails_with_the_regression_and_goes_stale(ci_state, tmp_path) -> None:
    episode, wt = ci_state("dev-04")
    ai = episode.agent_input
    policy = ai.edit_policy
    [mutant] = episode.ground_truth.injected_mutants
    test_id = ai.ci_event.failing_tests[0].test_id
    test_file = test_id.split("::", 1)[0]
    probe_code = probe_from_tests((wt / test_file).read_text(), [qualname_of_test(test_id)])
    record = MemoryRecord(
        record_id="dev-04-g1", repo_id=ai.repo.repo_id, function_qualname=mutant.function_qualname,
        behaviour_signature="strip_ansi keeps the input type", direction=RepairDirection.FIX_CODE,
        probe=ExecutableProbe(probe_id="p-dev-04", probe_code=probe_code, probe_selector=test_id),
        evidence_ids=["msg:bba729bf4539"], approved_at_commit=ai.ci_event.commit.sha,
    )
    context = EpisodeContext.from_episode_input(ai, wt)
    memory = store(tmp_path, "dev-04")
    memory.seed([record], worktree_path=wt)

    fix = inverse(mutant.unified_diff)
    apply_patch(wt, fix, policy=policy, scope=PatchScope.SRC)
    [fixed] = memory.read(context, mutant.function_qualname).checks
    assert fixed.probe_status is ProbeStatus.PASSED and fixed.valid_as_evidence

    revert_patch(wt, fix, policy=policy, scope=PatchScope.SRC)  # the regression is back
    # A bent test file and an added conftest cannot rescue the probe: it runs on the source roots only.
    (wt / test_file).write_text("def test_nothing():\n    pass\n")
    (wt / "tests" / "conftest.py").write_text("import pytest\ncollect_ignore_glob = ['*']\n")
    direct = run_executable_probe(wt, memory.records()[0].probe, policy=policy, timeout_seconds=120,
                                  python_executable=PYTHON)
    assert direct.status is ProbeStatus.FAILED

    first, second = (memory.read(context, mutant.function_qualname).checks[0] for _ in range(2))
    assert first.probe_status is ProbeStatus.FAILED and first.blocked_reason == "probe_failed"
    assert second.record.flagged_stale and not second.valid_as_evidence
    [third] = memory.read(context, mutant.function_qualname).checks  # stale: blocked without running again
    assert third.probe_status is ProbeStatus.NOT_RUN and third.blocked_reason == "probe_failed"
