"""Patch validation, application and reversal on real temporary git repositories."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from testorcode.errors import PatchError, ScopeViolationError, ToolError
from testorcode.schemas.episode_schemas import DecisionLabel, EditPolicy, PatchScope
from testorcode.tools.patch_tools import (
    apply_patch,
    is_protected_path,
    revert_patch,
    scope_for_decision,
    validate_patch,
)

POLICY = EditPolicy()
SRC, TEST = PatchScope.SRC, PatchScope.TEST

FILES = {
    "src/pkg/mod.py": "def add(a, b):\n    return a + b\n\n\ndef sub(a, b):\n    return a - b\n",
    "tests/test_mod.py": "from pkg.mod import add\n\n\ndef test_add():\n    assert add(1, 2) == 3\n",
    "tests/conftest.py": "import pytest\n",
    "conftest.py": "",
    "pyproject.toml": "[project]\nname = 'fixture'\n",
    "setup.cfg": "[metadata]\n",
    "requirements.txt": "pytest\n",
    ".github/workflows/ci.yml": "on: push\n",
    "docs/readme.md": "docs\n",
}


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    for rel, text in FILES.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)
    git(root, "init", "-q")
    git(root, "add", ".")
    git(root, "-c", "user.name=t", "-c", "user.email=t@example.invalid", "commit", "-qm", "init")
    return root


def diff_for(repo: Path, rel: str, new_text: str) -> str:
    """A real ``git diff`` for editing ``rel``; the file is restored afterwards."""
    path = repo / rel
    old = path.read_text()
    path.write_text(new_text)
    try:
        return git(repo, "diff", "--", rel)
    finally:
        path.write_text(old)


def new_file_diff(rel: str, text: str = "x = 1\n") -> str:
    lines = text.splitlines()
    body = "".join(f"+{line}\n" for line in lines)
    return (
        f"diff --git a/{rel} b/{rel}\nnew file mode 100644\n--- /dev/null\n+++ b/{rel}\n"
        f"@@ -0,0 +1,{len(lines)} @@\n{body}"
    )


def snapshot(repo: Path) -> dict[str, bytes]:
    """Every file outside .git, so 'nothing changed' can be checked exactly."""
    return {
        str(p.relative_to(repo)): p.read_bytes()
        for p in repo.rglob("*")
        if p.is_file() and ".git" not in p.relative_to(repo).parts
    }


SRC_FIX = FILES["src/pkg/mod.py"].replace("return a - b", "return a - b  # fixed")
# The one supported kind of test-only patch: a changed expected literal.
TEST_FIX = FILES["tests/test_mod.py"].replace("== 3", "== 4")


# --------------------------------------------------------------------------- #
# Scope
# --------------------------------------------------------------------------- #


def test_source_patch_is_valid_under_fix_code(repo: Path) -> None:
    result = validate_patch(
        repo, diff_for(repo, "src/pkg/mod.py", SRC_FIX), policy=POLICY, scope=SRC
    )
    assert result.ok and result.valid and result.allowed
    assert result.changed_paths == ("src/pkg/mod.py",)
    assert result.rejection_reasons == ()


def test_test_patch_is_valid_under_update_test(repo: Path) -> None:
    result = validate_patch(
        repo, diff_for(repo, "tests/test_mod.py", TEST_FIX), policy=POLICY, scope=TEST
    )
    assert result.ok
    assert result.changed_paths == ("tests/test_mod.py",)


def test_source_patch_is_rejected_under_update_test(repo: Path) -> None:
    result = validate_patch(
        repo, diff_for(repo, "src/pkg/mod.py", SRC_FIX), policy=POLICY, scope=TEST
    )
    assert result.valid and not result.allowed and not result.ok
    assert any("source file not editable" in r for r in result.rejection_reasons)


def test_test_patch_is_rejected_under_fix_code(repo: Path) -> None:
    result = validate_patch(
        repo, diff_for(repo, "tests/test_mod.py", TEST_FIX), policy=POLICY, scope=SRC
    )
    assert not result.allowed
    assert any("test file not editable" in r for r in result.rejection_reasons)


@pytest.mark.parametrize(
    "rel",
    [
        "tests/conftest.py",
        "conftest.py",
        ".github/workflows/ci.yml",
        "pyproject.toml",
        "setup.cfg",
        "requirements.txt",
    ],
)
@pytest.mark.parametrize("scope", [SRC, TEST])
def test_forbidden_files_are_rejected_in_every_scope(
    repo: Path, rel: str, scope: PatchScope
) -> None:
    diff = diff_for(repo, rel, (repo / rel).read_text() + "# changed\n")
    result = validate_patch(repo, diff, policy=POLICY, scope=scope)
    assert not result.allowed
    assert any(f"forbidden file: {rel}" == r for r in result.rejection_reasons)


@pytest.mark.parametrize(
    "rel", ["src/conftest.py", "src/pkg/conftest.py", ".github/new.yml", "Dockerfile"]
)
def test_new_forbidden_files_are_rejected(repo: Path, rel: str) -> None:
    result = validate_patch(repo, new_file_diff(rel), policy=POLICY, scope=SRC)
    assert not result.allowed
    assert any("forbidden file" in r for r in result.rejection_reasons)


def test_paths_outside_approved_roots_are_rejected(repo: Path) -> None:
    result = validate_patch(
        repo, diff_for(repo, "docs/readme.md", "changed\n"), policy=POLICY, scope=SRC
    )
    assert not result.allowed
    assert any("outside the approved" in r for r in result.rejection_reasons)


def test_escalate_never_maps_to_a_patch_scope() -> None:
    assert scope_for_decision(DecisionLabel.FIX_CODE) is SRC
    assert scope_for_decision(DecisionLabel.UPDATE_TEST) is TEST
    with pytest.raises(ScopeViolationError):
        scope_for_decision(DecisionLabel.ESCALATE)


def test_escalate_scope_cannot_apply_a_patch(repo: Path) -> None:
    before = snapshot(repo)
    diff = diff_for(repo, "src/pkg/mod.py", SRC_FIX)
    with pytest.raises(ScopeViolationError):
        apply_patch(repo, diff, policy=POLICY, scope=DecisionLabel.ESCALATE)  # type: ignore[arg-type]
    with pytest.raises(ScopeViolationError):
        validate_patch(repo, diff, policy=POLICY, scope="escalate")  # type: ignore[arg-type]
    assert snapshot(repo) == before


# --------------------------------------------------------------------------- #
# Unsafe paths
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("header_path", "reason"),
    [
        ("src/../.github/workflows/ci.yml", "traversal"),
        ("src/../../outside.py", "traversal"),
        ("src/./pkg/mod.py", "traversal"),
        ("src//pkg/x.py", "traversal"),
        ("/etc/passwd", "absolute"),
        ("src/.git/config", ".git"),
        ("src\\pkg\\x.py", "backslash"),
    ],
)
def test_unsafe_paths_are_rejected(repo: Path, header_path: str, reason: str) -> None:
    diff = f"--- /dev/null\n+++ b/{header_path}\n@@ -0,0 +1 @@\n+x = 1\n"
    result = validate_patch(repo, diff, policy=POLICY, scope=SRC)
    assert not result.allowed and not result.ok
    assert any(reason in r for r in result.rejection_reasons), result.rejection_reasons


def test_traversal_bypass_of_the_src_prefix_cannot_be_applied(repo: Path) -> None:
    """``src/../.github/...`` starts with ``src/`` but must never be treated as source."""
    diff = (
        "--- a/src/../.github/workflows/ci.yml\n+++ b/src/../.github/workflows/ci.yml\n"
        "@@ -1 +1 @@\n-on: push\n+on: [push, pull_request]\n"
    )
    before = snapshot(repo)
    with pytest.raises(ScopeViolationError):
        apply_patch(repo, diff, policy=POLICY, scope=SRC)
    assert snapshot(repo) == before


def test_old_path_without_prefix_is_rejected(repo: Path) -> None:
    diff = "--- /etc/passwd\n+++ b/src/pkg/x.py\n@@ -1 +1 @@\n-a\n+b\n"
    result = validate_patch(repo, diff, policy=POLICY, scope=SRC)
    assert not result.ok
    assert any("prefix" in r for r in result.rejection_reasons)


def test_symlink_directory_escape_is_rejected(repo: Path, tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (repo / "src" / "escape").symlink_to(outside, target_is_directory=True)
    diff = new_file_diff("src/escape/pwned.py")
    result = validate_patch(repo, diff, policy=POLICY, scope=SRC)
    assert not result.allowed
    assert any("symlink" in r for r in result.rejection_reasons)
    with pytest.raises(ScopeViolationError):
        apply_patch(repo, diff, policy=POLICY, scope=SRC)
    assert list(outside.iterdir()) == []


def test_symlink_file_escape_is_rejected(repo: Path, tmp_path: Path) -> None:
    target = tmp_path / "victim.py"
    target.write_text("a\n")
    (repo / "src" / "pkg" / "link.py").symlink_to(target)
    diff = "--- a/src/pkg/link.py\n+++ b/src/pkg/link.py\n@@ -1 +1 @@\n-a\n+b\n"
    with pytest.raises(ScopeViolationError, match="symlink"):
        apply_patch(repo, diff, policy=POLICY, scope=SRC)
    assert target.read_text() == "a\n"


@pytest.mark.parametrize(
    ("header", "reason"),
    [
        ("new file mode 120000", "unsupported file mode"),  # symlink
        ("new file mode 160000", "unsupported file mode"),  # submodule
        ("new file mode 100755", "unsupported file mode"),  # executable
    ],
)
def test_unsafe_file_modes_are_rejected(repo: Path, header: str, reason: str) -> None:
    diff = (
        f"diff --git a/src/pkg/x b/src/pkg/x\n{header}\n--- /dev/null\n+++ b/src/pkg/x\n"
        "@@ -0,0 +1 @@\n+/etc/passwd\n"
    )
    result = validate_patch(repo, diff, policy=POLICY, scope=SRC)
    assert not result.ok
    assert any(reason in r for r in result.rejection_reasons)


def test_mode_change_and_binary_patches_are_rejected(repo: Path) -> None:
    mode = "diff --git a/src/pkg/mod.py b/src/pkg/mod.py\nold mode 100644\nnew mode 100755\n"
    binary = (
        "diff --git a/src/pkg/b.bin b/src/pkg/b.bin\nnew file mode 100644\n"
        "index 0000000..1111111\nGIT binary patch\nliteral 1\nIcmZpX0000\n\nliteral 0\n"
        "HcmV?d00001\n"
    )
    assert any(
        "mode" in r for r in validate_patch(repo, mode, policy=POLICY, scope=SRC).rejection_reasons
    )
    assert any(
        "binary" in r
        for r in validate_patch(repo, binary, policy=POLICY, scope=SRC).rejection_reasons
    )


# --------------------------------------------------------------------------- #
# Malformed, non-applicable and multi-file diffs
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "diff",
    [
        "",
        "   \n",
        "this is not a diff\n",
        "--- a/src/pkg/mod.py\n+++ b/src/pkg/mod.py\n@@ garbage @@\n",
        "--- a/src/pkg/mod.py\n+++ b/src/pkg/mod.py\n@@ -1,3 +1,3 @@\n def add(a, b):\n",
        "--- a/src/pkg/mod.py\n+++ b/src/pkg/mod.py\n@@ -1 +1 @@\n-def add(a, b):\n+x\n*bad\n",
    ],
    ids=["empty", "blank", "prose", "bad-hunk-header", "truncated-hunk", "extra-garbage"],
)
def test_malformed_diffs_are_invalid_and_cannot_be_applied(repo: Path, diff: str) -> None:
    before = snapshot(repo)
    result = validate_patch(repo, diff, policy=POLICY, scope=SRC)
    assert not result.valid and not result.ok and result.rejection_reasons
    with pytest.raises(PatchError):
        apply_patch(repo, diff, policy=POLICY, scope=SRC)
    assert snapshot(repo) == before


def test_removed_line_starting_with_dashes_is_not_a_header(repo: Path) -> None:
    (repo / "src/pkg/notes.py").write_text("-- not a header\nkeep\n")
    git(repo, "add", ".")
    git(repo, "-c", "user.name=t", "-c", "user.email=t@example.invalid", "commit", "-qm", "notes")
    diff = diff_for(repo, "src/pkg/notes.py", "keep\n")
    assert "\n--- not a header" in diff
    result = validate_patch(repo, diff, policy=POLICY, scope=SRC)
    assert result.ok, result.rejection_reasons
    assert result.changed_paths == ("src/pkg/notes.py",)


def test_non_applicable_diff_is_invalid_but_in_scope(repo: Path) -> None:
    diff = "--- a/src/pkg/mod.py\n+++ b/src/pkg/mod.py\n@@ -1 +1 @@\n-def nope():\n+def yes():\n"
    before = snapshot(repo)
    result = validate_patch(repo, diff, policy=POLICY, scope=SRC)
    assert result.allowed and not result.valid
    assert any("cannot be applied cleanly" in r for r in result.rejection_reasons)
    with pytest.raises(PatchError) as info:
        apply_patch(repo, diff, policy=POLICY, scope=SRC)
    assert not isinstance(info.value, ScopeViolationError)
    assert snapshot(repo) == before


def test_multi_file_patch_with_one_forbidden_file_is_rejected_whole(repo: Path) -> None:
    diff = diff_for(repo, "src/pkg/mod.py", SRC_FIX) + diff_for(
        repo, "tests/conftest.py", "import pytest\npytest.skip('all', allow_module_level=True)\n"
    )
    before = snapshot(repo)
    result = validate_patch(repo, diff, policy=POLICY, scope=SRC)
    assert result.changed_paths == ("src/pkg/mod.py", "tests/conftest.py")
    assert not result.allowed
    with pytest.raises(ScopeViolationError):
        apply_patch(repo, diff, policy=POLICY, scope=SRC)
    assert snapshot(repo) == before


def test_mixed_source_and_test_patch_is_rejected_in_both_scopes(repo: Path) -> None:
    diff = diff_for(repo, "src/pkg/mod.py", SRC_FIX) + diff_for(repo, "tests/test_mod.py", TEST_FIX)
    for scope in (SRC, TEST):
        result = validate_patch(repo, diff, policy=POLICY, scope=scope)
        assert result.valid and not result.allowed


def test_partially_applicable_multi_file_patch_changes_nothing(repo: Path) -> None:
    good = diff_for(repo, "src/pkg/mod.py", SRC_FIX)
    bad = "--- a/src/pkg/other.py\n+++ b/src/pkg/other.py\n@@ -1 +1 @@\n-a\n+b\n"
    before = snapshot(repo)
    with pytest.raises(PatchError):
        apply_patch(repo, good + bad, policy=POLICY, scope=SRC)
    assert snapshot(repo) == before


def test_rename_checks_both_old_and_new_paths(repo: Path) -> None:
    rename = (
        "diff --git a/tests/test_mod.py b/src/pkg/test_mod.py\nsimilarity index 100%\n"
        "rename from tests/test_mod.py\nrename to src/pkg/test_mod.py\n"
    )
    result = validate_patch(repo, rename, policy=POLICY, scope=SRC)
    assert set(result.changed_paths) == {"tests/test_mod.py", "src/pkg/test_mod.py"}
    assert not result.allowed
    assert any("tests/test_mod.py" in r for r in result.rejection_reasons)


def test_rename_within_scope_is_allowed(repo: Path) -> None:
    rename = (
        "diff --git a/src/pkg/mod.py b/src/pkg/core.py\nsimilarity index 100%\n"
        "rename from src/pkg/mod.py\nrename to src/pkg/core.py\n"
    )
    result = validate_patch(repo, rename, policy=POLICY, scope=SRC)
    assert result.ok, result.rejection_reasons
    apply_patch(repo, rename, policy=POLICY, scope=SRC)
    assert (repo / "src/pkg/core.py").exists() and not (repo / "src/pkg/mod.py").exists()
    revert_patch(repo, rename)
    assert (repo / "src/pkg/mod.py").read_text() == FILES["src/pkg/mod.py"]
    assert not (repo / "src/pkg/core.py").exists()


# --------------------------------------------------------------------------- #
# Side effects, apply and revert
# --------------------------------------------------------------------------- #


def test_validation_never_modifies_files(repo: Path) -> None:
    before, status = snapshot(repo), git(repo, "status", "--porcelain")
    for diff in (
        diff_for(repo, "src/pkg/mod.py", SRC_FIX),
        new_file_diff("src/pkg/new.py"),
        "junk",
    ):
        for scope in (SRC, TEST):
            validate_patch(repo, diff, policy=POLICY, scope=scope)
    assert snapshot(repo) == before
    assert git(repo, "status", "--porcelain") == status


def test_apply_then_revert_a_valid_patch(repo: Path) -> None:
    diff = diff_for(repo, "src/pkg/mod.py", SRC_FIX)
    result = apply_patch(repo, diff, policy=POLICY, scope=SRC)
    assert result.ok and result.changed_paths == ("src/pkg/mod.py",)
    assert (repo / "src/pkg/mod.py").read_text() == SRC_FIX
    assert git(repo, "diff", "--cached") == "", "the index must not be touched"
    revert_patch(repo, diff, policy=POLICY, scope=SRC)
    assert (repo / "src/pkg/mod.py").read_text() == FILES["src/pkg/mod.py"]
    assert git(repo, "status", "--porcelain") == ""


def test_apply_and_revert_a_test_expectation_update(repo: Path) -> None:
    diff = diff_for(repo, "tests/test_mod.py", TEST_FIX)
    apply_patch(repo, diff, policy=POLICY, scope=TEST)
    assert (repo / "tests/test_mod.py").read_text() == TEST_FIX
    revert_patch(repo, diff)
    assert git(repo, "status", "--porcelain") == ""


def test_update_test_cannot_add_a_test_file(repo: Path) -> None:
    diff = new_file_diff("tests/test_new.py", "def test_new():\n    assert True")
    with pytest.raises(ScopeViolationError, match="may not add files"):
        apply_patch(repo, diff, policy=POLICY, scope=TEST)
    assert not (repo / "tests/test_new.py").exists()


def test_revert_preserves_unrelated_changes(repo: Path) -> None:
    diff = diff_for(repo, "src/pkg/mod.py", SRC_FIX)
    apply_patch(repo, diff, policy=POLICY, scope=SRC)
    # Unrelated edits: another file, an untracked file, and new lines above the hunk.
    (repo / "docs/readme.md").write_text("user edit\n")
    (repo / "scratch.txt").write_text("untracked\n")
    mod = repo / "src/pkg/mod.py"
    mod.write_text("USER_CONSTANT = 1\n\n\n" + mod.read_text())
    revert_patch(repo, diff)
    assert mod.read_text() == "USER_CONSTANT = 1\n\n\n" + FILES["src/pkg/mod.py"]
    assert (repo / "docs/readme.md").read_text() == "user edit\n"
    assert (repo / "scratch.txt").read_text() == "untracked\n"


def test_revert_fails_explicitly_when_the_patched_lines_changed(repo: Path) -> None:
    diff = diff_for(repo, "src/pkg/mod.py", SRC_FIX)
    apply_patch(repo, diff, policy=POLICY, scope=SRC)
    mod = repo / "src/pkg/mod.py"
    edited = mod.read_text().replace("# fixed", "# fixed differently")
    mod.write_text(edited)
    with pytest.raises(PatchError, match="reverted cleanly"):
        revert_patch(repo, diff)
    assert mod.read_text() == edited


def test_revert_of_a_patch_that_was_never_applied_fails(repo: Path) -> None:
    before = snapshot(repo)
    with pytest.raises(PatchError):
        revert_patch(repo, diff_for(repo, "src/pkg/mod.py", SRC_FIX))
    assert snapshot(repo) == before


def test_revert_rechecks_scope_when_given(repo: Path) -> None:
    diff = diff_for(repo, "src/pkg/mod.py", SRC_FIX)
    apply_patch(repo, diff, policy=POLICY, scope=SRC)
    with pytest.raises(ScopeViolationError):
        revert_patch(repo, diff, policy=POLICY, scope=TEST)
    assert (repo / "src/pkg/mod.py").read_text() == SRC_FIX


def test_apply_preserves_unrelated_uncommitted_changes(repo: Path) -> None:
    (repo / "docs/readme.md").write_text("user edit\n")
    apply_patch(repo, diff_for(repo, "src/pkg/mod.py", SRC_FIX), policy=POLICY, scope=SRC)
    assert (repo / "docs/readme.md").read_text() == "user edit\n"


def test_worktree_must_be_a_git_root(repo: Path, tmp_path: Path) -> None:
    diff = diff_for(repo, "src/pkg/mod.py", SRC_FIX)
    with pytest.raises(ToolError, match="not the root"):
        validate_patch(repo / "src", diff, policy=POLICY, scope=SRC)
    plain = tmp_path / "plain"
    plain.mkdir()
    with pytest.raises(ToolError, match="not a git work tree"):
        apply_patch(plain, diff, policy=POLICY, scope=SRC)
    with pytest.raises(ToolError, match="does not exist"):
        validate_patch(tmp_path / "missing", diff, policy=POLICY, scope=SRC)


def test_rejection_reasons_do_not_contain_patch_content(repo: Path) -> None:
    secret_line = "SECRET_TOKEN_VALUE_123"
    diff = new_file_diff("tests/conftest.py".replace("tests/", ".github/"), secret_line)
    result = validate_patch(repo, diff, policy=POLICY, scope=SRC)
    assert not result.ok
    assert all(secret_line not in r for r in result.rejection_reasons)


def test_fix_code_patch_turns_the_run_green_and_revert_restores_it(tmp_path: Path) -> None:
    """The tool chain the orchestrator will use: run -> validate -> apply -> run -> revert."""
    from testorcode.schemas.result_schemas import TestRunStatus
    from testorcode.tools.pytest_runner import run_tests

    root = tmp_path / "episode"
    files = {
        "pyproject.toml": "[tool.pytest.ini_options]\npythonpath = ['src']\n",
        "src/calc.py": "def double(x):\n    return x + x + 1\n",
        "tests/test_calc.py": "from calc import double\n\n\ndef test_double():\n    assert double(2) == 4\n",
    }
    for rel, text in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)
    git(root, "init", "-q")
    git(root, "add", ".")
    git(root, "-c", "user.name=t", "-c", "user.email=t@example.invalid", "commit", "-qm", "init")

    selector = ["tests/test_calc.py::test_double"]
    before = run_tests(root, selector, timeout_seconds=60)
    assert before.status is TestRunStatus.FAILED
    assert before.failing_test_ids == selector

    diff = diff_for(root, "src/calc.py", "def double(x):\n    return x + x\n")
    assert validate_patch(root, diff, policy=POLICY, scope=SRC).ok
    assert not validate_patch(root, diff, policy=POLICY, scope=TEST).ok
    apply_patch(root, diff, policy=POLICY, scope=SRC)
    assert run_tests(root, selector, timeout_seconds=60).status is TestRunStatus.PASSED

    revert_patch(root, diff, policy=POLICY, scope=SRC)
    assert git(root, "status", "--porcelain") == ""
    assert run_tests(root, selector, timeout_seconds=60).status is TestRunStatus.FAILED


def test_every_header_name_is_checked_when_git_headers_disagree(repo: Path) -> None:
    """git writes the ---/+++ file; the ``diff --git`` names are checked as well."""
    to_ci = (
        "diff --git a/src/pkg/mod.py b/src/pkg/mod.py\n"
        "--- a/.github/workflows/ci.yml\n+++ b/.github/workflows/ci.yml\n"
        "@@ -1 +1 @@\n-on: push\n+on: [push]\n"
    )
    from_ci = (
        "diff --git a/.github/workflows/ci.yml b/.github/workflows/ci.yml\n"
        + diff_for(repo, "src/pkg/mod.py", SRC_FIX).split("\n", 2)[2]
    )
    before = snapshot(repo)
    for diff in (to_ci, from_ci):
        result = validate_patch(repo, diff, policy=POLICY, scope=SRC)
        assert ".github/workflows/ci.yml" in result.changed_paths
        assert not result.allowed
        with pytest.raises(ScopeViolationError):
            apply_patch(repo, diff, policy=POLICY, scope=SRC)
    assert snapshot(repo) == before


# --------------------------------------------------------------------------- #
# Regression: protected configuration files in any directory
# --------------------------------------------------------------------------- #

NESTED_PROTECTED = [
    "tests/pytest.ini",
    "tests/.pytest.ini",
    "tests/tox.ini",
    "tests/setup.cfg",
    "tests/pyproject.toml",
    "tests/unit/conftest.py",
    "tests/PyTest.ini",
    "tests/Makefile",
    "tests/.github/workflows/ci.yml",
    "src/pyproject.toml",
    "src/pkg/setup.py",
    "src/pkg/requirements-dev.txt",
    "src/pkg/poetry.lock",
    "src/sitecustomize.py",
    "src/.coveragerc",
]


@pytest.mark.parametrize("rel", NESTED_PROTECTED)
@pytest.mark.parametrize("scope", [SRC, TEST])
def test_nested_protected_files_are_rejected(repo: Path, rel: str, scope: PatchScope) -> None:
    diff = new_file_diff(rel, "[pytest]")
    before = snapshot(repo)
    result = validate_patch(repo, diff, policy=POLICY, scope=scope)
    assert not result.allowed
    assert f"forbidden file: {rel}" in result.rejection_reasons
    with pytest.raises(ScopeViolationError):
        apply_patch(repo, diff, policy=POLICY, scope=scope)
    assert snapshot(repo) == before


def test_protected_files_are_rejected_even_with_a_permissive_policy(repo: Path) -> None:
    permissive = EditPolicy(allowed_test_roots=("tests/",), forbidden_globs=())
    result = validate_patch(repo, new_file_diff("tests/pytest.ini"), policy=permissive, scope=TEST)
    assert not result.allowed


@pytest.mark.parametrize(
    "rel",
    [
        "src/pkg/settings.py",
        "src/pkg/toml_utils.py",
        "src/pkg/lockfile.py",
    ],
)
def test_ordinary_files_with_config_like_names_stay_editable(repo: Path, rel: str) -> None:
    assert validate_patch(repo, new_file_diff(rel), policy=POLICY, scope=SRC).ok


def test_is_protected_path() -> None:
    assert is_protected_path("tests/pytest.ini")
    assert is_protected_path("a/b/c/TOX.INI")
    assert is_protected_path("src/.github/x.yml")
    assert not is_protected_path("src/pkg/github.py")
    assert not is_protected_path("tests/test_pyproject_toml.py")
    assert not is_protected_path("tests/test_setup.py")
    assert not is_protected_path("tests/test_config.py")


def test_nested_pytest_ini_cannot_hide_a_failing_test(tmp_path: Path) -> None:
    """The verified bypass: an UPDATE_TEST patch adding tests/pytest.ini with ``-k``."""
    from testorcode.schemas.result_schemas import TestRunStatus
    from testorcode.tools.pytest_runner import run_tests

    root = tmp_path / "episode"
    (root / "tests").mkdir(parents=True)
    (root / "tests/test_a.py").write_text(
        "def test_good():\n    assert True\n\n\ndef test_bad():\n    assert False\n"
    )
    git(root, "init", "-q")
    git(root, "add", ".")
    git(root, "-c", "user.name=t", "-c", "user.email=t@example.invalid", "commit", "-qm", "init")
    diff = new_file_diff("tests/pytest.ini", '[pytest]\naddopts = -k "not bad"')

    assert not validate_patch(root, diff, policy=POLICY, scope=TEST).allowed
    with pytest.raises(ScopeViolationError):
        apply_patch(root, diff, policy=POLICY, scope=TEST)
    assert not (root / "tests/pytest.ini").exists()
    assert run_tests(root, ["tests"], timeout_seconds=60).status is TestRunStatus.FAILED


# --------------------------------------------------------------------------- #
# Regression: revert_patch with default arguments
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("rel", "old_line"),
    [
        (".github/workflows/ci.yml", "on: evil"),
        ("tests/conftest.py", "import os"),
        ("pyproject.toml", "[evil]"),
        ("docs/readme.md", "evil"),
    ],
)
def test_revert_with_default_arguments_rejects_forbidden_files(
    repo: Path, rel: str, old_line: str
) -> None:
    """A crafted 'revert' must not become a way to edit protected files."""
    current = (repo / rel).read_text().splitlines()[0]
    diff = f"--- a/{rel}\n+++ b/{rel}\n@@ -1 +1 @@\n-{old_line}\n+{current}\n"
    before = snapshot(repo)
    with pytest.raises(ScopeViolationError):
        revert_patch(repo, diff)
    assert snapshot(repo) == before


def test_revert_with_default_arguments_still_reverts_allowed_patches(repo: Path) -> None:
    src = diff_for(repo, "src/pkg/mod.py", SRC_FIX)
    test = diff_for(repo, "tests/test_mod.py", TEST_FIX)
    apply_patch(repo, src, policy=POLICY, scope=SRC)
    apply_patch(repo, test, policy=POLICY, scope=TEST)
    revert_patch(repo, src)
    revert_patch(repo, test)
    assert git(repo, "status", "--porcelain") == ""


def test_failed_revert_leaves_unrelated_files_unchanged(repo: Path) -> None:
    good = diff_for(repo, "src/pkg/mod.py", SRC_FIX)
    apply_patch(repo, good, policy=POLICY, scope=SRC)
    (repo / "docs/readme.md").write_text("user edit\n")
    (repo / "scratch.txt").write_text("untracked\n")
    never_applied = "--- a/src/pkg/other.py\n+++ b/src/pkg/other.py\n@@ -1 +1 @@\n-a\n+b\n"
    before = snapshot(repo)
    with pytest.raises(PatchError):
        revert_patch(repo, good + never_applied)
    assert snapshot(repo) == before
    assert (repo / "src/pkg/mod.py").read_text() == SRC_FIX


# --------------------------------------------------------------------------- #
# Regression: shadowing production modules from test roots
# --------------------------------------------------------------------------- #


def _episode(tmp_path: Path, files: dict[str, str]) -> Path:
    root = tmp_path / "episode"
    for rel, text in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)
    git(root, "init", "-q")
    git(root, "add", ".")
    git(root, "-c", "user.name=t", "-c", "user.email=t@example.invalid", "commit", "-qm", "init")
    return root


CALC = {
    "pyproject.toml": "[tool.pytest.ini_options]\npythonpath = ['src']\n",
    "src/calc.py": "def double(x):\n    return x + x + 1\n",
    "tests/test_calc.py": "from calc import double\n\n\ndef test_double():\n    assert double(2) == 4\n",
}


def test_shadow_module_attack_is_blocked(tmp_path: Path) -> None:
    """The verified bypass: tests/calc.py made ``from calc import double`` load a fake."""
    from testorcode.schemas.result_schemas import TestRunStatus
    from testorcode.tools.pytest_runner import run_tests

    root = _episode(tmp_path, CALC)
    diff = new_file_diff("tests/calc.py", "def double(x):\n    return 2 * x")
    result = validate_patch(root, diff, policy=POLICY, scope=TEST)
    assert not result.allowed
    assert any("may not add files" in r for r in result.rejection_reasons)
    with pytest.raises(ScopeViolationError):
        apply_patch(root, diff, policy=POLICY, scope=TEST)
    assert not (root / "tests/calc.py").exists()
    assert run_tests(root, ["tests"], timeout_seconds=60).status is TestRunStatus.FAILED


@pytest.mark.parametrize(
    "rel",
    ["tests/unit/pkg.py", "tests/pkg/__init__.py", "tests/pkg/mod.py", "tests/helpers/pkg/x.py"],
)
def test_shadowing_modules_and_packages_are_rejected(repo: Path, rel: str) -> None:
    result = validate_patch(repo, new_file_diff(rel), policy=POLICY, scope=TEST)
    assert any("may not add files" in r for r in result.rejection_reasons)


def test_renaming_a_helper_onto_a_production_name_is_rejected(repo: Path) -> None:
    rename = (
        "diff --git a/tests/test_mod.py b/tests/pkg.py\nsimilarity index 100%\n"
        "rename from tests/test_mod.py\nrename to tests/pkg.py\n"
    )
    assert not validate_patch(repo, rename, policy=POLICY, scope=TEST).allowed


# --------------------------------------------------------------------------- #
# Regression: newly introduced skip / xfail / collection manipulation
# --------------------------------------------------------------------------- #

BASE_TEST = FILES["tests/test_mod.py"]
STRUCTURAL = "not limited to expected literal values"


def _test_patch(repo: Path, new_text: str) -> tuple[str, object]:
    diff = diff_for(repo, "tests/test_mod.py", new_text)
    return diff, validate_patch(repo, diff, policy=POLICY, scope=TEST)


# Every attack that previously reached a red -> green run, plus the earlier
# blocklist cases.  None needs a dedicated rule: the AST comparison rejects them.
ATTACKS = {
    "pt-skip-alias": BASE_TEST.replace(
        "    assert add(1, 2) == 3", "    import pytest as pt\n    pt.skip('x')"
    ),
    "from-pytest-import-skip-as": BASE_TEST.replace(
        "    assert add(1, 2) == 3", "    from pytest import skip as s\n    s('x')"
    ),
    "getattr-skip": BASE_TEST.replace(
        "    assert add(1, 2) == 3", "    import pytest\n    getattr(pytest, 'skip')('x')"
    ),
    "exec-skip": BASE_TEST.replace(
        "    assert add(1, 2) == 3", "    exec('import pytest; pytest.skip()')"
    ),
    "marker-alias": "import pytest\n"
    + BASE_TEST.replace("def test_add", "m = pytest.mark\n\n\n@m.skip\ndef test_add"),
    "skip-marker": "import pytest\n"
    + BASE_TEST.replace("def test_add", "@pytest.mark.skip\ndef test_add"),
    "xfail-module-mark": "import pytest\npytestmark = pytest.mark.xfail\n" + BASE_TEST,
    "autouse-skip-fixture": "import pytest\n\n\n@pytest.fixture(autouse=True)\ndef _s():\n"
    "    pytest.skip('x')\n\n\n" + BASE_TEST,
    "autouse-via-kwargs": "import pytest\n\n\n@pytest.fixture(**{'autouse': True})\ndef _s():\n"
    "    import pytest as p\n    p.skip('x')\n\n\n" + BASE_TEST,
    "unittest-skip": "import unittest\n\n\n@unittest.skip('x')\nclass TestX(unittest.TestCase):\n"
    "    def test_y(self):\n        pass\n\n\n" + BASE_TEST,
    "collection-hook": BASE_TEST
    + "\n\ndef pytest_collection_modifyitems(items):\n    items.clear()\n",
    "dunder-test-false": BASE_TEST + "\n\n__test__ = False\n",
    "sys-modules-alias": "import sys as s, types\nm = types.ModuleType('pkg.mod')\n"
    "m.add = lambda a, b: 3\ns.modules['pkg.mod'] = m\n" + BASE_TEST,
    "from-sys-import-path": "from sys import path\npath.insert(0, 'tests/fake')\n" + BASE_TEST,
    "sys-path-slice": "import sys\nsys.path[0:0] = ['tests/fake']\n" + BASE_TEST,
    "monkeypatch-stub": BASE_TEST.replace(
        "def test_add():\n",
        "def test_add(monkeypatch):\n    import pkg.mod\n"
        "    monkeypatch.setattr(pkg.mod, 'add', lambda a, b: 3)\n",
    ),
    "except-exception": BASE_TEST.replace(
        "    assert add(1, 2) == 3",
        "    try:\n        assert add(1, 2) == 3\n    except Exception:\n        pass",
    ),
    "contextlib-suppress": "import contextlib\n"
    + BASE_TEST.replace(
        "    assert add(1, 2) == 3",
        "    with contextlib.suppress(AssertionError):\n        assert add(1, 2) == 3",
    ),
    "os-exit": BASE_TEST.replace(
        "    assert add", "    import os\n    os._exit(0)\n    assert add"
    ),
    "pytest-internals": "import _pytest.runner\n" + BASE_TEST,
    "assertion-removed": BASE_TEST.replace("    assert add(1, 2) == 3", "    pass"),
    "test-renamed-out-of-collection": BASE_TEST.replace("def test_add", "def check_add"),
    "test-body-deleted": "from pkg.mod import add\n",
    "test-added": BASE_TEST + "\n\ndef test_more():\n    assert add(0, 0) == 0\n",
}

# Changes to the executable logic of an existing assertion.
LOGIC_CHANGES = {
    "vacuous-assert-true": BASE_TEST.replace("add(1, 2) == 3", "True"),
    "both-sides-literal": BASE_TEST.replace("add(1, 2) == 3", "3 == 3"),
    "operator-flipped": BASE_TEST.replace("== 3", "!= 3"),
    "input-changed": BASE_TEST.replace("add(1, 2)", "add(1, 1)"),
    "tested-expression-changed": BASE_TEST.replace("add(1, 2)", "add(2, 1)"),
    "expected-becomes-call": BASE_TEST.replace("== 3", "== add(1, 2)"),
    "or-true": BASE_TEST.replace("== 3", "== 3 or True"),
    "chained-compare": BASE_TEST.replace("== 3", "== 3 == 3"),
    "message-added": BASE_TEST.replace("== 3", "== 3, 'msg'"),
    "operands-swapped": BASE_TEST.replace("add(1, 2) == 3", "3 == add(1, 2)"),
    "approx": "import pytest\n" + BASE_TEST.replace("== 3", "== pytest.approx(3)"),
}


@pytest.mark.parametrize("case", sorted(ATTACKS))
def test_previous_bypass_attacks_are_rejected(repo: Path, case: str) -> None:
    diff, result = _test_patch(repo, ATTACKS[case])
    assert not result.allowed, case
    assert any(STRUCTURAL in r for r in result.rejection_reasons), result.rejection_reasons
    before = snapshot(repo)
    with pytest.raises(ScopeViolationError):
        apply_patch(repo, diff, policy=POLICY, scope=TEST)
    assert snapshot(repo) == before


@pytest.mark.parametrize("case", sorted(LOGIC_CHANGES))
def test_changed_assertion_logic_is_rejected(repo: Path, case: str) -> None:
    _, result = _test_patch(repo, LOGIC_CHANGES[case])
    assert not result.allowed, case
    assert any(STRUCTURAL in r for r in result.rejection_reasons), result.rejection_reasons


@pytest.mark.parametrize(
    ("base", "updated"),
    [
        ("    assert add(1, 2) == 3\n", "    assert add(1, 2) == 4\n"),
        ("    assert add(1, 2) == 3\n", "    assert add(1, 2) == 'three'\n"),
        ("    assert add(1, 2) != 3\n", "    assert add(1, 2) != 7\n"),
        ("    assert 3 == add(1, 2)\n", "    assert -3 == add(1, 2)\n"),
        ("    assert add(1, 2) == (1, 2)\n", "    assert add(1, 2) == (2, 1)\n"),
        ("    assert add(1, 2) == {'a': 1}\n", "    assert add(1, 2) == {'a': 2}  # updated\n"),
        (
            "    assert add(1, 2) == 3, 'sum'\n    assert add(0, 0) == 0\n",
            "    assert add(1, 2) == 4, 'sum'\n    assert add(0, 0) == 1\n",
        ),
    ],
    ids=[
        "int",
        "str",
        "inequality",
        "literal-on-left",
        "tuple",
        "dict-with-comment",
        "two-asserts",
    ],
)
def test_expected_literal_updates_are_accepted(repo: Path, base: str, updated: str) -> None:
    text = "from pkg.mod import add\n\n\ndef test_add():\n" + base
    (repo / "tests/test_mod.py").write_text(text)
    git(
        repo,
        "-c",
        "user.name=t",
        "-c",
        "user.email=t@example.invalid",
        "commit",
        "-qam",
        "base",
        "--allow-empty",
    )
    diff = diff_for(repo, "tests/test_mod.py", text.replace(base, updated))
    result = validate_patch(repo, diff, policy=POLICY, scope=TEST)
    assert result.ok, result.rejection_reasons
    apply_patch(repo, diff, policy=POLICY, scope=TEST)
    revert_patch(repo, diff)
    assert (repo / "tests/test_mod.py").read_text() == text


def test_test_patch_without_a_changed_expectation_is_rejected(repo: Path) -> None:
    _, result = _test_patch(repo, BASE_TEST.replace("== 3", "== 3  # just a comment"))
    assert any("changes no expected literal" in r for r in result.rejection_reasons)


def test_test_data_files_cannot_be_changed_by_test_patches(repo: Path) -> None:
    (repo / "tests/data.json").write_text('{"x": 1}\n')
    git(repo, "add", ".")
    git(repo, "-c", "user.name=t", "-c", "user.email=t@example.invalid", "commit", "-qm", "data")
    diff = diff_for(repo, "tests/data.json", '{"x": 2}\n')
    result = validate_patch(repo, diff, policy=POLICY, scope=TEST)
    assert any("existing Python test files only" in r for r in result.rejection_reasons)


def test_deleting_or_renaming_a_test_file_is_rejected(repo: Path) -> None:
    (repo / "tests/test_mod.py").unlink()
    deletion = git(repo, "diff", "--", "tests/test_mod.py")
    git(repo, "checkout", "--", "tests/test_mod.py")
    rename = (
        "diff --git a/tests/test_mod.py b/tests/mod_checks.py\nsimilarity index 100%\n"
        "rename from tests/test_mod.py\nrename to tests/mod_checks.py\n"
    )
    for diff in (deletion, rename):
        result = validate_patch(repo, diff, policy=POLICY, scope=TEST)
        assert any("may not delete or rename" in r for r in result.rejection_reasons), diff


def test_pre_existing_skip_does_not_block_an_expectation_update(repo: Path) -> None:
    base = (
        "import pytest\n\n\n@pytest.mark.skip(reason='flaky')\ndef test_flaky():\n    pass\n\n\n"
        "def test_ok():\n    assert len('a') == 1\n"
    )
    (repo / "tests/test_legacy.py").write_text(base)
    git(repo, "add", ".")
    git(repo, "-c", "user.name=t", "-c", "user.email=t@example.invalid", "commit", "-qm", "legacy")
    diff = diff_for(repo, "tests/test_legacy.py", base.replace("== 1", "== 2"))
    assert validate_patch(repo, diff, policy=POLICY, scope=TEST).ok


@pytest.mark.parametrize(
    ("addition", "allowed"),
    [
        ("import pytest\n", False),
        ("import _pytest.runner\n", False),
        ("import os\nos._exit(0)\n", False),
        ("def pytest_configure(config):\n    pass\n", False),
        ("import sys\n\n\ndef main():\n    sys.exit(1)\n", True),
    ],
    ids=["pytest-import", "pytest-internals", "os-exit", "hook", "cli-sys-exit"],
)
def test_source_patches_cannot_tamper_with_the_test_run(
    repo: Path, addition: str, allowed: bool
) -> None:
    diff = diff_for(repo, "src/pkg/mod.py", addition + FILES["src/pkg/mod.py"])
    assert validate_patch(repo, diff, policy=POLICY, scope=SRC).ok is allowed


def test_patch_leaving_unparsable_python_is_rejected(repo: Path) -> None:
    diff = diff_for(
        repo, "tests/test_mod.py", BASE_TEST.replace("def test_add():", "def test_add(:")
    )
    result = validate_patch(repo, diff, policy=POLICY, scope=TEST)
    assert any("does not parse" in r for r in result.rejection_reasons)


# --------------------------------------------------------------------------- #
# Regression: protected generated files and backups in nested directories
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "rel",
    [
        "tests/__pycache__/test_mod.cpython-312.pyc",
        "tests/__pycache__/helper.py",
        "src/pkg/__pycache__/mod.cpython-312-pytest-9.1.1.pyc",
        "src/pkg/mod.pyc",
        "src/pkg/mod.pyo",
        "src/pkg/_speedups.so",
        "src/pkg/evil.pth",
        "tests/test_mod.py.bak",
        "tests/conftest.py.orig",
        "tests/test_mod.py.rej",
        "src/pkg/mod.py~",
        "tests/.test_mod.py.swp",
        "tests/.env",
        "src/pkg/.pdbrc",
        "tests/.pytest_cache/v/x",
        "src/pkg.egg-info/PKG-INFO",
    ],
)
def test_generated_and_backup_files_are_protected(repo: Path, rel: str) -> None:
    scope = TEST if rel.startswith("tests/") else SRC
    result = validate_patch(repo, new_file_diff(rel, "x"), policy=POLICY, scope=scope)
    assert f"forbidden file: {rel}" in result.rejection_reasons


# --------------------------------------------------------------------------- #
# Regression: revert only undoes patches apply_patch applied, in one direction
# --------------------------------------------------------------------------- #


def test_revert_refuses_a_patch_that_apply_patch_never_applied(repo: Path) -> None:
    """A crafted 'revert' whose reverse would add an autouse skip fixture."""
    crafted = git_reverse(repo, "tests/test_mod.py", ATTACKS["autouse-skip-fixture"])
    before = snapshot(repo)
    with pytest.raises(ScopeViolationError):
        revert_patch(repo, crafted)
    # Even a revert that passes every content check needs apply_patch's record.
    plausible = git_reverse(repo, "tests/test_mod.py", TEST_FIX)
    with pytest.raises(PatchError, match="not applied by apply_patch"):
        revert_patch(repo, plausible)
    assert snapshot(repo) == before


def git_reverse(repo: Path, rel: str, other_text: str) -> str:
    """A diff from ``other_text`` to the committed content (reverse applies cleanly now)."""
    path = repo / rel
    current = path.read_text()
    path.write_text(other_text)
    git(repo, "add", rel)
    path.write_text(current)
    try:
        return git(repo, "diff", "--", rel)
    finally:
        git(repo, "reset", "-q", "--", rel)


def test_revert_of_a_mixed_scope_patch_is_rejected(repo: Path) -> None:
    mixed = diff_for(repo, "src/pkg/mod.py", SRC_FIX) + diff_for(
        repo, "tests/test_mod.py", TEST_FIX
    )
    (repo / "src/pkg/mod.py").write_text(SRC_FIX)
    (repo / "tests/test_mod.py").write_text(TEST_FIX)
    before = snapshot(repo)
    with pytest.raises(ScopeViolationError, match="both source and test"):
        revert_patch(repo, mixed)
    assert snapshot(repo) == before


def test_revert_uses_the_recorded_scope(repo: Path) -> None:
    diff = diff_for(repo, "tests/test_mod.py", TEST_FIX)
    apply_patch(repo, diff, policy=POLICY, scope=TEST)
    with pytest.raises(ScopeViolationError, match="applied with scope test"):
        revert_patch(repo, diff, scope=SRC)
    revert_patch(repo, diff)  # recorded scope used
    assert git(repo, "status", "--porcelain") == ""
    with pytest.raises(PatchError):
        revert_patch(repo, diff)  # record consumed: no second revert


def test_unsuccessful_revert_keeps_unrelated_changes_and_the_record(repo: Path) -> None:
    diff = diff_for(repo, "src/pkg/mod.py", SRC_FIX)
    apply_patch(repo, diff, policy=POLICY, scope=SRC)
    (repo / "docs/readme.md").write_text("user edit\n")
    mod = repo / "src/pkg/mod.py"
    mod.write_text(mod.read_text().replace("# fixed", "# edited by someone else"))
    before = snapshot(repo)
    with pytest.raises(PatchError):
        revert_patch(repo, diff)
    assert snapshot(repo) == before
    mod.write_text(SRC_FIX)
    revert_patch(repo, diff)  # still revertible once the conflict is gone
    assert mod.read_text() == FILES["src/pkg/mod.py"]
