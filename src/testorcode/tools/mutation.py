"""Cached-mutant runner: measure whether repaired tests still detect wrong behaviour.

Three parts:

* ``generate_mutants`` - deterministic single-operator AST mutants of **one function** (the operator families of
  ``scripts/screening/mutation_probe.py``: comparison, arithmetic and boolean operator swaps, constant tweaks, negation
  removal, ``return x`` -> ``return None``).  Each mutant is a one-line ``MutantSpec`` with a unified diff, generated
  from the current source only (never from gold data) and memoised per (file content, function, limit, seed).
* ``run_mutants`` - copies the worktree (no ``.git``, caches, compiled files or symlinks) into a scratch directory,
  optionally replaces some files (``overrides``, e.g. the tests as they were before a repair), runs the selected tests
  once unmutated (baseline), then applies each mutant with ``git apply`` in the scratch copy, runs the tests in the
  sandbox and restores the file.  The episode worktree is never written.  Outcomes are conservative: ``killed`` only
  when a test that passed on the baseline now fails; ``timeout``, ``error`` (the run could not execute the tests),
  ``invalid`` (the mutant does not apply) and ``not_run`` (the overall deadline passed) are never kills.
* ``mutation_check`` - the Integrity Guard's check.  For UPDATE_TEST the non-leaking baseline is **the same test files
  before this repair** (the reference repair stays in the evaluation harness): a mutant killed before the repair but
  not after is *measured weakening* (a problem: REJECT); updated tests that kill none of the executable mutants, or
  mutants that could not be measured, leave the strength *unproven* (a gap: ESCALATE).  For FIX_CODE it is an
  oracle-sensitivity check: the untouched tests must kill at least one executable mutant of the repaired function.
* ``load_cached_mutants`` - validated mutant specifications from a catalogue (``scripts/build_mutant_cache.py``);
  stored outcomes are never reused as results.

``MutationResult`` (the run record): ``mutants_total`` = mutants measured cleanly (in both states for UPDATE_TEST);
``killed_by_repaired_tests`` = mutants killed by **the failure group's own tests**; ``killed_by_reference_tests`` stays
``None`` here (retention against the developer's reference tests is an evaluation metric).  Kills by the whole test
files are kept in ``MutationCheck`` as diagnostics.
"""

from __future__ import annotations

import ast
import copy
import difflib
import hashlib
import json
import posixpath
import random
import shutil
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Literal

from testorcode.errors import ToolError
from testorcode.logging_config import get_logger
from testorcode.schemas.contracts import EpisodeContext, FailureGroupDetail, PatchProposal
from testorcode.schemas.episode_schemas import (
    MutantSpec,
    MutationFixture,
    PathClass,
    RepairDirection,
    ToolName,
)
from testorcode.schemas.result_schemas import MutationResult, TestRunStatus, ToolCallStatus
from testorcode.tools.git_tools import run_git
from testorcode.tools.pytest_runner import run_pytest
from testorcode.tools.sandbox import Isolation
from testorcode.tools.trace import tool_call
from testorcode.tools.verification import locate_module

logger = get_logger(__name__)

DEFAULT_MAX_MUTANTS: Final[int] = 6
DEFAULT_SEED: Final[int] = 7  # the seed of the screening feasibility probe
_SKIPPED_NAMES: Final[tuple[str, ...]] = (".git", "__pycache__", ".pytest_cache", ".tox", ".nox", ".venv")
_SKIPPED_SUFFIXES: Final[tuple[str, ...]] = (".pyc", ".pyo")

_CMP: Final[dict[type[ast.cmpop], type[ast.cmpop]]] = {
    ast.Eq: ast.NotEq, ast.NotEq: ast.Eq, ast.Lt: ast.LtE, ast.LtE: ast.Lt, ast.Gt: ast.GtE, ast.GtE: ast.Gt,
    ast.Is: ast.IsNot, ast.IsNot: ast.Is, ast.In: ast.NotIn, ast.NotIn: ast.In,
}
_BIN: Final[dict[type[ast.operator], type[ast.operator]]] = {
    ast.Add: ast.Sub, ast.Sub: ast.Add, ast.Mult: ast.FloorDiv, ast.FloorDiv: ast.Mult, ast.Mod: ast.Mult,
    ast.Div: ast.Mult,
}

Outcome = Literal["killed", "survived", "timeout", "error", "invalid", "not_run"]
DEFAULT_DEADLINE_SECONDS: Final[int] = 300  # whole mutation check of one repair (all states, all mutants)
_CACHE_LIMIT: Final[int] = 256


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #

_CACHE: dict[tuple[str, str, int, int], tuple[MutantSpec, ...]] = {}


def generate_mutants(
    source: str,
    file_path: str,
    function: str,
    *,
    module: str | None = None,
    limit: int = DEFAULT_MAX_MUTANTS,
    seed: int = DEFAULT_SEED,
) -> list[MutantSpec]:
    """Up to ``limit`` deterministic mutants of ``function`` (``Class.method`` or ``func``) in ``source``.

    ``module`` prefixes ``MutantSpec.function_qualname`` (``boltons.cacheutils`` -> ``boltons.cacheutils.f``).  When
    there are more candidates than ``limit``, a seeded sample is taken (same input, same mutants).  Returns ``[]`` when
    the function does not exist or offers no mutation point.
    """
    key = (hashlib.sha256(source.encode()).hexdigest(), f"{file_path}:{function}", limit, seed)
    if key not in _CACHE:
        if len(_CACHE) >= _CACHE_LIMIT:
            _CACHE.pop(next(iter(_CACHE)))  # oldest entry first: the memo stays bounded
        _CACHE[key] = tuple(_generate(source, file_path, function, module, limit, seed))
    return list(_CACHE[key])


def _generate(source: str, file_path: str, function: str, module: str | None, limit: int, seed: int) -> list[MutantSpec]:
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return []
    target = _find_function(tree, function)
    if target is None:
        return []
    lines = source.splitlines(keepends=True)
    candidates: list[tuple[str, str]] = []  # (description, mutated source)
    seen: set[str] = set()

    def collect(points, room: int | None = None) -> None:
        for node, mutated, desc, statement in points:
            if room is not None and len(candidates) >= room:
                return
            if node.lineno != node.end_lineno:
                continue  # one-line edits only: the splice must not cross lines
            new_source = _splice(lines, node, mutated, statement)
            if new_source is None or new_source == source or new_source in seen:
                continue
            try:
                ast.parse(new_source)
            except (SyntaxError, ValueError):
                continue
            seen.add(new_source)
            candidates.append((f"{function} line {node.lineno}: {desc}", new_source))

    collect(_mutation_points(target))
    if len(candidates) < limit:
        # Small functions: top up with the extended operators, after the base ones, so a function that already had
        # ``limit`` base candidates keeps exactly the same mutants and ids.
        collect(_extended_points(target), room=limit)
    indices = list(range(len(candidates)))
    if len(indices) > limit:
        indices = sorted(random.Random(seed).sample(indices, limit))
    qualname = f"{module}.{function}" if module else function
    mutants = []
    for i in indices:
        desc, new_source = candidates[i]
        diff = "".join(difflib.unified_diff(
            source.splitlines(keepends=True), new_source.splitlines(keepends=True),
            f"a/{file_path}", f"b/{file_path}", n=3,
        ))
        mutants.append(MutantSpec(mutant_id=f"mut-{i + 1:02d}", function_qualname=qualname, file_path=file_path,
                                  operator=desc, unified_diff=diff))
    return mutants


def _find_function(tree: ast.Module, qualname: str) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    scope: list[ast.stmt] = tree.body
    node: ast.AST | None = None
    for part in qualname.split("."):
        node = next((n for n in scope if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                     and n.name == part), None)
        if node is None:
            return None
        scope = node.body
    return node if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) else None


def _mutation_points(function: ast.FunctionDef | ast.AsyncFunctionDef):
    """``(original node, mutated copy, description, is_statement)`` for every mutation point, in source order."""
    body = function.body
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
        body = body[1:]  # the docstring
    skip: set[int] = set()
    for stmt in body:
        for node in ast.walk(stmt):
            if isinstance(node, ast.JoinedStr):
                skip.update(id(n) for n in ast.walk(node))  # f-string parts cannot be spliced safely
    points = []
    for stmt in body:
        for node in ast.walk(stmt):
            if id(node) in skip or not hasattr(node, "lineno"):
                continue
            if isinstance(node, ast.Compare) and type(node.ops[0]) in _CMP:
                new = copy.deepcopy(node)
                new.ops[0] = _CMP[type(node.ops[0])]()
                points.append((node, new, f"{type(node.ops[0]).__name__} -> {type(new.ops[0]).__name__}", False))
            elif isinstance(node, ast.BinOp) and type(node.op) in _BIN:
                new = copy.deepcopy(node)
                new.op = _BIN[type(node.op)]()
                points.append((node, new, f"{type(node.op).__name__} -> {type(new.op).__name__}", False))
            elif isinstance(node, ast.BoolOp):
                new = copy.deepcopy(node)
                new.op = ast.Or() if isinstance(node.op, ast.And) else ast.And()
                points.append((node, new, f"{type(node.op).__name__} -> {type(new.op).__name__}", False))
            elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
                points.append((node, copy.deepcopy(node.operand), "remove not", False))
            elif isinstance(node, ast.Constant):
                value = node.value
                if isinstance(value, bool):
                    replacement: object = not value
                elif isinstance(value, int):
                    replacement = value + 1
                elif isinstance(value, str):
                    replacement = "" if value else "XX"
                else:
                    continue
                points.append((node, ast.Constant(replacement), f"constant {value!r} -> {replacement!r}", False))
            elif isinstance(node, ast.Return) and node.value is not None and not (
                isinstance(node.value, ast.Constant) and node.value.value is None
            ):
                points.append((node, ast.Return(value=ast.Constant(None)), "return value -> None", True))
    return sorted(points, key=lambda p: (p[0].lineno, p[0].col_offset))


def _extended_points(function: ast.FunctionDef | ast.AsyncFunctionDef):
    """Extra operators, used only to top up small functions (see ``_generate``), in source order: negate the test of an
    ``if``/``while``/conditional expression, swap an augmented-assignment operator, drop a unary minus, and swap
    ``break``/``continue``.  (A decisive branch is often a conditional expression, e.g. ``a if flag else b``, which the
    base operators never change.)"""
    body = _without_docstring(function.body)
    skip = _fstring_nodes(body)
    points = []
    for stmt in body:
        for node in ast.walk(stmt):
            if id(node) in skip or not hasattr(node, "lineno"):
                continue
            if isinstance(node, (ast.If, ast.While, ast.IfExp)) and id(node.test) not in skip:
                negated = ast.UnaryOp(op=ast.Not(), operand=copy.deepcopy(node.test))
                points.append((node.test, negated, f"negate {type(node).__name__} condition", False))
            elif isinstance(node, ast.AugAssign) and type(node.op) in _BIN:
                new = copy.deepcopy(node)
                new.op = _BIN[type(node.op)]()
                points.append((node, new, f"{type(node.op).__name__}= -> {type(new.op).__name__}=", True))
            elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
                points.append((node, copy.deepcopy(node.operand), "remove unary minus", False))
            elif isinstance(node, ast.Break):
                points.append((node, ast.Continue(), "break -> continue", True))
            elif isinstance(node, ast.Continue):
                points.append((node, ast.Break(), "continue -> break", True))
    return sorted(points, key=lambda p: (p[0].lineno, p[0].col_offset))


def _without_docstring(body: list[ast.stmt]) -> list[ast.stmt]:
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
        return body[1:]
    return body


def _fstring_nodes(body: list[ast.stmt]) -> set[int]:
    """Ids of every node inside an f-string (they cannot be spliced safely)."""
    return {id(n) for stmt in body for node in ast.walk(stmt) if isinstance(node, ast.JoinedStr) for n in ast.walk(node)}


def _splice(lines: list[str], node: ast.AST, mutated: ast.AST, statement: bool) -> str | None:
    """Replace ``node``'s text on its line by ``mutated`` (parenthesised unless it is a statement)."""
    index = node.lineno - 1
    if index >= len(lines):
        return None
    raw = lines[index].encode("utf-8")
    text = ast.unparse(mutated)
    replacement = (text if statement else f"({text})").encode("utf-8")
    new_line = raw[: node.col_offset] + replacement + raw[node.end_col_offset:]
    try:
        out = list(lines)
        out[index] = new_line.decode("utf-8")
    except UnicodeDecodeError:
        return None
    return "".join(out)


# --------------------------------------------------------------------------- #
# Execution
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class MutantOutcome:
    mutant_id: str
    outcome: Outcome
    killed_by: tuple[str, ...] = ()
    detail: str | None = None


@dataclass(frozen=True)
class MutationRun:
    """One state of the tests run against a set of mutants."""

    status: TestRunStatus  # PASSED: the baseline executed and every mutant was tried; ERROR: no usable baseline
    baseline_failing: frozenset[str] = frozenset()
    outcomes: tuple[MutantOutcome, ...] = ()
    reason: str | None = None

    def by_id(self) -> dict[str, MutantOutcome]:
        return {o.mutant_id: o for o in self.outcomes}


def run_mutants(
    worktree_path: Path,
    mutants: Sequence[MutantSpec],
    selectors: Sequence[str],
    *,
    timeout_seconds: int,
    python_executable: str | Path | None = None,
    isolation: Isolation | str = Isolation.AUTO,
    docker_image: str | None = None,
    overrides: Mapping[str, str | None] | None = None,
    deadline: float | None = None,
) -> MutationRun:
    """Run ``selectors`` against each mutant in a scratch copy of the worktree (never the worktree itself).

    ``overrides`` replaces file contents in the copy (``None`` deletes the file), e.g. tests before a repair.
    ``deadline`` (a ``time.monotonic()`` value) bounds the whole run: each pytest run gets at most the remaining time,
    and mutants left when it has passed are ``not_run`` (never a kill).

    Raises:
        ToolError: no sandbox is available, or the worktree is missing.
    """
    root = Path(worktree_path).resolve()
    if not root.is_dir():
        raise ToolError(f"worktree does not exist or is not a directory: {root}")
    if not selectors:
        return MutationRun(TestRunStatus.ERROR, reason="no test selectors to run against the mutants")
    if deadline is not None and deadline - time.monotonic() < 1:
        return MutationRun(TestRunStatus.ERROR, reason="mutation deadline reached before the baseline run")
    with tempfile.TemporaryDirectory(prefix="testorcode-mutants-") as tmp:
        scratch = Path(tmp) / "work"
        _copy_tree(root, scratch)
        for rel, text in (overrides or {}).items():
            target = _inside(scratch, rel)
            if target is None:
                return MutationRun(TestRunStatus.ERROR, reason=f"unsafe override path {rel!r}")
            if text is None:
                target.unlink(missing_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(text)
        run_git(scratch, ["init", "-q"])

        def remaining() -> int:
            if deadline is None:
                return timeout_seconds
            return min(timeout_seconds, int(deadline - time.monotonic()))

        def execute():
            return run_pytest(scratch, list(selectors), timeout_seconds=max(1, remaining()),
                              python_executable=python_executable, isolation=isolation,
                              docker_image=docker_image).execution

        baseline = execute()
        if baseline.status not in (TestRunStatus.PASSED, TestRunStatus.FAILED):
            return MutationRun(TestRunStatus.ERROR, reason=f"unmutated baseline run is {baseline.status.value}: "
                               f"{(baseline.error_message or '')[:200]}")
        base_failing = frozenset(baseline.failing_test_ids)
        outcomes = []
        for mutant in mutants:
            if remaining() < 1:
                outcomes.append(MutantOutcome(mutant.mutant_id, "not_run", detail="mutation deadline reached"))
                continue
            target = _inside(scratch, mutant.file_path)
            if target is None or not target.is_file() or not mutant.unified_diff:
                outcomes.append(MutantOutcome(mutant.mutant_id, "invalid", detail="mutant file or diff missing"))
                continue
            original = target.read_bytes()
            # Schema models strip edge whitespace: restore the final newline, and let git recount hunks whose trailing
            # blank context line was stripped.
            diff = mutant.unified_diff if mutant.unified_diff.endswith("\n") else mutant.unified_diff + "\n"
            applied = run_git(scratch, ["apply", "--recount", "--whitespace=nowarn", "-"], input_text=diff)
            if applied.returncode != 0:
                target.write_bytes(original)
                outcomes.append(MutantOutcome(mutant.mutant_id, "invalid", detail="mutant does not apply to the code"))
                continue
            try:
                run = execute()
            finally:
                target.write_bytes(original)
            if run.status is TestRunStatus.TIMEOUT:
                outcomes.append(MutantOutcome(mutant.mutant_id, "timeout", detail=f"stopped after {run.timeout_seconds}s"))
            elif run.status not in (TestRunStatus.PASSED, TestRunStatus.FAILED):
                outcomes.append(MutantOutcome(mutant.mutant_id, "error", detail=run.status.value))
            else:
                killers = tuple(sorted(set(run.failing_test_ids) - base_failing))
                outcomes.append(MutantOutcome(mutant.mutant_id, "killed" if killers else "survived", killers))
    return MutationRun(TestRunStatus.PASSED, base_failing, tuple(outcomes))


def _inside(root: Path, rel: str) -> Path | None:
    """``root / rel`` for a plain relative path that stays inside ``root``, else None."""
    if not rel or rel.startswith("/") or "\\" in rel or posixpath.normpath(rel) != rel or rel.split("/")[0] == "..":
        return None
    target = root / rel
    return target if target.resolve().is_relative_to(root.resolve()) else None


def _copy_tree(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for entry in src.iterdir():
        if entry.is_symlink() or entry.name in _SKIPPED_NAMES or entry.name.endswith(_SKIPPED_SUFFIXES):
            continue
        if entry.is_dir():
            _copy_tree(entry, dst / entry.name)
        elif entry.is_file():
            shutil.copyfile(entry, dst / entry.name)



# --------------------------------------------------------------------------- #
# Cached mutant catalogues
# --------------------------------------------------------------------------- #


def load_cached_mutants(path: str | Path) -> dict[str, list[MutantSpec]]:
    """Mutants per function from a catalogue written by ``scripts/build_mutant_cache.py``.

    Only the mutant *specifications* are read and validated (``MutationFixture``: unique ids; every mutant names the
    fixture's function and a plain relative file path).  Outcomes stored in the catalogue are ignored: a mutation
    result is only ever what ``run_mutants`` measures now, on the current code.  Mutants that no longer apply are
    reported ``invalid`` at run time.

    Raises:
        ValueError: the file is not a valid catalogue.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    catalogue: dict[str, list[MutantSpec]] = {}
    for entry in data.get("entries", []):
        fixture = MutationFixture.model_validate(entry["mutation_fixture"])
        for mutant in fixture.mutants:
            if mutant.function_qualname != fixture.function_qualname:
                raise ValueError(f"mutant {mutant.mutant_id} names {mutant.function_qualname}, "
                                 f"not {fixture.function_qualname}")
            if _inside(Path("/"), mutant.file_path) is None or not mutant.unified_diff:
                raise ValueError(f"mutant {mutant.mutant_id} has an unsafe path or no diff")
        catalogue.setdefault(fixture.function_qualname, []).extend(fixture.mutants)
    return catalogue


# --------------------------------------------------------------------------- #
# The Integrity Guard's check
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class MutationCheck:
    """Measured mutation evidence for one applied repair.

    ``result.killed_by_repaired_tests`` counts mutants killed by **the failure group's own tests** (the repaired tests
    for UPDATE_TEST, the untouched verifying tests for FIX_CODE).  ``killed_by_file_after``/``_before`` count kills by
    any test in the selected files after/before the repair (diagnostics; ``None`` when not run).
    """

    result: MutationResult = field(default_factory=MutationResult)
    problems: tuple[str, ...] = ()  # measured loss of test strength: REJECT
    gaps: tuple[str, ...] = ()  # strength not (sufficiently) measured or not demonstrated: ESCALATE
    killed_by_file_after: int | None = None
    killed_by_file_before: int | None = None
    outcomes_after: tuple[MutantOutcome, ...] = ()
    outcomes_before: tuple[MutantOutcome, ...] = ()
    # UPDATE_TEST only: no test in the touched files kills any executable mutant, before or after the repair, so these
    # mutants cannot measure the repair's strength either way.  A gap too unless ``allow_unmeasurable`` was given.
    unmeasurable: tuple[str, ...] = ()
    # UPDATE_TEST only: share of the mutants the pre-repair tests killed that the repaired tests still kill; None when
    # the pre-repair tests killed none (nothing to retain, so nothing measured).
    retention: float | None = None


def locate_function(context: EpisodeContext, function_qualname: str) -> tuple[str, str, str] | None:
    """``(file, module, qualname inside the module)`` of a dotted function name in the worktree's source roots."""
    parts = function_qualname.split(".")
    roots = [r.strip("/") for r in context.edit_policy.allowed_source_roots]
    for i in range(len(parts) - 1, 0, -1):
        module = ".".join(parts[:i])
        found = locate_module(context.worktree_path, roots, module)
        if found:
            return found[1], module, ".".join(parts[i:])
    return None


def mutation_check(
    context: EpisodeContext,
    group: FailureGroupDetail,
    proposal: PatchProposal,
    pre_patch: Mapping[str, str | None],
    *,
    timeout_seconds: int,
    python_executable: str | Path | None = None,
    isolation: Isolation | str = Isolation.AUTO,
    docker_image: str | None = None,
    mutants: Sequence[MutantSpec] | None = None,
    max_mutants: int = DEFAULT_MAX_MUTANTS,
    deadline_seconds: int = DEFAULT_DEADLINE_SECONDS,
    allow_unmeasurable: bool = False,
    min_retention: float = 1.0,
) -> MutationCheck:
    """Measured mutation evidence for an applied repair, on mutants of the group's function.

    * **UPDATE_TEST** (the oracle was edited): the touched test files are run after the repair and as they were just
      before it (``pre_patch``, rebuilt by the Guard).  *Retention* = mutants killed both before and after / mutants
      killed before.  Retention below ``min_retention`` is a *proven loss of strength* (problem: REJECT); the default
      1.0 rejects any lost mutant (the pilot rule), the frozen final configuration uses 0.90
      (``Settings.guard_min_retention``).  The Guard sees only the pre-repair tests: the developer's reference tests
      are evaluation-side (``integrity_audit``) and never reach the agent or the Guard.  Updated tests that kill none of the executable mutants, or mutants that could not
      be executed (errors, timeouts, deadline), leave the strength unproven (gap: ESCALATE).
    * **FIX_CODE** (the oracle is untouched): an oracle-sensitivity check on the repaired code.  If the group's tests
      kill none of the executable mutants of the repaired function, they cannot tell the repair from broken variants
      of it (gap: ESCALATE).  When no mutant can be generated the check is simply not run (no gap): the untouched tests
      remain the verification.

    ``mutants`` replaces generation (selected or cached mutants, executed now).  The whole check is bounded by
    ``deadline_seconds``.

    *Strength unmeasurable* (UPDATE_TEST): executable mutants exist but no test in the touched files kills any of them,
    before or after the repair, so nothing was lost and nothing can be shown.  It is reported in ``unmeasurable``; it is
    also a gap (ESCALATE) unless ``allow_unmeasurable`` is given (``Settings.guard_unmeasurable_strength``).

    Raises:
        ToolError: no sandbox is available.
    """
    fix = proposal.decision is RepairDirection.FIX_CODE
    function = group.group.function_qualname
    if not function:
        return MutationCheck() if fix else MutationCheck(
            gaps=("the failure group has no mapped function, so no mutants can be chosen",))
    if mutants is None:
        located = locate_function(context, function)
        if located is None:
            return MutationCheck() if fix else MutationCheck(
                gaps=(f"source of {function} was not found in the source roots",))
        path, module, inner = located
        source = (context.worktree_path / path).read_text(encoding="utf-8", errors="replace")
        mutants = generate_mutants(source, path, inner, module=module, limit=max_mutants)
    if not mutants:
        return MutationCheck() if fix else MutationCheck(gaps=(f"no mutant could be generated for {function}",))
    policy = context.edit_policy
    test_files = {t.test_file for t in group.failing_tests}
    if not fix:
        test_files |= {p for p in pre_patch if policy.classify_path(p) is PathClass.TEST}
    selectors = [p for p in sorted(test_files) if (context.worktree_path / p).is_file()]
    run = {"timeout_seconds": timeout_seconds, "python_executable": python_executable, "isolation": isolation,
           "docker_image": docker_image, "deadline": time.monotonic() + deadline_seconds}
    with tool_call(ToolName.RUN_MUTANTS, function=function, mutants=len(mutants), selectors=selectors[:10],
                   direction=proposal.decision.value) as call:
        after = run_mutants(context.worktree_path, mutants, selectors, **run)
        before = None if fix else run_mutants(context.worktree_path, mutants, selectors, overrides=dict(pre_patch), **run)
        failed = [r for r in (after, before) if r is not None and r.status is not TestRunStatus.PASSED]
        if failed:
            reason = failed[0].reason or "mutation run failed"
            call.fail(ToolCallStatus.ERROR, reason)
            return MutationCheck(MutationResult(status=TestRunStatus.ERROR),
                                 gaps=(f"mutation check could not run: {reason}",))
    a = after.by_id()
    b = before.by_id() if before is not None else {}
    measured = ("killed", "survived")
    valid = [m.mutant_id for m in mutants
             if a[m.mutant_id].outcome in measured and (fix or b[m.mutant_id].outcome in measured)]
    unexecuted = [m.mutant_id for m in mutants if m.mutant_id not in valid]
    group_ids = set(group.group.test_ids)
    killed_after = {m for m in valid if a[m].outcome == "killed"}
    killed_before = {m for m in valid if b.get(m) is not None and b[m].outcome == "killed"}
    by_group = {m for m in killed_after if group_ids & set(a[m].killed_by)}
    problems, gaps, unmeasurable = [], [], []
    lost = sorted(killed_before - killed_after)
    retention = len(killed_before & killed_after) / len(killed_before) if killed_before else None
    if lost and retention is not None and retention < min_retention:
        problems.append(
            f"measured test weakening (retention {retention:.2f} < {min_retention:.2f}): {len(lost)} cached mutant(s) of {function} killed by the tests before the "
            f"repair survive the repaired tests ({', '.join(lost)}); the files kill {len(killed_after)} of "
            f"{len(valid)} after the repair, {len(killed_before)} before"
        )
    if not valid:
        gaps.append(f"none of the {len(mutants)} mutants of {function} could be executed cleanly")
    elif not by_group and not fix and not killed_after and not killed_before:
        unmeasurable.append(f"the updated tests detect none of the {len(valid)} executable mutants of {function}, "
                            "and no test in the touched files killed any of them before the repair either, so its "
                            "strength cannot be measured")
        if not allow_unmeasurable:
            gaps.append(unmeasurable[-1])
    elif not by_group:
        what = "tests that verify this fix" if fix else "updated tests"
        gaps.append(f"the {what} detect none of the {len(valid)} executable mutants of {function}, "
                    "so their strength is not demonstrated")
    if unexecuted and not fix:
        not_run = sum(a[m].outcome == "not_run" or b.get(m) is not None and b[m].outcome == "not_run"
                      for m in unexecuted)
        gaps.append(f"{len(unexecuted)} of {len(mutants)} mutants of {function} were not measured "
                    f"(timeout, error, invalid{' or deadline' if not_run else ''})")
    logger.info("mutation check %s (%s): %d mutants, %d valid, killed by group=%d, file after=%d before=%s",
                function, proposal.decision.value, len(mutants), len(valid), len(by_group), len(killed_after),
                None if fix else len(killed_before))
    result = MutationResult(
        status=TestRunStatus.PASSED if valid else TestRunStatus.ERROR,
        mutants_total=len(valid), killed_by_repaired_tests=len(by_group),
    )
    return MutationCheck(result, tuple(problems), tuple(gaps), len(killed_after),
                         None if fix else len(killed_before), after.outcomes, before.outcomes if before else (),
                         tuple(unmeasurable), retention)
