"""Deterministic test-weakening analysis (proposal section 11.1, Table 4 "test-weakening rate").

``analyse_test_changes(before, after)`` compares the before/after text of the test files one patch touched and fills
``TestWeakeningFindings`` completely.  Everything is AST-based and judged **per test function**, so changing an expected
value (``== 'a'`` -> ``== 'A'``: an intended-behaviour update) is not weakening, while these are:

* **removed assertions**: a test has fewer *reachable* assertion nodes (``assert``, ``self.assert*``/``assert_*``
  helpers, ``pytest.raises``/``warns``) than before - assertions after a ``return``/``raise``/``break``/``continue``
  or under a constant-false condition (``if False:``, ``if 0:``) do not count; a deleted test loses all of them;
* **added skip / xfail**: ``@pytest.mark.skip/skipif/xfail`` (also through ``from pytest import mark``),
  ``pytest.skip/xfail/importorskip(...)``, ``@unittest.skip*``/``expectedFailure``, ``self.skipTest(...)``,
  ``raise SkipTest``, ``pytestmark = ...`` and ``__test__ = False``; markers are compared by content, so a changed
  ``skipif``/``xfail`` condition counts as an added marker;
* **loosened comparisons**: an assertion at the same position became weaker (``==`` -> ``!=``/``in``/``<``, a
  comparison -> plain truthiness, a tautology such as ``x == x`` or ``assert True``, ``assertEqual`` ->
  ``assertTrue``, exact -> ``pytest.approx``, a wider ``approx`` tolerance), or the expected value of an
  *inequality* (``!=``, ``is not``, ``not in``) changed - a new excluded value pins no new behaviour;
* **swallowed exceptions / broad expectations**: a new ``try`` whose broad handler (bare, ``Exception``,
  ``BaseException``) or ``AssertionError`` handler does not re-raise, ``contextlib.suppress(...)``,
  ``pytest.raises(Exception)``;
* **deleted or disabled tests**: a test function that disappeared or was renamed out of collection.

``changed_tests`` names every test function whose syntax tree changed, so the Integrity Guard can reject edits
outside the failure group.  The analysis reports; deciding is the Guard's job.
"""

from __future__ import annotations

import ast
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

from testorcode.schemas.result_schemas import TestWeakeningFindings

_EXACT, _ORDERED, _WEAK, _NONE = 3, 2, 1, 0
_RANK_OPS: Final[dict[type[ast.cmpop], int]] = {
    ast.Eq: _EXACT, ast.Is: _EXACT,
    ast.In: _ORDERED, ast.NotIn: _ORDERED, ast.Lt: _ORDERED, ast.LtE: _ORDERED, ast.Gt: _ORDERED, ast.GtE: _ORDERED,
    ast.NotEq: _WEAK, ast.IsNot: _WEAK,
}
_UNITTEST_RANK: Final[dict[str, int]] = {
    **dict.fromkeys(("assertEqual", "assertEquals", "assertIs", "assertListEqual", "assertDictEqual",
                     "assertSetEqual", "assertTupleEqual", "assertSequenceEqual", "assertMultiLineEqual",
                     "assertCountEqual", "assertIsNone"), _EXACT),
    **dict.fromkeys(("assertAlmostEqual", "assertIn", "assertGreater", "assertGreaterEqual", "assertLess",
                     "assertLessEqual", "assertRegex", "assertRaises", "assertRaisesRegex", "assertWarns",
                     "assertIsInstance", "assertNotIn"), _ORDERED),
    **dict.fromkeys(("assertTrue", "assertFalse", "assertNotEqual", "assertIsNot", "assertIsNotNone",
                     "assertNotAlmostEqual", "assertNotIsInstance", "assertNotRegex"), _WEAK),
}
_EXPECTATION_CALLS: Final[frozenset[str]] = frozenset({"raises", "warns", "deprecated_call", "assertRaises",
                                                       "assertRaisesRegex", "assertWarns", "assertWarnsRegex"})
_BROAD: Final[frozenset[str]] = frozenset({"Exception", "BaseException"})
_SKIP_DECORATORS: Final[frozenset[str]] = frozenset({"skip", "skipif", "skipIf", "skipUnless"})
_SKIP_CALLS: Final[frozenset[str]] = frozenset({"skip", "importorskip", "skipTest"})
_SKIP_EXCEPTIONS: Final[frozenset[str]] = frozenset({"SkipTest", "Skipped"})


@dataclass(frozen=True)
class WeakeningAnalysis:
    findings: TestWeakeningFindings
    problems: tuple[str, ...]
    changed_tests: frozenset[str]  # "path::Qual.name" of every test function whose syntax tree changed


def analyse_test_changes(
    before: Mapping[str, str | None], after: Mapping[str, str | None]
) -> WeakeningAnalysis:
    """Weakening findings for the change ``before`` -> ``after`` of the given test files (``None`` = absent)."""
    totals: Counter[str] = Counter()
    problems: list[str] = []
    changed: set[str] = set()
    for path in sorted(set(before) | set(after)):
        old, new = before.get(path), after.get(path)
        old_tree, new_tree = _parse(old), _parse(new)
        if new is not None and new_tree is None:
            problems.append(f"{path}: does not parse after the patch")
        old_tests, new_tests = _tests(old_tree), _tests(new_tree)
        totals["before"] += _count_assertions(old_tree)
        totals["after"] += _count_assertions(new_tree)
        changed |= {f"{path}::{n}" for n in old_tests.keys() & new_tests.keys()
                    if ast.dump(old_tests[n]) != ast.dump(new_tests[n])}
        changed |= {f"{path}::{n}" for n in old_tests.keys() ^ new_tests.keys()}

        deleted = sorted(old_tests.keys() - new_tests.keys())
        totals["deleted"] += len(deleted)
        totals["removed"] += sum(_count_assertions(old_tests[n]) for n in deleted)
        for name in sorted(old_tests.keys() & new_tests.keys()):
            a, b = old_tests[name], new_tests[name]
            totals["removed"] += max(0, _reachable_assertions(a.body) - _reachable_assertions(b.body))
            totals["loosened"] += _loosened(a, b)
        added_marks = _markers(new_tree) - _markers(old_tree)
        totals["skip"] += sum(n for (kind, _), n in added_marks.items() if kind == "skip")
        totals["xfail"] += sum(n for (kind, _), n in added_marks.items() if kind == "xfail")
        totals["swallowed"] += max(0, _swallowing(new_tree) - _swallowing(old_tree))
        if deleted:
            problems.append(f"{path}: test(s) deleted or disabled: {', '.join(deleted)}")

    removed = max(totals["removed"], totals["before"] - totals["after"])
    labels = (("removed", removed, "assertion(s) removed"), ("skip", totals["skip"], "skip marker(s) added"),
              ("xfail", totals["xfail"], "xfail marker(s) added"),
              ("loosened", totals["loosened"], "comparison(s) loosened"),
              ("swallowed", totals["swallowed"], "exception(s) swallowed or expected too broadly"))
    problems += [f"test weakening: {count} {text}" for _, count, text in labels if count]
    findings = TestWeakeningFindings(
        checks_run=True,
        assertions_before=totals["before"],
        assertions_after=totals["after"],
        removed_assertions=removed,
        added_skip_markers=totals["skip"],
        added_xfail_markers=totals["xfail"],
        loosened_comparisons=totals["loosened"],
        swallowed_exceptions=totals["swallowed"],
        deleted_tests=totals["deleted"],
        weakening_detected=bool(problems),
    )
    return WeakeningAnalysis(findings, tuple(problems), frozenset(changed))


# --------------------------------------------------------------------------- #
# Structure
# --------------------------------------------------------------------------- #


def _parse(text: str | None) -> ast.Module | None:
    if text is None:
        return None
    try:
        return ast.parse(text)
    except (SyntaxError, ValueError):
        return None


def _tests(tree: ast.Module | None) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    """Collected test functions: ``test*`` at module level and ``test*`` methods of ``Test*`` classes.

    A class or module that sets ``__test__ = False`` collects nothing, so its tests count as disabled.
    """
    found: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}
    if tree is None or _disables_collection(tree.body):
        return found
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test"):
            found[node.name] = node
        elif isinstance(node, ast.ClassDef) and node.name.startswith("Test") and not _disables_collection(node.body):
            for member in node.body:
                if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)) and member.name.startswith("test"):
                    found[f"{node.name}.{member.name}"] = member
    return found


def _disables_collection(body: list[ast.stmt]) -> bool:
    return any(
        isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "__test__" for t in n.targets)
        and isinstance(n.value, ast.Constant) and not n.value.value
        for n in body
    )


def _dotted(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    if isinstance(node, ast.Call):
        return _dotted(node.func)
    return ""


def _last(node: ast.AST) -> str:
    return _dotted(node).rsplit(".", 1)[-1]


# --------------------------------------------------------------------------- #
# Assertions and their strength
# --------------------------------------------------------------------------- #


def _assertion_nodes(tree: ast.AST | None) -> list[ast.AST]:
    """Assertion-like nodes in source order: ``assert`` statements and assertion/expectation helper calls."""
    if tree is None:
        return []
    nodes = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Assert)
        or (isinstance(n, ast.Call) and (_last(n.func).startswith(("assert", "assert_")) or _last(n.func) in _EXPECTATION_CALLS))
    ]
    return sorted(nodes, key=lambda n: (n.lineno, n.col_offset))


def _count_assertions(tree: ast.AST | None) -> int:
    return len(_assertion_nodes(tree))


def _reachable_assertions(block: list[ast.stmt]) -> int:
    """Assertion nodes in ``block`` that can execute: nothing after a ``return``/``raise``/``break``/``continue`` at
    the same level, and only the taken branch of an ``if``/``while`` on a constant."""
    total = 0
    for statement in block:
        total += _statement_assertions(statement)
        if isinstance(statement, (ast.Return, ast.Raise, ast.Break, ast.Continue)):
            break
    return total


def _statement_assertions(statement: ast.stmt) -> int:
    if isinstance(statement, (ast.If, ast.While)) and isinstance(statement.test, ast.Constant):
        if statement.test.value:
            return _reachable_assertions(statement.body)
        return _reachable_assertions(statement.orelse)
    blocks = [getattr(statement, f) for f in ("body", "orelse", "finalbody")
              if isinstance(getattr(statement, f, None), list)]
    if isinstance(statement, ast.Try | getattr(ast, "TryStar", ast.Try)):
        blocks += [h.body for h in statement.handlers]
    if not blocks or isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return _count_assertions(statement)
    inner = {id(n) for b in blocks for s in b for n in ast.walk(s)}
    header = sum(id(n) not in inner for n in _assertion_nodes(statement))  # e.g. ``with pytest.raises(...)``
    return header + sum(_reachable_assertions(b) for b in blocks)


def _rank(node: ast.AST) -> int:
    """How strictly an assertion pins the behaviour (3 exact ... 0 vacuous)."""
    if isinstance(node, ast.Assert):
        return _expr_rank(node.test)
    name = _last(node.func)  # an assertion helper call
    if name in _EXPECTATION_CALLS:
        return _WEAK if _broad_expectation(node) else _ORDERED
    return _UNITTEST_RANK.get(name, _ORDERED)


def _expr_rank(expr: ast.expr) -> int:
    if isinstance(expr, ast.Constant):
        return _NONE if expr.value else _EXACT  # assert True is vacuous; assert False always fails
    if isinstance(expr, ast.BoolOp):
        ranks = [_expr_rank(v) for v in expr.values]
        return max(ranks) if isinstance(expr.op, ast.And) else max(_NONE, min(ranks) - 1)
    if isinstance(expr, ast.Compare):
        if any(ast.dump(a) == ast.dump(b) for a, b in zip([expr.left, *expr.comparators], expr.comparators, strict=False)):
            return _NONE  # x == x
        rank = min(_RANK_OPS.get(type(op), _WEAK) for op in expr.ops)
        if rank == _EXACT and any(_last(side) == "approx" for side in (expr.left, *expr.comparators)):
            return _ORDERED
        return rank
    return _WEAK  # plain truthiness, isinstance(...), not x, ...


def _approx_tolerance(node: ast.AST) -> float | None:
    for call in (n for n in ast.walk(node) if isinstance(n, ast.Call) and _last(n.func) == "approx"):
        for kw in call.keywords:
            if kw.arg in ("rel", "abs") and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, int | float):
                return float(kw.value.value)
    return None


def _loosened(old: ast.AST, new: ast.AST) -> int:
    """Assertions at the same position whose strength dropped (only when the count is unchanged)."""
    a, b = _assertion_nodes(old), _assertion_nodes(new)
    if len(a) != len(b):
        return 0  # removals are counted separately; positions no longer correspond
    count = 0
    for x, y in zip(a, b, strict=True):
        if _rank(y) < _rank(x):
            count += 1
            continue
        tx, ty = _approx_tolerance(x), _approx_tolerance(y)
        if tx is not None and ty is not None and ty > tx:
            count += 1
        elif _is_inequality(x) and _is_inequality(y) and ast.dump(x) != ast.dump(y):
            count += 1  # a different excluded value pins no new behaviour: not an intended-behaviour update
    return count


def _is_inequality(node: ast.AST) -> bool:
    return (isinstance(node, ast.Assert) and isinstance(node.test, ast.Compare) and len(node.test.ops) == 1
            and isinstance(node.test.ops[0], (ast.NotEq, ast.IsNot, ast.NotIn)))


def _broad_expectation(call: ast.Call) -> bool:
    if not call.args:
        return False
    first = call.args[0]
    names = [_last(e) for e in first.elts] if isinstance(first, ast.Tuple) else [_last(first)]
    return any(n in _BROAD for n in names)


# --------------------------------------------------------------------------- #
# Skips, xfails and swallowed exceptions (counted over the whole module)
# --------------------------------------------------------------------------- #


def _markers(tree: ast.Module | None) -> Counter[tuple[str, str]]:
    """Skip/xfail mechanisms as ``(kind, content)`` keys, so the difference of two counters counts added markers
    *and* markers whose content changed (``skipif(False)`` -> ``skipif(True)``)."""
    found: Counter[tuple[str, str]] = Counter()
    if tree is None:
        return found
    # Calls such as ``pytest.mark.skipif(...)`` used as decorators are counted once, as markers.
    decorators = {
        id(d) for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        for d in n.decorator_list
    }
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            for decorator in node.decorator_list:
                name = _last(decorator)
                if name in _SKIP_DECORATORS:
                    found["skip", ast.dump(decorator)] += 1
                elif name in ("xfail", "expectedFailure"):
                    found["xfail", ast.dump(decorator)] += 1
        elif isinstance(node, ast.Call) and id(node) not in decorators:
            name = _last(node.func)
            if name in _SKIP_CALLS:
                found["skip", ast.dump(node)] += 1
            elif name == "xfail":
                found["xfail", ast.dump(node)] += 1
        elif isinstance(node, ast.Raise) and node.exc is not None and _last(node.exc) in _SKIP_EXCEPTIONS:
            found["skip", ast.dump(node)] += 1
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in ("pytestmark", "__test__"):
                    found["skip", ast.dump(node)] += 1
    return found


def _swallowing(tree: ast.Module | None) -> int:
    count = 0
    if tree is None:
        return count
    for node in ast.walk(tree):
        if isinstance(node, ast.Try):
            for handler in node.handlers:
                broad = handler.type is None or (
                    [_last(e) for e in handler.type.elts] if isinstance(handler.type, ast.Tuple) else [_last(handler.type)]
                )
                swallows = handler.type is None or any(n in _BROAD or n == "AssertionError" for n in broad)
                if swallows and not any(
                    isinstance(n, ast.Raise) for n in ast.walk(handler)
                ):
                    count += 1
        elif isinstance(node, ast.Call):
            name = _last(node.func)
            if name == "suppress" or (name in _EXPECTATION_CALLS and _broad_expectation(node)):
                count += 1
    return count
