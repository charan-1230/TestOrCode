"""Content checks on the Python files a patch changes.

Run by ``patch_tools`` after the path checks, on the before/after content of
every touched file (computed in a throw-away copy, never in the worktree).

**Test files (UPDATE_TEST): one narrow, structural rule.**  A test-side change
is accepted only if

1. it modifies an *existing* ``.py`` file under a test root (no new, deleted,
   renamed or non-Python test files);
2. the file parses before and after;
3. after replacing the *expected literal* of every eligible assertion and the
   *literal keyword arguments* of every eligible setup call with placeholders,
   the two syntax trees are identical;
4. no literal keyword argument was removed from a setup call; and
5. at least one expected literal or setup keyword argument actually changed.

An eligible setup call is a call inside the body of a test function
(``test*`` at module level, or a ``test*`` method of a ``Test*`` class) that is
not part of an ``assert`` statement, not in a decorator, and is not an
assertion, approximation, skip, mark, fixture, patching or reflection helper
(``pytest.*``, ``unittest.*``, ``mock.*`` and names containing assert, approx,
raises, warn, skip, xfail, fail, mark, fixture, patch, parametrize, setattr,
getattr, exec, eval, import or compile).  Its literal keyword arguments may be
added or changed, never removed; its callee, positional arguments and
non-literal keyword arguments must stay the same.  This admits setup changes such
as ``cached(lru)`` -> ``cached(lru, scoped=False)`` when the intended behaviour
changed a default.  Whether the changed test functions are the failing ones is
checked by the Integrity Guard, which knows the failure group.

An eligible assertion is ``assert <expr> == <literal>`` or
``assert <expr> != <literal>`` (either operand order), where exactly one side
is a literal: a constant, a signed number, or a tuple/list/set/dict of
literals.  The tested expression, the operator, the message and every other
statement must stay byte-for-byte identical as syntax trees (comments and
formatting may change).  So imports, aliases, calls, decorators, fixtures,
tests, hooks, skips, ``sys.modules``/``sys.path`` changes, monkeypatching,
``exec``/``eval`` and ``try``/``except`` blocks can neither be added nor
altered - no list of suspicious names is involved.

Not supported (rejected, to be escalated): new tests or assertions, changed
inputs (positional arguments, or anything inside an assertion), removed keyword
arguments, expected values held in variables, ``pytest.approx``,
``self.assertEqual``, parametrize tables, fixtures and test data files.

**Source files (FIX_CODE)** keep the earlier basic checks: the file must parse
and must not newly import pytest/``_pytest`` or add pytest hooks, skips or
hard process exits.  Source behaviour is judged by trusted verification
(``tools.verification``), not by these checks.
"""

from __future__ import annotations

import ast
import tempfile
from collections import Counter
from pathlib import Path
from typing import Final

from testorcode.errors import ToolError
from testorcode.schemas.episode_schemas import EditPolicy, PathClass
from testorcode.tools.git_tools import run_git

_PLACEHOLDER: Final[str] = "__testorcode_expected_literal__"

# Source-file constructs (kind -> description used in rejection reasons).
SKIP_CALL = "skip/xfail call"
SKIP_MARK = "skip/xfail marker"
SKIP_RAISE = "raised skip exception"
AUTOUSE = "autouse fixture"
HOOK = "pytest hook function"
HARD_EXIT = "hard process exit"
PYTEST_INTERNALS = "_pytest import"
PYTEST_IMPORT = "pytest import"

_SKIP_CALLS: Final[frozenset[str]] = frozenset(
    {
        "pytest.skip",
        "pytest.xfail",
        "pytest.importorskip",
        "skip",
        "xfail",
        "importorskip",
        "unittest.skip",
        "unittest.skipIf",
        "unittest.skipUnless",
        "unittest.expectedFailure",
    }
)
_HARD_EXITS: Final[frozenset[str]] = frozenset({"pytest.exit", "os._exit", "os.abort", "os.kill"})
_MARK_SUFFIXES: Final[tuple[str, ...]] = ("mark.skip", "mark.skipif", "mark.xfail")
_SKIP_EXCEPTIONS: Final[tuple[str, ...]] = ("SkipTest", "Skipped", "skip.Exception", "XFailed")


# --------------------------------------------------------------------------- #
# Before/after content
# --------------------------------------------------------------------------- #


def forward_states(
    root: Path, diff: str, paths: list[str], *, reverse: bool
) -> tuple[dict[str, bytes | None], dict[str, bytes | None]]:
    """(before, after) content of ``paths`` for applying ``diff`` *forwards*.

    The diff is applied (``reverse=False``) or reversed in a temporary copy of
    the touched files.  For a revert the current worktree is the "after" state
    of the original patch, so checks always judge the patch as it was applied.

    Raises:
        ToolError: the diff does not apply to the copy (callers check first).
    """
    current = {p: _read(root / p) for p in paths}
    with tempfile.TemporaryDirectory(prefix="testorcode-patch-") as tmp:
        work = Path(tmp)
        for path, content in current.items():
            if content is not None:
                (work / path).parent.mkdir(parents=True, exist_ok=True)
                (work / path).write_bytes(content)
        run_git(work, ["init", "-q"])
        args = ["apply", "--whitespace=nowarn", *(["-R"] if reverse else []), "-"]
        result = run_git(work, args, input_text=diff)
        if result.returncode != 0:
            raise ToolError("patch could not be applied to a scratch copy for content checks")
        other = {p: _read(work / p) for p in paths}
    return (other, current) if reverse else (current, other)


def _read(path: Path) -> bytes | None:
    return path.read_bytes() if path.is_file() and not path.is_symlink() else None


# --------------------------------------------------------------------------- #
# Checks
# --------------------------------------------------------------------------- #


def content_problems(
    before: dict[str, bytes | None],
    after: dict[str, bytes | None],
    policy: EditPolicy,
) -> list[str]:
    """Rejection reasons for the change ``before`` -> ``after`` (empty if acceptable)."""
    problems: list[str] = []
    test_paths = 0
    changes = 0
    for path, old in before.items():
        new = after.get(path)
        if policy.classify_path(path) is PathClass.TEST:
            test_paths += 1
            changed, problem = _test_change(path, old, new)
            changes += changed
            if problem:
                problems.append(problem)
        elif path.endswith(".py"):
            problems.extend(_source_problems(path, old, new))
    if test_paths and not problems and changes == 0:
        problems.append(
            "test-only patch changes no expected literal value or setup keyword argument; only expected "
            "values in existing `assert <expr> ==/!= <literal>` statements and literal keyword arguments "
            "of setup calls in test functions may change"
        )
    return problems


def _test_change(path: str, old: bytes | None, new: bytes | None) -> tuple[int, str | None]:
    """(number of changed expectations/setup calls, rejection reason or None) for one test file."""
    if not path.endswith(".py"):
        return 0, f"{path}: test-only patches may change existing Python test files only"
    if old is None:
        return 0, f"{path}: test-only patches may not add files (new tests, modules or packages)"
    if new is None:
        return 0, f"{path}: test-only patches may not delete or rename test files"
    old_tree, new_tree = _parse(old), _parse(new)
    if old_tree is None or new_tree is None:
        return 0, f"{path}: does not parse as Python before or after the patch"
    old_shape, old_literals, old_calls = _normalise(old_tree)
    new_shape, new_literals, new_calls = _normalise(new_tree)
    if old_shape != new_shape:
        return 0, (
            f"{path}: change is not limited to expected literal values in existing "
            "equality/inequality assertions or literal keyword arguments of setup calls "
            "(code, tests, imports, inputs or assertions changed)"
        )
    removed = sorted({k for a, b in zip(old_calls, new_calls, strict=True) for k in a.keys() - b.keys()})
    if removed:
        return 0, f"{path}: removes keyword argument(s) from a setup call: {', '.join(removed)}"
    changed_literals = sum(a != b for a, b in zip(old_literals, new_literals, strict=True))
    changed_calls = sum(a != b for a, b in zip(old_calls, new_calls, strict=True))
    return changed_literals + changed_calls, None


def _normalise(tree: ast.Module) -> tuple[str, list[str], list[dict[str, str]]]:
    """Tree dump with every eligible expected literal and setup-call literal keyword replaced,
    plus those literals and keyword maps in order."""
    literals: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assert) or not isinstance(node.test, ast.Compare):
            continue
        test = node.test
        if len(test.ops) != 1 or not isinstance(test.ops[0], (ast.Eq, ast.NotEq)):
            continue
        left_literal, right_literal = _is_literal(test.left), _is_literal(test.comparators[0])
        if left_literal == right_literal:  # both literal (ambiguous) or neither
            continue
        placeholder = ast.Name(id=_PLACEHOLDER, ctx=ast.Load())
        if left_literal:
            literals.append(ast.dump(test.left))
            test.left = placeholder
        else:
            literals.append(ast.dump(test.comparators[0]))
            test.comparators = [placeholder]
    calls: list[dict[str, str]] = []
    for function in _test_functions(tree):
        for call in _setup_calls(function):
            literal = [kw for kw in call.keywords if kw.arg is not None and _is_literal(kw.value)]
            calls.append({kw.arg: ast.dump(kw.value) for kw in literal})
            call.keywords = [kw for kw in call.keywords if kw not in literal]
    return ast.dump(tree, include_attributes=False), literals, calls


_EXCLUDED_CALL_ROOTS: Final[frozenset[str]] = frozenset({"pytest", "unittest", "mock"})
_EXCLUDED_CALL_WORDS: Final[tuple[str, ...]] = (
    "assert", "approx", "raises", "warn", "skip", "xfail", "fail", "mark", "fixture", "patch",
    "parametrize", "setattr", "getattr", "exec", "eval", "import", "compile",
)


def _test_functions(tree: ast.Module) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    """``test*`` functions at module level and ``test*`` methods of ``Test*`` classes."""
    found = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test"):
            found.append(node)
        elif isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
            found += [
                m for m in node.body
                if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)) and m.name.startswith("test")
            ]
    return found


def _setup_calls(function: ast.FunctionDef | ast.AsyncFunctionDef) -> list[ast.Call]:
    """Eligible setup calls in the body of ``function`` (never in asserts, decorators or nested scopes)."""
    calls: list[ast.Call] = []

    def visit(node: ast.AST) -> None:
        if isinstance(node, (ast.Assert, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            return
        if isinstance(node, ast.Call) and _setup_call_allowed(node):
            calls.append(node)
        for child in ast.iter_child_nodes(node):
            visit(child)

    for statement in function.body:
        visit(statement)
    return calls


def _setup_call_allowed(call: ast.Call) -> bool:
    name = _dotted(call.func).lower()
    if not name or name.split(".", 1)[0] in _EXCLUDED_CALL_ROOTS:
        return False
    return not any(word in name for word in _EXCLUDED_CALL_WORDS)


def _is_literal(node: ast.expr) -> bool:
    if isinstance(node, ast.Constant):
        return node.value is not Ellipsis
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        return (
            isinstance(node.operand, ast.Constant)
            and isinstance(node.operand.value, (int, float, complex))
            and not isinstance(node.operand.value, bool)
        )
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return all(_is_literal(e) for e in node.elts)
    if isinstance(node, ast.Dict):
        return all(k is not None and _is_literal(k) for k in node.keys) and all(
            _is_literal(v) for v in node.values
        )
    return False


def _source_problems(path: str, old: bytes | None, new: bytes | None) -> list[str]:
    if new is None:
        return []
    new_tree = _parse(new)
    if new_tree is None:
        return [f"{path}: does not parse as Python after the patch"]
    added = _source_constructs(new_tree) - _source_constructs(_parse(old))
    return [f"{path}: introduces {kind} ({count}x)" for kind, count in sorted(added.items())]


def _parse(content: bytes | None) -> ast.Module | None:
    if content is None:
        return None
    try:
        return ast.parse(content)
    except (SyntaxError, ValueError):
        return None


def _dotted(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    if isinstance(node, ast.Call):
        return _dotted(node.func)
    return ""


def _source_constructs(tree: ast.Module | None) -> Counter[str]:
    """Constructs a source file has no business adding (it would only tamper with tests)."""
    found: Counter[str] = Counter()
    if tree is None:
        return found
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = _dotted(node.func)
            if name in _SKIP_CALLS:
                found[SKIP_CALL] += 1
            elif name in _HARD_EXITS:
                found[HARD_EXIT] += 1
            if name.endswith("fixture") and any(kw.arg == "autouse" for kw in node.keywords):
                found[AUTOUSE] += 1
        elif isinstance(node, ast.Attribute) and _dotted(node).endswith(_MARK_SUFFIXES):
            found[SKIP_MARK] += 1
        elif isinstance(node, ast.Raise) and node.exc is not None:
            if _dotted(node.exc).endswith(_SKIP_EXCEPTIONS):
                found[SKIP_RAISE] += 1
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("pytest_"):
                found[HOOK] += 1
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            modules = (
                [a.name for a in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
            )
            for module in modules:
                top = module.split(".", 1)[0]
                if top == "_pytest":
                    found[PYTEST_INTERNALS] += 1
                elif top == "pytest":
                    found[PYTEST_IMPORT] += 1
    return found
