"""Small helpers shared by the agents: bounded prompt text and reading worktree files.

Agents read only files inside the episode worktree and never log what they read.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

from testorcode.errors import ToolError

_HUNK_NEW_RANGE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@", re.MULTILINE)


def clip(text: str | None, limit: int) -> str:
    """``text`` cut to ``limit`` characters, marking the cut."""
    if not text:
        return ""
    return text if len(text) <= limit else text[:limit] + f"\n... [{len(text) - limit} characters cut]"


def read_worktree_file(worktree: Path, rel_path: str) -> str | None:
    """Text of ``rel_path`` inside ``worktree``; None if missing.

    Raises:
        ToolError: the path escapes the worktree.
    """
    root = worktree.resolve()
    target = (root / rel_path).resolve()
    if root != target and root not in target.parents:
        raise ToolError(f"path escapes the worktree: {rel_path}")
    if not target.is_file():
        return None
    return target.read_text(encoding="utf-8", errors="replace")


def function_source(module_text: str, name: str) -> str | None:
    """Source of the top-level function or ``Class.method`` called ``name`` (dotted), with line numbers."""
    try:
        tree = ast.parse(module_text)
    except SyntaxError:
        return None
    parts = name.split(".")
    body: list[ast.stmt] = tree.body
    node: ast.AST | None = None
    for part in parts:
        node = next(
            (n for n in body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.name == part),
            None,
        )
        if node is None:
            return None
        body = getattr(node, "body", [])
    assert node is not None
    start = min([node.lineno, *(d.lineno for d in getattr(node, "decorator_list", []))])
    lines = module_text.splitlines()
    return numbered(lines[start - 1 : node.end_lineno], start)


def numbered(lines: list[str], first: int) -> str:
    return "\n".join(f"{n:>5}| {line}" for n, line in enumerate(lines, first))


def changed_windows(module_text: str, file_diff: str, *, context: int = 30, limit: int = 12000) -> str:
    """Numbered windows of the *current* file around every hunk of ``file_diff``."""
    lines = module_text.splitlines()
    ranges: list[tuple[int, int]] = []
    for match in _HUNK_NEW_RANGE.finditer(file_diff):
        start = int(match.group(1))
        length = int(match.group(2)) if match.group(2) is not None else 1
        lo, hi = max(1, start - context), min(len(lines), start + length + context)
        if ranges and lo <= ranges[-1][1] + 1:
            ranges[-1] = (ranges[-1][0], max(ranges[-1][1], hi))
        else:
            ranges.append((lo, hi))
    if not ranges:
        return clip(numbered(lines, 1), limit)
    windows = [numbered(lines[lo - 1 : hi], lo) for lo, hi in ranges]
    return clip("\n     ...\n".join(windows), limit)


def split_diff_by_file(diff: str) -> dict[str, str]:
    """``{path: diff}`` for a multi-file unified diff (keyed by the ``+++ b/`` path)."""
    out: dict[str, list[str]] = {}
    current: list[str] | None = None
    pending: list[str] = []
    for line in diff.splitlines():
        if line.startswith("diff --git ") or (line.startswith("--- ") and current is not None and not pending):
            pending = [line]
            current = None
            continue
        if line.startswith("--- ") and current is None:
            pending.append(line)
            continue
        if line.startswith("+++ "):
            path = line[4:].split("\t", 1)[0]
            path = path.removeprefix("b/")
            current = out.setdefault(path, [])
            current.extend([*pending, line])
            pending = []
            continue
        if current is not None:
            current.append(line)
        else:
            pending.append(line)
    return {path: "\n".join(lines) for path, lines in out.items()}
