"""Run selected tests and return a structured result.  Must eventually run sandboxed."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from testorcode.schemas.contracts import PytestExecution


def run_tests(
    worktree_path: Path, selectors: Sequence[str], *, timeout_seconds: int
) -> PytestExecution:
    """Run pytest on ``selectors`` inside ``worktree_path``.

    Test failures are returned (``status=FAILED``); a missing selector or
    collection problem is ``ERROR`` / ``COLLECTION_ERROR``; a kill is ``TIMEOUT``.
    None of these raise.

    Raises:
        ToolError: pytest cannot be started at all.
    """
    raise NotImplementedError("the pytest runner is not implemented yet")
