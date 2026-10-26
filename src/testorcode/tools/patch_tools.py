"""Patch validation and application.  Scope rules live in ``EditPolicy``."""

from __future__ import annotations

from pathlib import Path

from testorcode.schemas.contracts import PatchValidation
from testorcode.schemas.episode_schemas import EditPolicy, PatchScope


def validate_patch(
    worktree_path: Path, unified_diff: str, *, policy: EditPolicy, scope: PatchScope
) -> PatchValidation:
    """Static check: diff is well-formed, applies cleanly, touches only allowed paths.

    Never modifies the worktree.  A bad patch is reported in the result, not raised.
    """
    raise NotImplementedError("patch validation is not implemented yet")


def apply_patch(
    worktree_path: Path, unified_diff: str, *, policy: EditPolicy, scope: PatchScope
) -> PatchValidation:
    """Apply a validated patch to the episode worktree (never the user's checkout).

    Raises:
        ScopeViolationError: a path is outside ``policy`` for ``scope``.
        PatchError: the patch does not apply.
    """
    raise NotImplementedError("patch application is not implemented yet")


def revert_patch(worktree_path: Path, unified_diff: str) -> None:
    """Undo a previously applied patch.

    Raises:
        PatchError.
    """
    raise NotImplementedError("patch reverting is not implemented yet")
