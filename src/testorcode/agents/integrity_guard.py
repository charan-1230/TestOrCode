"""Integrity Guard: independently verify a repair; the only path to memory writes."""

from __future__ import annotations

from testorcode.llm_client import LLMClient
from testorcode.schemas.contracts import (
    EpisodeContext,
    FailureGroupDetail,
    GuardResult,
    IntentAnalysis,
    MemoryApproval,
    PatchProposal,
)
from testorcode.schemas.episode_schemas import MemoryRecord


def verify_repair(
    context: EpisodeContext,
    group: FailureGroupDetail,
    analysis: IntentAnalysis,
    proposal: PatchProposal,
    *,
    llm: LLMClient | None = None,
) -> GuardResult:
    """Check the applied repair: tests, weakening, scope, evidence, mutants.

    Passing tests alone never yields APPROVE.  The worktree already contains
    the applied patch.

    Raises:
        ToolError, ToolTimeoutError.
    """
    raise NotImplementedError("the integrity guard is not implemented yet")


def authorize_memory_write(
    result: GuardResult, record: MemoryRecord, *, maintainer_approved: bool
) -> MemoryApproval:
    """Build the approval ``MemoryStore.write`` requires.

    Raises:
        MemoryAuthorizationError: ``result`` is not APPROVE or the maintainer did not approve.
    """
    raise NotImplementedError("memory authorization is not implemented yet")
