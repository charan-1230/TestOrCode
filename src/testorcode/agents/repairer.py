"""Repairer: propose a scoped test or source patch (never applies it)."""

from __future__ import annotations

from testorcode.llm_client import LLMClient
from testorcode.schemas.contracts import (
    EpisodeContext,
    FailureGroupDetail,
    GuardResult,
    IntentAnalysis,
    RepairerOutput,
)


def propose_repair(
    context: EpisodeContext,
    group: FailureGroupDetail,
    analysis: IntentAnalysis,
    *,
    llm: LLMClient,
    previous_rejection: GuardResult | None = None,
) -> RepairerOutput:
    """Return a ``PatchProposal``, or an ``EscalationRequest`` if it cannot repair safely.

    ``previous_rejection`` is set on the single permitted revision.  The proposal
    must stay inside ``context.edit_policy``; the orchestrator re-validates it.

    Raises:
        LLMError, ToolError.
    """
    raise NotImplementedError("the repairer is not implemented yet")
