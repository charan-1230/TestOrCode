"""Intent Analyst: retrieve change-intent evidence and form a hypothesis per group."""

from __future__ import annotations

from testorcode.llm_client import LLMClient
from testorcode.memory import MemoryStore
from testorcode.retrieval import Retriever
from testorcode.schemas.contracts import EpisodeContext, FailureGroupDetail, IntentAnalysis


def analyse_intent(
    context: EpisodeContext,
    group: FailureGroupDetail,
    *,
    retriever: Retriever,
    memory: MemoryStore | None,
    llm: LLMClient,
) -> IntentAnalysis:
    """Return a hypothesis and recommended decision for one failure group.

    Every cited id must come from ``retriever`` or ``memory`` (the orchestrator
    verifies with ``unknown_evidence_ids``).  Insufficient or conflicting
    evidence yields ``ESCALATE``, not a guess.  ``memory=None`` = disabled.

    Raises:
        LLMError, ToolError.
    """
    raise NotImplementedError("the intent analyst is not implemented yet")
