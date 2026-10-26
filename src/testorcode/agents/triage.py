"""Triage: run tests, confirm stable failures, group them by affected function."""

from __future__ import annotations

from testorcode.config import Settings
from testorcode.llm_client import LLMClient
from testorcode.schemas.contracts import EpisodeContext, TriageResult
from testorcode.schemas.episode_schemas import CIEvent


def run_triage(
    context: EpisodeContext, ci_event: CIEvent, *, llm: LLMClient, settings: Settings
) -> TriageResult:
    """Run the failing tests, drop unstable ones, group the rest.

    Raises:
        ToolError, ToolTimeoutError: tests cannot be run.
        LLMError: provider failure.
    """
    raise NotImplementedError("triage is not implemented yet")
