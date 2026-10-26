"""Episode orchestration: agent order, budgets, revision limit, escalation.

Planned flow for one episode::

    Triage -> for each failure group:
        Intent Analyst -> (ESCALATE?) -> Repairer -> validate_patch -> apply
            -> Integrity Guard --APPROVE--> (maintainer approval) -> memory write
                               --REJECT---> one revision, then ESCALATE

Responsibilities: build ``EpisodeContext``; share one ``LLMClient`` with all
agents; enforce budgets (``BudgetExceededError``); allow at most
``settings.max_repair_revisions`` revisions per rejected repair; turn exceptions
into ``ErrorRecord`` entries; never let an execution failure masquerade as an
escalation; return the complete ``ExperimentRunRecord``.
"""

from __future__ import annotations

from pathlib import Path

from testorcode.config import Settings
from testorcode.llm_client import LLMClient
from testorcode.memory import MemoryStore
from testorcode.schemas.episode_schemas import EpisodeInput
from testorcode.schemas.result_schemas import ExperimentRunRecord


def run_episode(
    episode: EpisodeInput,
    *,
    worktree_path: Path,
    settings: Settings,
    llm: LLMClient,
    memory: MemoryStore | None = None,
) -> ExperimentRunRecord:
    """Run one episode and return its run record.

    Takes ``EpisodeInput`` (never ``Episode``) so gold labels cannot reach the
    agents.  ``memory=None`` disables memory (condition A2).  Execution
    failures are recorded in the returned record, not raised, wherever possible.
    """
    raise NotImplementedError("the orchestrator is not implemented yet")
