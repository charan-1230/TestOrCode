"""Experimental conditions as capability switches (one code path, not one application per condition).

Numbering follows the 5-day plan (the priority source), which ``result_schemas.ExperimentCondition`` already uses:
A1 no intent retrieval, A2 no memory, A3 no Integrity Guard, A4 coarse memory, A5 single agent.  **The detailed
proposal swaps A3 and A4** (its A3 is coarse memory, its A4 no Integrity Guard).  The ids stay as they are (saved runs
carry them); ``PROPOSAL_NAME`` and ``DESCRIPTION`` give every table and document the mapping.

``episode_for_condition`` returns the episode view a condition is allowed to see; the orchestrator builds its retriever
from ``profile.intent_retrieval`` / ``profile.standard_rag`` (see ``retrieval``).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

from testorcode.retrieval import ci_message_evidence_id
from testorcode.schemas.episode_schemas import Availability, EpisodeInput, ToolName
from testorcode.schemas.result_schemas import ExperimentCondition as C


@dataclass(frozen=True)
class ConditionProfile:
    intent_retrieval: bool = False  # Intent Analyst retrieval with intent queries
    standard_rag: bool = False  # B1-style search_repo tool with the failing test's text
    memory: bool = False
    memory_scope: Literal["function", "module"] = "function"  # "module" = coarse notes without probes (A4)
    integrity_guard: bool = False
    multi_agent: bool = False
    sees_commit_message: bool = True
    uses_llm: bool = True


_FULL = ConditionProfile(intent_retrieval=True, memory=True, integrity_guard=True, multi_agent=True)

PROFILES: dict[C, ConditionProfile] = {
    C.FULL: _FULL,
    C.B0: ConditionProfile(),  # plain LLM: failing output and diff, no tools or memory
    C.B1: ConditionProfile(standard_rag=True),  # single tool-using agent, standard RAG, no memory
    C.B2: ConditionProfile(standard_rag=True),  # B1 plus a stronger prompt (prompt differs, capabilities do not)
    C.A1_NO_INTENT_RETRIEVAL: ConditionProfile(
        memory=True, integrity_guard=True, multi_agent=True, sees_commit_message=False
    ),
    C.A2_NO_MEMORY: ConditionProfile(intent_retrieval=True, integrity_guard=True, multi_agent=True),
    C.A3_NO_INTEGRITY_GUARD: ConditionProfile(intent_retrieval=True, memory=True, multi_agent=True),
    # Coarse memory: FULL with module-level notes that no probe gates (the store's ``scope="module"``).
    C.A4_COARSE_MEMORY: replace(_FULL, memory_scope="module"),
    C.A5_SINGLE_AGENT: ConditionProfile(intent_retrieval=True, memory=True, integrity_guard=True),
    C.TRIVIAL_ALWAYS_UPDATE_TEST: ConditionProfile(uses_llm=False),
    C.TRIVIAL_ALWAYS_FIX_CODE: ConditionProfile(uses_llm=False),
}


# The condition's name in the detailed proposal (Table "Ablations"/"Baselines"), where it differs from the id.
PROPOSAL_NAME: dict[C, str] = {
    C.A3_NO_INTEGRITY_GUARD: "A4 (proposal)",
    C.A4_COARSE_MEMORY: "A3 (proposal)",
}
DESCRIPTION: dict[C, str] = {
    C.FULL: "Full system",
    C.B0: "Plain LLM: failing output and diff, no tools or memory",
    C.B1: "Single tool-using agent, standard RAG, no memory",
    C.B2: "B1 plus a strong instruction prompt",
    C.A1_NO_INTENT_RETRIEVAL: "No intent retrieval",
    C.A2_NO_MEMORY: "No memory",
    C.A3_NO_INTEGRITY_GUARD: "No Integrity Guard",
    C.A4_COARSE_MEMORY: "Coarse memory: module-level notes without probes",
    C.A5_SINGLE_AGENT: "Single agent, everything else kept",
    C.TRIVIAL_ALWAYS_UPDATE_TEST: "Always UPDATE_TEST",
    C.TRIVIAL_ALWAYS_FIX_CODE: "Always FIX_CODE",
}


def condition_label(condition: C) -> str:
    """``A3 - No Integrity Guard [A4 (proposal)]``: the id, what it removes, and the proposal's name when different."""
    other = PROPOSAL_NAME.get(condition)
    return f"{condition.value} - {DESCRIPTION[condition]}" + (f" [{other}]" if other else "")


# Tools that return change-intent text: withheld from a condition without any retrieval (A1).
_RETRIEVAL_TOOLS = frozenset({ToolName.SEARCH_REPO, ToolName.GIT_LOG, ToolName.GIT_SHOW, ToolName.GIT_BLAME})


def profile_for(condition: C) -> ConditionProfile:
    return PROFILES[condition]


def episode_for_condition(episode: EpisodeInput, condition: C) -> EpisodeInput:
    """The view of ``episode`` that ``condition`` may see.

    * No memory -> ``visible_memory=None`` (the schema's "memory disabled"; ``[]`` would mean enabled but empty).
    * A1 -> the commit message is withheld (``WITHHELD``) because the Analyst sees only failing output and the diff.
    * No retrieval at all (A1: neither intent retrieval nor standard RAG) -> the corpus is withheld (``items=[]``,
      genuinely empty) and ``search_repo`` and the history tools (``git_log``/``git_show``/``git_blame``: earlier commit
      messages are change-intent text) are removed from ``allowed_tools``.  Withholding the commit message alone is
      not enough: every pilot corpus also holds the CI commit's own message (``msg:<sha>``), which any search would
      return.  Memory stays available when the profile has it.
    """
    profile = profile_for(condition)
    update: dict = {}
    if not profile.memory:
        update["visible_memory"] = None
    elif episode.visible_memory is None:
        update["visible_memory"] = []
    if not profile.sees_commit_message:
        event = episode.ci_event.model_copy(
            update={"commit_message": None, "commit_message_availability": Availability.WITHHELD}
        )
        update["ci_event"] = event
        # The corpus must not leak what the prompt withholds: drop the CI commit's own message item (msg:<sha12>).
        drop = ci_message_evidence_id(episode.ci_event.commit.sha)
        if episode.corpus.items:
            update["corpus"] = episode.corpus.model_copy(
                update={"items": [i for i in episode.corpus.items if i.evidence_id != drop]}
            )
    if profile.uses_llm and not (profile.intent_retrieval or profile.standard_rag):
        update["corpus"] = episode.corpus.model_copy(update={"items": []})
        update["allowed_tools"] = [t for t in episode.allowed_tools if t not in _RETRIEVAL_TOOLS]
    return episode.model_copy(update=update) if update else episode
