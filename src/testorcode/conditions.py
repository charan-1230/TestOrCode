"""Experimental conditions as capability switches (one code path, not one application per condition).

Numbering follows the 5-day plan (the priority source), which ``result_schemas.ExperimentCondition`` already uses:
A1 no intent retrieval, A2 no memory, A3 no Integrity Guard, A4 coarse memory, A5 single agent.

``episode_for_condition`` returns the episode view a condition is allowed to see; the orchestrator builds its retriever
from ``profile.intent_retrieval`` / ``profile.standard_rag`` (see ``retrieval``).
"""

from __future__ import annotations

from dataclasses import dataclass

from testorcode.schemas.episode_schemas import Availability, EpisodeInput, ToolName
from testorcode.schemas.result_schemas import ExperimentCondition as C


@dataclass(frozen=True)
class ConditionProfile:
    intent_retrieval: bool = False  # Intent Analyst retrieval with intent queries
    standard_rag: bool = False  # B1-style search_repo tool with the failing test's text
    memory: bool = False
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
    C.A4_COARSE_MEMORY: _FULL,  # coarse memory is a memory-store setting, not a retrieval switch
    C.A5_SINGLE_AGENT: ConditionProfile(intent_retrieval=True, memory=True, integrity_guard=True),
    C.TRIVIAL_ALWAYS_UPDATE_TEST: ConditionProfile(uses_llm=False),
    C.TRIVIAL_ALWAYS_FIX_CODE: ConditionProfile(uses_llm=False),
}


def profile_for(condition: C) -> ConditionProfile:
    return PROFILES[condition]


def episode_for_condition(episode: EpisodeInput, condition: C) -> EpisodeInput:
    """The view of ``episode`` that ``condition`` may see.

    * No memory -> ``visible_memory=None`` (the schema's "memory disabled"; ``[]`` would mean enabled but empty).
    * A1 -> the commit message is withheld (``WITHHELD``) because the Analyst sees only failing output and the diff.
    * No retrieval at all (A1: neither intent retrieval nor standard RAG) -> the corpus is withheld (``items=[]``,
      genuinely empty) and ``search_repo`` is removed from ``allowed_tools``.  Withholding the commit message alone is
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
    if profile.uses_llm and not (profile.intent_retrieval or profile.standard_rag):
        update["corpus"] = episode.corpus.model_copy(update={"items": []})
        update["allowed_tools"] = [t for t in episode.allowed_tools if t is not ToolName.SEARCH_REPO]
    return episode.model_copy(update=update) if update else episode
