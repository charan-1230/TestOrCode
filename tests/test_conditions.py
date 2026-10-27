from __future__ import annotations

import pytest

from testorcode.conditions import PROFILES, episode_for_condition, profile_for
from testorcode.dataset import agent_view_path, load_agent_inputs
from testorcode.schemas.episode_schemas import Availability
from testorcode.schemas.result_schemas import ExperimentCondition as C


@pytest.fixture(scope="module")
def episodes():
    return load_agent_inputs(agent_view_path("dev"))


def test_every_condition_has_a_profile() -> None:
    assert set(PROFILES) == set(C)


def test_core_condition_capabilities() -> None:
    full, b1 = profile_for(C.FULL), profile_for(C.B1)
    assert full.intent_retrieval and full.memory and full.integrity_guard and full.multi_agent
    assert b1.standard_rag and not b1.intent_retrieval and not b1.memory and not b1.multi_agent
    a1, a2, a3 = profile_for(C.A1_NO_INTENT_RETRIEVAL), profile_for(C.A2_NO_MEMORY), profile_for(C.A3_NO_INTEGRITY_GUARD)
    assert not a1.intent_retrieval and not a1.sees_commit_message and a1.memory
    assert a2.intent_retrieval and not a2.memory
    assert a3.intent_retrieval and a3.memory and not a3.integrity_guard
    assert not profile_for(C.TRIVIAL_ALWAYS_FIX_CODE).uses_llm


def test_episode_views(episodes) -> None:
    s4 = next(e for e in episodes if e.visible_memory)
    s1 = next(e for e in episodes if e.visible_memory == [])
    assert episode_for_condition(s4, C.FULL) is s4
    assert episode_for_condition(s4, C.A2_NO_MEMORY).visible_memory is None  # disabled, not empty
    assert episode_for_condition(s4, C.B1).visible_memory is None
    assert episode_for_condition(s1, C.FULL).visible_memory == []
    a1 = episode_for_condition(s4, C.A1_NO_INTENT_RETRIEVAL)
    assert a1.ci_event.commit_message is None and a1.ci_event.commit_message_availability is Availability.WITHHELD
    assert a1.visible_memory == s4.visible_memory
    assert s4.ci_event.commit_message  # the original is untouched
