"""Trivial always-UPDATE_TEST / always-FIX_CODE baselines: deterministic, no model, no gold labels."""

from __future__ import annotations

import pytest

from testorcode.baselines import TRIVIAL_ACTIONS, is_trivial, trivial_record
from testorcode.dataset import agent_view_path, load_agent_inputs
from testorcode.schemas.episode_schemas import DecisionLabel
from testorcode.schemas.result_schemas import (
    ExperimentCondition,
    ExperimentRunRecord,
    RunStatus,
    TestRunStatus,
)

EPISODES = [e for split in ("dev", "test") for e in load_agent_inputs(agent_view_path(split))]


@pytest.mark.parametrize("condition", sorted(TRIVIAL_ACTIONS, key=lambda c: c.value))
def test_one_fixed_decision_per_failing_test_on_every_episode(condition: ExperimentCondition) -> None:
    for episode in EPISODES:
        record = trivial_record(episode, condition, seed=0)
        assert [p.test_id for p in record.predictions] == [t.test_id for t in episode.ci_event.failing_tests]
        assert {p.predicted_action for p in record.predictions} == {TRIVIAL_ACTIONS[condition]}
        assert ExperimentRunRecord.model_validate_json(record.to_jsonl()) == record  # a valid, round-trippable record


def test_it_makes_no_model_call_and_does_nothing_to_a_repository() -> None:
    record = trivial_record(EPISODES[0], ExperimentCondition.TRIVIAL_ALWAYS_FIX_CODE, seed=3)
    assert record.provenance.model_id is None and record.provenance.seed == 3
    assert record.usage.reported and record.usage.llm_calls == 0 and record.usage.total_tokens == 0
    assert not record.patches and not record.tool_events and not record.test_runs
    assert record.final_test_status is TestRunStatus.NOT_RUN  # never claims a test run it did not make
    assert record.outcome.status is RunStatus.COMPLETED


def test_the_decision_does_not_depend_on_the_episode_content() -> None:
    kinds = {
        (p.predicted_action, p.hypothesis, tuple(p.cited_evidence_ids))
        for e in EPISODES
        for p in trivial_record(e, ExperimentCondition.TRIVIAL_ALWAYS_UPDATE_TEST, seed=0).predictions
    }
    assert kinds == {(DecisionLabel.UPDATE_TEST, None, ())}


@pytest.mark.parametrize("condition", [ExperimentCondition.FULL, ExperimentCondition.B1, ExperimentCondition.A2_NO_MEMORY])
def test_other_conditions_are_refused(condition: ExperimentCondition) -> None:
    assert not is_trivial(condition)
    with pytest.raises(ValueError, match="not a trivial baseline"):
        trivial_record(EPISODES[0], condition, seed=0)
