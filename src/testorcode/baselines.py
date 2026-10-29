"""Deterministic trivial baselines: always UPDATE_TEST and always FIX_CODE.

They make no model call, no tool call and never touch a repository, so they need no worktree, network or API key.
The record is a pure function of the agent view (``EpisodeInput``); gold labels are never read.  A trivial baseline
never repairs, so it carries no patch and no test run (``final_test_status`` stays ``NOT_RUN``) and is scored on
decisions only.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Final

from testorcode.schemas.episode_schemas import DecisionLabel, EpisodeInput
from testorcode.schemas.result_schemas import (
    AgentRole,
    ExperimentCondition,
    ExperimentRunRecord,
    PredictedDecision,
    RunOutcome,
    RunProvenance,
    RunStatus,
    TokenUsage,
)

TRIVIAL_ACTIONS: Final[dict[ExperimentCondition, DecisionLabel]] = {
    ExperimentCondition.TRIVIAL_ALWAYS_UPDATE_TEST: DecisionLabel.UPDATE_TEST,
    ExperimentCondition.TRIVIAL_ALWAYS_FIX_CODE: DecisionLabel.FIX_CODE,
}


def is_trivial(condition: ExperimentCondition) -> bool:
    return condition in TRIVIAL_ACTIONS


def trivial_record(
    episode: EpisodeInput,
    condition: ExperimentCondition,
    *,
    seed: int | None,
    code_revision: str | None = None,
) -> ExperimentRunRecord:
    """The run record of a trivial baseline on ``episode``: one fixed decision per failing test.

    Raises:
        ValueError: ``condition`` is not a trivial baseline.
    """
    if condition not in TRIVIAL_ACTIONS:
        msg = f"{condition.value} is not a trivial baseline; use {sorted(c.value for c in TRIVIAL_ACTIONS)}"
        raise ValueError(msg)
    action = TRIVIAL_ACTIONS[condition]
    started = datetime.now(UTC)
    predictions = [
        PredictedDecision(
            test_id=t.test_id,
            predicted_action=action,
            hypothesis=None,
            rationale=f"trivial baseline: always {action.value}",
            decided_by=AgentRole.ORCHESTRATOR,
        )
        for t in episode.ci_event.failing_tests
    ]
    return ExperimentRunRecord(
        provenance=RunProvenance(
            run_id=f"{episode.episode_id}-{uuid.uuid4().hex[:12]}",
            episode_id=episode.episode_id,
            condition=condition,
            seed=seed,
            model_id=None,
            code_revision=code_revision,
            started_at=started,
            ended_at=datetime.now(UTC),
        ),
        predictions=predictions,
        usage=TokenUsage(reported=True, input_tokens=0, output_tokens=0, total_tokens=0, llm_calls=0),
        outcome=RunOutcome(status=RunStatus.COMPLETED),
    )
