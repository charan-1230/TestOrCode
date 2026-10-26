"""Project-specific metrics over run records.

Run-level output is ``result_schemas.RunEvaluation`` (decision, evidence, repair,
integrity and escalation dimensions); the harness fills ``ExperimentRunRecord.evaluation``.
Only this module sees ground truth.  ``ConditionSummary`` is a provisional
aggregate shape owned by the evaluation branch and may change.
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict

from testorcode.schemas.episode_schemas import Episode
from testorcode.schemas.result_schemas import (
    ExperimentCondition,
    ExperimentRunRecord,
    RunEvaluation,
)


class ConditionSummary(BaseModel):
    """Metrics for one experimental condition.  ``None`` = not computable, never 0."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    condition: ExperimentCondition
    n_runs: int
    metrics: dict[str, float | None]


def evaluate_run(record: ExperimentRunRecord, episode: Episode) -> RunEvaluation:
    """Score one run against its hidden gold labels.

    Raises:
        ValueError: the record and episode have different episode ids.
    """
    raise NotImplementedError("evaluation is not implemented yet")


def summarize(
    records: Sequence[ExperimentRunRecord], episodes: Sequence[Episode]
) -> list[ConditionSummary]:
    """Aggregate evaluated runs per condition (macro-F1, rates, Recall@k, ...)."""
    raise NotImplementedError("evaluation is not implemented yet")
