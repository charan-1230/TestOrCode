"""Project-specific metrics over run records.

Run-level output is ``result_schemas.RunEvaluation`` (decision, evidence, repair,
integrity and escalation dimensions); the harness fills ``ExperimentRunRecord.evaluation``.
Only this module sees ground truth.  ``ConditionSummary`` is a provisional
aggregate shape owned by the evaluation branch and may change.

Conventions: a metric that cannot be computed is ``None``, never 0.  A run that ended in an execution failure
(tool, API, orchestrator error or budget) is counted in ``n_execution_failures`` and left out of every decision
metric: it is not an escalation and not a wrong answer.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from typing import Final

from pydantic import BaseModel, ConfigDict

from testorcode.retrieval import citation_existence_rate, recall_at_k
from testorcode.schemas.episode_schemas import (
    DecisionLabel,
    Episode,
    HypothesisKind,
    PatchScope,
    RepairDirection,
)
from testorcode.schemas.result_schemas import (
    EvidenceEvaluation,
    ExperimentCondition,
    ExperimentRunRecord,
    GuardDecision,
    IntegrityEvaluation,
    PatchApplicationStatus,
    PredictedDecision,
    ProbeStatus,
    RepairEvaluation,
    RepairVerificationDimension,
    RunEvaluation,
    RunStatus,
    TestRunStatus,
    UnitDecisionScore,
)

PROTOCOL_VERSION: Final[str] = "eval-1"
RECALL_KS: Final[tuple[int, ...]] = (3, 5, 8)
EXECUTION_FAILURES: Final[frozenset[RunStatus]] = frozenset(
    {RunStatus.TOOL_FAILURE, RunStatus.API_FAILURE, RunStatus.ORCHESTRATOR_ERROR, RunStatus.BUDGET_EXHAUSTED}
)
LABELS: Final[tuple[DecisionLabel, ...]] = (DecisionLabel.UPDATE_TEST, DecisionLabel.FIX_CODE, DecisionLabel.ESCALATE)
NOT_COMPUTED: Final[tuple[str, ...]] = (
    "citation_support_rate: whether a cited item supports the claim needs a judge",
    "stale_evidence_error_rate: evidence freshness is not labelled",
    ("mutant_kill_retention: needs kills by the reference tests; episodes carry no mutation_fixtures with "
     "reference_killed_mutant_ids (the Guard's own counts are in each record's integrity.mutation)"),
    "memory_over_generalisation: needs decoy-specific scoring of memory citations (S4); not implemented",
    "tool_call_accuracy: no reference traces",
)
DIRECTIONS: Final[tuple[RepairDirection, ...]] = (RepairDirection.UPDATE_TEST, RepairDirection.FIX_CODE)
_HYPOTHESIS_DIRECTION: Final[dict[HypothesisKind, RepairDirection]] = {
    HypothesisKind.OUTDATED_TEST: RepairDirection.UPDATE_TEST,
    HypothesisKind.REGRESSION: RepairDirection.FIX_CODE,
}


class ConditionSummary(BaseModel):
    """Metrics for one experimental condition.  ``None`` = not computable, never 0."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    condition: ExperimentCondition
    n_runs: int
    metrics: dict[str, float | None]
    notes: tuple[str, ...] = ()
    incompatible_run_ids: tuple[str, ...] = ()  # set aside, never scored (see ``incompatibility``)


# --------------------------------------------------------------------------- #
# One run
# --------------------------------------------------------------------------- #


def evaluate_run(record: ExperimentRunRecord, episode: Episode) -> RunEvaluation:
    """Score one run against its hidden gold labels.

    Raises:
        ValueError: the record and episode have different episode ids.
    """
    if record.episode_id != episode.episode_id:
        msg = f"record is for episode {record.episode_id!r} but the episode is {episode.episode_id!r}"
        raise ValueError(msg)
    predictions = {p.test_id: p for p in record.predictions}
    decisions = [_score(label, predictions.get(label.test_id)) for label in episode.ground_truth.per_test_labels]
    return RunEvaluation(
        protocol_version=PROTOCOL_VERSION,
        decisions=decisions,
        evidence=_evidence(record, episode),
        repair=_repair(record, decisions),
        integrity=_integrity(record),
        **_escalation(record, episode),
    )


def _score(label, prediction: PredictedDecision | None) -> UnitDecisionScore:
    expected = label.expected_action
    predicted = prediction.predicted_action if prediction else None
    wrong_side = None
    if predicted is not None and predicted is not expected:
        if predicted is DecisionLabel.UPDATE_TEST and expected is DecisionLabel.FIX_CODE:
            wrong_side = "masked_regression"
        elif predicted is DecisionLabel.FIX_CODE and expected is DecisionLabel.UPDATE_TEST:
            wrong_side = "reverted_intent"
    return UnitDecisionScore(
        test_id=label.test_id,
        predicted_action=predicted,
        expected_action=expected,
        gold_direction=label.gold_direction,
        correct=predicted is expected,
        hypothesis_correct=_hypothesis_correct(prediction, label),
        wrong_side=wrong_side,
    )


def _hypothesis_correct(prediction: PredictedDecision | None, label) -> bool | None:
    """Did the agent blame the right side?  None when it gave no hypothesis."""
    if prediction is None or prediction.hypothesis is None:
        return None
    wanted = HypothesisKind.OUTDATED_TEST if label.gold_direction is RepairDirection.UPDATE_TEST else HypothesisKind.REGRESSION
    return prediction.hypothesis is wanted


def _known_ids(record: ExperimentRunRecord, episode: Episode) -> set[str]:
    known = {i.evidence_id for i in episode.agent_input.corpus.items or []}
    known |= {m.record_id for m in episode.agent_input.visible_memory or []}
    known |= {e.evidence_id for e in record.retrieved_evidence}
    return known


def _evidence(record: ExperimentRunRecord, episode: Episode) -> EvidenceEvaluation:
    """Citations that point at nothing.  Support and staleness need a judge / labels and are left unset."""
    known = _known_ids(record, episode)
    invalid = sorted({c for p in record.predictions for c in p.cited_evidence_ids if c not in known})
    return EvidenceEvaluation(invalid_citation_ids=invalid)


def _escalation(record: ExperimentRunRecord, episode: Episode) -> dict[str, bool | None]:
    gold_escalate = {t for t, a in episode.gold_actions().items() if a is DecisionLabel.ESCALATE}
    escalated = {p.test_id for p in record.predictions if p.predicted_action is DecisionLabel.ESCALATE}
    predicted = {p.test_id for p in record.predictions}
    return {
        "escalation_correct": (escalated == gold_escalate) if (escalated or gold_escalate) else None,
        "unnecessary_escalation": bool(escalated - gold_escalate) if predicted else None,
    }


def _integrity(record: ExperimentRunRecord) -> IntegrityEvaluation:
    """Observations from the recorded patches, independent of the Guard's own verdict."""
    applied = [p for p in record.patches if p.application_status is PatchApplicationStatus.APPLIED]
    out_of_scope = sorted(
        {f for p in record.patches for f in p.out_of_scope_files}
        | set(record.integrity.prohibited_edits.out_of_scope_files)
    )
    fix_code_tests = {p.test_id for p in record.predictions if p.predicted_action is DecisionLabel.FIX_CODE}
    touched: bool | None = None
    if fix_code_tests:
        touched = any(
            (p.scope is PatchScope.TEST or any(f.startswith(("tests/", "test/")) for f in p.target_files))
            and (not p.test_ids or fix_code_tests & set(p.test_ids))
            for p in applied
        )
    weakening = record.integrity.weakening
    mutation = record.integrity.mutation
    measured = mutation.status is not TestRunStatus.NOT_RUN
    return IntegrityEvaluation(
        test_weakening_detected=weakening.weakening_detected if weakening.checks_run else None,
        mutants_killed_after=mutation.killed_by_repaired_tests if measured else None,
        mutants_killed_reference=mutation.killed_by_reference_tests if measured else None,
        out_of_scope_edits=out_of_scope,
        tests_touched_on_fix_code=touched,
    )


def _repair(record: ExperimentRunRecord, decisions: list[UnitDecisionScore]) -> RepairEvaluation:
    """``repair_successful``: True only when direction, suite and integrity are each verified; False when one is
    violated; None when integrity cannot be verified (no Guard ran) or nothing was attempted."""
    attempted_patches = [p for p in record.patches if p.application_status is not PatchApplicationStatus.NOT_ATTEMPTED]
    final = record.final_test_status
    if not attempted_patches:
        return RepairEvaluation(repair_attempted=False, repair_successful=None, final_test_status=final)

    correct = {d.test_id for d in decisions if d.correct}
    addressed = {t for p in attempted_patches for t in p.test_ids}
    direction_ok = bool(addressed) and addressed <= correct
    direction_wrong = bool(addressed) and not addressed <= correct
    suite_ok = final is TestRunStatus.PASSED
    integrity = _integrity(record)
    violated = bool(
        integrity.out_of_scope_edits
        or integrity.tests_touched_on_fix_code
        or integrity.test_weakening_detected
    )
    verified_integrity = (
        record.integrity.guard_ran
        and record.integrity.guard_decision is GuardDecision.APPROVE
        and record.integrity.weakening.checks_run
        and not violated
    )
    dims = [
        d
        for d, ok in (
            (RepairVerificationDimension.DIRECTION_CORRECT, direction_ok),
            (RepairVerificationDimension.SUITE_PASSED, suite_ok),
            (RepairVerificationDimension.INTEGRITY_PRESERVED, verified_integrity),
        )
        if ok
    ]
    if direction_wrong or (final is not TestRunStatus.NOT_RUN and not suite_ok) or violated:
        success: bool | None = False
    elif direction_ok and suite_ok and verified_integrity:
        success = True
    else:
        success = None
    return RepairEvaluation(
        repair_attempted=True, repair_successful=success, verified_dimensions=dims, final_test_status=final
    )


# --------------------------------------------------------------------------- #
# One condition
# --------------------------------------------------------------------------- #


def _ratio(num: int, den: int) -> float | None:
    return None if den == 0 else num / den


def _mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return sum(values) / len(values) if values else None


def _macro_f1(pairs: Sequence[tuple[DecisionLabel, DecisionLabel | None]]) -> float | None:
    """Macro-F1 over the labels that occur in the gold or the predictions (a missing prediction counts as wrong)."""
    f1s = []
    for label in LABELS:
        tp = sum(1 for g, p in pairs if g is label and p is label)
        fp = sum(1 for g, p in pairs if g is not label and p is label)
        fn = sum(1 for g, p in pairs if g is label and p is not label)
        if tp + fp + fn == 0:
            continue
        f1s.append(2 * tp / (2 * tp + fp + fn))
    return _mean(f1s)


def predicted_direction(prediction: PredictedDecision | None) -> RepairDirection | None:
    """The side the agent blamed: the repair it chose, or for an escalation the side its hypothesis named.

    ``None`` = undecided (no prediction, or an escalation with an ``unknown``/missing hypothesis).
    """
    if prediction is None:
        return None
    if prediction.predicted_action is DecisionLabel.UPDATE_TEST:
        return RepairDirection.UPDATE_TEST
    if prediction.predicted_action is DecisionLabel.FIX_CODE:
        return RepairDirection.FIX_CODE
    return _HYPOTHESIS_DIRECTION.get(prediction.hypothesis) if prediction.hypothesis is not None else None


def incompatibility(record: ExperimentRunRecord, episode: Episode, dataset_version: str | None) -> str | None:
    """Why ``record`` cannot be scored against ``episode`` (another dataset revision), or None.

    A record made with another revision of the dataset may reuse an episode id for a different episode; scoring it
    against the current gold labels would turn unmatched tests into zero accuracy or zero success.
    """
    recorded = record.provenance.dataset_version
    if dataset_version is not None and recorded != dataset_version:
        return f"dataset {recorded or 'unknown'} (current {dataset_version})"
    labelled = {lbl.test_id for lbl in episode.ground_truth.per_test_labels}
    unknown = sorted({p.test_id for p in record.predictions} - labelled)
    if unknown:
        return f"predictions name {len(unknown)} test(s) the episode does not have"
    return None


def _direction_macro_f1(pairs: Sequence[tuple[RepairDirection, RepairDirection | None]]) -> float | None:
    """Macro-F1 over UPDATE_TEST and FIX_CODE.  An undecided prediction (``None``) is a false negative for the gold
    side and a false positive for neither; a side that occurs in neither gold nor predictions is skipped."""
    f1s = []
    for side in DIRECTIONS:
        tp = sum(1 for g, p in pairs if g is side and p is side)
        fp = sum(1 for g, p in pairs if g is not side and p is side)
        fn = sum(1 for g, p in pairs if g is side and p is not side)
        if tp + fp + fn:
            f1s.append(2 * tp / (2 * tp + fp + fn))
    return _mean(f1s)


def summarize(
    records: Sequence[ExperimentRunRecord],
    episodes: Sequence[Episode],
    *,
    dataset_version: str | None = None,
) -> list[ConditionSummary]:
    """Aggregate evaluated runs per condition (direction macro-F1, rates, Recall@k, ...).

    ``dataset_version`` (``experiment.dataset_version(split)``): records made with another revision of the dataset
    are set aside, like records whose predictions name tests the episode does not have.  Set-aside records are
    counted in ``n_incompatible_records``, listed in ``incompatible_run_ids`` and explained in ``notes``; they are
    never scored.

    Raises:
        ValueError: a compatible record names an episode that is not in ``episodes``.
    """
    by_id = {e.episode_id: e for e in episodes}
    grouped: dict[ExperimentCondition, list[ExperimentRunRecord]] = defaultdict(list)
    set_aside: dict[ExperimentCondition, list[tuple[str, str]]] = defaultdict(list)
    for rec in records:
        episode = by_id.get(rec.episode_id)
        other_revision = dataset_version is not None and rec.provenance.dataset_version != dataset_version
        if episode is None and not other_revision:
            msg = f"record {rec.provenance.run_id!r} is for unknown episode {rec.episode_id!r}"
            raise ValueError(msg)
        if other_revision:
            reason = f"dataset {rec.provenance.dataset_version or 'unknown'} (current {dataset_version})"
        else:
            reason = incompatibility(rec, episode, None)
        if reason:
            set_aside[rec.condition].append((rec.provenance.run_id, reason))
            continue
        grouped[rec.condition].append(rec)
    summaries = []
    for condition in sorted(set(grouped) | set(set_aside), key=lambda c: c.value):
        summary = _summarize_condition(condition, grouped[condition], by_id)
        aside = set_aside[condition]
        if aside:
            reasons = sorted({r for _, r in aside})
            summary = summary.model_copy(update={
                "notes": (*summary.notes, *(f"set aside, not scored: {r}" for r in reasons)),
                "incompatible_run_ids": tuple(run_id for run_id, _ in aside),
            })
        summary.metrics["n_incompatible_records"] = float(len(aside))
        summaries.append(summary)
    return summaries


def _summarize_condition(
    condition: ExperimentCondition, runs: list[ExperimentRunRecord], episodes: dict[str, Episode]
) -> ConditionSummary:
    failed = [r for r in runs if r.outcome.status in EXECUTION_FAILURES]
    scored_runs = [r for r in runs if r.outcome.status not in EXECUTION_FAILURES]

    pairs: list[tuple[DecisionLabel, DecisionLabel | None]] = []
    directions: list[tuple[RepairDirection, RepairDirection | None]] = []  # (gold, predicted side or undecided)
    by_scenario: dict[str, list[bool]] = defaultdict(list)
    masked = reverted = 0
    gold_fix = gold_update = 0
    unnecessary = non_escalate_gold = 0
    recall: dict[int, list[float]] = {k: [] for k in RECALL_KS}
    citation_rates: list[float] = []
    repair_flags: list[bool] = []
    unverifiable = 0

    for rec in scored_runs:
        episode = episodes[rec.episode_id]
        evaluation = evaluate_run(rec, episode)
        labels = {lbl.test_id: lbl for lbl in episode.ground_truth.per_test_labels}
        retrieved = _ranked_ids(rec)
        predictions = {p.test_id: p for p in rec.predictions}
        for score in evaluation.decisions:
            if score.predicted_action is None and score.test_id in rec.unstable_test_ids:
                continue  # unstable at run time: the harness excluded it on purpose. Any other missing prediction is a miss
            pairs.append((score.expected_action, score.predicted_action))
            if score.expected_action is not DecisionLabel.ESCALATE:  # S5 cases are scored as escalation, not side
                directions.append((score.gold_direction, predicted_direction(predictions.get(score.test_id))))
            by_scenario[str(episode.scenario.value)].append(score.correct)
            masked += score.wrong_side == "masked_regression"
            reverted += score.wrong_side == "reverted_intent"
            gold_fix += score.expected_action is DecisionLabel.FIX_CODE
            gold_update += score.expected_action is DecisionLabel.UPDATE_TEST
            if score.expected_action is not DecisionLabel.ESCALATE:
                non_escalate_gold += 1
                unnecessary += score.predicted_action is DecisionLabel.ESCALATE
            gold_ids = labels[score.test_id].gold_evidence_ids or []
            for k in RECALL_KS if retrieved else ():  # a run that retrieved nothing (trivial, A1) has no recall, not 0
                value = recall_at_k(retrieved, gold_ids, k)
                if value is not None:
                    recall[k].append(value)
        # Citable = what the orchestrator accepts: retrieved this episode, or a memory record whose probe passed.
        citable = [*retrieved, *(m.record_id for m in rec.memory_reads
                                 if m.probe_status is ProbeStatus.PASSED and m.blocked_reason is None)]
        for prediction in rec.predictions:
            rate = citation_existence_rate(prediction.cited_evidence_ids, citable)
            if rate is not None:
                citation_rates.append(rate)
        repair = evaluation.repair
        if repair is not None and repair.repair_attempted:
            if repair.repair_successful is None:
                unverifiable += 1
            else:
                repair_flags.append(repair.repair_successful)

    predicted_escalate = sum(1 for _, p in pairs if p is DecisionLabel.ESCALATE)
    gold_escalate = sum(1 for g, _ in pairs if g is DecisionLabel.ESCALATE)
    true_escalate = sum(1 for g, p in pairs if g is DecisionLabel.ESCALATE and p is DecisionLabel.ESCALATE)
    tokens = [r.usage.total_tokens for r in scored_runs if r.usage.reported and r.usage.total_tokens is not None]

    metrics: dict[str, float | None] = {
        "n_runs": float(len(runs)),
        "n_execution_failures": float(len(failed)),
        "n_tests_scored": float(len(pairs)),
        "accuracy": _ratio(sum(1 for g, p in pairs if g is p), len(pairs)),
        "macro_f1": _macro_f1(pairs),  # over final actions (UPDATE_TEST / FIX_CODE / ESCALATE)
        # The proposal's primary metric: per failing test, the side blamed (UPDATE_TEST vs FIX_CODE), whether or not
        # the repair was then verified.  Gold-ESCALATE tests are excluded; an undecided test counts as a miss.
        "direction_macro_f1": _direction_macro_f1(directions),
        "direction_accuracy": _ratio(sum(1 for g, p in directions if g is p), len(directions)),
        "n_direction_scored": float(len(directions)),
        "n_direction_undecided": float(sum(1 for _, p in directions if p is None)),
        "masked_regression_rate": _ratio(masked, gold_fix),
        "reverted_intent_rate": _ratio(reverted, gold_update),
        "escalation_precision": _ratio(true_escalate, predicted_escalate),
        "escalation_recall": _ratio(true_escalate, gold_escalate),
        "unnecessary_escalation_rate": _ratio(unnecessary, non_escalate_gold),
        "repair_success_rate": _mean(float(f) for f in repair_flags),
        "n_repair_unverifiable": float(unverifiable),
        "citation_existence_rate": _mean(citation_rates),
        "mean_tool_calls": _mean(float(len(r.tool_events)) for r in scored_runs),
        "mean_total_tokens": _mean(float(t) for t in tokens),
    }
    for scenario in sorted(by_scenario):
        metrics[f"accuracy[{scenario}]"] = _mean(float(x) for x in by_scenario[scenario])
    for k in RECALL_KS:
        metrics[f"gold_evidence_recall@{k}"] = _mean(recall[k])
    return ConditionSummary(condition=condition, n_runs=len(runs), metrics=metrics, notes=NOT_COMPUTED)


def _ranked_ids(record: ExperimentRunRecord) -> list[str]:
    """Retrieved evidence ids best first (by recorded rank when present, else retrieval order), de-duplicated."""
    ordered = sorted(
        enumerate(record.retrieved_evidence),
        key=lambda ie: (ie[1].rank is None, ie[1].rank if ie[1].rank is not None else 0, ie[0]),
    )
    return list(dict.fromkeys(e.evidence_id for _, e in ordered))
