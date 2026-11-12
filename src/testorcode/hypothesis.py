"""The pre-registered hypothesis (proposal section 7) and its check against saved runs.

Thresholds (``HypothesisThresholds``, defaults = the proposal's numbers) are fixed in the frozen config before the
test split is touched.  ``check_hypothesis`` reads them, scores the runs with the frozen ``ScoringRules`` and reports
every criterion with its observed value, threshold, ``n`` and outcome (``True``/``False``, or ``None`` = not
computable, e.g. a condition that did not run).  The verdict follows the proposal's rejection rule only:

* **rejected** - the paired-bootstrap 95% CI of the direction macro-F1 difference FULL - B1 includes 0, or B2 closes
  the gap to under ``b2_gap`` (FULL - B2 direction macro-F1 < ``b2_gap``);
* **not decidable** - a rejection test cannot be computed (B1 or B2 missing, too few episodes);
* **not rejected** - neither rejection test fires.  The other thresholds are reported as met or missed, but they do
  not change the verdict; "not rejected" is not "proven".
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from testorcode.evaluation import DEFAULT_RULES, ScoringRules, summarize
from testorcode.report_tables import audit_counts
from testorcode.schemas.episode_schemas import Episode
from testorcode.schemas.result_schemas import ExperimentCondition as C
from testorcode.schemas.result_schemas import ExperimentRunRecord
from testorcode.stats import BOOTSTRAP_SAMPLES, memory_savings, paired_bootstrap


class HypothesisThresholds(BaseModel):
    """Proposal section 7 (provisional there; frozen on dev before the test split)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    min_direction_f1_gain: float = Field(default=0.15, description="FULL - B1 direction macro-F1.")
    max_masked_regression_full: float = 0.05
    min_masked_regression_b1: float = 0.20
    min_retention_full: float = 0.9
    max_retention_b1: float = 0.7
    min_covered_tool_call_saving: float = Field(default=0.25, description="FULL against A2 on covered episodes.")
    max_overgeneralisation_scoped: float = 0.10
    min_overgeneralisation_coarse: float = 0.30
    b2_gap: float = Field(default=0.05, description="Rejected when FULL - B2 direction macro-F1 is below this.")


def _criterion(name: str, observed: float | None, op: str, threshold: float, n: float | None = None) -> dict:
    passed = None if observed is None else (observed >= threshold if op == ">=" else observed <= threshold)
    return {"criterion": name, "observed": observed, "rule": f"{op} {threshold}", "n": n, "met": passed}


def _retention(audit: dict, condition: C) -> tuple[float | None, int]:
    c = audit.get(condition.value, {})
    denominator = c.get("retention_denominator", 0)
    return (c.get("retained", 0) / denominator if denominator else None), denominator


def check_hypothesis(
    records: Sequence[ExperimentRunRecord], episodes: Sequence[Episode], *,
    thresholds: HypothesisThresholds | None = None, rules: ScoringRules = DEFAULT_RULES,
    dataset_version: str | None = None, audit_jsonl: Path | None = None, samples: int = BOOTSTRAP_SAMPLES,
) -> dict:
    t = thresholds or HypothesisThresholds()
    metrics = {s.condition: s.metrics for s in summarize(records, episodes, dataset_version=dataset_version, rules=rules)}

    def metric(condition: C, key: str) -> float | None:
        return metrics.get(condition, {}).get(key)

    full_f1, b1_f1, b2_f1 = (metric(c, "direction_macro_f1") for c in (C.FULL, C.B1, C.B2))
    audit = audit_counts(audit_jsonl) if audit_jsonl is not None and audit_jsonl.is_file() else {}
    retention_full, n_full = _retention(audit, C.FULL)
    retention_b1, n_b1 = _retention(audit, C.B1)
    saving = memory_savings(records, episodes, C.FULL) if C.A2_NO_MEMORY in metrics and C.FULL in metrics else None
    tool_saving = saving["agent_tool_calls"]["saving"] if saving else None

    criteria = [
        _criterion("direction macro-F1 gain, FULL - B1", None if full_f1 is None or b1_f1 is None else full_f1 - b1_f1,
                   ">=", t.min_direction_f1_gain, metric(C.FULL, "n_direction_scored")),
        _criterion("masked-regression rate, FULL", metric(C.FULL, "masked_regression_rate"), "<=",
                   t.max_masked_regression_full),
        _criterion("masked-regression rate, B1 (expected high)", metric(C.B1, "masked_regression_rate"), ">=",
                   t.min_masked_regression_b1),
        _criterion("mutant-kill retention, FULL (audit)", retention_full, ">=", t.min_retention_full, n_full),
        _criterion("mutant-kill retention, B1 (audit, expected low)", retention_b1, "<=", t.max_retention_b1, n_b1),
        _criterion("agent tool-call saving on covered episodes, FULL vs A2", tool_saving, ">=",
                   t.min_covered_tool_call_saving, saving["n_covered"] if saving else None),
        _criterion("over-generalisation on decoys, FULL (scoped memory)",
                   metric(C.FULL, "memory_over_generalisation_rate"), "<=", t.max_overgeneralisation_scoped,
                   metric(C.FULL, "n_decoy_runs_scored")),
        _criterion("over-generalisation on decoys, A4 (coarse memory)",
                   metric(C.A4_COARSE_MEMORY, "memory_over_generalisation_rate"), ">=", t.min_overgeneralisation_coarse,
                   metric(C.A4_COARSE_MEMORY, "n_decoy_runs_scored")),
    ]

    present = {r.condition for r in records}
    ci = paired_bootstrap(records, C.FULL, C.B1, episodes, samples=samples, rules=rules) \
        if {C.FULL, C.B1} <= present else None
    ci_includes_zero = None if ci is None or ci.get("ci95") is None else not ci["ci_excludes_zero"]
    b2_gap = None if full_f1 is None or b2_f1 is None else full_f1 - b2_f1
    b2_closes = None if b2_gap is None else b2_gap < t.b2_gap
    rejection = [
        {"test": "bootstrap 95% CI of FULL - B1 direction macro-F1 includes 0", "fires": ci_includes_zero,
         "ci95": ci.get("ci95") if ci else None, "n_episodes": ci.get("n_episodes") if ci else None},
        {"test": f"B2 closes the gap: FULL - B2 direction macro-F1 < {t.b2_gap}", "fires": b2_closes,
         "observed_gap": b2_gap},
    ]
    if any(r["fires"] for r in rejection):
        verdict = "rejected"
    elif any(r["fires"] is None for r in rejection):
        verdict = "not decidable"
    else:
        verdict = "not rejected"
    return {"verdict": verdict, "rejection_tests": rejection, "criteria": criteria,
            "thresholds": t.model_dump(), "scoring_rules": rules.model_dump()}


def markdown(result: dict) -> list[str]:
    def f(v) -> str:
        return "-" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))

    verdict = f"**Verdict: {result['verdict']}** (rejection rule only; thresholds below are reported, not voted)."
    lines = ["## Pre-registered hypothesis (proposal section 7)", "", verdict, "",
             "| Rejection test | Fires | Detail |", "|---|---|---|"]
    for r in result["rejection_tests"]:
        detail = f"CI {r['ci95']}, {r['n_episodes']} episodes" if "ci95" in r else f"gap {f(r['observed_gap'])}"
        lines.append(f"| {r['test']} | {f(r['fires'])} | {detail} |")
    lines += ["", "| Criterion | Observed | Threshold | n | Met |", "|---|---|---|---|---|"]
    for c in result["criteria"]:
        lines.append(f"| {c['criterion']} | {f(c['observed'])} | {c['rule']} | {f(c['n'])} | {f(c['met'])} |")
    return lines
