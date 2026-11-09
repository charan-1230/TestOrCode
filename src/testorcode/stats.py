"""Paired statistics between conditions (proposal section 11: "paired bootstrap and McNemar tests, 2 seeds").

All comparisons pair runs of two conditions on the **same episode and seed**; a pair where either run ended in an
execution failure is dropped (and counted).  Pure Python and deterministic (fixed resampling seed).

* ``paired_bootstrap`` - 95% percentile CI of the difference in direction macro-F1 (or final-action accuracy) between
  two conditions, resampling **episodes** with replacement (the tests of one episode are not independent, so the
  episode is the unit).  The proposal's hypothesis is rejected when this CI includes 0.
* ``mcnemar`` - exact two-sided McNemar test on paired per-test correctness (direction or final action).
* ``seed_summary`` - per condition, the metric for each seed and its mean and sample standard deviation.
* ``memory_savings`` - agent tool calls and model calls of a memory condition on episodes covered by an earlier
  approved record, against the same episodes and seeds in A2 (no memory).

Small samples give wide intervals and large p-values; the numbers are reported with their ``n``, never alone.
"""

from __future__ import annotations

import math
import random
import statistics
from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import Final, Literal

from testorcode.evaluation import (
    EXECUTION_FAILURES,
    _direction_macro_f1,
    agent_tool_events,
    covered_by_memory,
    evaluate_run,
    predicted_direction,
)
from testorcode.schemas.episode_schemas import DecisionLabel, Episode, RepairDirection
from testorcode.schemas.result_schemas import ExperimentCondition, ExperimentRunRecord

BOOTSTRAP_SAMPLES: Final[int] = 10000
BOOTSTRAP_SEED: Final[int] = 20261108

Metric = Literal["direction_macro_f1", "final_action_accuracy"]
Correctness = Literal["direction", "final_action"]

# One scored unit: (episode id, seed, test id, gold direction, predicted direction, final action correct).
_Unit = tuple[str, int | None, str, RepairDirection | None, RepairDirection | None, bool]


def _units(records: Sequence[ExperimentRunRecord], episodes: dict[str, Episode]) -> dict[tuple[str, int | None], list[_Unit]]:
    """Per (episode, seed): the scored tests of one condition's completed run."""
    out: dict[tuple[str, int | None], list[_Unit]] = {}
    for rec in records:
        if rec.outcome.status in EXECUTION_FAILURES or rec.episode_id not in episodes:
            continue
        episode = episodes[rec.episode_id]
        predictions = {p.test_id: p for p in rec.predictions}
        units = []
        for d in evaluate_run(rec, episode).decisions:
            if d.predicted_action is None and d.test_id in rec.unstable_test_ids:
                continue
            gold_side = None if d.expected_action is DecisionLabel.ESCALATE else d.gold_direction
            units.append((rec.episode_id, rec.provenance.seed, d.test_id, gold_side,
                          predicted_direction(predictions.get(d.test_id)), d.correct))
        out[(rec.episode_id, rec.provenance.seed)] = units
    return out


def _metric(units: Sequence[_Unit], metric: Metric) -> float | None:
    if metric == "final_action_accuracy":
        return sum(u[5] for u in units) / len(units) if units else None
    return _direction_macro_f1([(u[3], u[4]) for u in units if u[3] is not None])


def _paired(records: Sequence[ExperimentRunRecord], a: ExperimentCondition, b: ExperimentCondition,
            episodes: Sequence[Episode]):
    by_id = {e.episode_id: e for e in episodes}
    ua = _units([r for r in records if r.condition is a], by_id)
    ub = _units([r for r in records if r.condition is b], by_id)
    keys = sorted(set(ua) & set(ub), key=lambda k: (k[0], -1 if k[1] is None else k[1]))
    dropped = len(set(ua) ^ set(ub))
    return ua, ub, keys, dropped


def paired_bootstrap(
    records: Sequence[ExperimentRunRecord], a: ExperimentCondition, b: ExperimentCondition, episodes: Sequence[Episode],
    *, metric: Metric = "direction_macro_f1", samples: int = BOOTSTRAP_SAMPLES, seed: int = BOOTSTRAP_SEED,
) -> dict:
    """``metric(a) - metric(b)`` on paired runs, with a 95% percentile bootstrap CI over episodes.

    Every seed of a resampled episode is kept together (the episode is the resampling unit).
    """
    ua, ub, keys, dropped = _paired(records, a, b, episodes)
    episode_ids = sorted({k[0] for k in keys})
    by_episode: dict[str, list[tuple[str, int | None]]] = defaultdict(list)
    for k in keys:
        by_episode[k[0]].append(k)

    def diff(ids: Sequence[str]) -> float | None:
        sa = [u for e in ids for k in by_episode[e] for u in ua[k]]
        sb = [u for e in ids for k in by_episode[e] for u in ub[k]]
        ma, mb = _metric(sa, metric), _metric(sb, metric)
        return None if ma is None or mb is None else ma - mb

    observed = diff(episode_ids)
    result = {"a": a.value, "b": b.value, "metric": metric, "n_episodes": len(episode_ids), "n_pairs": len(keys),
              "n_unpaired_dropped": dropped, "difference": observed, "ci95": None, "samples": samples}
    if observed is None or len(episode_ids) < 2:
        return result
    rng = random.Random(seed)
    values = sorted(v for v in (diff([rng.choice(episode_ids) for _ in episode_ids]) for _ in range(samples))
                    if v is not None)
    if values:
        result["ci95"] = (_percentile(values, 0.025), _percentile(values, 0.975))
        result["ci_excludes_zero"] = result["ci95"][0] > 0 or result["ci95"][1] < 0
    return result


def _percentile(sorted_values: Sequence[float], q: float) -> float:
    position = q * (len(sorted_values) - 1)
    low, high = math.floor(position), math.ceil(position)
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * (position - low)


def mcnemar(
    records: Sequence[ExperimentRunRecord], a: ExperimentCondition, b: ExperimentCondition, episodes: Sequence[Episode],
    *, correctness: Correctness = "direction",
) -> dict:
    """Exact two-sided McNemar test on paired per-test correctness.

    ``b_only``: tests ``a`` got wrong and ``b`` right; ``a_only``: the reverse.  p = P(X <= min) * 2 under
    Binomial(a_only + b_only, 1/2), capped at 1.  Gold-ESCALATE tests count only for ``final_action``.
    """
    ua, ub, keys, dropped = _paired(records, a, b, episodes)

    def right(u: _Unit) -> bool | None:
        if correctness == "final_action":
            return u[5]
        return None if u[3] is None else u[4] is u[3]

    a_only = b_only = both = neither = 0
    for k in keys:
        rb = {u[2]: right(u) for u in ub[k]}
        for u in ua[k]:
            x, y = right(u), rb.get(u[2])
            if x is None or y is None:
                continue
            a_only += x and not y
            b_only += y and not x
            both += x and y
            neither += not x and not y
    n = a_only + b_only
    p = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(a_only, b_only) + 1)) / 2 ** n)
    return {"a": a.value, "b": b.value, "correctness": correctness, "a_only_correct": a_only,
            "b_only_correct": b_only, "both_correct": both, "neither_correct": neither, "p_value": p,
            "n_pairs": len(keys), "n_unpaired_dropped": dropped}


def seed_summary(records: Sequence[ExperimentRunRecord], episodes: Sequence[Episode], *,
                 metric: Metric = "direction_macro_f1") -> dict[str, dict]:
    """Per condition: ``metric`` for each seed (over that seed's completed runs), mean and sample std."""
    by_id = {e.episode_id: e for e in episodes}
    out: dict[str, dict] = {}
    for condition in sorted({r.condition for r in records}, key=lambda c: c.value):
        per_seed = {}
        for seed in sorted({r.provenance.seed for r in records if r.condition is condition}, key=lambda s: -1 if s is None else s):
            units = _units([r for r in records if r.condition is condition and r.provenance.seed == seed], by_id)
            per_seed[seed] = _metric([u for us in units.values() for u in us], metric)
        values = [v for v in per_seed.values() if v is not None]
        out[condition.value] = {
            "metric": metric, "per_seed": per_seed, "n_seeds": len(values),
            "mean": statistics.fmean(values) if values else None,
            "std": statistics.stdev(values) if len(values) >= 2 else None,
        }
    return out


def memory_savings(records: Sequence[ExperimentRunRecord], episodes: Sequence[Episode], condition: ExperimentCondition,
                   *, baseline: ExperimentCondition = ExperimentCondition.A2_NO_MEMORY) -> dict:
    """Agent tool calls and model calls on memory-covered episodes, paired with ``baseline`` on the same episodes/seeds.

    ``saving`` = 1 - mean(condition) / mean(baseline) over the pairs (proposal: at least 25% fewer tool calls).
    """
    by_id = {e.episode_id: e for e in episodes}
    base = {(r.episode_id, r.provenance.seed): r for r in records
            if r.condition is baseline and r.outcome.status not in EXECUTION_FAILURES}
    pairs = []
    for r in records:
        if r.condition is not condition or r.outcome.status in EXECUTION_FAILURES or r.episode_id not in by_id:
            continue
        other = base.get((r.episode_id, r.provenance.seed))
        if other is not None and covered_by_memory(r, by_id[r.episode_id]):
            pairs.append((r, other))

    def mean(values: list[float]) -> float | None:
        return statistics.fmean(values) if values else None

    def saving(f: Callable[[ExperimentRunRecord], float | None]) -> dict:
        xs = [(f(x), f(y)) for x, y in pairs]
        xs = [(x, y) for x, y in xs if x is not None and y is not None]
        mx, my = mean([x for x, _ in xs]), mean([y for _, y in xs])
        return {"condition_mean": mx, "baseline_mean": my, "n_pairs": len(xs),
                "saving": None if not xs or not my else 1 - mx / my}

    return {"condition": condition.value, "baseline": baseline.value, "n_covered": len(pairs),
            "agent_tool_calls": saving(lambda r: float(len(agent_tool_events(r)))),
            "llm_calls": saving(lambda r: None if r.usage.llm_calls is None else float(r.usage.llm_calls))}
