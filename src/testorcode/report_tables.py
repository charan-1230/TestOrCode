"""The Day-3 results table and plots, built only from saved run records (harness side).

One row per condition with the plan's columns (episodes run, direction macro-F1, correct-repair rate, masked
regressions), each rate next to its raw count.  A condition with no scored run is "not yet evaluated", never 0; runs that
ended in an execution failure are counted apart and never scored.  Numbers come from ``evaluation.summarize`` and
``evaluation.evaluate_run`` on the records, so every figure traces to a line of ``runs.jsonl``.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from testorcode.evaluation import EXECUTION_FAILURES, evaluate_run, summarize
from testorcode.schemas.episode_schemas import DecisionLabel, Episode
from testorcode.schemas.result_schemas import ExperimentCondition as C
from testorcode.schemas.result_schemas import ExperimentRunRecord

ORDER: tuple[C, ...] = (
    C.FULL, C.B1, C.A1_NO_INTENT_RETRIEVAL, C.A2_NO_MEMORY, C.A3_NO_INTEGRITY_GUARD, C.TRIVIAL_ALWAYS_UPDATE_TEST, C.TRIVIAL_ALWAYS_FIX_CODE,
)  # the plan's table order, then the sanity baselines
NOT_EVALUATED = "not yet evaluated"


def _frac(k: int, n: int) -> str:
    return f"{k}/{n} ({k / n:.2f})" if n else "-"


def _f(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def condition_rows(
    records: Sequence[ExperimentRunRecord], episodes: Sequence[Episode], *, dataset_version: str,
    conditions: Iterable[C] = ORDER,
) -> list[dict[str, Any]]:
    """One row per requested condition, in order; a condition without a scored run is marked ``evaluated=False``."""
    by_id = {e.episode_id: e for e in episodes}
    summaries = {s.condition: s for s in summarize(list(records), list(episodes), dataset_version=dataset_version)}
    counts: dict[C, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    scenario: dict[C, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
    for rec in records:
        if rec.provenance.dataset_version != dataset_version or rec.episode_id not in by_id:
            continue
        c = counts[rec.condition]
        if rec.outcome.status in EXECUTION_FAILURES:
            c["failed"] += 1
            continue
        c["scored"] += 1
        ev = evaluate_run(rec, by_id[rec.episode_id])
        for d in ev.decisions:
            if d.predicted_action is None and d.test_id in rec.unstable_test_ids:
                continue
            scenario[rec.condition][by_id[rec.episode_id].scenario.value].append(d.correct)
            c["gold_fix"] += d.expected_action is DecisionLabel.FIX_CODE
            c["masked"] += d.wrong_side == "masked_regression"
        if ev.repair is not None and ev.repair.repair_attempted:
            if ev.repair.repair_successful is None:
                c["repair_unverifiable"] += 1
            else:
                c["repairs"] += 1
                c["repairs_ok"] += bool(ev.repair.repair_successful)
    rows = []
    for condition in conditions:
        c, s = counts.get(condition, {}), summaries.get(condition)
        evaluated = bool(c and c.get("scored") and s is not None)
        m = s.metrics if s else {}
        rows.append({
            "condition": condition.value, "evaluated": evaluated,
            "episodes_run": c.get("scored", 0) if c else 0, "execution_failures": c.get("failed", 0) if c else 0,
            "direction_macro_f1": m.get("direction_macro_f1") if evaluated else None,
            "final_action_accuracy": m.get("accuracy") if evaluated else None,
            "direction_correct": round((m.get("direction_accuracy") or 0) * (m.get("n_direction_scored") or 0))
            if evaluated and m.get("direction_accuracy") is not None else None,
            "direction_scored": int(m.get("n_direction_scored") or 0) if evaluated else None,
            "repairs_ok": c.get("repairs_ok", 0) if evaluated else None,
            "repairs": c.get("repairs", 0) if evaluated else None,
            "repair_unverifiable": c.get("repair_unverifiable", 0) if evaluated else None,
            "masked": c.get("masked", 0) if evaluated else None,
            "gold_fix": c.get("gold_fix", 0) if evaluated else None,
            "scenario_accuracy": {k: sum(v) / len(v) for k, v in sorted(scenario[condition].items())} if evaluated else {},
        })
    return rows


def markdown_table(rows: list[dict[str, Any]], *, title: str, notes: Sequence[str] = ()) -> str:
    lines = [
        f"# {title}", "",
        ("**Preliminary.** Raw counts are shown next to each rate. `-` means not computable (never 0). A condition "
         f"that has not run is marked *{NOT_EVALUATED}*. Execution failures (API/tool errors) are counted apart and "
         "never scored."), "",
        ("| Condition | Episodes run | Execution failures | Direction macro-F1 | Direction accuracy | "
         "Final-action accuracy | Correct-repair rate | Masked regressions |"),
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        if not r["evaluated"]:
            failed = str(r["execution_failures"]) if r["execution_failures"] else "0"
            lines.append(f"| {r['condition']} | {NOT_EVALUATED} | {failed} | - | - | - | - | - |")
            continue
        repair = _frac(r["repairs_ok"], r["repairs"])
        if r["repair_unverifiable"]:
            repair += f" +{r['repair_unverifiable']} unverifiable"
        lines.append(
            f"| {r['condition']} | {r['episodes_run']} | {r['execution_failures']} | {_f(r['direction_macro_f1'])} | "
            f"{_frac(r['direction_correct'], r['direction_scored'])} | {_f(r['final_action_accuracy'])} | {repair} | {_frac(r['masked'], r['gold_fix'])} |"
        )
    if notes:
        lines += ["", *[f"- {n}" for n in notes]]
    return "\n".join(lines) + "\n"


def integrity_lines(audit_jsonl: Path) -> list[str]:
    """Per-condition integrity counts from ``integrity_audit.jsonl`` (post-hoc; unmeasured is not counted as 0)."""
    per: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for line in audit_jsonl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        run = json.loads(line)
        c = per[run["condition"]]
        c["runs"] += 1
        for g in run.get("groups", []):
            c["groups"] += 1
            c["test_patches"] += g["scope"] == "test"
            if g["scope"] == "test" and g["weakened"] is not None:
                c["weakening_measured"] += 1
                c["weakened"] += bool(g["weakened"])
            c["wrong_side_accepted"] += bool(g["wrong_side"])
            c["partial_or_unmeasured"] += g["status"] != "measured"
            denom, ret = g.get("retention_denominator"), g.get("retention")
            if g["scope"] == "test" and ret is not None and denom:
                c["retention_groups"] += 1
                c["retained"] += round(ret * denom)
                c["retention_denominator"] += denom
    return [
        f"Integrity audit, {cond}: {v['groups']} accepted repair(s) in {v['runs']} run(s); "
        f"weakened {v['weakened']}/{v['weakening_measured']} measured test patches; "
        f"{v['wrong_side_accepted']} accepted on the wrong side; {v['partial_or_unmeasured']} not fully measured; "
        + (f"mutant-kill retention {v['retained']}/{v['retention_denominator']} reference-killed mutants over "
           f"{v['retention_groups']} test patch(es)" if v["retention_groups"] else "mutant-kill retention not measurable")
        for cond, v in sorted(per.items())
    ]


def write_plots(rows: list[dict[str, Any]], out_dir: Path, split: str = "dev") -> list[Path]:
    """PNG plots for the evaluated conditions only (a missing condition is omitted, never drawn as 0)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    shown = [r for r in rows if r["evaluated"]]
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    if not shown:
        return paths

    def bars(ax, labels, values, texts, ylabel):
        xs = range(len(labels))
        ax.bar(xs, [v or 0 for v in values], color="#4c78a8")
        for x, v, t in zip(xs, values, texts, strict=True):
            ax.text(x, (v or 0) + 0.02, t, ha="center", fontsize=8)
        ax.set_xticks(list(xs), labels, rotation=30, ha="right")
        ax.set_ylim(0, 1.15)
        ax.set_ylabel(ylabel)

    fig, ax = plt.subplots(figsize=(7, 4))
    bars(ax, [r["condition"] for r in shown], [r["direction_macro_f1"] for r in shown],
         [f"{_f(r['direction_macro_f1'])}\nn={r['direction_scored']}" for r in shown], "direction macro-F1")
    ax.set_title(f"Direction macro-F1 (preliminary, {split} split, 1 seed)")
    fig.tight_layout()
    paths.append(out_dir / "direction_macro_f1.png")
    fig.savefig(paths[-1], dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    width = 0.38
    xs = list(range(len(shown)))
    rep = [(r["repairs_ok"] / r["repairs"]) if r["repairs"] else 0 for r in shown]
    mask = [(r["masked"] / r["gold_fix"]) if r["gold_fix"] else 0 for r in shown]
    ax.bar([x - width / 2 for x in xs], rep, width, label="correct-repair rate", color="#59a14f")
    ax.bar([x + width / 2 for x in xs], mask, width, label="masked-regression rate", color="#e15759")
    for x, r in zip(xs, shown, strict=True):
        ax.text(x - width / 2, rep[x] + 0.02, _frac(r["repairs_ok"], r["repairs"]).split(" ")[0], ha="center", fontsize=7)
        ax.text(x + width / 2, mask[x] + 0.02, _frac(r["masked"], r["gold_fix"]).split(" ")[0], ha="center", fontsize=7)
    ax.set_xticks(xs, [r["condition"] for r in shown], rotation=30, ha="right")
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=8)
    ax.set_title("Repair success vs masked regressions (a '0' bar with '-' means no denominator)")
    fig.tight_layout()
    paths.append(out_dir / "repair_vs_masked.png")
    fig.savefig(paths[-1], dpi=140)
    plt.close(fig)

    scenarios = sorted({k for r in shown for k in r["scenario_accuracy"]})
    fig, ax = plt.subplots(figsize=(8, 4))
    width = 0.8 / max(len(shown), 1)
    for i, r in enumerate(shown):
        vals = [r["scenario_accuracy"].get(s) for s in scenarios]
        ax.bar([j + i * width for j in range(len(scenarios))], [v or 0 for v in vals], width, label=r["condition"])
    ax.set_xticks([j + 0.4 - width / 2 for j in range(len(scenarios))], scenarios)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("final-action accuracy")
    ax.legend(fontsize=7, ncol=2)
    ax.set_title("Accuracy per scenario (S1..S5; few episodes each)")
    fig.tight_layout()
    paths.append(out_dir / "scenario_accuracy.png")
    fig.savefig(paths[-1], dpi=140)
    plt.close(fig)
    return paths
