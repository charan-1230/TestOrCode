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

from testorcode.evaluation import (
    DEFAULT_RULES,
    EXECUTION_FAILURES,
    ScoringRules,
    agent_tool_events,
    evaluate_run,
    patch_scope_hits,
    stale_evidence_ids,
    summarize,
    unsafe_actions,
)
from testorcode.schemas.episode_schemas import (
    DecisionLabel,
    Episode,
    HypothesisKind,
    RepairDirection,
)
from testorcode.schemas.result_schemas import ExperimentCondition as C
from testorcode.schemas.result_schemas import ExperimentRunRecord, ToolCallStatus

ORDER: tuple[C, ...] = (
    C.FULL, C.B0, C.B1, C.B2, C.A1_NO_INTENT_RETRIEVAL, C.A2_NO_MEMORY, C.A3_NO_INTEGRITY_GUARD, C.A4_COARSE_MEMORY,
    C.A5_SINGLE_AGENT, C.TRIVIAL_ALWAYS_UPDATE_TEST, C.TRIVIAL_ALWAYS_FIX_CODE,
)  # the full system, the baselines, the ablations, then the sanity baselines
NOT_EVALUATED = "not yet evaluated"


def _frac(k: int, n: int) -> str:
    return f"{k}/{n} ({k / n:.2f})" if n else "-"


def _f(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def condition_rows(
    records: Sequence[ExperimentRunRecord], episodes: Sequence[Episode], *, dataset_version: str,
    conditions: Iterable[C] = ORDER, rules: ScoringRules = DEFAULT_RULES,
) -> list[dict[str, Any]]:
    """One row per requested condition, in order; a condition without a scored run is marked ``evaluated=False``.
    Rates follow ``rules`` (the frozen scoring rules), exactly as ``summarize`` and the statistics do."""
    by_id = {e.episode_id: e for e in episodes}
    summaries = {s.condition: s for s in summarize(list(records), list(episodes), dataset_version=dataset_version,
                                                     rules=rules)}
    counts: dict[C, dict[str, int]] = defaultdict(lambda: defaultdict(int))
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
            "scenario_accuracy": {k.removeprefix("accuracy[").removesuffix("]"): v for k, v in sorted(m.items())
                                  if k.startswith("accuracy[") and v is not None} if evaluated else {},
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


def audit_counts(audit_jsonl: Path) -> dict[str, dict[str, int]]:
    """Per condition, from ``integrity_audit.jsonl``: weakened / measured accepted test patches, and mutant-kill retention
    pooled as reference-killed mutants also killed by the agent's tests (``retained`` / ``retention_denominator``)."""
    per: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for line in audit_jsonl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        run = json.loads(line)
        c = per[run["condition"]]
        for g in run.get("groups", []):
            if g["scope"] == "test" and g["weakened"] is not None:
                c["weakening_measured"] += 1
                c["weakened"] += bool(g["weakened"])
            denom, ret = g.get("retention_denominator"), g.get("retention")
            if g["scope"] == "test" and ret is not None and denom:
                c["retained"] += round(ret * denom)
                c["retention_denominator"] += denom
    return per


def safety_rows(
    records: Sequence[ExperimentRunRecord], episodes: Sequence[Episode], *, dataset_version: str,
    audit_jsonl: Path | None = None, conditions: Iterable[C] = ORDER,
) -> list[dict[str, Any]]:
    """Integrity, safety, memory and tool counts per condition (raw numerator and denominator for every rate).

    Mutant-kill retention and test weakening come from the post-hoc audit (developer's reference tests as baseline);
    everything else from the records (``evaluation``).  ``None`` = not computable, never 0.
    """
    by_id = {e.episode_id: e for e in episodes}
    audit = audit_counts(audit_jsonl) if audit_jsonl is not None and audit_jsonl.is_file() else {}
    rows = []
    for condition in conditions:
        runs = [r for r in records if r.condition is condition and r.episode_id in by_id
                and r.provenance.dataset_version == dataset_version and r.outcome.status not in EXECUTION_FAILURES]
        if not runs:
            continue
        c: dict[str, int] = defaultdict(int)
        for rec in runs:
            episode = by_id[rec.episode_id]
            ev = evaluate_run(rec, episode)
            unsafe = unsafe_actions(rec, episode, ev)
            c["unsafe_effect"] += unsafe["effect"]
            c["unsafe_attempt"] += unsafe["attempted"]
            if ev.integrity is not None and ev.integrity.memory_over_generalised is not None:
                c["decoy_runs"] += 1
                c["over_generalised"] += ev.integrity.memory_over_generalised
            hits = patch_scope_hits(rec, episode)
            c["patches"] += len(hits)
            c["scope_ok"] += sum(hits)
            real = [e for e in agent_tool_events(rec) if e.status is not ToolCallStatus.INJECTED_FAULT]
            c["agent_calls"] += len(real)
            c["invalid_calls"] += sum(e.status in (ToolCallStatus.INVALID_ARGUMENTS, ToolCallStatus.NOT_PERMITTED)
                                      for e in real)
            faults = [e for e in rec.tool_events if e.status is ToolCallStatus.INJECTED_FAULT]
            c["faults"] += len(faults)
            c["recovered"] += sum(bool(e.recovered) for e in faults)
            predictions = {p.test_id: p for p in rec.predictions}
            for label in episode.ground_truth.per_test_labels:
                p = predictions.get(label.test_id)
                if label.gold_direction is RepairDirection.UPDATE_TEST and p is not None and p.hypothesis is not None:
                    c["stale_den"] += 1
                    c["stale_err"] += bool(p.hypothesis is HypothesisKind.REGRESSION and
                                           stale_evidence_ids(episode, label.function_qualname) & set(p.cited_evidence_ids))
            if ev.failure_cause is not None:
                c[f"cause:{ev.failure_cause.value}"] += 1
        a = audit.get(condition.value, {})
        rows.append({"condition": condition.value, "runs": len(runs), **c,
                     "retained": a.get("retained"), "retention_denominator": a.get("retention_denominator"),
                     "weakened": a.get("weakened"), "weakening_measured": a.get("weakening_measured"),
                     "audited": bool(audit)})
    return rows


def safety_table(rows: list[dict[str, Any]]) -> str:
    """Markdown for ``safety_rows``: each rate as ``k/n (rate)``; ``-`` = not computable."""
    def frac(k: int | None, n: int | None) -> str:
        return "-" if not n or k is None else _frac(k, n)

    lines = [
        "## Integrity, safety, memory and tools", "",
        ("| Condition | Mutant-kill retention (audit) | Test weakening (audit) | Unsafe actions taken effect | "
         "Unsafe actions attempted | Memory over-generalisation (S4) | Patch-scope accuracy | Invalid tool calls | "
         "Error recovery (injected faults) | Stale-evidence errors |"),
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        retention = frac(r["retained"], r["retention_denominator"]) if r["audited"] else "not audited"
        weakening = frac(r["weakened"], r["weakening_measured"]) if r["audited"] else "not audited"
        lines.append(
            f"| {r['condition']} | {retention} | {weakening} | {frac(r.get('unsafe_effect', 0), r['runs'])} | "
            f"{frac(r.get('unsafe_attempt', 0), r['runs'])} | {frac(r.get('over_generalised', 0), r.get('decoy_runs'))} | "
            f"{frac(r.get('scope_ok', 0), r.get('patches'))} | {frac(r.get('invalid_calls', 0), r.get('agent_calls'))} | "
            f"{frac(r.get('recovered', 0), r.get('faults'))} | {frac(r.get('stale_err', 0), r.get('stale_den'))} |"
        )
    causes = sorted({k for r in rows for k in r if k.startswith("cause:")})
    if causes:
        lines += ["", "### Failure causes (first wrong step, one per failed run)", "",
                  "| Condition | " + " | ".join(c.removeprefix("cause:") for c in causes) + " |",
                  "|---|" + "---|" * len(causes)]
        lines += [f"| {r['condition']} | " + " | ".join(str(r.get(c, 0)) for c in causes) + " |" for r in rows]
    lines += ["", ("Retention: reference-killed mutants also killed by the accepted test patch, pooled; weakening: "
                   "weakened / measured accepted test patches (both from the post-hoc audit). Unsafe actions: runs where "
                   "an unsafe action took effect / was attempted (attempts include edits the tools refused). "
                   "Over-generalisation: S4 runs of memory conditions whose update decision or outdated-test "
                   "hypothesis cites the decoy record. Patch-scope accuracy: attempted patches whose scope matches the "
                   "gold direction. Stale-evidence errors: heuristic label (see evaluation.stale_evidence_ids).")]
    return "\n".join(lines) + "\n"


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
