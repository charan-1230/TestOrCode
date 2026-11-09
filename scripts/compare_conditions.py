"""Paired statistics between conditions from saved run records (``testorcode.stats``).

    python scripts/compare_conditions.py results/<exp>/runs.jsonl                 # FULL against every other condition
    python scripts/compare_conditions.py runs.jsonl --pairs FULL:B1,FULL:A1 --samples 20000

For each pair: the direction macro-F1 difference with a 95% paired-bootstrap CI over episodes, the same for final-action
accuracy, and exact McNemar tests on paired per-test correctness (direction and final action).  Then per-seed variance for
every condition and, for each memory condition, tool and model calls on memory-covered episodes against A2.  Writes
``stats.json`` and ``stats.md`` next to the runs file.  No model is called.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from testorcode.conditions import profile_for
from testorcode.dataset import episode_path, load_episodes
from testorcode.experiment import read_records
from testorcode.schemas.result_schemas import ExperimentCondition as C
from testorcode.stats import (
    BOOTSTRAP_SAMPLES,
    mcnemar,
    memory_savings,
    paired_bootstrap,
    seed_summary,
)


def _f(v) -> str:
    return "-" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", type=Path)
    parser.add_argument("--pairs", default="", help="comma separated A:B pairs (default: FULL against every other)")
    parser.add_argument("--samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--dataset", default="pilot")
    args = parser.parse_args(argv)
    records = read_records(args.runs)
    splits = {r.provenance.split for r in records}
    if len(splits) != 1 or None in splits:
        sys.stderr.write(f"records must come from exactly one known split, found {sorted(map(str, splits))}\n")
        return 2
    episodes = load_episodes(episode_path(splits.pop(), dataset=args.dataset))
    present = sorted({r.condition for r in records}, key=lambda c: c.value)
    if args.pairs:
        pairs = [tuple(C(x.strip()) for x in p.split(":")) for p in args.pairs.split(",") if p.strip()]
    else:
        pairs = [(C.FULL, c) for c in present if c is not C.FULL] if C.FULL in present else []
    out = {"pairs": [], "seeds": seed_summary(records, episodes), "memory_savings": []}
    lines = ["# Paired statistics", "", f"Source: `{args.runs}`; {len(records)} records; bootstrap samples {args.samples}.",
             "Differences are A - B; CI = 95% percentile bootstrap over episodes (seeds of an episode resampled together).",
             "", ("| A | B | Episodes | Direction macro-F1 diff [95% CI] | Final-action acc. diff [95% CI] | "
                  "McNemar direction (A only / B only, p) | McNemar final action (A only / B only, p) |"),
             "|---|---|---|---|---|---|---|"]
    for a, b in pairs:
        f1 = paired_bootstrap(records, a, b, episodes, samples=args.samples)
        acc = paired_bootstrap(records, a, b, episodes, metric="final_action_accuracy", samples=args.samples)
        md = mcnemar(records, a, b, episodes)
        mf = mcnemar(records, a, b, episodes, correctness="final_action")
        out["pairs"].append({"direction_macro_f1": f1, "final_action_accuracy": acc, "mcnemar_direction": md,
                             "mcnemar_final_action": mf})

        def ci(r):
            return f"{_f(r['difference'])} [{_f(r['ci95'][0])}, {_f(r['ci95'][1])}]" if r["ci95"] else _f(r["difference"])

        lines.append(f"| {a.value} | {b.value} | {f1['n_episodes']} | {ci(f1)} | {ci(acc)} | "
                     f"{md['a_only_correct']} / {md['b_only_correct']}, p={md['p_value']:.3f} | "
                     f"{mf['a_only_correct']} / {mf['b_only_correct']}, p={mf['p_value']:.3f} |")
    lines += ["", "## Direction macro-F1 per seed", "", "| Condition | Per seed | Mean | Std | Seeds |", "|---|---|---|---|---|"]
    for cond, s in out["seeds"].items():
        per = ", ".join(f"{k}: {_f(v)}" for k, v in s["per_seed"].items())
        lines.append(f"| {cond} | {per} | {_f(s['mean'])} | {_f(s['std'])} | {s['n_seeds']} |")
    memory_conditions = [c for c in present if profile_for(c).memory]
    if C.A2_NO_MEMORY in present and memory_conditions:
        lines += ["", "## Memory-covered episodes against A2 (same episode and seed)", "",
                  "| Condition | Covered pairs | Agent tool calls (cond / A2, saving) | Model calls (cond / A2, saving) |",
                  "|---|---|---|---|"]
        for c in memory_conditions:
            m = memory_savings(records, episodes, c)
            out["memory_savings"].append(m)
            t, llm = m["agent_tool_calls"], m["llm_calls"]
            lines.append(f"| {c.value} | {m['n_covered']} | {_f(t['condition_mean'])} / {_f(t['baseline_mean'])}, "
                         f"{_f(t['saving'])} | {_f(llm['condition_mean'])} / {_f(llm['baseline_mean'])}, {_f(llm['saving'])} |")
    lines += ["", "Small samples: wide intervals and large p-values are expected; read every number with its n."]
    target = args.runs.parent
    (target / "stats.json").write_text(json.dumps(out, indent=2, default=str) + "\n", encoding="utf-8")
    (target / "stats.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    sys.stdout.write("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
