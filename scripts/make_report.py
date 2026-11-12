"""Write the Day-3 results table (report.md) and plots (PNG) from an experiment's saved ``runs.jsonl``.

    python scripts/make_report.py results/day3/pilot-dev-5cond/runs.jsonl [--audit integrity_audit.jsonl] [--no-plots]
    python scripts/make_report.py results/final/test/runs.jsonl --dataset final --freeze configs/final.json

Everything comes from the saved records (see ``testorcode.report_tables``); a condition that has no scored run is
listed as "not yet evaluated", never as zero.  Output goes next to the runs file.  Needs matplotlib for the plots.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from testorcode.conditions import PROPOSAL_NAME, condition_label
from testorcode.dataset import DEFAULT_DATASET, episode_path, load_episodes
from testorcode.experiment import dataset_version, read_records
from testorcode.freeze import load_rules
from testorcode.report_tables import (
    condition_rows,
    integrity_lines,
    markdown_table,
    safety_rows,
    safety_table,
    write_plots,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", type=Path)
    parser.add_argument("--audit", type=Path, default=None, help="integrity_audit.jsonl (default: next to runs if present)")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--freeze", default=None, help="frozen config whose scoring rules the tables use")
    args = parser.parse_args(argv)
    rules, _, _ = load_rules(args.freeze)
    records = read_records(args.runs)
    if not records:
        sys.stderr.write(f"no records in {args.runs}\n")
        return 2
    splits = {r.provenance.split for r in records}
    if len(splits) != 1 or None in splits:
        sys.stderr.write(f"records must come from exactly one known split, found {sorted(map(str, splits))}\n")
        return 2
    split = splits.pop()
    episodes = load_episodes(episode_path(split, dataset=args.dataset))
    version = dataset_version(split, args.dataset)
    rows = condition_rows(records, episodes, dataset_version=version, rules=rules)
    models = sorted({r.provenance.model_id for r in records if r.provenance.model_id and r.condition.value in
                     {row["condition"] for row in rows if row["evaluated"]} and not r.condition.value.startswith("TRIVIAL")})
    seeds = sorted({r.provenance.seed for r in records})
    notes = [(f"Split `{split}` ({len(episodes)} episodes), dataset `{version}`; seeds {seeds}; "
              f"model(s): {', '.join(models) or 'none (baselines only)'}."),
             rules.describe() + ".",
             "Condition names: " + "; ".join(condition_label(c) for c in PROPOSAL_NAME) + ".",
             ("One seed and a small episode set: not a statistical comparison." if len(seeds) < 2 or len(episodes) < 20
              else "For confidence intervals and paired tests see stats.md (scripts/compare_conditions.py)."),
             "Mutant-kill retention comes from the post-hoc integrity audit lines below (pooled raw counts; undefined where the reference tests kill no mutant)."]
    audit = args.audit or args.runs.parent / "integrity_audit.jsonl"
    if audit.is_file():
        notes += integrity_lines(audit)
    else:
        notes.append("No integrity audit found next to the runs; integrity columns are not reported.")
    out = args.runs.parent
    plots = [] if args.no_plots else write_plots(rows, out, split)
    text = markdown_table(rows, title=f"Results table, {args.runs.parent.name}", notes=notes)
    text += "\n" + safety_table(safety_rows(records, episodes, dataset_version=version,
                                             audit_jsonl=audit if audit.is_file() else None))
    if plots:
        text += "\n" + "\n".join(f"![{p.stem}]({p.name})" for p in plots) + "\n"
    (out / "report.md").write_text(text, encoding="utf-8")
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
