"""Summarise a runs.jsonl into summary.json and summary.md (one row per condition).

    python scripts/summarize_results.py results/experiments/latest/runs.jsonl

Only the harness reads the gold labels: they are taken from the split recorded in each record's provenance.
Records made with another revision of that split (``provenance.dataset_version``) are set aside and counted in
``n_incompatible_records``, never scored.  ``direction_macro_f1`` is the proposal's per-test UPDATE_TEST-vs-FIX_CODE
metric (an escalation counts by the side its hypothesis blamed); ``macro_f1``/``accuracy`` score final actions.
Metrics that cannot be computed are shown as '-' (never 0); see the notes under the table.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from testorcode.dataset import episode_path, load_episodes
from testorcode.evaluation import ConditionSummary, summarize
from testorcode.experiment import dataset_version, read_records

SHOWN = (
    "n_runs", "n_incompatible_records", "n_execution_failures", "n_direction_scored", "direction_macro_f1",
    "direction_accuracy", "n_direction_undecided", "n_tests_scored", "accuracy", "macro_f1", "masked_regression_rate",
    "reverted_intent_rate", "escalation_precision", "escalation_recall", "unnecessary_escalation_rate",
    "repair_success_rate", "n_repair_unverifiable", "gold_evidence_recall@3", "gold_evidence_recall@5",
    "gold_evidence_recall@8", "citation_existence_rate", "mean_tool_calls", "mean_total_tokens",
)  # fmt: skip


def cell(value: float | None) -> str:
    if value is None:
        return "-"
    return str(int(value)) if float(value).is_integer() else f"{value:.3f}"


def markdown(summaries: list[ConditionSummary]) -> str:
    keys = [k for k in SHOWN if any(k in s.metrics for s in summaries)]
    extra = sorted({k for s in summaries for k in s.metrics if k.startswith("accuracy[")})
    columns = keys + extra
    lines = ["| condition | " + " | ".join(columns) + " |", "|---|" + "---|" * len(columns)]
    for s in summaries:
        lines.append(f"| {s.condition.value} | " + " | ".join(cell(s.metrics.get(k)) for k in columns) + " |")
    notes = sorted({n for s in summaries for n in s.notes})
    if notes:
        lines += ["", "Not computed (shown as '-' or absent):", *[f"- {n}" for n in notes]]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", type=Path)
    args = parser.parse_args()
    records = read_records(args.runs)
    if not records:
        sys.stderr.write(f"no records in {args.runs}\n")
        return 2
    splits = {r.provenance.split for r in records}
    if len(splits) != 1 or None in splits:
        sys.stderr.write(f"records must come from exactly one known split, found {sorted(map(str, splits))}\n")
        return 2
    split = splits.pop()
    episodes = load_episodes(episode_path(split))
    # Records made with another revision of the dataset are set aside and reported, never scored against it.
    summaries = summarize(records, episodes, dataset_version=dataset_version(split))
    (args.runs.parent / "summary.json").write_text(
        json.dumps([s.model_dump(mode="json") for s in summaries], indent=2) + "\n", encoding="utf-8"
    )
    table = markdown(summaries)
    (args.runs.parent / "summary.md").write_text(table, encoding="utf-8")
    sys.stdout.write(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
