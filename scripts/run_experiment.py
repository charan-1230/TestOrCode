"""Run an experiment: episodes x conditions x seeds, scored, appended to <out>/runs.jsonl (resumable).

    python scripts/run_experiment.py --conditions TRIVIAL_UPDATE_TEST,TRIVIAL_FIX_CODE      # no API key needed
    python scripts/run_experiment.py --conditions A2,B1 --episodes dev-02,dev-05 --seeds 0,1  # needs .env + ../boltons

The default split is dev.  The frozen test split is refused unless ``--final`` is given, and the gold labels are
used only by the harness (simulated maintainer and scoring), never by an agent.  Then summarise with
``python scripts/summarize_results.py <out>/runs.jsonl``.  A run that ended in ``api_failure`` counts as done and is
not re-run unless ``--retry-failed`` is given (then its line is replaced).

Several libraries: ``--repos repos.json`` maps each episode's ``repo_id`` to its clone and test interpreter, e.g.
``{"boltons": {"path": "../boltons", "python": "../boltons-venv/bin/python"}, "tabulate": {...}}``.  Each episode's
tool-fault plan (scenario T) is applied unless ``--no-tool-faults`` is given.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pydantic import ValidationError

from testorcode.experiment import ROOT, ExperimentConfig, RepoEnvironment, run_experiment
from testorcode.schemas.result_schemas import ExperimentCondition


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--conditions", required=True, help="comma separated, e.g. TRIVIAL_UPDATE_TEST,A2,B1")
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--final", action="store_true", help="required for the frozen test split")
    parser.add_argument("--retry-failed", action="store_true",
                        help="re-run runs that ended in api_failure, replacing their line (refused for a memory "
                             "condition when a later episode of the stream already ran)")
    parser.add_argument("--episodes", default="", help="comma separated ids; default: the whole split")
    parser.add_argument("--seeds", default="0", help="comma separated integers")
    parser.add_argument("--out", default=str(ROOT / "results" / "experiments" / "latest"))
    parser.add_argument("--repo", default=str(ROOT.parent / "boltons"), help="clone of the pinned repository")
    parser.add_argument("--repos", default=None,
                        help="JSON file: repo_id -> {path, python, docker_image} for multi-library experiments")
    parser.add_argument("--dataset", default="pilot", help="dataset file prefix (datasets/<dataset>_<split>.jsonl)")
    parser.add_argument("--no-tool-faults", action="store_true", help="do not inject the episodes' tool faults")
    parser.add_argument("--worktrees", default=str(ROOT.parent / "episode-worktrees"),
                        help="directory for the disposable episode worktrees (outside the clone and this project)")
    args = parser.parse_args()
    try:
        repos = {}
        if args.repos:
            raw = json.loads(Path(args.repos).read_text(encoding="utf-8"))
            base = Path(args.repos).resolve().parent  # relative paths are relative to the mapping file
            repos = {k: RepoEnvironment.model_validate({**v, **{f: str((base / v[f]).absolute()) for f in ("path", "python")
                                                              if v.get(f) and not Path(v[f]).is_absolute()}})
                     for k, v in raw.items()}
        config = ExperimentConfig(
            split=args.split,
            final=args.final,
            retry_failed=args.retry_failed,
            episode_ids=tuple(e for e in args.episodes.split(",") if e),
            conditions=tuple(ExperimentCondition(c.strip()) for c in args.conditions.split(",") if c.strip()),
            seeds=tuple(int(s) for s in args.seeds.split(",") if s.strip()),
            out=Path(args.out),
            repo=Path(args.repo),
            repos=repos,
            dataset=args.dataset,
            tool_faults=not args.no_tool_faults,
            worktrees=Path(args.worktrees),
        )
    except (ValidationError, ValueError, OSError) as exc:
        sys.stderr.write(f"refusing: {exc}\n")
        return 2
    try:
        result = run_experiment(config)
    except ValueError as exc:
        sys.stderr.write(f"{exc}\n")
        return 2
    sys.stdout.write(f"ran {result.ran} ({result.retried} retried), already done {result.skipped_done}, unavailable {len(result.skipped_unavailable)}\n")
    for line in result.skipped_unavailable:
        sys.stdout.write(f"  unavailable: {line}\n")
    sys.stdout.write(f"records: {config.out / 'runs.jsonl'}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
