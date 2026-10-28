"""Run an experiment: episodes x conditions x seeds, scored, appended to <out>/runs.jsonl (resumable).

    python scripts/run_experiment.py --conditions TRIVIAL_UPDATE_TEST,TRIVIAL_FIX_CODE      # no API key needed
    python scripts/run_experiment.py --conditions A2,B1 --episodes dev-02,dev-05 --seeds 0,1  # needs .env + ../boltons

The default split is dev.  The frozen test split is refused unless ``--final`` is given, and the gold labels are
used only by the harness (simulated maintainer and scoring), never by an agent.  Then summarise with
``python scripts/summarize_results.py <out>/runs.jsonl``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pydantic import ValidationError

from testorcode.experiment import ROOT, ExperimentConfig, run_experiment
from testorcode.schemas.result_schemas import ExperimentCondition


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--conditions", required=True, help="comma separated, e.g. TRIVIAL_UPDATE_TEST,A2,B1")
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--final", action="store_true", help="required for the frozen test split")
    parser.add_argument("--episodes", default="", help="comma separated ids; default: the whole split")
    parser.add_argument("--seeds", default="0", help="comma separated integers")
    parser.add_argument("--out", default=str(ROOT / "results" / "experiments" / "latest"))
    parser.add_argument("--repo", default=str(ROOT.parent / "boltons"), help="clone of the pinned repository")
    args = parser.parse_args()
    try:
        config = ExperimentConfig(
            split=args.split,
            final=args.final,
            episode_ids=tuple(e for e in args.episodes.split(",") if e),
            conditions=tuple(ExperimentCondition(c.strip()) for c in args.conditions.split(",") if c.strip()),
            seeds=tuple(int(s) for s in args.seeds.split(",") if s.strip()),
            out=Path(args.out),
            repo=Path(args.repo),
        )
    except (ValidationError, ValueError) as exc:
        sys.stderr.write(f"refusing: {exc}\n")
        return 2
    result = run_experiment(config)
    sys.stdout.write(f"ran {result.ran}, already done {result.skipped_done}, unavailable {len(result.skipped_unavailable)}\n")
    for line in result.skipped_unavailable:
        sys.stdout.write(f"  unavailable: {line}\n")
    sys.stdout.write(f"records: {config.out / 'runs.jsonl'}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
