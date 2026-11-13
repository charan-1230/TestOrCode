"""Run an experiment: episodes x conditions x seeds, scored, appended to <out>/runs.jsonl (resumable).

    python scripts/run_experiment.py --conditions TRIVIAL_UPDATE_TEST,TRIVIAL_FIX_CODE      # no API key needed
    python scripts/run_experiment.py --conditions A2,B1 --episodes dev-02,dev-05 --seeds 0,1  # needs .env + ../boltons

The default split is dev.  The frozen test split is refused unless ``--final`` is given, and the gold labels are
used only by the harness (simulated maintainer and scoring), never by an agent.  Then summarise with
``python scripts/summarize_results.py <out>/runs.jsonl``.  A run that ended in ``api_failure`` counts as done and is
not re-run unless ``--retry-failed`` is given (then its line is replaced).

Several libraries: ``--repos repos.json`` maps each episode's ``repo_id`` to its clone and test interpreter, e.g.
``{"boltons": {"path": "../boltons", "python": "../boltons-venv/bin/python"}, "tabulate": {...}}``.  Each episode's
tool-fault plan (scenario T) is applied unless ``--no-tool-faults`` is given; ``--tool-fault-rate 0.1`` also fails about
10% of every agent's tool calls.

Final runs: ``--freeze configs/final.json`` applies the frozen configuration (``testorcode.freeze``) on top of ``.env``
and records it in ``config.json``.  The model check on a subset: ``--env-file .env.mistral`` and/or ``--model <id>``
into its own ``--out``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from pydantic import ValidationError

from testorcode.errors import ConfigError
from testorcode.experiment import ROOT, ExperimentConfig, load_repos, run_experiment
from testorcode.freeze import load_frozen
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
    parser.add_argument("--tool-fault-rate", type=float, default=None,
                        help="also fail this share of the agents' tool calls (scenario T), e.g. 0.1")
    parser.add_argument("--freeze", default=None, help="frozen configuration JSON (testorcode.freeze)")
    parser.add_argument("--env-file", default=None, help="settings file instead of .env (e.g. a second model)")
    parser.add_argument("--model", default=None, help="LLM_MODEL override for every run")
    parser.add_argument("--worktrees", default=str(ROOT.parent / "episode-worktrees"),
                        help="directory for the disposable episode worktrees (outside the clone and this project)")
    parser.add_argument("--retrieval-mode", choices=("bm25", "hybrid"), default=None,
                        help="RETRIEVAL_MODE for every retrieving condition (same model/budget; only the index differs)")
    parser.add_argument("--embedding-backend", choices=("voyage", "local"), default=None)
    parser.add_argument("--rerank", choices=("none", "voyage", "local"), default=None)
    parser.add_argument("--embedding-cache-dir", default=None)
    args = parser.parse_args()
    # Settings.from_env lets the process environment win over the env file, so these overrides reach every run.
    for var, value in (("RETRIEVAL_MODE", args.retrieval_mode), ("EMBEDDING_BACKEND", args.embedding_backend),
                       ("RERANK_BACKEND", args.rerank), ("EMBEDDING_CACHE_DIR", args.embedding_cache_dir)):
        if value:
            os.environ[var] = value
    try:
        frozen, frozen_sha = load_frozen(args.freeze) if args.freeze else (None, None)
        options = frozen.experiment if frozen is not None else None
        tool_faults = not args.no_tool_faults
        rate = args.tool_fault_rate
        if options is not None:  # the frozen file decides unless the command line contradicts it
            if options.tool_faults is not None and args.no_tool_faults and options.tool_faults:
                raise ValueError("--no-tool-faults contradicts the frozen config (tool_faults=true)")
            if options.tool_fault_rate is not None and rate is not None and rate != options.tool_fault_rate:
                raise ValueError("--tool-fault-rate contradicts the frozen config")
            tool_faults = options.tool_faults if options.tool_faults is not None else tool_faults
            rate = options.tool_fault_rate if options.tool_fault_rate is not None else rate
        if args.final and frozen is not None and frozen.name.upper().startswith("DRAFT"):
            raise ValueError(f"--final needs a frozen configuration, not the draft {frozen.name!r}")
        if args.final and frozen is None:
            sys.stderr.write("warning: --final without --freeze: the runs record no frozen configuration\n")
        repos = load_repos(args.repos) if args.repos else {}
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
            tool_faults=tool_faults,
            tool_fault_rate=rate,
            env_file=Path(args.env_file) if args.env_file else None,
            model=args.model,
            frozen=frozen,
            frozen_sha256=frozen_sha,
            worktrees=Path(args.worktrees),
        )
    except (ValidationError, ValueError, OSError, ConfigError) as exc:
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
