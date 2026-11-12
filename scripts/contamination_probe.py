"""Contamination probe: ask the plain model to complete each episode's real commit without context (proposal §11).

    python scripts/contamination_probe.py --split dev --repo ../boltons --out results/final/contamination-dev.json
    python scripts/contamination_probe.py --split dev --repos repos.json --env-file .env.mistral --out ...  # check model

Needs the pinned clone(s) and a model (``.env``, or ``--env-file``/``--model``/``--freeze`` as for the experiment).
Spends one model call per changed source file per episode.  The frozen test split needs ``--final`` like every other
script.  Writes one JSON file (per-episode recall of the commit's added lines, pooled recall, episodes over the
"memorised" threshold) and prints the summary; see ``testorcode.contamination`` for the scoring.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from testorcode.config import Settings
from testorcode.contamination import run_probe
from testorcode.dataset import DEFAULT_DATASET, episode_path, load_episodes
from testorcode.experiment import ROOT
from testorcode.freeze import load_frozen
from testorcode.llm_client import create_llm_client
from testorcode.logging_config import configure_logging


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--final", action="store_true", help="required for the frozen test split")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--episodes", default="", help="comma separated ids; default: the whole split")
    parser.add_argument("--repo", default=str(ROOT.parent / "boltons"), help="clone of the pinned repository")
    parser.add_argument("--repos", default=None, help="JSON file: repo_id -> {path, ...} (as for run_experiment)")
    parser.add_argument("--env-file", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--freeze", default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.split == "test" and not args.final:
        parser.error("the test split is frozen; pass --final only for the frozen evaluation")

    settings = Settings.from_env(env_file=Path(args.env_file) if args.env_file else ROOT / ".env")
    if args.freeze:
        settings = load_frozen(args.freeze)[0].apply(settings)
    if args.model:
        settings = settings.model_copy(update={"llm_model": args.model})
    configure_logging(settings.log_level, secrets=settings.secret_values())

    wanted = {e for e in args.episodes.split(",") if e}
    episodes = [e for e in load_episodes(episode_path(args.split, dataset=args.dataset))
                if not wanted or e.episode_id in wanted]
    repos: dict[str, Path] = {}
    if args.repos:
        base = Path(args.repos).resolve().parent
        repos = {k: (base / v["path"]).resolve() for k, v in json.loads(Path(args.repos).read_text()).items()}

    llm = create_llm_client(settings)
    results = []
    for repo_id in dict.fromkeys(e.agent_input.repo.repo_id for e in episodes):
        repo = repos.get(repo_id, Path(args.repo))
        group = [e for e in episodes if e.agent_input.repo.repo_id == repo_id]
        results.append(run_probe(group, repo, llm, model_id=settings.llm_model,
                                 progress=lambda eid: sys.stderr.write(f"probing {eid}\n")))
    merged = {
        **results[0],
        "episodes": [r for part in results for r in part["episodes"]],
        "split": args.split,
        "dataset": args.dataset,
    } if results else {"episodes": []}
    if len(results) > 1:  # pool across repositories
        scored = [r for r in merged["episodes"] if r["added_line_recall"] is not None]
        added = sum(r["added_lines"] for r in scored)
        merged["summary"] = {
            "episodes_scored": len(scored),
            "pooled_added_line_recall": sum(r["recovered_lines"] for r in scored) / added if added else None,
            "episodes_memorised": sum(bool(r["memorised"]) for r in scored),
        }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
    sys.stdout.write(json.dumps(merged.get("summary", {}), indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
