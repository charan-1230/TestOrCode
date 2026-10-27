"""Run one pilot episode end to end on the real repository and save the run record.

    real pytest failure -> Gemini decision -> restricted patch or escalation -> test rerun -> record

Steps: materialize the episode's CI state in a git worktree of the pinned clone, build ``Settings`` from
``.env``, run ``orchestrator.run_episode`` with the real agents and tools, write the ``ExperimentRunRecord``
to ``results/day1/<run_id>.json`` and print a summary.  Only after the run, the harness reads the gold
labels (``pilot_<split>.jsonl``) to score the prediction; they are never passed to the agents.

The condition is recorded as A2 (no memory): ``MemoryStore`` is not implemented yet, and A2 is exactly
the multi-agent pipeline with intent retrieval and the Integrity Guard.

    python scripts/run_episode.py --episode dev-05 --repo ../boltons
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from functools import partial
from pathlib import Path

from testorcode.agents.integrity_guard import verify_repair
from testorcode.conditions import episode_for_condition
from testorcode.config import Settings
from testorcode.dataset import agent_view_path, episode_path, load_agent_inputs, load_episodes
from testorcode.llm_client import create_llm_client
from testorcode.logging_config import configure_logging
from testorcode.orchestrator import Agents, run_episode
from testorcode.schemas.episode_schemas import DecisionLabel, HypothesisKind
from testorcode.schemas.result_schemas import ExperimentCondition, ExperimentRunRecord
from testorcode.tools.git_tools import run_git

ROOT = Path(__file__).resolve().parents[1]
CONDITION = ExperimentCondition.A2_NO_MEMORY
_DIRECTION = {HypothesisKind.OUTDATED_TEST: DecisionLabel.UPDATE_TEST, HypothesisKind.REGRESSION: DecisionLabel.FIX_CODE}


def _materializer():
    spec = importlib.util.spec_from_file_location("materialize_episode", ROOT / "scripts" / "pilot" / "materialize_episode.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.materialize


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--episode", required=True)
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--allow-test-split", action="store_true", help="the test split is frozen; refuse it by default")
    parser.add_argument("--repo", default=str(ROOT.parent / "boltons"), help="clone of the pinned repository")
    parser.add_argument("--worktree", default=None, help="default: ../episode-worktrees/<episode>")
    parser.add_argument("--out", default=str(ROOT / "results" / "day1"))
    args = parser.parse_args()
    if args.split == "test" and not args.allow_test_split:
        parser.error("the test split is frozen; pass --allow-test-split only for the frozen evaluation")

    settings = Settings.from_env(env_file=ROOT / ".env").model_copy(update={"experiment_condition": CONDITION})
    configure_logging(settings.log_level, log_file=settings.log_file, secrets=settings.secret_values())

    episode = next(e for e in load_agent_inputs(agent_view_path(args.split)) if e.episode_id == args.episode)
    episode = episode_for_condition(episode, CONDITION)
    worktree = Path(args.worktree or ROOT.parent / "episode-worktrees" / args.episode).absolute()
    _materializer()(args.repo, worktree, episode.model_dump(mode="json"))

    llm = create_llm_client(settings)
    agents = Agents(
        integrity_guard=partial(
            verify_repair, python_executable=settings.test_python, timeout_seconds=settings.test_timeout_seconds
        )
    )
    record = run_episode(episode, worktree_path=worktree, settings=settings, llm=llm, memory=None, agents=agents)

    # Harness side only from here on: gold labels are read after the run, for scoring.
    full = next(e for e in load_episodes(episode_path(args.split)) if e.episode_id == args.episode)
    summary = _summary(record, full.gold_actions(), full.scenario, worktree)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{record.provenance.run_id}.json").write_text(record.model_dump_json(indent=2) + "\n")
    (out / f"{record.provenance.run_id}.summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    _print(record, summary, out)
    return 0


def _summary(record: ExperimentRunRecord, gold: dict, scenario, worktree: Path) -> dict:
    status = run_git(worktree, ["status", "--porcelain", "--", "boltons", "tests"]).stdout
    ci_dirty = run_git(worktree, ["diff", "--name-only"]).stdout.split()
    return {
        "run_id": record.provenance.run_id,
        "episode_id": record.provenance.episode_id,
        "scenario": str(scenario),
        "condition": record.provenance.condition,
        "model_id": record.provenance.model_id,
        "run_status": record.outcome.status,
        "final_test_status": record.final_test_status,
        "guard_decision": record.integrity.guard_decision,
        "guard_reasons": record.integrity.rejection_reasons,
        "revisions_used": record.outcome.revisions_used,
        "tokens": record.usage.model_dump(mode="json"),
        "test_runs": [
            {"phase": r.phase, "status": r.status, "passed": r.passed, "failed": r.failed, "failing": r.failing_test_ids}
            for r in record.test_runs
        ],
        "patches": [
            {"patch_id": p.patch_id, "scope": p.scope, "files": p.target_files, "status": p.application_status,
             "apply_error": p.apply_error}
            for p in record.patches
        ],
        # "predicted" is the final action (an unverified repair ends as ESCALATE); "direction" is the side the
        # model blamed (from its hypothesis), scored separately against the gold direction.
        "predictions": [
            {"test_id": p.test_id, "predicted": p.predicted_action, "hypothesis": p.hypothesis,
             "direction": _DIRECTION.get(p.hypothesis), "cited": p.cited_evidence_ids, "decided_by": p.decided_by,
             "gold": gold.get(p.test_id), "match": gold.get(p.test_id) == p.predicted_action,
             "direction_match": _DIRECTION.get(p.hypothesis) == gold.get(p.test_id)}
            for p in record.predictions
        ],
        "escalation": record.outcome.escalation.model_dump(mode="json") if record.outcome.escalation else None,
        "errors": [e.model_dump(mode="json") for e in record.errors],
        "worktree_changed_files_after_run": ci_dirty,
        "worktree_status_after_run": status.splitlines(),
    }


def _print(record: ExperimentRunRecord, summary: dict, out: Path) -> None:
    w = sys.stdout.write
    w(f"\n=== {summary['episode_id']} ({summary['scenario']})  condition={summary['condition']}  model={summary['model_id']}\n")
    for run in summary["test_runs"]:
        w(f"test run [{run['phase']}]: {run['status']} passed={run['passed']} failed={run['failed']}\n")
    for p in summary["predictions"]:
        w(f"decision {p['test_id']}: {p['predicted']} (hypothesis {p['hypothesis']}, cited {p['cited']}) "
          f"direction={p['direction']} gold={p['gold']} action_match={p['match']} direction_match={p['direction_match']}\n")
    for patch in record.patches:
        w(f"\npatch {patch.patch_id} scope={patch.scope} status={patch.application_status} "
          f"error={patch.apply_error or '-'}\n{patch.unified_diff}\n")
    w(f"guard: {summary['guard_decision']}  reasons: {summary['guard_reasons'] or '-'}\n")
    if summary["escalation"]:
        w(f"escalation: {summary['escalation']['trigger']}: {summary['escalation']['summary']}\n")
    for err in summary["errors"]:
        w(f"error: {err['category']} at {err['operation']}: {err['message']}\n")
    w(f"run status: {summary['run_status']}   tokens: {summary['tokens']['total_tokens']}   "
      f"calls: {summary['tokens']['llm_calls']}\nrecord: {out / (summary['run_id'] + '.json')}\n")


if __name__ == "__main__":
    raise SystemExit(main())
