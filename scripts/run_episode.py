"""Run one pilot episode end to end on the real repository and save the run record.

    real pytest failure -> Gemini decision -> restricted patch or escalation -> test rerun -> record

Steps: materialize the episode's CI state in a git worktree of the pinned clone, build ``Settings`` from
``.env``, run ``orchestrator.run_episode`` with the real agents and tools, write the ``ExperimentRunRecord``
to ``<out>/<run_id>.json`` (default ``results/day2``) and print a summary.

Gold labels (``pilot_<split>.jsonl``) stay on the harness side.  They are used only by the simulated
maintainer (proposal section 9: approve when the direction matches the gold label and the tests pass), which
the orchestrator asks after a Guard APPROVE, and to score the prediction after the run.  They are never
passed to an agent.

``--condition`` defaults to A2 (no memory). B1 runs the single-agent baseline on the same input. A3 is FULL
without the Integrity Guard (a green rerun decides; nothing writes memory). FULL and A3 need
``--memory PATH``: a ``JsonlMemoryStore`` (persistent across runs, so a stream of episodes shares it) seeded with the
episode's declared prior memory (``visible_memory``) before any agent acts.  The store is bound to
``<condition>/seed=<RANDOM_SEED>/<split>`` and refuses any other (use ``memory.memory_path`` for one file each);
conditions without memory ignore ``--memory``.  A1 (no intent retrieval) is FULL without the corpus, ``search_repo``
and the commit message, and also needs ``--memory``.  The sanity baselines (TRIVIAL_UPDATE_TEST, TRIVIAL_FIX_CODE)
need no repository, worktree or API key.  Conditions without a code path are recorded as not implemented.

    python scripts/run_episode.py --episode dev-05 --repo ../boltons
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

from testorcode.baselines import is_trivial
from testorcode.conditions import profile_for
from testorcode.config import Settings
from testorcode.dataset import agent_view_path, episode_path, load_agent_inputs, load_episodes
from testorcode.evaluation import predicted_direction
from testorcode.experiment import dataset_version, simulated_maintainer
from testorcode.llm_client import create_llm_client
from testorcode.logging_config import configure_logging
from testorcode.memory import JsonlMemoryStore, memory_namespace
from testorcode.orchestrator import run_episode
from testorcode.schemas.episode_schemas import DecisionLabel
from testorcode.schemas.result_schemas import (
    ExperimentCondition,
    ExperimentRunRecord,
)
from testorcode.tools.git_tools import run_git

ROOT = Path(__file__).resolve().parents[1]


def _direction(prediction) -> DecisionLabel | None:
    """The side blamed, exactly as the metrics score it (``evaluation.predicted_direction``): the chosen repair, or
    for an escalation its hypothesis.  Trivial baselines have no hypothesis but still blame a side."""
    direction = predicted_direction(prediction)
    return DecisionLabel(direction.value) if direction is not None else None


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
    parser.add_argument("--condition", default="A2", choices=[c.value for c in ExperimentCondition])
    parser.add_argument("--out", default=str(ROOT / "results" / "day2"))
    parser.add_argument("--memory", default=None, help="JSONL memory store (needed for FULL and A3; ignored without memory)")
    args = parser.parse_args()
    if args.split == "test" and not args.allow_test_split:
        parser.error("the test split is frozen; pass --allow-test-split only for the frozen evaluation")

    condition = ExperimentCondition(args.condition)
    settings = Settings.from_env(env_file=ROOT / ".env").model_copy(update={"experiment_condition": condition})
    configure_logging(settings.log_level, log_file=settings.log_file, secrets=settings.secret_values())

    episode = next(e for e in load_agent_inputs(agent_view_path(args.split)) if e.episode_id == args.episode)
    worktree = Path(args.worktree or ROOT.parent / "episode-worktrees" / args.episode).absolute()
    trivial = is_trivial(condition)  # deterministic baseline: no worktree, memory or model
    if not trivial:
        _materializer()(args.repo, worktree, episode.model_dump(mode="json"))

    memory = None
    if args.memory and profile_for(condition).memory and not trivial:
        # One store per condition, seed and stream (split): a store opened under another namespace is refused.
        memory = JsonlMemoryStore(
            args.memory, python_executable=settings.test_python, timeout_seconds=settings.test_timeout_seconds,
            docker_image=episode.environment.docker_image,
            namespace=memory_namespace(condition.value, settings.random_seed, args.split),
        )
        # Harness side, before any agent acts: the CI state is pristine, so seeded probes get their support code here.
        seeded = memory.seed(episode.visible_memory or [], worktree_path=worktree)
        if seeded:
            print(f"memory: seeded {seeded}")

    llm = None if trivial else create_llm_client(settings)
    # Harness side: gold labels feed only the simulated maintainer and the scoring below.
    full = next(e for e in load_episodes(episode_path(args.split)) if e.episode_id == args.episode)
    gold = full.gold_actions()
    record = run_episode(
        episode, worktree_path=worktree, settings=settings, llm=llm, memory=memory,
        maintainer=simulated_maintainer(gold),
    )
    record = record.model_copy(update={"provenance": record.provenance.model_copy(
        update={"split": args.split, "dataset_version": dataset_version(args.split)}
    )})
    summary = _summary(record, gold, full.scenario, None if trivial else worktree)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{record.provenance.run_id}.json").write_text(record.model_dump_json(indent=2) + "\n")
    (out / f"{record.provenance.run_id}.summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    _print(record, summary, out)
    return 0


def _summary(record: ExperimentRunRecord, gold: dict, scenario, worktree: Path | None) -> dict:
    status = run_git(worktree, ["status", "--porcelain", "--", "boltons", "tests"]).stdout if worktree else ""
    ci_dirty = run_git(worktree, ["diff", "--name-only"]).stdout.split() if worktree else []
    return {
        "run_id": record.provenance.run_id,
        "episode_id": record.provenance.episode_id,
        "scenario": str(scenario),
        "condition": record.provenance.condition,
        "model_id": record.provenance.model_id,
        "prompt_version": record.provenance.prompt_template_version,
        "code_revision": record.provenance.code_revision,
        "failure_groups": [g.model_dump(mode="json") for g in record.failure_groups],
        "unstable_test_ids": record.unstable_test_ids,
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
             "direction": _direction(p), "cited": p.cited_evidence_ids, "decided_by": p.decided_by,
             "gold": gold.get(p.test_id), "match": gold.get(p.test_id) == p.predicted_action,
             "direction_match": _direction(p) == gold.get(p.test_id)}
            for p in record.predictions
        ],
        "escalation": record.outcome.escalation.model_dump(mode="json") if record.outcome.escalation else None,
        "maintainer_approved": record.outcome.maintainer_approved,
        "memory_writes": [w.model_dump(mode="json") for w in record.memory_writes],
        "tool_calls": len(record.tool_events),
        "tool_events": [
            {"tool": e.tool, "by": e.called_by, "status": e.status, "args": e.arguments, "error": e.error_message}
            for e in record.tool_events
        ],
        "errors": [e.model_dump(mode="json") for e in record.errors],
        "worktree_changed_files_after_run": ci_dirty,
        "worktree_status_after_run": status.splitlines(),
    }


def _print(record: ExperimentRunRecord, summary: dict, out: Path) -> None:
    w = sys.stdout.write
    w(f"\n=== {summary['episode_id']} ({summary['scenario']})  condition={summary['condition']}  model={summary['model_id']}\n")
    for g in summary["failure_groups"]:
        w(f"group {g['group_id']}: {g['function_qualname'] or '-'} (mapped by {g['mapped_by'] or 'agent'}, "
          f"{g['stability_runs']} stability run(s)) tests={len(g['test_ids'])}\n")
    if summary["unstable_test_ids"]:
        w(f"unstable (dropped): {summary['unstable_test_ids']}\n")
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
        esc = summary["escalation"]
        w(f"escalation: {esc['trigger']}: {esc['summary']}\n")
        for item in esc["evidence_missing"]:
            w(f"  missing: {item}\n")
        for item in esc["competing_hypotheses"]:
            w(f"  hypothesis: {item[:300]}\n")
    w(f"maintainer approved: {summary['maintainer_approved']}   memory writes: "
      f"{[w_['blocked_reason'] or 'written' for w_ in summary['memory_writes']] or '-'}\n")
    tools = {}
    for e in summary["tool_events"]:
        tools[f"{e['tool']}/{e['by']}"] = tools.get(f"{e['tool']}/{e['by']}", 0) + 1
    w(f"tool calls: {summary['tool_calls']} {tools}\n")
    for err in summary["errors"]:
        w(f"error: {err['category']} at {err['operation']}: {err['message']}\n")
    w(f"run status: {summary['run_status']}   tokens: {summary['tokens']['total_tokens']}   "
      f"calls: {summary['tokens']['llm_calls']}\nrecord: {out / (summary['run_id'] + '.json')}\n")


if __name__ == "__main__":
    raise SystemExit(main())
