"""One live hosted-model call: the Intent Analyst prompt for one dev episode, validated as ``IntentAnalysis``.

Day-1 check that credentials, the endpoint and structured output work.  No tests are run; the failure
group is taken from the episode's CI record.  Writes ``results/day1/llm_smoke.json`` (model id, token
usage, parsed decision; never the key or the prompt).

    python scripts/llm_smoke.py --episode dev-05 [--worktree ../episode-worktrees/dev-05]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from testorcode.agents.intent_analyst import group_queries
from testorcode.agents.prompts import INTENT_SYSTEM, INTENT_VERSION, build_intent_user
from testorcode.config import Settings
from testorcode.dataset import agent_view_path, load_agent_inputs
from testorcode.llm_client import create_llm_client
from testorcode.logging_config import configure_logging
from testorcode.retrieval import build_retriever, search_many
from testorcode.schemas.contracts import (
    CommitInfo,
    EpisodeContext,
    FailureGroupDetail,
    IntentAnalysis,
)
from testorcode.schemas.episode_schemas import FailureGroup
from testorcode.schemas.result_schemas import AgentRole

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--episode", default="dev-05")
    parser.add_argument("--worktree", default=None, help="materialized episode (only to show the test source)")
    parser.add_argument("--out", default=str(ROOT / "results" / "day1" / "llm_smoke.json"))
    args = parser.parse_args()

    settings = Settings.from_env(env_file=ROOT / ".env")
    configure_logging(settings.log_level, secrets=settings.secret_values())
    episode = next(e for e in load_agent_inputs(agent_view_path("dev")) if e.episode_id == args.episode)
    ci = episode.ci_event
    worktree = Path(args.worktree).absolute() if args.worktree else ROOT
    context = EpisodeContext.from_episode_input(episode, worktree)
    group = FailureGroupDetail(
        group=FailureGroup(group_id="g1", test_ids=[t.test_id for t in ci.failing_tests]),
        failing_tests=tuple(ci.failing_tests),
        affected_files=tuple(c.path for c in ci.changed_files),
        context_commits=(CommitInfo(sha=ci.commit.sha, message=ci.commit_message),) if ci.commit_message else (),
        diff_context="\n".join(c.unified_diff or "" for c in ci.changed_files) or None,
    )
    evidence = search_many(build_retriever(episode.corpus), group_queries(group), top_k=8)

    llm = create_llm_client(settings)
    started = time.monotonic()
    response = llm.complete_structured(
        role=AgentRole.INTENT_ANALYST,
        system=INTENT_SYSTEM,
        user=build_intent_user(context, group, evidence, []),
        response_model=IntentAnalysis,
    )
    elapsed = time.monotonic() - started
    parsed = response.parsed
    retrieved = [e.evidence_id for e in evidence]
    unknown = [i for i in parsed.cited_evidence_ids if i not in retrieved]

    record = {
        "kind": "llm_smoke",
        "at": datetime.now(UTC).isoformat(),
        "episode_id": episode.episode_id,
        "provider": settings.llm_provider,
        "base_url": settings.llm_base_url,
        "requested_model": settings.llm_model,
        "model_id": response.model_id,
        "structured_mode": settings.llm_structured_mode,
        "prompt_version": INTENT_VERSION,
        "latency_seconds": round(elapsed, 2),
        "usage": response.usage.model_dump(mode="json"),
        "retrieved_evidence_ids": retrieved,
        "cited_ids_not_retrieved": unknown,
        "validated_as": "IntentAnalysis",
        "response": parsed.model_dump(mode="json"),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2) + "\n")
    sys.stdout.write(json.dumps(record, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
