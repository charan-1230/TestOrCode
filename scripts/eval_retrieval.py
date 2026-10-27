"""Gold-evidence retrieval check on the pilot episodes (BM25, offline, no LLM).

Reports, per episode that has gold evidence ids, the rank of the best gold item and Recall@k under:

* ``naive``       B1-style standard RAG: the failing test's own text;
* ``intent``      Full/A2 intent queries over the whole CI diff, message-free;
* ``intent+group`` the same, but per failure group (oracle grouping from gold function names, i.e. an upper bound that
                   assumes Triage groups correctly);
* ``intent+msg``  intent queries that also use the commit subject (degenerate: the gold item *is* that text; shown only
                   to document why message-free numbers are the honest ones).

Gold ids are read from the episode file, so corrections made during human review are picked up automatically.
The frozen test split is refused unless ``--final`` is given.  Results go to ``results/retrieval_<split>.json``.

    python scripts/eval_retrieval.py            # dev split
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from testorcode.dataset import episode_path, load_episodes
from testorcode.retrieval import (
    build_retriever,
    hit_rank,
    intent_queries,
    naive_query,
    recall_at_k,
    search_many,
)
from testorcode.schemas.contracts import FailureGroupDetail
from testorcode.schemas.episode_schemas import Episode, FailureGroup

DEPTH = 50
KS = (1, 3, 5, 8)


def oracle_groups(episode: Episode) -> list[tuple[FailureGroupDetail, set[str]]]:
    """Per gold function: a failure group (upper bound for Triage) and the gold evidence of its tests."""
    tests = {t.test_id: t for t in episode.agent_input.ci_event.failing_tests}
    by_function: dict[str, list] = {}
    for label in episode.ground_truth.per_test_labels:
        if label.function_qualname:
            by_function.setdefault(label.function_qualname, []).append(label)
    groups = []
    for function, labels in sorted(by_function.items()):
        gold = {g for lbl in labels for g in (lbl.gold_evidence_ids or [])}
        if not gold:
            continue
        module = function.split(".")[1] if function.count(".") >= 2 else function
        ids = [lbl.test_id for lbl in labels]
        detail = FailureGroupDetail(
            group=FailureGroup(group_id=f"g-{function}", function_qualname=function, test_ids=ids),
            failing_tests=tuple(tests[i] for i in ids),
            affected_files=(f"boltons/{module}.py",),
        )
        groups.append((detail, gold))
    return groups


def score(ranked: list[str], gold: set[str]) -> dict:
    return {"rank": hit_rank(ranked, gold), **{f"recall@{k}": recall_at_k(ranked, gold, k) for k in KS}}


def evaluate(split: str) -> list[dict]:
    rows = []
    for episode in load_episodes(episode_path(split)):
        ai = episode.agent_input
        retriever = build_retriever(ai.corpus)
        whole_gold = {g for lbl in episode.ground_truth.per_test_labels for g in (lbl.gold_evidence_ids or [])}
        if not whole_gold:
            continue
        ranked = {
            "naive": [h.evidence_id for h in retriever.search(naive_query(ai.ci_event), top_k=DEPTH)],
            "intent": [h.evidence_id for h in search_many(retriever, intent_queries(ai.ci_event, include_message=False), top_k=DEPTH)],
            "intent+msg": [h.evidence_id for h in search_many(retriever, intent_queries(ai.ci_event), top_k=DEPTH)],
        }
        row = {"episode": episode.episode_id, "scenario": episode.scenario.value, "gold": sorted(whole_gold),
               "methods": {name: score(ids, whole_gold) for name, ids in ranked.items()}}
        grouped = []
        for detail, gold in oracle_groups(episode):
            hits = search_many(retriever, intent_queries(ai.ci_event, detail, include_message=False), top_k=DEPTH)
            ids = [h.evidence_id for h in hits]
            grouped.append({"function": detail.group.function_qualname, **score(ids, gold),
                            "top8_types": dict(Counter(h.source_type.value for h in hits[:8]))})
        row["methods"]["intent+group"] = grouped
        rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    parser.add_argument("--final", action="store_true", help="allow the frozen test split (do this once, after freezing)")
    parser.add_argument("--out", default=str(ROOT / "results"))
    args = parser.parse_args()
    if args.split == "test" and not args.final:
        sys.stderr.write("refusing to read the frozen test split without --final (see datasets/README.md)\n")
        return 2
    rows = evaluate(args.split)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"retrieval_{args.split}.json").write_text(json.dumps(rows, indent=1, sort_keys=True) + "\n", encoding="utf-8")

    def fmt(rank: int | None) -> str:
        return f">{DEPTH}" if rank is None else str(rank)

    sys.stdout.write(f"{'episode':8} {'scen':4} {'naive':>6} {'intent':>7} {'+group':>7} {'+msg*':>6}   (gold rank; * = degenerate)\n")
    for r in rows:
        m = r["methods"]
        group = ",".join(fmt(g["rank"]) for g in m["intent+group"]) or "-"
        sys.stdout.write(f"{r['episode']:8} {r['scenario']:4} {fmt(m['naive']['rank']):>6} {fmt(m['intent']['rank']):>7} "
                         f"{group:>7} {fmt(m['intent+msg']['rank']):>6}\n")
    for name in ("naive", "intent"):
        for k in KS:
            vals = [r["methods"][name][f"recall@{k}"] for r in rows if r["methods"][name][f"recall@{k}"] is not None]
            sys.stdout.write(f"{name:7} recall@{k}: {sum(vals) / len(vals):.2f} over {len(vals)} episodes\n" if vals else "")
    sys.stdout.write(f"n={len(rows)} episodes with gold evidence: a smoke measurement, not a result\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
