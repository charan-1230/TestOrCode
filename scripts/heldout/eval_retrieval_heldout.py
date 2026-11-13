"""Retrieval-only evaluation on the 72 held-out episodes (no LLM, no orchestrator).

Compares BM25 with hybrid (BM25 + Voyage ``voyage-code-3`` embeddings, RRF) against the gold evidence ids in
``ground_truth``.  Gold labels are read here only; agents never see them.

    python scripts/heldout/eval_retrieval_heldout.py --estimate          # unique texts / tokens / ETA, no network
    python scripts/heldout/eval_retrieval_heldout.py --warm              # embed everything into the cache (resumable)
    python scripts/heldout/eval_retrieval_heldout.py --mode bm25         # offline
    python scripts/heldout/eval_retrieval_heldout.py --mode hybrid       # cache only once warmed (no API call)
    python scripts/heldout/eval_retrieval_heldout.py --compare           # tables + paired bootstrap from both result files

Protocols: ``standard`` = the corpus as shipped; ``discoverable`` also drops gold the CI diff already shows; ``msgfree`` = the CI commit's own ``msg:`` item removed (the agent already
reads that text, so finding it proves nothing).  Gold that was removed leaves the episode out of that protocol's recall.
Methods: ``naive`` (B1: failing test text), ``intent`` (whole-episode intent queries, message-free), ``group`` (the Intent
Analyst's own ``group_queries`` over oracle failure groups) and ``intent+msg`` (degenerate; shown for completeness).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from testorcode.agents.intent_analyst import group_queries  # noqa: E402
from testorcode.config import Settings  # noqa: E402
from testorcode.dataset import load_episodes  # noqa: E402
from testorcode.evaluation import stale_evidence_ids  # noqa: E402
from testorcode.retrieval import (  # noqa: E402
    Retriever,
    ci_message_evidence_id,
    build_retriever,
    hit_rank,
    intent_queries,
    naive_query,
    recall_at_k,
    search_many,
)
from testorcode.retrieval_hybrid import _document_text, estimate_tokens, retrieval_config_id  # noqa: E402
from testorcode.schemas.contracts import FailureGroupDetail  # noqa: E402
from testorcode.schemas.episode_schemas import Episode, FailureGroup, RepairDirection, RetrievalCorpusSpec  # noqa: E402

LIBRARIES = ("more_itertools", "attrs", "packaging")
KS = (1, 3, 5, 10)
DEPTH = 50
METHODS = ("naive", "intent", "group", "intent+msg")
PROTOCOLS = ("standard", "msgfree", "discoverable")
DIFF_VISIBLE_FRACTION = 0.5  # an item whose non-trivial lines are at least this share "added by the CI diff" is in the prompt
RESULTS = ROOT / "results"


def load_all() -> list[tuple[str, Episode]]:
    out = []
    for lib in LIBRARIES:
        out += [(lib, e) for e in load_episodes(ROOT / "datasets" / f"heldout_{lib}.jsonl")]
    return out


def gold_of(episode: Episode) -> set[str]:
    return {g for lbl in episode.ground_truth.per_test_labels for g in (lbl.gold_evidence_ids or [])}


def ci_message_id(episode: Episode) -> str:
    return ci_message_evidence_id(episode.agent_input.ci_event.commit.sha)


def gold_visibility(episode: Episode, evidence_id: str) -> str:
    """Is this gold item already in the agent's prompt?

    ``prompt_message``: the CI commit's own message (``ci_event.commit_message``).  ``prompt_diff``: at least half of the
    item's non-trivial lines are lines the CI diff adds, so the diff the agent reads already shows it.  Otherwise
    ``discoverable``: only retrieval (or reading files) can surface it.
    """
    if evidence_id == ci_message_id(episode):
        return "prompt_message"
    item = next(i for i in episode.agent_input.corpus.items if i.evidence_id == evidence_id)
    added = {ln[1:].strip() for c in episode.agent_input.ci_event.changed_files
             for ln in (c.unified_diff or "").splitlines() if ln.startswith("+") and not ln.startswith("+++")}
    lines = [ln.strip() for ln in (item.text or "").splitlines() if len(ln.strip()) > 12]
    if lines and sum(ln in added for ln in lines) / len(lines) >= DIFF_VISIBLE_FRACTION:
        return "prompt_diff"
    return "discoverable"


def protocol_gold(episode: Episode, gold: set[str], protocol: str) -> set[str]:
    """Gold that ``protocol`` counts: all of it, minus the CI message, or only what the prompt does not already show."""
    if protocol == "standard":
        return set(gold)
    if protocol == "msgfree":
        return gold - {ci_message_id(episode)}
    return {g for g in gold if gold_visibility(episode, g) == "discoverable"}


def corpus_for(episode: Episode, protocol: str) -> RetrievalCorpusSpec:
    corpus = episode.agent_input.corpus
    if protocol == "standard":
        return corpus  # msgfree and discoverable below: the CI message item is the visible evidence, so it is not indexed
    drop = ci_message_id(episode)
    return corpus.model_copy(update={"items": [i for i in corpus.items or [] if i.evidence_id != drop]})


def oracle_groups(episode: Episode) -> list[tuple[FailureGroupDetail, set[str], bool]]:
    """Per gold function: the failure group Triage would ideally build, the gold evidence of its tests and whether the
    group is a gold UPDATE_TEST one.  Oracle: ``function_qualname`` comes from the gold labels, so this is an upper bound."""
    ci = episode.agent_input.ci_event
    tests = {t.test_id: t for t in ci.failing_tests}
    source = [c for c in ci.changed_files if "test" not in c.path.lower().split("/")[-1]]
    diff = "\n".join(c.unified_diff or "" for c in source)
    by_function: dict[str, list] = defaultdict(list)
    for label in episode.ground_truth.per_test_labels:
        if label.function_qualname and label.test_id in tests:
            by_function[label.function_qualname].append(label)
    groups = []
    for function, labels in sorted(by_function.items()):
        gold = {g for lbl in labels for g in (lbl.gold_evidence_ids or [])}
        if not gold:
            continue
        ids = [lbl.test_id for lbl in labels]
        detail = FailureGroupDetail(
            group=FailureGroup(group_id=f"g-{function}", function_qualname=function, test_ids=ids),
            failing_tests=tuple(tests[i] for i in ids),
            affected_files=tuple(c.path for c in source),
            diff_context=diff or None,
        )
        groups.append((detail, gold, any(lbl.gold_direction is RepairDirection.UPDATE_TEST for lbl in labels)))
    return groups


def episode_queries(episode: Episode) -> dict[str, list[list[str]]]:
    """Every query the evaluation sends for ``episode``, by method (a list of query lists: group has one per group)."""
    ci = episode.agent_input.ci_event
    return {
        "naive": [[naive_query(ci)]],
        "intent": [intent_queries(ci, include_message=False)],
        "intent+msg": [intent_queries(ci)],
        "group": [group_queries(detail) for detail, _, _ in oracle_groups(episode)],
    }


def ranked_ids(retriever: Retriever, method: str, query_lists: list[list[str]]) -> list[list[str]]:
    if method == "naive":
        return [[h.evidence_id for h in retriever.search(query_lists[0][0], top_k=DEPTH)]]
    return [[h.evidence_id for h in search_many(retriever, qs, top_k=DEPTH)] for qs in query_lists]


def metrics(ranked: list[str], gold: set[str], stale: set[str]) -> dict:
    row: dict = {"rank": hit_rank(ranked, gold)}
    for k in KS:
        top = ranked[:k]
        row[f"recall@{k}"] = recall_at_k(ranked, gold, k)
        row[f"hit@{k}"] = float(bool(gold & set(top)))  # evidence coverage: some gold item is in the top k
    row["precision@5"] = len(gold & set(ranked[:5])) / 5  # citation-precision proxy: share of the top 5 that is gold
    row["stale@5"] = len(stale & set(ranked[:5])) / 5  # share of the top 5 that is (heuristically) stale
    row["stale_in_top5"] = float(bool(stale & set(ranked[:5])))
    return row


def stale_for_episode(episode: Episode) -> set[str]:
    """Stale items for the episode.  Stale text only matters where the behaviour changed on purpose (gold UPDATE_TEST),
    as in ``evaluation._evidence``; a FIX_CODE function's docstring is still true."""
    functions = {lbl.function_qualname for lbl in episode.ground_truth.per_test_labels
                 if lbl.function_qualname and lbl.gold_direction is RepairDirection.UPDATE_TEST}
    return set().union(*(stale_evidence_ids(episode, f) for f in functions)) if functions else set()


def evaluate(settings: Settings) -> dict:
    rows = []
    for lib, episode in load_all():
        gold_all = gold_of(episode)
        if not gold_all:
            continue
        stale = stale_for_episode(episode)
        queries = episode_queries(episode)
        groups = oracle_groups(episode)
        visibility = {g: gold_visibility(episode, g) for g in sorted(gold_all)}
        for protocol in PROTOCOLS:
            gold = protocol_gold(episode, gold_all, protocol)
            if not gold:
                continue  # this protocol has nothing to find in this episode: it is left out of its denominator
            retriever = build_retriever(corpus_for(episode, protocol), settings=settings)
            base = {"library": lib, "episode": episode.episode_id, "scenario": episode.scenario.value, "protocol": protocol}
            for method in METHODS:
                if method == "group":
                    lists = ranked_ids(retriever, method, queries[method]) if groups else []
                    for (detail, ggold, update), ids in zip(groups, lists, strict=True):
                        g = protocol_gold(episode, ggold, protocol)
                        if g:
                            gstale = stale_evidence_ids(episode, detail.group.function_qualname) if update else set()
                            rows.append({**base, "method": method, "unit": detail.group.group_id, "gold": sorted(g),
                                         "gold_visibility": {x: visibility[x] for x in sorted(g)}, **metrics(ids, g, gstale)})
                else:
                    ids = ranked_ids(retriever, method, queries[method])[0]
                    rows.append({**base, "method": method, "unit": "episode", "gold": sorted(gold),
                                 "gold_visibility": {x: visibility[x] for x in sorted(gold)}, **metrics(ids, gold, stale)})
    return {"config": retrieval_config_id(settings), "depth": DEPTH, "metric_definitions": METRIC_DEFINITIONS, "rows": rows}


METRIC_DEFINITIONS = {
    "unit": "one row = one (episode, protocol, method); method 'group' has one row per oracle failure group with gold",
    "recall@k": "|gold in top k| / |gold| for the row, then the mean over rows; k = 1, 3, 5, 10 over a depth-50 ranking",
    "hit@k (evidence coverage)": "1 if at least one gold item is in the top k, else 0; mean over rows (the proposal names no coverage metric)",
    "precision@5": "|gold in top 5| / 5 (fixed denominator 5, even when fewer than 5 items come back); with one gold item the maximum is 0.2",
    "stale@5": "|stale items in top 5| / 5, stale = stale_evidence_ids() heuristic, only for gold-UPDATE_TEST functions (else 0 stale items)",
    "stale_in_top5": "1 if any stale item is in the top 5",
    "protocols": {
        "standard": "all gold; includes the CI commit's own message, which the prompt already shows (not a discovery)",
        "msgfree": "CI message item removed from corpus and gold; episodes with no other gold are left out",
        "discoverable": "msgfree, and also gold whose text the CI diff already shows (>= half of its lines are added by the diff) removed",
    },
    "ci": "paired percentile bootstrap (2000 resamples, seed 0) of the per-row hybrid - bm25 difference; uninformative for n < 10",
}


# --------------------------------------------------------------------------- #
# Warm-up: unique texts to embed
# --------------------------------------------------------------------------- #


def unique_texts() -> tuple[list[str], list[str]]:
    docs: dict[str, None] = {}
    queries: dict[str, None] = {}
    for _, episode in load_all():
        if not gold_of(episode):
            continue  # only gold-bearing episodes are evaluated
        for protocol in PROTOCOLS:
            for item in corpus_for(episode, protocol).items or []:
                docs[_document_text(item)] = None
        for lists in episode_queries(episode).values():
            for qs in lists:
                queries.update(dict.fromkeys(q for q in qs if q.strip()))
    return list(docs), list(queries)


def warm(settings: Settings) -> None:
    from testorcode.retrieval_hybrid import build_components

    embedder, _ = build_components(settings)
    docs, queries = unique_texts()
    sys.stdout.write(f"embedding {len(docs)} documents and {len(queries)} queries (cached ones are skipped)\n")
    sys.stdout.flush()
    for name, texts in (("query", queries), ("document", docs)):
        for start in range(0, len(texts), 160):
            embedder.embed(texts[start:start + 160], input_type=name)
            sys.stdout.write(f"{name}: {min(start + 160, len(texts))}/{len(texts)}\n")
            sys.stdout.flush()


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def aggregate(rows: list[dict]) -> dict:
    out = {}
    for protocol in PROTOCOLS:
        for method in METHODS:
            sel = [r for r in rows if r["protocol"] == protocol and r["method"] == method]
            if not sel:
                continue
            cell = {"n": len(sel)}
            for k in KS:
                cell[f"recall@{k}"] = mean([r[f"recall@{k}"] for r in sel if r[f"recall@{k}"] is not None])
                cell[f"hit@{k}"] = mean([r[f"hit@{k}"] for r in sel])
            cell["precision@5"] = mean([r["precision@5"] for r in sel])
            cell["stale@5"] = mean([r["stale@5"] for r in sel])
            cell["stale_in_top5"] = mean([r["stale_in_top5"] for r in sel])
            out[f"{protocol}/{method}"] = cell
    return out


def by_gold_type(rows: list[dict]) -> dict:
    out: dict = {}
    for r in rows:
        if r["method"] != "intent":
            continue
        for kind in sorted({g.split(":", 1)[0] for g in r["gold"]}):
            gold = {g for g in r["gold"] if g.startswith(kind + ":")}
            out.setdefault(f"{r['protocol']}/{kind}", []).append(gold)
    return {k: len(v) for k, v in out.items()}


def bootstrap(diffs: list[float], n: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(diffs) for _ in diffs) / len(diffs) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n)]


def fmt(v: float | None) -> str:
    return "  -  " if v is None else f"{v:.2f}"


def compare() -> str:
    a = json.loads((RESULTS / "retrieval_heldout_bm25.json").read_text())
    b = json.loads((RESULTS / "retrieval_heldout_hybrid.json").read_text())
    lines = [f"BM25: {a['config']}", f"Hybrid: {b['config']}", "", "Metric definitions (denominators):"]
    for name, text in METRIC_DEFINITIONS.items():
        if isinstance(text, dict):
            lines += [f"- protocol `{k}`: {v}" for k, v in text.items()]
        else:
            lines.append(f"- {name}: {text}")
    lines.append("")
    agg_a, agg_b = aggregate(a["rows"]), aggregate(b["rows"])
    cols = [f"recall@{k}" for k in KS] + [f"hit@{k}" for k in (1, 5, 10)] + ["precision@5", "stale@5"]
    lines.append("| protocol/method | system | n | " + " | ".join(cols) + " |")
    lines.append("|" + "---|" * (len(cols) + 3))
    for key in agg_a:
        for name, agg in (("bm25", agg_a), ("hybrid", agg_b)):
            c = agg.get(key)
            if c:
                lines.append(f"| {key} | {name} | {c['n']} | " + " | ".join(fmt(c[col]) for col in cols) + " |")
    lines.append("")
    index_b = {(r["protocol"], r["method"], r["episode"], r["unit"]): r for r in b["rows"]}
    for protocol in PROTOCOLS:
        for method in ("intent", "group"):
            for metric in ("recall@5", "recall@10"):
                diffs = [index_b[k][metric] - r[metric]
                         for r in a["rows"] if r["protocol"] == protocol and r["method"] == method
                         and (k := (protocol, method, r["episode"], r["unit"])) in index_b and r[metric] is not None]
                if diffs:
                    lo, hi = bootstrap(diffs)
                    note = "  [n < 10: interval not informative]" if len(diffs) < 10 else ""
                    lines.append(f"{protocol}/{method} {metric}: hybrid - bm25 = {sum(diffs)/len(diffs):+.3f} "
                                 f"(95% bootstrap CI {lo:+.3f}..{hi:+.3f}, n={len(diffs)}){note}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["bm25", "hybrid"])
    parser.add_argument("--rerank", choices=["none", "voyage", "local"], default="none")
    parser.add_argument("--embedding-backend", choices=["voyage", "local"], default="voyage")
    parser.add_argument("--cache-dir", default=str(ROOT / "results" / "embedding_cache"))
    parser.add_argument("--estimate", action="store_true")
    parser.add_argument("--warm", action="store_true")
    parser.add_argument("--compare", action="store_true")
    args = parser.parse_args()
    if args.compare:
        text = compare()
        (RESULTS / "retrieval_heldout_comparison.md").write_text(text + "\n", encoding="utf-8")
        sys.stdout.write(text + "\n")
        return 0
    base = Settings.from_env(env_file=ROOT / ".env") if (ROOT / ".env").is_file() else Settings.from_env({})
    settings = base.model_copy(update={
        "retrieval_mode": "hybrid" if (args.warm or args.mode == "hybrid") else "bm25",
        "embedding_backend": args.embedding_backend, "rerank_backend": args.rerank,
        "embedding_cache_dir": Path(args.cache_dir)})
    if args.estimate:
        docs, queries = unique_texts()
        tokens = sum(estimate_tokens(t) for t in docs + queries)
        keys = max(1, len(settings.voyage_api_keys()))
        minutes = tokens / (9000 * keys)
        sys.stdout.write(f"{len(docs)} documents + {len(queries)} queries = ~{tokens} est. tokens; "
                         f"{keys} key(s) at ~9k tokens/min each: ~{minutes:.0f} min\n")
        return 0
    if args.warm:
        warm(settings)
        return 0
    if args.mode is None:
        parser.error("choose --mode, --estimate, --warm or --compare")
    result = evaluate(settings)
    RESULTS.mkdir(exist_ok=True)
    suffix = args.mode if args.rerank == "none" else f"{args.mode}-{args.rerank}"
    (RESULTS / f"retrieval_heldout_{suffix}.json").write_text(json.dumps(result, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    agg = aggregate(result["rows"])
    for key, cell in agg.items():
        sys.stdout.write(f"{key:22} n={cell['n']:3} " + " ".join(f"R@{k}={fmt(cell[f'recall@{k}'])}" for k in KS)
                         + f" P@5={fmt(cell['precision@5'])} stale@5={fmt(cell['stale@5'])}\n")
    sys.stdout.write(f"gold types in intent rows: {by_gold_type(result['rows'])}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
