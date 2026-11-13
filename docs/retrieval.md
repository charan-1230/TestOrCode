# Retrieval (BM25 + hybrid)

Implements the 5-day plan's retrieval work: BM25 over commit messages and documentation (plus docstrings and changelog), evidence
IDs, the A1/A2/B1 switches, per-run evidence logging and the retrieval metrics. Hybrid retrieval (BM25 + Voyage embeddings, RRF) sits behind the same interface.
Code: `src/testorcode/retrieval_hybrid.py`, `src/testorcode/retrieval.py`, `src/testorcode/conditions.py`; check: `scripts/eval_retrieval.py`; tests: `tests/test_retrieval.py`,
`tests/test_conditions.py`.

## What is indexed
The episode's materialised corpus (`EpisodeInput.corpus.items`, see `datasets/README.md`): last 100 non-merge commit messages, README,
last 3 CHANGELOG sections, `docs/<module>.rst`, docstrings of the affected modules and (corpus `pilot-2`) source comments inside functions
changed by the CI diff, all at or before the CI commit. Items beyond
`history_cutoff_index` raise `HistoryCutoffError`. Evidence ids are the corpus ids (`msg:<sha12>`, `doc:<path>#<n>`, `changelog:<n>`,
`docstring:<module>.<qualname>`, `comment:<module>.<qualname>#L<line>`); results carry only corpus ids and refs, so a citation is traceable. `corpus.items is None` (build from a
worktree at run time) raises `NotImplementedError`: deferred.

## API
| Function | Purpose |
|---|---|
| `build_retriever(corpus, worktree_path=None, *, settings=None) -> Retriever` | BM25Okapi (deterministic: score desc, then `evidence_id`; items sharing no query term are dropped), or `HybridRetriever` when `settings.retrieval_mode == "hybrid"` |
| `Retriever.search(query, *, top_k)` | `RetrievedEvidence` list, `rank` from 1, `retrieval_method=BM25` |
| `naive_query(ci_event)` | B1 "standard RAG": the failing test's own text |
| `intent_queries(ci_event, group=None, *, include_message=True, include_failure_text=False)` | Full/A2: changed functions + modules, behaviour words (changed identifiers); never test ids |
| `search_many(retriever, queries, *, top_k)` / `reciprocal_rank_fusion` | multi-query fusion (also the hook for hybrid fusion later) |
| `SearchRepoTool(retriever, allowed_tools=..., top_k, max_calls, log)` | the `search_repo` agent tool: `ToolError` if not allowed, `BudgetExceededError` past `max_calls`, logs everything |
| `RetrievalLog` | `records()` for `ExperimentRunRecord.retrieved_evidence`; `ids()` / `unknown_citations()` for citation checks |
| `memory_evidence(MemoryReadResult)` | only records whose probe passed and that are not stale become `MEMORY_RECORD` evidence |
| `hit_rank`, `recall_at_k`, `citation_existence_rate` | pure metric helpers (support-of-claim and stale-evidence rates stay in evaluation) |

## Conditions (`testorcode.conditions`)
Numbering follows the 5-day plan (A3 = no Integrity Guard).

| Condition | Retrieval | Memory | Guard | Commit message visible |
|---|---|---|---|---|
| FULL | intent | yes | yes | yes |
| B1 | standard RAG (`search_repo`, naive) | no | n/a | yes |
| A1 no intent retrieval | none | yes | yes | **no** |
| A2 no memory | intent | no | yes | yes |
| A3 no Integrity Guard | intent | yes | no | yes |

`episode_for_condition(episode, condition)` returns the view a condition may see: memory off -> `visible_memory=None` (the schema's
"disabled"; `[]` means enabled but empty); A1 -> commit message withheld. Assumption to confirm: A1 hides the message (detailed document:
"Analyst sees only failing output and diff").

## Hybrid retrieval (`retrieval_hybrid.py`)
`HybridRetriever(items, embedder, reranker=None, *, cutoff_index=None)` ranks the corpus twice - BM25 top 50 and cosine similarity of dense
embeddings top 50 (numpy) - fuses the two lists with reciprocal-rank fusion (`RRF_K = 60`, fixed, not tuned) and, optionally, reorders
the fused top 30 with a reranker. Each hit carries `retrieval_method` (`hybrid` / `rerank`), `commit_index` and `component_ranks`
(`{"bm25": r, "dense": r}`), so a ranking can be audited after a run. `search_many` fuses several queries on top of this in the same way.

| Setting (`.env`) | Meaning |
|---|---|
| `RETRIEVAL_MODE=bm25\|hybrid` | index used by every retrieving condition (default `bm25`) |
| `EMBEDDING_BACKEND=voyage\|local` | `voyage-code-3` (the proposal's choice) or a local sentence-transformers model |
| `RERANK_BACKEND=none\|voyage\|local` | optional reranker: `rerank-2.5` or a local cross-encoder |
| `VOYAGE_API_KEY1..9` / `VOYAGE_API_KEY` | key pool; calls rotate over the keys |
| `EMBEDDING_CACHE_DIR` | append-only JSONL cache per model; resumable, one line per (input type, text) |

`scripts/run_experiment.py --retrieval-mode hybrid [--rerank ...] [--embedding-cache-dir ...]` switches the index for a whole experiment, so
B1/A2/FULL etc. keep the same model and budget and differ only in the index. `ExperimentRunRecord.provenance.retrieval_config_version`
records the setup (`retrieval_config_id(settings)`, e.g. `hybrid:bm25+voyage-code-3/rrf60/depth50/rerank30`).

**Voyage rate limits.** Each key is limited to 3 requests and 10k tokens per minute. `VoyageKeyPool` rotates the keys, keeps a rolling
60 s request/token ledger per key (estimated at 3 characters per token, 9k budget), sleeps until the freest key has room, benches a key
for 65 s after a 429 and retries transient connection errors. Documents are embedded by their first 1500 characters. Each text is
embedded once and cached, so a full evaluation re-runs from the cache with no network call.

## Chronology and leakage
* The corpus is built at the CI commit (`history_cutoff_commit`): history is the last 100 non-merge commit messages up to and including
  the CI commit; docs, changelog, docstrings and source comments are read from the tree at that commit.
* Items with `commit_index > history_cutoff_index` raise `HistoryCutoffError` in `build_retriever` and are also filtered inside both
  retrievers (defence in depth). `tests/test_heldout_retrieval.py` checks all 72 held-out corpora: unique ids, cutoff-clean, none of
  `future_commit_shas`, and that every gold id exists in the corpus.
* **Cutoff rule (decided in the audit).** What the agent can see at decision time is the prompt (including the CI commit's own message and
  diff), the tree at the CI commit, and history through the parent via the git tools. The corpus snapshot (CI commit) matches that, so
  indexing `msg:<ci sha12>` adds nothing the prompt lacks *while the message is visible*. When a condition withholds the message,
  `episode_for_condition` now also drops `msg:<ci sha12>` from a retained corpus (A1 already empties the corpus).
* Limits: only commit messages have a per-item `commit_index`; docs, changelog and docstring items inherit the CI commit as a snapshot.
  The corpus keeps the CI commit's own message (the git tools stop at the parent) and truncates it to 1200 characters in 3 episodes
  (att-20, pkg-18, pkg-20); the prompt carries the full text, so nothing extra is exposed.
* Memory is **not** in the index. It is read separately through `read_memory_lines` / `memory_evidence`, only records whose executable probe
  passed become citable `MEMORY_RECORD` evidence, and A2 (no memory) passes `visible_memory=None`. A1 (no intent retrieval) empties the
  corpus and removes `search_repo` and the git tools.

## Evaluation on the 72 held-out episodes (retrieval only, no LLM)
`scripts/heldout/eval_retrieval_heldout.py` (`--estimate`, `--warm`, `--mode bm25|hybrid`, `--compare`) reads gold ids from
`ground_truth`; results are in `results/retrieval_heldout_{bm25,hybrid}.json` (each row records `gold_visibility`) and
`results/retrieval_heldout_comparison.md`. The audit that shaped this section is `docs/retrieval_audit.md`.

**What gold is, and what counts as finding it.** 42 of the 72 episodes have gold evidence (S1 and the UPDATE_TEST tests of S3). In every
one of the 40 episodes whose gold includes a `msg:` item, that item is the **CI commit's own message** (never an earlier commit's), and
the agent receives that message in `ci_event.commit_message`. Finding it is therefore not a retrieval discovery. The gold items are
split by what the prompt already shows:

| Tier | Meaning | Gold items |
|---|---|---|
| `prompt_message` | the CI commit's own message | 40 |
| `prompt_diff` | at least half of the item's lines are lines the CI diff adds (docstring or comment written by the CI commit) | 3 |
| `discoverable` | only retrieval or reading files can surface it | 2 (docstrings of `recipes.running_median`, `recipes.nth_combination`) |

Three protocols follow: `standard` (all gold, as shipped; n = 42 episodes), `msgfree` (CI message item removed from the corpus and the
gold; n = 5 episodes) and `discoverable` (also drops diff-visible gold; n = 2). The small `msgfree`/`discoverable` samples are a property
of the frozen gold (37 gold episodes have the CI message as their only gold item), not an evaluation bug: the filter is checked in
`tests/test_heldout_retrieval.py`, which pins 42 / 5 / 2. The dataset is not changed to enlarge them.

**Metric definitions.** One row is one (episode, protocol, method); the `group` method has one row per oracle failure group with gold.
* Recall@k (k = 1, 3, 5, 10): |gold in top k| / |gold| for the row, averaged over rows, from a depth-50 ranking.
* Evidence coverage = hit@k: 1 if at least one gold item is in the top k, averaged over rows (the proposal defines no such metric).
* precision@5: |gold in top 5| / 5. The denominator is fixed at 5 even if fewer items come back, and with a single gold item the maximum
  is 0.2. It is a proxy for citation precision: no model cites here, so the proposal's "cited item exists and supports the claim" is
  measured only in the final runs (`evaluation.py`, `scripts/citation_support.py`).
* stale@5 (stale-evidence *exposure*): |stale items in top 5| / 5. "Stale" is the heuristic `stale_evidence_ids` label (docstring or doc
  section about a function whose behaviour the CI diff changed on purpose, not updated by the diff). It applies only to gold-UPDATE_TEST
  functions, as in `evaluation._evidence`; a FIX_CODE function's docstring is still true. The proposal's stale-evidence *error rate*
  (stale text causing a wrong hypothesis) needs model runs and is measured in the final experiments.
* Paired CI: percentile bootstrap (2000 resamples, seed 0) of the per-row hybrid minus BM25 difference; uninformative for n < 10.

Methods: `naive` (B1: failing-test text), `intent` (whole-episode intent queries, message-free), `group` (the Intent Analyst's own
`group_queries` over **oracle** failure groups: the function name comes from the gold labels, so this is an upper bound on Triage), and
`intent+msg` (adds the commit subject; degenerate).

| protocol/method | system | n | recall@1 | recall@3 | recall@5 | recall@10 | hit@1 | hit@5 | hit@10 | precision@5 | stale@5 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| standard/naive | bm25 | 42 | 0.10 | 0.29 | 0.36 | 0.51 | 0.12 | 0.38 | 0.52 | 0.08 | 0.06 |
| standard/naive | hybrid | 42 | 0.12 | 0.31 | 0.35 | 0.61 | 0.14 | 0.36 | 0.62 | 0.08 | 0.05 |
| standard/intent | bm25 | 42 | 0.05 | 0.24 | 0.33 | 0.51 | 0.07 | 0.36 | 0.52 | 0.08 | 0.09 |
| standard/intent | hybrid | 42 | 0.12 | 0.24 | 0.42 | 0.68 | 0.14 | 0.43 | 0.69 | 0.10 | 0.06 |
| standard/group | bm25 | 42 | 0.11 | 0.36 | 0.50 | 0.63 | 0.14 | 0.52 | 0.64 | 0.11 | 0.09 |
| standard/group | hybrid | 42 | 0.17 | 0.38 | 0.48 | 0.73 | 0.19 | 0.50 | 0.74 | 0.10 | 0.08 |
| msgfree/naive | bm25 | 5 | 0.40 | 0.60 | 0.60 | 0.80 | 0.40 | 0.60 | 0.80 | 0.12 | 0.08 |
| msgfree/naive | hybrid | 5 | 0.40 | 0.60 | 1.00 | 1.00 | 0.40 | 1.00 | 1.00 | 0.20 | 0.08 |
| msgfree/intent | bm25 | 5 | 0.40 | 0.60 | 0.80 | 1.00 | 0.40 | 0.80 | 1.00 | 0.16 | 0.08 |
| msgfree/intent | hybrid | 5 | 0.40 | 0.60 | 1.00 | 1.00 | 0.40 | 1.00 | 1.00 | 0.20 | 0.04 |
| msgfree/group | bm25 | 5 | 0.60 | 0.60 | 0.80 | 1.00 | 0.60 | 0.80 | 1.00 | 0.16 | 0.08 |
| msgfree/group | hybrid | 5 | 0.40 | 0.60 | 0.80 | 1.00 | 0.40 | 0.80 | 1.00 | 0.16 | 0.04 |
| discoverable/naive | bm25 | 2 | 0.50 | 0.50 | 0.50 | 1.00 | 0.50 | 0.50 | 1.00 | 0.10 | 0.10 |
| discoverable/naive | hybrid | 2 | 0.50 | 0.50 | 1.00 | 1.00 | 0.50 | 1.00 | 1.00 | 0.20 | 0.10 |
| discoverable/intent | bm25 | 2 | 0.50 | 0.50 | 1.00 | 1.00 | 0.50 | 1.00 | 1.00 | 0.20 | 0.00 |
| discoverable/intent | hybrid | 2 | 0.50 | 0.50 | 1.00 | 1.00 | 0.50 | 1.00 | 1.00 | 0.20 | 0.00 |
| discoverable/group | bm25 | 2 | 0.50 | 0.50 | 1.00 | 1.00 | 0.50 | 1.00 | 1.00 | 0.20 | 0.00 |
| discoverable/group | hybrid | 2 | 0.50 | 0.50 | 1.00 | 1.00 | 0.50 | 1.00 | 1.00 | 0.20 | 0.00 |

Paired difference, hybrid minus BM25 (95% bootstrap CI over rows):

* standard/intent recall@5: hybrid - bm25 = +0.083 (95% bootstrap CI +0.012..+0.167, n=42)
* standard/intent recall@10: hybrid - bm25 = +0.167 (95% bootstrap CI +0.048..+0.310, n=42)
* standard/group recall@5: hybrid - bm25 = -0.024 (95% bootstrap CI -0.119..+0.071, n=42)
* standard/group recall@10: hybrid - bm25 = +0.095 (95% bootstrap CI +0.024..+0.190, n=42)
* msgfree/intent recall@5: hybrid - bm25 = +0.200 (95% bootstrap CI +0.000..+0.600, n=5)  [n < 10: interval not informative]
* msgfree/intent recall@10: hybrid - bm25 = +0.000 (95% bootstrap CI +0.000..+0.000, n=5)  [n < 10: interval not informative]
* msgfree/group recall@5: hybrid - bm25 = +0.000 (95% bootstrap CI +0.000..+0.000, n=5)  [n < 10: interval not informative]
* msgfree/group recall@10: hybrid - bm25 = +0.000 (95% bootstrap CI +0.000..+0.000, n=5)  [n < 10: interval not informative]
* discoverable/intent recall@5: hybrid - bm25 = +0.000 (95% bootstrap CI +0.000..+0.000, n=2)  [n < 10: interval not informative]
* discoverable/intent recall@10: hybrid - bm25 = +0.000 (95% bootstrap CI +0.000..+0.000, n=2)  [n < 10: interval not informative]
* discoverable/group recall@5: hybrid - bm25 = +0.000 (95% bootstrap CI +0.000..+0.000, n=2)  [n < 10: interval not informative]
* discoverable/group recall@10: hybrid - bm25 = +0.000 (95% bootstrap CI +0.000..+0.000, n=2)  [n < 10: interval not informative]

Reading:
* On `standard`, hybrid gains at k = 10 for intent queries (+0.167, CI above zero) and a little at k = 5; for grouped queries only k = 10
  improves. These numbers mix visible and discovered evidence, so they say how often the Analyst's search surfaces the gold message
  alongside everything else, not how often it finds something it did not already have.
* `intent+msg` Recall@1 is low (0.12) although the subject line alone ranks the CI message first in 37 of 37 message-only episodes
  (`docs/retrieval_audit.md`): the Analyst fuses three queries with RRF, and the other two queries dilute the exact match. This is a
  property of `search_many`, not an evaluation bug, and the retrieval configuration was not changed to improve it.
* On the evidence the agent could not already see, only 2 episodes (`discoverable`) and 5 (`msgfree`) are available: hybrid and BM25 are
  indistinguishable there, and no conclusion about hybrid on discovered evidence can be drawn from this dataset.

## Pilot dev split (earlier smoke test, BM25 only)
Rank of the best gold item, `python scripts/eval_retrieval.py` (ids as of dataset revision 2; gold types in brackets):

| Episode | gold | naive (B1) | intent | intent, per group (oracle) |
|---|---|---:|---:|---:|
| dev-05 (S1, tbutils) | message | 10 | 2 | 4 |
| dev-06 (S1, cacheutils) | message | 48 | 1 | 10 |
| dev-08 (S3, strutils) | source comment only | not in top 50 | not in top 50 | 33 |
| dev-09 (S1, iterutils) | message + source comment | 14 | 1 | 13 |

Mean Recall@k over the 4 episodes (dev-09 has two gold ids): naive 0.00 at k = 1, 3, 5, 8; intent 0.38 / 0.62 / 0.62 / 0.62.

- The naive result reproduces the proposal's claim: the failing test's text retrieves tests and docstrings, not intent.
- Caveats: only 4 episodes; the query design (drop failure text from intent queries, group function names) was chosen on the first three
  dev episodes of the earlier revision, so treat the intent numbers as optimistic until checked once on the frozen test split (`--final`).
- **Message gold is degenerate:** the gold item is the CI commit's own message, which the agent already receives in
  `ci_event.commit_message`. Querying with that message is rank 1 by construction, so all numbers above are message-free.
- **Source-comment gold (dataset revision 2):** comments are indexed by their text plus `<path>::<qualname>`. A comment added by the CI diff
  is also visible verbatim in the diff, so it is not independent of the input either. dev-08's comment gold ("pick the unit after
  rounding...") shares few words with function-name and diff-identifier queries, so keyword retrieval misses it (best rank 33, not in
  the top 8). This is a consequence of the evidence policy, not a tuning target: report Recall@k per gold type (`msg:` / `comment:`).
  Aligning `source_ref` with the docstring convention (`<path>::<qualname>`) improved that rank from 49 (when it was `<path>` only) to 33.
- Per-group numbers assume Triage groups failures correctly (oracle grouping from gold function names); for dev-06 and dev-09 grouping
  made ranks worse than ungrouped.

## Not done / limitations
* **Reranking is implemented but not evaluated.** Voyage `rerank-2.5` needs one ~30-document call per query, which does not fit the
  10k tokens-per-minute key limit at this scale, and the local cross-encoder is not downloaded here. Earlier notes (`docs/final-code-plan.md`)
  report that local reranking hurt Recall@1 on the pilot.
* Stale-evidence labels are heuristic; docs, changelog and docstring items carry no per-item timestamp.
* Citation *support* (does the cited item support the claim) needs run records and `scripts/citation_support.py`; here only precision@5
  against gold is reported.
* Fusion parameters were not tuned on held-out data (`RRF_K = 60`, depth 50).
* `intent_analyst.group_queries` and `retrieval.intent_queries` remain two builders (the Analyst has a failure group, not a CI event);
  `TOP_K = 8` is fixed in the Analyst.
* Building the corpus from a worktree at run time, LLM query rewriting and freshness labelling remain out of scope.
