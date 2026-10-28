# Retrieval (BM25 first)

Implements the 5-day plan's retrieval work: BM25 over commit messages and documentation (plus docstrings and changelog), evidence
IDs, the A1/A2/B1 switches, per-run evidence logging and the retrieval metrics. Embeddings are deferred behind the same interface.
Code: `src/testorcode/retrieval.py`, `src/testorcode/conditions.py`; check: `scripts/eval_retrieval.py`; tests: `tests/test_retrieval.py`,
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
| `build_retriever(corpus, worktree_path=None) -> Retriever` | BM25Okapi, deterministic (score desc, then `evidence_id`); items sharing no query term are dropped |
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

## Measured on the pilot dev split (smoke test: 4 episodes with gold ids; corpus `pilot-2`)
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

## Not done (hooks only)
Embeddings or Voyage reranking (add as another `Retriever` and fuse with `reciprocal_rank_fusion`; `sentence-transformers` is an optional
extra, Voyage needs `VOYAGE_API_KEY`), building the corpus from a worktree, LLM query rewriting, freshness/stale labelling. The corpus is
bounded to the affected modules plus history, so evidence about other modules cannot be found.
