# Final experiment protocol (frozen as `configs/final.json`, name `final-v1`)

The decisions below are written into `configs/final.json` and applied with `--freeze configs/final.json`; each run's `config.json`
copies the file, its SHA-256 and the effective non-secret settings. The model (`LLM_PROVIDER`, `LLM_BASE_URL`, `LLM_MODEL`,
`LLM_STRUCTURED_MODE`) is deliberately not in the file: it comes from `.env`, must be one dated model for every condition, and is
recorded per run. Commit `configs/final.json` (with the model fixed) before the first `--final` run.

| Area | Decision | Where it lives |
|---|---|---|
| Retrieval | hybrid BM25 + Voyage `voyage-code-3` (RRF), reranking off, top-k 8, warmed cache `results/embedding_cache`; the same retriever and budget for FULL and B1 | `RETRIEVAL_MODE`, `EMBEDDING_*`, `RERANK_BACKEND` |
| Retrieval comparison | BM25-only results are the separate study `results/retrieval_heldout_*` (retrieval only, no LLM) | `docs/retrieval.md`, `docs/retrieval_audit.md` |
| Ablations | A1 no intent retrieval (corpus and history tools withheld; memory kept), A2 no memory, A3 no Integrity Guard | `conditions.py` (A3 = no Guard is the proposal's numbering; the Detailed Idea's A3 is coarse memory, code name `A4_COARSE_MEMORY`) |
| Guard, test patches | mutant-kill retention >= 0.90 against the mutants the **pre-repair** tests killed, else reject; strength no mutant can measure is UNMEASURABLE and escalates (never a pass, no invented score); structural checks always apply (removed assertions, skip/xfail, loosened comparisons, swallowed exceptions, scope) | `GUARD_MIN_RETENTION=0.9`, `GUARD_UNMEASURABLE_STRENGTH=escalate`, `tools/mutation.py`, `tools/weakening.py` |
| Guard, source patches | no test-file edits; the relevant tests must pass; oracle-sensitivity check | `tools/patch_tools.py`, `tools/mutation.py` |
| Patch policy | UPDATE_TEST edits relevant test files only, FIX_CODE relevant source only, nothing outside src/tests; one revision, then escalate; relaxed `expected_value` test-patch rule (the pilot's literal-only rule is not imposed; weakening stays impossible at that level) | `MAX_REPAIR_REVISIONS=1`, `TEST_PATCH_RULE=expected_value` |
| Tool faults | seeded, about 10% of the agents' tool calls (patch does not apply, invalid test selector, timeout), on top of each episode's own scenario-T plan; kind, tool and recovery recorded; API/rate-limit errors are separate (`api_failure`, never injected) | `experiment.tool_fault_rate=0.1`, `tools/faults.py` |
| Scoring | direction macro-F1 over UPDATE_TEST and FIX_CODE only (an escalation takes the side its hypothesis names); final-action accuracy and macro-F1 over all three actions including ESCALATE; a correct repair needs correct direction, passing tests and preserved integrity, so a Guard escalation after a correct repair is a wrong final action; retention reported only with a nonzero reference denominator, else N/A; incomplete and API-failed runs are set aside, never scored as agent failures | `scoring` in the config, `evaluation.py` |
| Hypotheses | the Detailed Idea's provisional thresholds: F1 gain >= 0.15 (paired-bootstrap CI excluding 0), masked regression <= 5% vs >= 20% for B1, retention >= 0.90 vs <= 0.70 for B1, >= 25% fewer tool calls on covered episodes, over-generalisation <= 10% vs >= 30% for coarse memory | `hypothesis` in the config, `hypothesis.py` |
| B2 | not in the final matrix: the "prompting alone" comparison is reported as **not evaluated** (the code has a B2 profile, which has never been run live) | `hypothesis.py` marks it not decidable |

## Deviations and clarifications

1. **Retention reference.** The decision compares kills with the developer's reference tests. The Guard runs at decision time and must
   never see the reference tests (gold), so the operational 0.90 threshold uses the tests as they were before the repair. The
   comparison against the developer's reference tests is the evaluation-side audit (`scripts/audit_integrity.py`, mutant-kill
   retention), reported with the same 0.90 threshold. Before this change the Guard rejected any lost mutant (retention 1.0).
2. **Relaxed test-patch rule.** The pilot ran with `literal`; `expected_value` has unit tests but no live run yet.
3. **Thresholds are provisional.** They are the Detailed Idea's values. Finalising them "on development data" needs dev runs
   (LLM calls), which have not been made; confirm or change them in a commit before the first `--final` run, never against the 72 episodes.
4. **Hybrid retrieval at run time.** The warmed cache holds the documents and the evaluation's queries. Queries the agents write at run
   time are embedded live through the Voyage key pool (5 keys, 3 requests/min each), so the first runs wait on the rate limit; the
   cache grows as they run.
5. **Tool-fault rate and episode plans.** `tool_fault_rate` is applied to every episode and adds to the 9 episodes' own plans, so those
   episodes see more than 10% faults.
6. **Dev stream.** Only the 9-episode boltons pilot exists (the proposal plans about 24); a limitation to state in the report.

## Commands

Run from the repository root with `../myenv/bin/python`; LLM calls start at step 4.

```
# 1. regenerate / verify the merged held-out files and the environment (no network, no LLM)
../myenv/bin/python scripts/heldout/merge_heldout.py --check
../myenv/bin/python -m pytest -q tests
# 2. (optional) refresh the retrieval study
../myenv/bin/python scripts/heldout/eval_retrieval_heldout.py --mode bm25
../myenv/bin/python scripts/heldout/eval_retrieval_heldout.py --mode hybrid
../myenv/bin/python scripts/heldout/eval_retrieval_heldout.py --compare
# 3. provider check, one dated model (decision still open)
../myenv/bin/python scripts/provider_check.py
# 4. final runs: ONE --out (the report scripts read a single runs.jsonl); resumable, so a stopped command can simply be repeated.
#    Seeds 0 and 1 for FULL and B1, seed 0 for A1-A3. Condition values are FULL, B1, A1 (no intent retrieval), A2 (no memory),
#    A3 (no Integrity Guard).
RUN="../myenv/bin/python scripts/run_experiment.py --dataset heldout --split test --final --repos configs/repos.heldout.json --freeze configs/final.json --out results/final/heldout"
$RUN --conditions FULL,B1 --seeds 0,1
$RUN --conditions A1,A2,A3 --seeds 0
# 5. after the runs
../myenv/bin/python scripts/audit_integrity.py --runs results/final/heldout/runs.jsonl --dataset heldout --repos configs/repos.heldout.json
../myenv/bin/python scripts/make_report.py results/final/heldout/runs.jsonl --dataset heldout --freeze configs/final.json
../myenv/bin/python scripts/compare_conditions.py results/final/heldout/runs.jsonl --dataset heldout --freeze configs/final.json
```
