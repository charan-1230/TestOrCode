# Final experiment protocol (frozen as `configs/final.json`, name `final-v1`)

The decisions below are written into `configs/final.json` and applied with `--freeze configs/final.json`; each run's `config.json`
copies the file, its SHA-256 and the effective non-secret settings. The model is pinned in the file (`LLM_PROVIDER=gemini`,
`LLM_BASE_URL` = the Gemini OpenAI-compatible endpoint, `LLM_MODEL=gemini-3.5-flash-lite`, `LLM_STRUCTURED_MODE=json_schema`): one model
for every condition, and a wrong `.env` value cannot change it (API keys stay in `.env`). Commit `configs/final.json` before the first
`--final` run. `gemma-4-31b-it` was tested and rejected (`docs/PRE_FREEZE_REVIEW.md` section 9).

| Area | Decision | Where it lives |
|---|---|---|
| Model | `gemini-3.5-flash-lite` (Gemini OpenAI-compatible endpoint, structured mode `json_schema`), one model for every condition; pilot and dev pass used the same model | `LLM_*` in `configs/final.json` |
| Retrieval | hybrid BM25 + Voyage `voyage-code-3` (RRF), reranking off, top-k 8, warmed cache `results/embedding_cache`; the same retriever and budget for FULL and B1 | `RETRIEVAL_MODE`, `EMBEDDING_*`, `RERANK_BACKEND` |
| Retrieval comparison | BM25-only results are the separate study `results/retrieval_heldout_*` (retrieval only, no LLM) | `docs/retrieval.md`, `docs/retrieval_audit.md` |
| Ablations | A1 no intent retrieval (corpus and history tools withheld; memory kept), A2 no memory, A3 no Integrity Guard | `conditions.py` (A3 = no Guard is the proposal's numbering; the Detailed Idea's A3 is coarse memory, code name `A4_COARSE_MEMORY`) |
| Guard, test patches | mutant-kill retention >= 0.90 against the mutants the **pre-repair** tests killed, else reject; strength no mutant can measure is UNMEASURABLE and escalates (never a pass, no invented score); structural checks always apply (removed assertions, skip/xfail, loosened comparisons, swallowed exceptions, scope) | `GUARD_MIN_RETENTION=0.9`, `GUARD_UNMEASURABLE_STRENGTH=escalate`, `tools/mutation.py`, `tools/weakening.py` |
| Guard, source patches | no test-file edits; the relevant tests must pass; oracle-sensitivity check | `tools/patch_tools.py`, `tools/mutation.py` |
| Patch policy | UPDATE_TEST edits relevant test files only, FIX_CODE relevant source only, nothing outside src/tests; one revision, then escalate; relaxed `expected_value` test-patch rule (the pilot's literal-only rule is not imposed; weakening stays impossible at that level) | `MAX_REPAIR_REVISIONS=1`, `TEST_PATCH_RULE=expected_value` |
| Tool faults | seeded, about 10% of the agents' tool calls (patch does not apply, invalid test selector, timeout), on top of each episode's own scenario-T plan; kind, tool and recovery recorded; API/rate-limit errors are separate (`api_failure`, never injected) | `experiment.tool_fault_rate=0.1`, `tools/faults.py` |
| Scoring | **Primary direction macro-F1 = hypothesis-based** (`direction_from = "hypothesis"`): per test, the side the agent blamed, UPDATE_TEST vs FIX_CODE only, where an escalation takes the side its hypothesis names (the proposal's per-test direction prediction). **Secondary direction macro-F1 = action-based**: an escalation is undecided. Both are always computed and reported side by side (`direction_macro_f1`, `direction_macro_f1_hypothesis`, `direction_macro_f1_action`); the primary is never switched because the secondary gives a larger gap. Final-action accuracy and macro-F1 (including ESCALATE), **escalation rate** and **repair coverage** (share of scored tests whose group got an accepted repair) are reported separately, each with its denominator (scored tests). A correct repair needs correct direction, passing tests and preserved integrity, so a Guard escalation after a correct repair is a wrong final action. Retention is reported only with a nonzero reference denominator, else N/A. Incomplete and API-failed runs are set aside, never scored as agent failures. | `scoring` in the config, `evaluation.py` (`summarize`), `report_tables.py`, `scripts/compare_conditions.py` (primary and secondary paired CIs) |
| Hypotheses | the Detailed Idea's thresholds, **confirmed unchanged after the dev pass** (nine episodes cannot calibrate them and none is tuned): F1 gain >= 0.15 on the **primary** (hypothesis-based) direction macro-F1 with the paired-bootstrap CI excluding 0, masked regression <= 5% vs >= 20% for B1, retention >= 0.90 vs <= 0.70 for B1, >= 25% fewer tool calls on covered episodes, over-generalisation <= 10% vs >= 30% for coarse memory. The verdict (`hypothesis.py`) uses the primary metric only; if it is not supported on the held-out set it is reported as such. | `hypothesis` in the config, `hypothesis.py` |
| B2 | not in the final matrix: the "prompting alone" comparison is reported as **not evaluated** (the code has a B2 profile, which has never been run live) | `hypothesis.py` marks it not decidable |

## Deviations and clarifications

1. **Retention reference.** The decision compares kills with the developer's reference tests. The Guard runs at decision time and must
   never see the reference tests (gold), so the operational 0.90 threshold uses the tests as they were before the repair. The
   comparison against the developer's reference tests is the evaluation-side audit (`scripts/audit_integrity.py`, mutant-kill
   retention), reported with the same 0.90 threshold. Before this change the Guard rejected any lost mutant (retention 1.0).
2. **Relaxed test-patch rule.** The pilot ran with `literal`; `expected_value` has unit tests but no live run yet.
3. **Thresholds.** They are the Detailed Idea's values and stay unchanged. The dev pass (9 episodes, one seed) could not calibrate them:
   hypothesis-based direction macro-F1 was 1.00 for FULL and B1 there, so the primary F1-gain hypothesis may well come out unsupported on the
   held-out set; the action-based secondary showed a +0.278 gap on dev. Neither is used to change the rule or the thresholds.
4. **Hybrid retrieval at run time.** The warmed cache holds the documents and the evaluation's queries. Queries the agents write at run
   time are embedded live through the Voyage key pool (5 keys, 3 requests/min each), so the first runs wait on the rate limit; the
   cache grows as they run.
5. **Tool-fault rate and episode plans.** `tool_fault_rate` is applied to every episode and adds to the 9 episodes' own plans, so those
   episodes see more than 10% faults.
6. **Dev stream.** Only the 9-episode boltons pilot exists (the proposal plans about 24); a limitation to state in the report.

## Commands

Run from the repository root with `../myenv/bin/python`; LLM calls start at step 3.

```
# 1. verify (no network, no LLM)
../myenv/bin/python scripts/heldout/merge_heldout.py --check
../myenv/bin/python -m pytest -q tests
# 2. (optional) refresh the retrieval study (no LLM)
../myenv/bin/python scripts/heldout/eval_retrieval_heldout.py --mode bm25
../myenv/bin/python scripts/heldout/eval_retrieval_heldout.py --mode hybrid
../myenv/bin/python scripts/heldout/eval_retrieval_heldout.py --compare
# 3. held-out stage: contamination probe FIRST, then the runs. One --out for the runs (the report scripts read a single runs.jsonl);
#    the runner resumes, so a stopped command can simply be repeated. Condition values: FULL, B1, A1 (no intent retrieval),
#    A2 (no memory), A3 (no Integrity Guard).
../myenv/bin/python scripts/contamination_probe.py --split test --final --dataset heldout --repos configs/repos.heldout.json --freeze configs/final.json --out results/final/heldout/contamination.json
RUN="../myenv/bin/python scripts/run_experiment.py --dataset heldout --split test --final --repos configs/repos.heldout.json --freeze configs/final.json --out results/final/heldout"
$RUN --conditions FULL,B1,A1,A2,A3 --seeds 0     # 360 runs
$RUN --conditions FULL,B1 --seeds 1              # 144 more
# 4. after the runs
../myenv/bin/python scripts/audit_integrity.py --runs results/final/heldout/runs.jsonl --dataset heldout --repos configs/repos.heldout.json
../myenv/bin/python scripts/make_report.py results/final/heldout/runs.jsonl --dataset heldout --freeze configs/final.json
../myenv/bin/python scripts/compare_conditions.py results/final/heldout/runs.jsonl --dataset heldout --freeze configs/final.json
../myenv/bin/python scripts/summarize_results.py results/final/heldout/runs.jsonl --dataset heldout --freeze configs/final.json
```
If seed 1 does not finish, report the seed-0 matrix as the main result and say seed 1 is partial.
