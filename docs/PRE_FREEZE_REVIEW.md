# Pre-freeze review (steps 1-3 of the final plan)

Written for the decisions before freezing (step 4). Nothing here touched the 72 held-out episodes with an LLM. All numbers are from
files you can open; paths are given. Date of the work: 13 Nov 2026.

## 1. Step 1 - your decisions, as applied
| Decision | Applied how |
|---|---|
| Model | `gemini-3.5-flash-lite` through the 3 keys of `.env` (key pool). `scripts/provider_check.py`: structured mode `json_schema` parsed and correct on all four probes; the endpoint returned the model id `gemini-3.5-flash-lite`. |
| Quota / order | Final order: FULL, B1, A1, A2, A3 with seed 0 (5 x 72 = 360 runs), then FULL and B1 with seed 1 (144 more). Not yet in `docs/final_protocol.md`'s commands, see section 6. |
| Thresholds | The DPI values stay in `configs/final.json` for now; section 5 lists what the dev pass says about them. |
| Optional work | Not done (second model, B0, B2, A4, A5). B2 stays "not evaluated". |

## 2. Step 2 - code and validation gaps (no LLM)
| Item | Result |
|---|---|
| Final report builder | `docs/report/build_final_report.py` now takes `--dataset heldout` (test) and `--dev-dataset pilot` (dev); defaults `results/final/dev` and `results/final/heldout`. It builds on the held-out dry-run records. |
| Post-run scripts on held-out | `make_report.py`, `compare_conditions.py`, `summarize_results.py` and `citation_support.py export` all run with `--dataset heldout --freeze configs/final.json` on the trivial-baseline dry run. |
| Mutant audit on held-out clones | `audit_integrity.py` machinery on the reference repair of 3 episodes (one per library, synthetic accepted record, no LLM): 5-7 s each. mit-01 fully measured (retention 1.0); att-01 partial (4 of 6 mutants end in `error`, so unmeasured); pkg-01 partial (the reference tests kill none of the measured mutants, retention undefined). So the Guard will meet unmeasured/unmeasurable cases on attrs and packaging; by design they escalate (a gap), which lowers FULL's repair rate there. It is a property of the data, not tuned. |
| Hybrid retrieval | **Feasible.** A live Voyage query through the 5-key pool takes about 1.4 s. The 532 documents of the dev corpora were embedded into the cache (`scripts/heldout/warm_dev_embeddings.py`, new). Agent-written queries are embedded live and appended to the cache. No fallback to BM25 was needed. |
| Stray `boltons` string | Harmless (the default of a help text); left. |

## 3. Step 3 - dev pass with the frozen configuration
Command: `run_experiment.py --dataset pilot --split dev --freeze configs/final.json --conditions FULL,B1,A1,A2,A3 --seeds 0 --out results/final/dev`
(9 boltons dev episodes x 5 conditions = 45 runs; `config.json` confirms the frozen settings were applied: hybrid, `expected_value`,
retention 0.90, one revision, fault rate 0.1, model gemini-3.5-flash-lite).

| Check | Result |
|---|---|
| Execution | 45 of 45 records, 0 execution failures, 0 `api_failure`. Gemini returned HTTP 503 several times; the client's retries (up to 6) absorbed them. |
| Cost / time | mean 14 s per run (median 12, max 40), 619 s for all 45; about 14 model calls and 10-12k tokens per run in the retrieving conditions. At that rate 360-504 runs is roughly 1.5-2 h; the held-out suites (attrs, packaging) will be slower. |
| Hybrid retrieval live | every retrieved item in FULL, B1, A2, A3 is `hybrid`; `retrieval_config_version` = `hybrid:bm25+voyage-code-3/rrf60/depth50/rerank30` on all 36 retrieving runs; A1 has none, as it should. |
| Tool faults | 4 injected faults in 45 runs (timeouts on read_file: 3 recovered; one patch-does-not-apply, not recovered). Rate 0.1 on the agents' own calls gives few events at this scale; expect about 10x more on 72 episodes x 7 condition-seeds, still thin per condition. |
| Patch rule | 14 `apply_patch` calls were refused as "not limited to expected values"; the relaxed rule is active (config check). They are the agent trying edits beyond expected values (for example new tests), which the rule is meant to refuse. |
| Memory | 30 memory reads, 5 writes in FULL; A2 reads none. |
| Contamination probe (dev) | pooled added-line recall 0.026, 1 of 9 episodes flagged "memorised" (`results/final/dev/contamination.json`). Run it on the held-out set with `--final` at the start of step 5. |

Dev results, seed 0, 9 episodes each (`results/final/dev/summary.md`, `stats.md`, `integrity_audit.md`):

| Condition | Final-action accuracy | Direction macro-F1 (from hypothesis) | Direction macro-F1 (from action) | Escalated runs | Accepted repairs |
|---|---|---|---|---|---|
| FULL | 0.750 | 1.00 | 0.778 | 5 / 9 | 4 (retention vs reference 0.83) |
| B1 | 0.250 | 1.00 | 0.500 | 5 / 9 | 4 (1 on the wrong side) |
| A1 | 0.125 | 0.81 | 0.000 | 9 / 9 | 0 |
| A2 | 0.438 | 1.00 | 0.564 | 5 / 9 | 4 |
| A3 | 0.625 | 1.00 | 0.708 | 5 / 9 | 4 |

Paired FULL minus B1 on 9 episodes: final-action accuracy +0.500 (95% CI 0.000 to 0.750, McNemar 8/0, p = 0.008); direction macro-F1
0.000 under "hypothesis" and +0.278 (CI 0.000 to 0.500) under "action". With one seed and nine episodes every interval is wide.
Masked regressions: 0 in every condition. Weakened test patches: 0 in every condition.

## 4. Things the dev pass tells you
1. **The direction metric saturates under the current scoring rule.** With `direction_from = "hypothesis"` an escalation still takes the side
   its hypothesis names, and even B1's hypotheses are right on dev (1.00 for FULL, B1, A2, A3). The DPI target "+0.15 direction macro-F1
   over B1" cannot be met, and the pre-registered hypothesis comes out **rejected** (the CI of FULL - B1 is 0 to 0). With
   `direction_from = "action"` (only an applied repair counts; an escalation is undecided) the gap is +0.278 on dev.
2. Final-action accuracy shows the clear separation (0.75 vs 0.25), because B1 escalates or fails more and the Guard gates FULL.
3. Reference-test retention of accepted test patches is 0.83 on dev for every condition (2 test patches each), below the 0.90 evaluation
   threshold. Two patches are too few to say anything; the Guard's own operational threshold (0.90 against the pre-repair tests) fired
   no rejection on dev.
4. The "25% fewer tool calls on covered episodes" and the over-generalisation hypotheses have almost no data on dev (one memory-covered
   pair; FULL used 4 tool calls vs 3 for A2).
5. A1 escalates in 9 of 9 runs, as designed (no retrieval, no commit message).

## 5. Decisions to take before freezing (with a recommendation)
| # | Question | Evidence | Recommendation |
|---|---|---|---|
| D1 | Direction rule: `hypothesis` or `action` | section 4.1 | Use `action` for the headline direction macro-F1 so it measures decisions, keep the hypothesis-based number as a secondary column; decide on dev data only, commit before the held-out run. Alternatively keep `hypothesis` and accept that the F1-gain hypothesis is not testable. Either way say it in the report. |
| D2 | Keep the F1-gain threshold at +0.15 | dev gap under `action` is 0.278 but the CI includes 0 | Keep 0.15; do not tune it from nine episodes. |
| D3 | Masked-regression (<= 5% vs >= 20%), retention (>= 0.90 vs <= 0.70), tool-call saving (>= 25%), over-generalisation (<= 10% vs >= 30%) | dev has no signal (0 masked regressions in B1, retention based on 2 patches, 1 covered pair) | Keep as pre-registered; the held-out result is the test. State plainly that dev could not calibrate them. |
| D4 | Run matrix commands | section 1 | Use the two commands in section 6. |

## 6. Updated final run commands (priority order, seed 0 first)
```
RUN="../myenv/bin/python scripts/run_experiment.py --dataset heldout --split test --final --repos configs/repos.heldout.json --freeze configs/final.json --out results/final/heldout"
$RUN --conditions FULL,B1,A1,A2,A3 --seeds 0     # 360 runs
$RUN --conditions FULL,B1 --seeds 1              # 144 more
../myenv/bin/python scripts/contamination_probe.py --split test --final --dataset heldout --repos configs/repos.heldout.json --freeze configs/final.json --out results/final/heldout/contamination.json
```
The runner resumes (same `--out`), so a stopped command can be repeated. If seed 1 does not finish, report the seed-0 matrix as the main
result and say seed 1 is partial.

## 7. Risks still open
- Gemini free-tier limits: no quota error appeared in 45 runs (the 503s were retried), but 504 runs is about 7k calls; watch for
  daily-limit errors and use `--retry-failed` for `api_failure` runs.
- attrs/packaging mutants that error, and unmeasurable test strength, make the Guard escalate (section 2). That lowers FULL's repair
  rate on those libraries and is reported as is.
- Only 9 dev episodes (the proposal plans about 24) and thresholds that dev could not calibrate: limitations to state.
- The dev results are not comparable with the Day 1-4 pilot runs (new prompts, git tools, hybrid retrieval, fault rate).
- Docker is unreachable; the sandbox is bubblewrap only.
- Files not yet committed: `results/final/dev/` (untracked; whitelist or ignore it), the grown `results/embedding_cache` (dev documents and
  live queries), the new/edited scripts and docs listed below.

## 8. Files changed or added in steps 2-3
`docs/report/build_final_report.py` (dev/test datasets), `scripts/heldout/warm_dev_embeddings.py` (new), `docs/PRE_FREEZE_REVIEW.md` (this file),
`results/final/dev/*` (dev pass outputs: `runs.jsonl`, `config.json`, `summary.md`, `stats.md`, `integrity_audit.*`, `contamination.json`),
`results/embedding_cache/*.jsonl` (grown). No source file under `src/` changed in this round.

## 9. Alternative model test: `gemma-4-31b-it` (14 Nov) - verdict: fall back to `gemini-3.5-flash-lite`
Tried because of its limits (30 rpm, 16k tpm, 14.4k rpd) on the same keys. Evidence (all reproducible from `results/final/dev-gemma/` and the log):
| Check | Result |
|---|---|
| `.env` | `LLM_MODEL=ggemma-4-31b-it` is a typo (double g): every call failed in under a second. The tests below used `--model gemma-4-31b-it`; fix `.env` (or use the override) before any run. |
| Direct call, one-word reply, 4 keys | 2 of 4 keys returned HTTP 500 (`INTERNAL`); the other two answered but took 18-23 s, and the reply starts with a `<thought>` block (reasoning text leaks into the content). `gemini-3.5-flash-lite` answered all four in about 1 s. |
| Dev slice (dev-05, dev-06 x FULL, B1; seed 0; frozen config) | 0 of 4 runs completed: 2 `api_failure` (HTTP 500/503 persisted through the client's 6 retries) and 2 `incomplete` (`reply does not validate as RepairStep/B1Step: Invalid JSON`). 20 server-error retries and 2 HTTP 429 in the log. |
| Speed | 125-335 s per run against 14 s mean for flash-lite (about 20x slower). 504 runs would take days, not hours, even before retries. |
| Quota fit | The 16k tokens per minute limit also bites: one run uses 10-12k tokens. |
Conclusion: not usable for the final runs. Keep `gemini-3.5-flash-lite` (dev pass in section 3: 45 of 45 runs, 0 failures). The model deviation from the proposal's Qwen3-Coder/Devstral
stays as described in section 1. The `results/final/dev-gemma/` files are a failed experiment; delete them or keep them as evidence of the model check (do not mix them into the final tables).

## 10. Decisions taken (14 Nov) and where they are encoded
D1: direction macro-F1 hypothesis-based is PRIMARY, action-based SECONDARY (both always reported; escalation rate and repair coverage added as separate
columns); D2/D3: all pre-registered thresholds, the Guard (retention 0.90, UNMEASURABLE escalates, conservative) and the relaxed scope-restricted patch rule
unchanged; model `gemini-3.5-flash-lite` pinned in `configs/final.json`; hybrid retrieval, warmed cache, no rerank and 10% tool faults unchanged; run order
seed 0 (FULL, B1, A1, A2, A3), then FULL and B1 at seed 1, contamination probe first. See `docs/final_protocol.md`.

