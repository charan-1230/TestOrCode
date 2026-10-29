# Day 3, Human 3 (dataset, evaluation, experiments): what was done and what the first table shows

Built on the merge of Human 1 and Human 2 (`719cb6b`). Uncommitted.

| Slot | Plan task (Human 3) | Status |
|---|---|---|
| 09:00-11:00 | Finalize pilot labels and gold evidence | **Done as a recorded review**, no dataset content changed (§1) |
| 11:00-13:00 | A2 switch and experiment configuration | **Done.** A2 is the `Settings` default; one `run_experiment.py` call accepts every condition (§2) |
| 13:00-15:00 | Test all five conditions on 2-3 episodes | **Done offline** (fake model, all in the existing runner tests) and **live** (A2 on dev-05 first, then everything) |
| 15:00-18:00 | Run Full, B1, A1, A2, A3 on the pilot | **Done on the dev split**: 9 episodes, seed 0, all five conditions plus the two trivial baselines (§3) |
| 18:00-20:00 | First results table and plots | **Done** (§4): `report.md` and three PNGs |

## 1. Labels and gold evidence

- The pilot has **13 episodes: 9 dev + 4 frozen test**. Hashes match `datasets/pilot_manifest.json`; no dataset file was edited.
- Gold evidence ids exist for **5 of 13** episodes (dev-05, dev-06, dev-08 on its UPDATE_TEST test, dev-09, test-03), all UPDATE_TEST
  labels. The other 8 have none on purpose (nothing states a changed intent). Recall@k is undefined there and is shown as `-`.
- `datasets/README.md` has a new "Review status" section recording this. `tests/test_dataset.py::test_gold_evidence_coverage_is_the_documented_one`
  fails if the data and that statement drift apart.
- Not done: no gold id was invented, and dev-01 (S5) remains provisional. A per-id audit file `docs/dataset_revision_audit.md` is
  referenced but was never committed; the references now say so.
- Stale test names in `tests/test_dataset.py` (the dev-07/dev-08 numbering from before revision 2) were renamed. The tests are SHA-based, so
  behaviour is unchanged.

## 2. Code changes

- `evaluation.py`: a completed run with **no prediction for a test now counts that test as wrong** (before, it was dropped from the
  metrics). A test the harness itself found unstable (`unstable_test_ids`) is still excluded. Two tests in `tests/test_evaluation.py`.
- `src/testorcode/report_tables.py` and `scripts/make_report.py` (new): the plan's table, built only from saved records, with raw counts next to
  each rate. A condition with no scored run prints "not yet evaluated". API/tool failures are counted apart. Plots need
  matplotlib (`pip install -e .[report]`, new optional extra in `pyproject.toml`; installed in `../myenv`). `tests/test_report_tables.py` (6 tests).
- Merge note: the earlier edit to the citation metric (memory reads count as citable) is in `evaluation.py` and is still waiting for your review.

## 3. The live runs

`results/day3/pilot-dev-5cond/` (new directory; no earlier result file was touched). Command per condition:
`scripts/run_experiment.py --split dev --conditions <C> --seeds 0 --out results/day3/pilot-dev-5cond`, then `scripts/audit_integrity.py`,
`scripts/summarize_results.py`, `scripts/make_report.py`.

- Model for every condition: **gemini-3.5-flash-lite** (the `.env` default). The earlier `results/day2` and `results/day3/{a3-pilot,dev07-full-b1}`
  runs used gemini-3.5-flash and must not be compared with these.
- 9 dev episodes x 5 conditions + 2 trivial baselines = 63 runs. **No execution failures** (no `api_failure`; transient 503s were retried inside the client).
- A2 was first run on dev-05 alone as a smoke test; the full A2 run resumed and skipped it (same output directory), so it is one record, not two.
- The frozen test split was not touched.

## 4. First results table (preliminary: 9 dev episodes, one seed, one model)

| Condition | Direction macro-F1 | Direction accuracy | Final-action accuracy | Correct-repair rate | Masked regressions | Tokens (sum) |
|---|---|---|---|---|---|---|
| FULL | 1.000 | 14/14 | 0.563 | 3/6 | 0/10 | 95,782 |
| B1 | 0.944 | 12/14 | 0.250 | 0/4 (+2 unverifiable) | 0/10 | 69,201 |
| A1 | 0.676 | 11/14 | 0.125 | - | 0/10 | 46,662 |
| A2 | 1.000 | 14/14 | 0.688 | 5/7 | 0/10 | 99,700 |
| A3 | 1.000 | 14/14 | 0.438 | 0/3 (+3 unverifiable) | 0/10 | 87,668 |
| always UPDATE_TEST | 0.222 | 4/14 | 0.250 | - | 10/10 | 0 |
| always FIX_CODE | 0.417 | 10/14 | 0.625 | - | 0/10 | 0 |

How to read it, and what not to conclude:

- **Direction is scored from the agent's hypothesis, so it is generous.** FULL escalated on 9 of its 16 test decisions, yet its direction
  score is 14/14, because an escalation counts by the side it blamed. **Final-action accuracy is much lower** (column above, `summary.md`), and only A2 (0.688) beats the always-FIX_CODE baseline (0.625) on it;
  FULL, A3, B1 and A1 fall below it because they escalate a lot. Do not read "1.000" as "FULL fixed everything".
- **Correct-repair rates have small and different denominators** (3-7 repairs per condition). "Unverifiable" means a repair was accepted but
  no trusted probe confirmed it; A3 and B1 have no Guard, so most of their repairs are in that bucket. A3's 0/3 is not "A3 failed".
- **Masked regressions are 0 for every real condition**, because no condition edited a test to hide a regression. The metric is only
  informative against the always-UPDATE_TEST baseline (10/10). With 10 gold FIX_CODE tests this does not show that the Guard is needed.
- **A1 escalated every decision** (16/16), so its repair rate is undefined and its 0.676 comes from hypothesis only. This is the A1
  semantics question Human 1 raised in `day3-human1.md` §2.3, still open.
- Integrity audit (post-hoc, same method for every condition): **0 accepted repairs were weakened and 0 were accepted on the wrong side**
  in any condition (A2 6 accepted repairs, A3 4, B1 4, FULL 4, A1 0). One repair per condition (dev-06) is only partly measured because the
  reference tests kill none of its 2 cached mutants, so retention is undefined; retention was measurable for 1-2 repairs per condition, not enough to compare.
- Not computed: `mutant_kill_retention` as a table metric, citation support, stale-evidence rate, over-generalisation on decoys, confidence intervals,
  seed variance.

## 5. One failure and one success, from the saved logs (`scripts/review_runs.py`)

- **Success, dev-03 (FULL):** the Analyst blamed a regression (`FIX_CODE`, cited `msg:42586df3dd67`, `docstring:funcutils.once`), the Repairer's
  source patch applied, the Guard ran mutants and approved it, the whole suite passed, and the record `dev-03-g1` was written to memory.
  A3 reached the same repair without a Guard; the audit confirms the source patch was oracle-sensitive.
- **Failure, dev-05 (FULL, A2, A3):** the Analyst correctly judged `outdated_test`, but both repair attempts edited
  the test in a way the patch-content check rejected ("change is not limited to expected literal values in existing equality/inequality
  assertions"). Result: ESCALATE, gold UPDATE_TEST, so the final action is wrong even though the direction was right. This is the tool-level
  safety rule working as designed, and it shows a cost of the strict rule, not a Guard decision. B1 failed differently: its test edits applied
  but the target test still failed, so both were reverted and it escalated.
- **Second failure, dev-04 (FULL):** escalated for "no cited evidence that the change was intended" although the gold label is FIX_CODE; A3 on the same
  episode accepted the source patch after citing a source comment. Single runs: this may be sampling noise, not a condition effect.

## 6. Remaining limitations

- One seed, one model, 9 episodes. No statistics. The frozen test split has had only the trivial baselines run on it (earlier), not these conditions.
- Docker is untested (bwrap only). Tool-fault injection (scenario T) is not wired and `git_log`/`git_show`/`git_blame` are still stubs, so dev-01 stays provisional.
- `tests/test_cli_memory.py` still reads the developer's `.env`.
- Retrying a failed run is the open item of §8 (awaiting approval). Day 4 (not started): the full matrix, CIs/McNemar, the report text, and the decision on how to score A1 and "ESCALATE after a correct, probe-less repair".

## 7. Throughput (plan: "benchmark several episodes on Day 3, then measure throughput")

Measured from `provenance.started_at/ended_at` and `usage` in `results/day3/pilot-dev-5cond/runs.jsonl` (no new API calls were made to get these).
Sequential runs, gemini-3.5-flash-lite, one bwrap sandbox, 9 dev episodes per condition:

| Condition | Mean per run | Total (9 runs) | Model calls | Tokens |
|---|---|---|---|---|
| FULL | 106 s | 15.9 min | 30 | 95,782 |
| A2 | 93 s | 13.9 min | 32 | 99,700 |
| A3 | 79 s | 11.9 min | 26 | 87,668 |
| A1 | 48 s | 7.2 min | 18 | 46,662 |
| B1 | 42 s | 6.3 min | 12 | 69,201 |

- The five real conditions took about **55 minutes of run time** for 45 runs (about 74 s per run) and about **400k tokens**. Trivial baselines cost nothing.
- Rough extrapolation, **not a measured plan**: at about 74 s per run, 504 sequential runs would take about 10 hours and about 4.5 M tokens
  (the 504-run matrix of the proposal has a different condition and seed mix, and a larger test set may have different episode sizes).
  Wall-clock was also affected by retried 503s. Whether that schedule is achievable also depends on the provider quota, which was not measured here.

## 8. FULL/A1 recovery after an `api_failure` (investigated; production code unchanged)

Question from `day3-human1.md` §6: can an interrupted or failed FULL/A1 stream be resumed without breaking memory order or duplicating memory writes?
Tested with scripted failures only (`tests/test_experiment_recovery.py`, no network, real runner/orchestrator/Guard/memory store):

| Case | Result |
|---|---|
| An episode ends in `api_failure` | It writes no memory (the Guard writes only after approval); later episodes still run and read nothing from it. **Passes** |
| Resume after that | `run_experiment` treats the failed record as done: nothing is re-run, no model call, no duplicate line, memory unchanged. **Passes** (so a failed run is never retried automatically) |
| Crash after the Guard wrote memory but before the run line was appended, then an identical replay | The episode is re-run once; its memory write is idempotent, so no duplicate record or line. **Passes** |
| Same crash, but the replay decides differently (so its record differs) | The stored record is never overwritten and ids stay unique. **Passes** (the status of that replay run was not characterised) |
| Manual retry (delete the failed first episode's line, resume) | The failed *earlier* episode was re-run after later episodes had written memory, so it **read a record from its own future**. This was the hazard; it is now prevented by `--retry-failed` (below) |

Conclusion: the default behaviour is safe but never retries.

**`--retry-failed` (implemented after approval; `experiment.run_experiment`, `scripts/run_experiment.py`, `ExperimentConfig.retry_failed`).**
- Opt-in; the default is unchanged (a failed run counts as done). It retries only `api_failure` runs.
- A retry replaces the failed line in one atomic rewrite of `runs.jsonl`, so there is never a second line for a run. If the retry cannot start, the failed line stays.
- For conditions with memory it is refused, before anything runs (`ValueError`, CLI exit 2, message says to re-run the stream into a fresh `--out`), when
  a later episode of the same condition and seed already has a record that wrote memory or whose episode declares seeded memory, or when the failed attempt itself already wrote memory.
  So a failed last episode (or a trailing run of failed ones) can be retried in order; an early failure cannot.
- Memory-free conditions (A2, B1) retry freely. A memory record already stored is never overwritten by a different one (unchanged store rule).
- Tests: `tests/test_experiment_recovery.py` (12). Nothing was run live.
