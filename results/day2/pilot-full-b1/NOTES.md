# Day-2 live pilot: FULL vs B1 (dev split)

**Live hosted-model runs**, not scripted.

- **Produced with:** the experiment runner (`experiment.run_experiment` and `default_run_one`), on 2026-10-29.
- **Model:** `gemini-3.5-flash` through Gemini's OpenAI-compatible endpoint, the same model for both conditions.
- **Seed:** 0.
- **Dataset:** `pilot@87d1398471c0`, the current revision.
- **Code:** `d7b361f5` plus the uncommitted Day-2 integration fixes on `feat/day2-integration`, recorded as
  `code_revision ...-dirty`.
- **Budgets:** the episodes' own budgets, identical for both conditions.
- **Repository and sandbox:** boltons at the pinned commit, with `../boltons-venv` (pytest 8.4.2) running in the
  bubblewrap sandbox.

| Episode (scenario) | FULL | B1 |
|---|---|---|
| dev-04 (S2, gold FIX_CODE) | FIX_CODE. Guard APPROVE (tests untouched, suite green, 6/6 mutants killed by the group test). Maintainer approved, memory record `dev-04-g1` written. Final suite passed. 5 LLM calls | FIX_CODE. Patch applied, final suite passed, no Guard. 2 LLM calls |
| dev-06 (S1, gold UPDATE_TEST) | Hypothesis `outdated_test` (the correct side). Guard **ESCALATE**: the updated test kills 0 of 2 executable mutants, so its strength is not demonstrated. Patch reverted, final suite failed. 7 LLM calls | UPDATE_TEST patch applied, final suite passed, no Guard. 5 LLM calls |
| dev-07 (S4 decoy, gold FIX_CODE) | **api_failure** (HTTP 429, free-tier quota exhausted) after the Intent Analyst. The seeded decoy record `mem-dev-s1-cacheutils` was read with its probe PASSED and cited | **api_failure** (HTTP 429) |

`summary.md` and `summary.json` were made with `scripts/summarize_results.py runs.jsonl`.

| Metric | FULL | B1 |
|---|---|---|
| Direction macro-F1 (2 scored tests each) | 1.0 | 1.0 |
| Final-action accuracy | 0.5 (dev-06 escalated) | 1.0 |
| Execution failures | 1 | 1 |

**This is a smoke-level pilot, not a comparison:** 2 scorable tests per condition, one seed. B1's repairs are
unverifiable by design (no Guard), so its `repair_success_rate` is `-`.

FULL's memory store is in `memory/FULL/seed-0/dev.jsonl`, with namespace `FULL/seed=0/dev`. It holds `dev-04-g1`
(approved) and the seeded `mem-dev-s1-cacheutils`.
