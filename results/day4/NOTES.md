# Day-4 live runs

`pilot-test-5cond/`: FULL, B1, A1, A2, A3 and the two trivial baselines on the 4 **frozen test** episodes
(test-01..04), run once with `--final`, seed 0, model `gemini-3.5-flash-lite` (the `.env` default, as for
`results/day3/pilot-dev-5cond`). 28 runs, 0 execution failures, nothing retried. Files: `runs.jsonl`, `config.json`,
`integrity_audit.{md,jsonl}` (audit v2), `summary.{md,json}`, `report.md` and three PNGs.

Smoke-level only: 4 episodes, one seed, one library. No condition is shown to be better than another. Prompts, retrieval and
thresholds were not tuned on this split. Read the table with the denominators in `report.md`.
