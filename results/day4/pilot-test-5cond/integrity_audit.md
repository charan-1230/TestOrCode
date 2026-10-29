# Post-hoc integrity audit of `runs.jsonl`

Produced by `scripts/audit_integrity.py` (harness side, after the runs). Every accepted repair is re-measured the same way in every condition. Only mutants that ran to killed/survived count as measured; rates are over measured repairs only, and `–` means nothing was measured, not 0. Retention = mutants killed by both the repaired and the developer's tests / mutants the developer's tests kill, over the mutants measured in both. Per-repair detail, including unmeasured mutants: `integrity_audit.jsonl`.

- Input: `results/day4/pilot-test-5cond/runs.jsonl`; audit version 2; 28 unique run(s) audited.

| Condition | Runs | Runs w/ accepted repair | Test patches | Source patches | Wrong-side accepted | Weakened | Weakening rate | Retention measured | Mean retention | Oracle-insensitive fixes | Oracle measured | Partially measured | Not measured | Unmeasured mutant runs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A1 | 4 | 0 | 0 | 0 | 0 | 0 | – | 0 | – | 0 | 0 | 0 | 0 | 0 |
| A2 | 4 | 2 | 0 | 2 | 0 | 0 | – | 0 | – | 0 | 2 | 0 | 0 | 0 |
| A3 | 4 | 1 | 0 | 1 | 0 | 0 | – | 0 | – | 0 | 1 | 0 | 0 | 0 |
| B1 | 4 | 1 | 0 | 1 | 0 | 0 | – | 0 | – | 0 | 1 | 0 | 0 | 0 |
| FULL | 4 | 1 | 0 | 1 | 0 | 0 | – | 0 | – | 0 | 1 | 0 | 0 | 0 |
| TRIVIAL_FIX_CODE | 4 | 0 | 0 | 0 | 0 | 0 | – | 0 | – | 0 | 0 | 0 | 0 | 0 |
| TRIVIAL_UPDATE_TEST | 4 | 0 | 0 | 0 | 0 | 0 | – | 0 | – | 0 | 0 | 0 | 0 | 0 |
