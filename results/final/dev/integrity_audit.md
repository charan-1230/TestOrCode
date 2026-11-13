# Post-hoc integrity audit of `runs.jsonl`

Produced by `scripts/audit_integrity.py` (harness side, after the runs). Every accepted repair is re-measured the same way in every condition. Only mutants that ran to killed/survived count as measured; rates are over measured repairs only, and `–` means nothing was measured, not 0. Retention = mutants killed by both the repaired and the developer's tests / mutants the developer's tests kill, over the mutants measured in both. Per-repair detail, including unmeasured mutants: `integrity_audit.jsonl`.

- Input: `results/final/dev/runs.jsonl`; audit version 2; 45 unique run(s) audited.

| Condition | Runs | Runs w/ accepted repair | Test patches | Source patches | Wrong-side accepted | Weakened | Weakening rate | Retention measured | Mean retention | Oracle-insensitive fixes | Oracle measured | Partially measured | Not measured | Unmeasured mutant runs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A1 | 9 | 0 | 0 | 0 | 0 | 0 | – | 0 | – | 0 | 0 | 0 | 0 | 0 |
| A2 | 9 | 4 | 2 | 2 | 0 | 0 | 0.000 | 2 | 0.833 | 0 | 2 | 0 | 0 | 0 |
| A3 | 9 | 4 | 2 | 3 | 0 | 0 | 0.000 | 2 | 0.833 | 0 | 3 | 0 | 0 | 0 |
| B1 | 9 | 4 | 2 | 2 | 1 | 0 | 0.000 | 2 | 0.833 | 0 | 2 | 0 | 0 | 0 |
| FULL | 9 | 4 | 2 | 3 | 0 | 0 | 0.000 | 2 | 0.833 | 0 | 3 | 0 | 0 | 0 |
