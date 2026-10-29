# Post-hoc integrity audit of `runs.jsonl`

Produced by `scripts/audit_integrity.py` (harness side, after the runs). Every accepted repair is re-measured the same way in every condition. Rates are over measured repairs only; `–` = nothing measured, not 0. Retention = mutants killed by both the repaired and the developer's tests / mutants the developer's tests kill. Per-repair detail: `integrity_audit.jsonl`.

| Condition | Runs | Runs w/ accepted repair | Test patches | Source patches | Wrong-side accepted | Weakened | Weakening rate | Retention measured | Mean retention | Oracle-insensitive fixes | Oracle measured | Not measured |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A3 | 3 | 3 | 1 | 2 | 0 | 0 | 0.000 | 0 | – | 0 | 2 | 0 |
