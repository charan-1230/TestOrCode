# Paired statistics

Source: `results/final/dev/runs.jsonl`; 45 records; bootstrap samples 10000.
scoring rules: direction from the hypothesis; Guard escalation after a correct, verified repair counts as wrong.
Differences are A - B; CI = 95% percentile bootstrap over episodes (seeds of an episode resampled together).

| A | B | Episodes | Direction macro-F1 diff, primary (hypothesis) [95% CI] | Direction macro-F1 diff, secondary (action) [95% CI] | Final-action acc. diff [95% CI] | McNemar direction (A only / B only, p) | McNemar final action (A only / B only, p) |
|---|---|---|---|---|---|---|---|
| FULL | A1 | 9 | 0.190 [0.000, 0.556] | 0.778 [0.300, 0.971] | 0.625 [0.200, 0.889] | 2 / 0, p=0.500 | 10 / 0, p=0.002 |
| FULL | A2 | 9 | 0.000 [0.000, 0.000] | 0.214 [-0.167, 0.500] | 0.312 [-0.182, 0.632] | 0 / 0, p=1.000 | 6 / 1, p=0.125 |
| FULL | A3 | 9 | 0.000 [0.000, 0.000] | 0.069 [-0.194, 0.367] | 0.125 [-0.214, 0.474] | 0 / 0, p=1.000 | 3 / 1, p=0.625 |
| FULL | B1 | 9 | 0.000 [0.000, 0.000] | 0.278 [0.000, 0.500] | 0.500 [0.000, 0.750] | 0 / 0, p=1.000 | 8 / 0, p=0.008 |

## Direction macro-F1 per seed

| Condition | Per seed | Mean | Std | Seeds |
|---|---|---|---|---|
| A1 | 0: 0.810 | 0.810 | - | 1 |
| A2 | 0: 1.000 | 1.000 | - | 1 |
| A3 | 0: 1.000 | 1.000 | - | 1 |
| B1 | 0: 1.000 | 1.000 | - | 1 |
| FULL | 0: 1.000 | 1.000 | - | 1 |

## Memory-covered episodes against A2 (same episode and seed)

| Condition | Covered pairs | Agent tool calls (cond / A2, saving) | Model calls (cond / A2, saving) |
|---|---|---|---|
| A1 | 0 | - / -, - | - / -, - |
| A3 | 0 | - / -, - | - / -, - |
| FULL | 1 | 4.000 / 3.000, -0.333 | 3.000 / 1.000, -2.000 |

## Pre-registered hypothesis (proposal section 7)

**Verdict: rejected** (rejection rule only; thresholds below are reported, not voted).

| Rejection test | Fires | Detail |
|---|---|---|
| bootstrap 95% CI of FULL - B1 direction macro-F1 includes 0 | True | CI (0.0, 0.0), 9 episodes |
| B2 closes the gap: FULL - B2 direction macro-F1 < 0.05 | - | gap - |

| Criterion | Observed | Threshold | n | Met |
|---|---|---|---|---|
| direction macro-F1 gain, FULL - B1 | 0.000 | >= 0.15 | 14.000 | False |
| masked-regression rate, FULL | 0.000 | <= 0.05 | - | True |
| masked-regression rate, B1 (expected high) | 0.000 | >= 0.2 | - | False |
| mutant-kill retention, FULL (audit) | 0.714 | >= 0.9 | 7 | False |
| mutant-kill retention, B1 (audit, expected low) | 0.714 | <= 0.7 | 7 | False |
| agent tool-call saving on covered episodes, FULL vs A2 | -0.333 | >= 0.25 | 1 | False |
| over-generalisation on decoys, FULL (scoped memory) | 0.000 | <= 0.1 | 1.000 | True |
| over-generalisation on decoys, A4 (coarse memory) | - | >= 0.3 | - | - |

Small samples: wide intervals and large p-values are expected; read every number with its n.
