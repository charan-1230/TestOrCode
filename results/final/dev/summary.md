| condition | n_runs | n_incompatible_records | n_execution_failures | n_direction_scored | direction_macro_f1 | direction_accuracy | n_direction_undecided | n_tests_scored | accuracy | macro_f1 | masked_regression_rate | reverted_intent_rate | escalation_precision | escalation_recall | unnecessary_escalation_rate | repair_success_rate | n_repair_unverifiable | gold_evidence_recall@3 | gold_evidence_recall@5 | gold_evidence_recall@8 | citation_existence_rate | mean_tool_calls | mean_total_tokens | accuracy[S1] | accuracy[S2] | accuracy[S3] | accuracy[S4] | accuracy[S5] |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A1 | 9 | 0 | 0 | 14 | 0.810 | 0.857 | 1 | 16 | 0.125 | 0.074 | 0 | 0 | 0.125 | 1 | 1 | - | 0 | - | - | - | - | 7.111 | 5466 | 0 | 0 | 0 | 0 | 1 |
| A2 | 9 | 0 | 0 | 14 | 1 | 1 | 0 | 16 | 0.438 | 0.479 | 0 | 0 | 0.182 | 1 | 0.643 | 0.500 | 0 | 0.750 | 0.750 | 0.750 | 1 | 12 | 9511.333 | 0.333 | 0.750 | 0.250 | 0 | 1 |
| A3 | 9 | 0 | 0 | 14 | 1 | 1 | 0 | 16 | 0.625 | 0.606 | 0 | 0 | 0.250 | 1 | 0.429 | 0 | 4 | 0.750 | 0.750 | 0.750 | 1 | 13.111 | 10681.667 | 0.333 | 0.750 | 1 | 0 | 1 |
| B1 | 9 | 0 | 0 | 14 | 1 | 1 | 0 | 16 | 0.250 | 0.317 | 0 | 0 | 0 | 0 | 0.714 | 0 | 2 | 0.250 | 0.375 | 0.375 | 1 | 10.222 | 11757.889 | 0.333 | 0.500 | 0.250 | 0 | 0 |
| FULL | 9 | 0 | 0 | 14 | 1 | 1 | 0 | 16 | 0.750 | 0.685 | 0 | 0 | 0.333 | 1 | 0.286 | 0.667 | 0 | 0.750 | 0.750 | 0.750 | 1 | 14.222 | 10859.667 | 0.333 | 0.500 | 1 | 1 | 1 |

Not computed (shown as '-' or absent):
- citation_support_rate: needs labels for whether a cited item supports the claim; computed by scripts/citation_support.py from human labels (optionally an LLM judge validated against them), not here
- mutant_kill_retention / test_weakening_rate against the developer's reference tests: measured post hoc by scripts/audit_integrity.py (integrity_audit.jsonl), joined into the report by report_tables
- scoring rules: direction from the hypothesis; Guard escalation after a correct, verified repair counts as wrong
- stale_evidence_error_rate uses a heuristic staleness label (a docstring/doc item about the labelled function that the CI diff did not update), not a human label
- tool_call_accuracy against full reference traces: no reference traces exist; patch_scope_accuracy is the reference check the data supports (the scope of each patch against the gold direction of its tests)
