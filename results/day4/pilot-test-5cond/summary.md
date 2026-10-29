| condition | n_runs | n_incompatible_records | n_execution_failures | n_direction_scored | direction_macro_f1 | direction_accuracy | n_direction_undecided | n_tests_scored | accuracy | macro_f1 | masked_regression_rate | reverted_intent_rate | escalation_precision | escalation_recall | unnecessary_escalation_rate | repair_success_rate | n_repair_unverifiable | gold_evidence_recall@3 | gold_evidence_recall@5 | gold_evidence_recall@8 | citation_existence_rate | mean_tool_calls | mean_total_tokens | accuracy[S2] | accuracy[S3] | accuracy[S4] |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A1 | 4 | 0 | 0 | 6 | 0.333 | 0.500 | 0 | 6 | 0 | 0 | 0 | 0 | 0 | - | 1 | - | 0 | - | - | - | 1 | 7.750 | 5248.250 | 0 | 0 | 0 |
| A2 | 4 | 0 | 0 | 6 | 1 | 1 | 0 | 6 | 0.333 | 0.222 | 0 | 0 | 0 | - | 0.667 | 0.500 | 0 | 1 | 1 | 1 | 1 | 12.750 | 9102.250 | 0.500 | 0.333 | 0 |
| A3 | 4 | 0 | 0 | 6 | 1 | 1 | 0 | 6 | 0.167 | 0.133 | 0 | 0 | 0 | - | 0.833 | 0 | 1 | 1 | 1 | 1 | 1 | 12.500 | 7721.750 | 0.500 | 0 | 0 |
| B1 | 4 | 0 | 0 | 6 | 0.429 | 0.500 | 2 | 6 | 0.167 | 0.133 | 0 | 0 | 0 | - | 0.833 | 0 | 1 | 0 | 0 | 0 | 1 | 9.250 | 7914.750 | 0 | 0 | 1 |
| FULL | 4 | 0 | 0 | 6 | 1 | 1 | 0 | 6 | 0.167 | 0.133 | 0 | 0 | 0 | - | 0.833 | 0.500 | 0 | 1 | 1 | 1 | 1 | 13 | 8006 | 0.500 | 0 | 0 |
| TRIVIAL_FIX_CODE | 4 | 0 | 0 | 6 | 0.400 | 0.667 | 0 | 6 | 0.667 | 0.400 | 0 | 1 | - | - | 0 | - | 0 | - | - | - | - | 0 | 0 | 1 | 0.333 | 1 |
| TRIVIAL_UPDATE_TEST | 4 | 0 | 0 | 6 | 0.250 | 0.333 | 0 | 6 | 0.333 | 0.250 | 1 | 0 | - | - | 0 | - | 0 | - | - | - | - | 0 | 0 | 0 | 0.667 | 0 |

Not computed (shown as '-' or absent):
- citation_support_rate: whether a cited item supports the claim needs a judge
- memory_over_generalisation: needs decoy-specific scoring of memory citations (S4); not implemented
- mutant_kill_retention: needs kills by the reference tests; episodes carry no mutation_fixtures with reference_killed_mutant_ids (the Guard's own counts are in each record's integrity.mutation)
- stale_evidence_error_rate: evidence freshness is not labelled
- tool_call_accuracy: no reference traces
