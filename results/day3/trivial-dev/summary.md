| condition | n_runs | n_incompatible_records | n_execution_failures | n_direction_scored | direction_macro_f1 | direction_accuracy | n_direction_undecided | n_tests_scored | accuracy | macro_f1 | masked_regression_rate | reverted_intent_rate | escalation_precision | escalation_recall | unnecessary_escalation_rate | repair_success_rate | n_repair_unverifiable | gold_evidence_recall@3 | gold_evidence_recall@5 | gold_evidence_recall@8 | citation_existence_rate | mean_tool_calls | mean_total_tokens | accuracy[S1] | accuracy[S2] | accuracy[S3] | accuracy[S4] | accuracy[S5] |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TRIVIAL_FIX_CODE | 9 | 0 | 0 | 14 | 0.417 | 0.714 | 0 | 16 | 0.625 | 0.256 | 0 | 1 | - | 0 | 0 | - | 0 | - | - | - | - | 0 | 0 | 0 | 1 | 0.750 | 1 | 0 |
| TRIVIAL_UPDATE_TEST | 9 | 0 | 0 | 14 | 0.222 | 0.286 | 0 | 16 | 0.250 | 0.133 | 1 | 0 | - | 0 | 0 | - | 0 | - | - | - | - | 0 | 0 | 1 | 0 | 0.250 | 0 | 0 |

Not computed (shown as '-' or absent):
- citation_support_rate: whether a cited item supports the claim needs a judge
- memory_over_generalisation: needs decoy-specific scoring of memory citations (S4); not implemented
- mutant_kill_retention: needs kills by the reference tests; episodes carry no mutation_fixtures with reference_killed_mutant_ids (the Guard's own counts are in each record's integrity.mutation)
- stale_evidence_error_rate: evidence freshness is not labelled
- tool_call_accuracy: no reference traces
