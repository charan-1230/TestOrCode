BM25: bm25-okapi/rrf60
Hybrid: hybrid:bm25+voyage-code-3/rrf60/depth50/rerank30

Metric definitions (denominators):
- unit: one row = one (episode, protocol, method); method 'group' has one row per oracle failure group with gold
- recall@k: |gold in top k| / |gold| for the row, then the mean over rows; k = 1, 3, 5, 10 over a depth-50 ranking
- hit@k (evidence coverage): 1 if at least one gold item is in the top k, else 0; mean over rows (the proposal names no coverage metric)
- precision@5: |gold in top 5| / 5 (fixed denominator 5, even when fewer than 5 items come back); with one gold item the maximum is 0.2
- stale@5: |stale items in top 5| / 5, stale = stale_evidence_ids() heuristic, only for gold-UPDATE_TEST functions (else 0 stale items)
- stale_in_top5: 1 if any stale item is in the top 5
- protocol `standard`: all gold; includes the CI commit's own message, which the prompt already shows (not a discovery)
- protocol `msgfree`: CI message item removed from corpus and gold; episodes with no other gold are left out
- protocol `discoverable`: msgfree, and also gold whose text the CI diff already shows (>= half of its lines are added by the diff) removed
- ci: paired percentile bootstrap (2000 resamples, seed 0) of the per-row hybrid - bm25 difference; uninformative for n < 10

| protocol/method | system | n | recall@1 | recall@3 | recall@5 | recall@10 | hit@1 | hit@5 | hit@10 | precision@5 | stale@5 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| standard/naive | bm25 | 42 | 0.10 | 0.29 | 0.36 | 0.51 | 0.12 | 0.38 | 0.52 | 0.08 | 0.06 |
| standard/naive | hybrid | 42 | 0.12 | 0.31 | 0.35 | 0.61 | 0.14 | 0.36 | 0.62 | 0.08 | 0.05 |
| standard/intent | bm25 | 42 | 0.05 | 0.24 | 0.33 | 0.51 | 0.07 | 0.36 | 0.52 | 0.08 | 0.09 |
| standard/intent | hybrid | 42 | 0.12 | 0.24 | 0.42 | 0.68 | 0.14 | 0.43 | 0.69 | 0.10 | 0.06 |
| standard/group | bm25 | 42 | 0.11 | 0.36 | 0.50 | 0.63 | 0.14 | 0.52 | 0.64 | 0.11 | 0.09 |
| standard/group | hybrid | 42 | 0.17 | 0.38 | 0.48 | 0.73 | 0.19 | 0.50 | 0.74 | 0.10 | 0.08 |
| standard/intent+msg | bm25 | 42 | 0.12 | 0.42 | 0.58 | 0.80 | 0.14 | 0.60 | 0.81 | 0.13 | 0.09 |
| standard/intent+msg | hybrid | 42 | 0.12 | 0.42 | 0.68 | 0.89 | 0.14 | 0.69 | 0.90 | 0.15 | 0.06 |
| msgfree/naive | bm25 | 5 | 0.40 | 0.60 | 0.60 | 0.80 | 0.40 | 0.60 | 0.80 | 0.12 | 0.08 |
| msgfree/naive | hybrid | 5 | 0.40 | 0.60 | 1.00 | 1.00 | 0.40 | 1.00 | 1.00 | 0.20 | 0.08 |
| msgfree/intent | bm25 | 5 | 0.40 | 0.60 | 0.80 | 1.00 | 0.40 | 0.80 | 1.00 | 0.16 | 0.08 |
| msgfree/intent | hybrid | 5 | 0.40 | 0.60 | 1.00 | 1.00 | 0.40 | 1.00 | 1.00 | 0.20 | 0.04 |
| msgfree/group | bm25 | 5 | 0.60 | 0.60 | 0.80 | 1.00 | 0.60 | 0.80 | 1.00 | 0.16 | 0.08 |
| msgfree/group | hybrid | 5 | 0.40 | 0.60 | 0.80 | 1.00 | 0.40 | 0.80 | 1.00 | 0.16 | 0.04 |
| msgfree/intent+msg | bm25 | 5 | 0.60 | 0.60 | 0.60 | 0.80 | 0.60 | 0.60 | 0.80 | 0.12 | 0.04 |
| msgfree/intent+msg | hybrid | 5 | 0.40 | 0.60 | 0.60 | 1.00 | 0.40 | 0.60 | 1.00 | 0.12 | 0.08 |
| discoverable/naive | bm25 | 2 | 0.50 | 0.50 | 0.50 | 1.00 | 0.50 | 0.50 | 1.00 | 0.10 | 0.10 |
| discoverable/naive | hybrid | 2 | 0.50 | 0.50 | 1.00 | 1.00 | 0.50 | 1.00 | 1.00 | 0.20 | 0.10 |
| discoverable/intent | bm25 | 2 | 0.50 | 0.50 | 1.00 | 1.00 | 0.50 | 1.00 | 1.00 | 0.20 | 0.00 |
| discoverable/intent | hybrid | 2 | 0.50 | 0.50 | 1.00 | 1.00 | 0.50 | 1.00 | 1.00 | 0.20 | 0.00 |
| discoverable/group | bm25 | 2 | 0.50 | 0.50 | 1.00 | 1.00 | 0.50 | 1.00 | 1.00 | 0.20 | 0.00 |
| discoverable/group | hybrid | 2 | 0.50 | 0.50 | 1.00 | 1.00 | 0.50 | 1.00 | 1.00 | 0.20 | 0.00 |
| discoverable/intent+msg | bm25 | 2 | 0.50 | 0.50 | 0.50 | 1.00 | 0.50 | 0.50 | 1.00 | 0.10 | 0.00 |
| discoverable/intent+msg | hybrid | 2 | 0.50 | 0.50 | 0.50 | 1.00 | 0.50 | 0.50 | 1.00 | 0.10 | 0.00 |

standard/intent recall@5: hybrid - bm25 = +0.083 (95% bootstrap CI +0.012..+0.167, n=42)
standard/intent recall@10: hybrid - bm25 = +0.167 (95% bootstrap CI +0.048..+0.310, n=42)
standard/group recall@5: hybrid - bm25 = -0.024 (95% bootstrap CI -0.119..+0.071, n=42)
standard/group recall@10: hybrid - bm25 = +0.095 (95% bootstrap CI +0.024..+0.190, n=42)
msgfree/intent recall@5: hybrid - bm25 = +0.200 (95% bootstrap CI +0.000..+0.600, n=5)  [n < 10: interval not informative]
msgfree/intent recall@10: hybrid - bm25 = +0.000 (95% bootstrap CI +0.000..+0.000, n=5)  [n < 10: interval not informative]
msgfree/group recall@5: hybrid - bm25 = +0.000 (95% bootstrap CI +0.000..+0.000, n=5)  [n < 10: interval not informative]
msgfree/group recall@10: hybrid - bm25 = +0.000 (95% bootstrap CI +0.000..+0.000, n=5)  [n < 10: interval not informative]
discoverable/intent recall@5: hybrid - bm25 = +0.000 (95% bootstrap CI +0.000..+0.000, n=2)  [n < 10: interval not informative]
discoverable/intent recall@10: hybrid - bm25 = +0.000 (95% bootstrap CI +0.000..+0.000, n=2)  [n < 10: interval not informative]
discoverable/group recall@5: hybrid - bm25 = +0.000 (95% bootstrap CI +0.000..+0.000, n=2)  [n < 10: interval not informative]
discoverable/group recall@10: hybrid - bm25 = +0.000 (95% bootstrap CI +0.000..+0.000, n=2)  [n < 10: interval not informative]
