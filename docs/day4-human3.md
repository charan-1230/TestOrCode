# Day 4, Human 3 (evaluation): report, tables and dataset facts

Local `main`, on top of `12ae1f8`.

| Day-4 slot (Human 3) | Status |
|---|---|
| Validate labels and result logs | **Done from the saved files** (dataset hashes unchanged; label counts in report §4) |
| Run A1 and A2, collect metrics | **Dev: Day 3. Frozen test split: done** (`results/day4/pilot-test-5cond`) |
| Compute macro-F1, repair rate, error counts | **Done for dev**; new "Final-action acc." column and pooled mutant-kill retention (audit) in the table |
| Draft results, dataset, baseline sections | **Done**: `docs/report/` |
| Challenges, mitigations, 15-day plan | **Done**: report §7 and §9 |

## Changes
- `report_tables.py`: `final_action_accuracy` column; `integrity_lines` now pools mutant-kill retention as raw counts over test patches
  whose reference kills at least one mutant ("not measurable" otherwise). `scripts/make_report.py` note updated. Test:
  `test_integrity_lines_pool_retention_as_raw_counts_and_skip_undefined`.
- `docs/report/build_report.py` (new): builds `report.tex` and `report.pdf` from one content definition; every result cell is read
  from saved `runs.jsonl` files, so the report cannot drift from the runs.

## Report formats
`docs/report/report.tex` is the LaTeX source. **No TeX distribution was available**, so `report.pdf` is rendered with reportlab from the
same content, not compiled from the `.tex`. Compile the `.tex` with `pdflatex` (twice) on a machine that has TeX; it uses only
article, geometry, graphicx, longtable, array, hyperref and parskip.

## Not computed (still)
Citation support, stale-evidence rate, memory over-generalisation, confidence intervals, seed variance. Reported as such.

## Test-split review (hand review of the saved logs, `results/day4/pilot-test-5cond`)
Method: `scripts/review_runs.py` on `runs.jsonl`. One run per cell; observations only.
- test-01 (S2): FULL, A2 and A3 fixed the code (FULL: Guard approve, memory written). B1 hypothesised an outdated test, and the
  patch-content rule refused both of its test edits. A1 escalated.
- test-02 (S2): all five escalated for "no cited evidence", with the correct (regression) hypothesis.
- test-03 (S3): outdated half escalated everywhere (FULL/A2: literal-only rule refused the test patch; A3/B1: diff did not apply). Regression
  half fixed only by A2 (Guard approve).
- test-04 (S4 decoy): FULL/A3 read the passing decoy record but did not cite it, and escalated; A1 cited it for a wrong hypothesis and escalated;
  B1 fixed the code.
These are the report's §6 "Reading the frozen test-split runs". The direction ordering is driven by test-01 and test-03.
