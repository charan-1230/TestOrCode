# Day 4 final review (submission readiness)

Scope rule followed: no change to the Full/B1 strict patch-content or evidence policies, no new experiments, no API calls, no production
behaviour change. Saved results, datasets and `.env` were not modified (SHA-256 of both `runs.jsonl` files re-checked after the demo).

## Requirement status (authoritative plan, Day 4 table and exit criteria)

| Requirement | Status | Evidence / note |
|---|---|---|
| H1 Fix Full/B1 failures | **Complete (reviewed, no change)** | Failures are the strict patch rule and the evidence policy; left unchanged on instruction (`docs/day4-human1.md` §1) |
| H1 Run Full and B1 on the frozen pilot | **Complete** | `results/day4/pilot-test-5cond` (4 test episodes, 28 runs, 0 execution failures); dev already in `results/day3/pilot-dev-5cond` |
| H1 Test ambiguous and mixed cases | **Covered by existing S3 episodes only** | dev-08 and test-03 (mixed) and the hand review in the report; no new experiment (not required by the plan) |
| H1 Document architecture and decisions | **Complete** | Report §2, §3; `docs/day4-human1.md` |
| H1 Help integrate report and demo | **Complete** | Report; `scripts/demo.sh`, `docs/demo/` |
| H2 Fix sandbox and integrity bugs | **Complete (hygiene); dev-06 strength gap deferred** | `docs/day4-human2.md` §1, §2 |
| H2 Run integrity checks and A3 | **Complete** | A3 and the post-hoc audit on dev and test |
| H2 Test unsafe patches and probe failures | **Complete via Day-3 tests** | `tests/test_real_episode_integrity.py` ran in the 1097-test suite (also in a fresh venv) |
| H2 Document safety design and limitations | **Complete** | Report §2, §3, §7; `docs/day4-human2.md` §3 |
| H2 Validate setup instructions, clean Git state | **Setup validated; Git state = your call** | Fresh Python 3.12 venv install + 1097 tests passed. Your commit `a187c5a` holds the earlier Day-4 work; later edits (this review, demo, report fixes) are uncommitted |
| H3 Validate labels and result logs | **Complete** | Dataset unchanged; counts generated from the dataset in report §4 |
| H3 Run A1 and A2, collect metrics | **Complete** | Both splits, 1 seed |
| H3 Macro-F1, repair rate, error counts | **Complete** | Tables with raw counts; execution failures counted apart |
| H3 Draft results, dataset, baseline sections | **Complete** | Report §4 to §6 |
| H3 Challenges, mitigations, 15-day plan | **Complete** | Report §7, §9 |
| Review report (all, 19:00) | **Automated review done; human read-through pending** | Below |
| Exit: every report result traces to a saved run | **Met** | Tables are generated from `runs.jsonl`; hand-written observations were checked against `review_runs.py` output |
| Exit: dataset table with count and label distribution | **Met** | 9 dev + 4 test; per-test direction and action counts |
| Exit: Full-versus-B1 comparison | **Met (preliminary)** | Both splits |
| Exit: ablations reported only for completed runs | **Met** | A1, A2, A3 shown only where run |
| Exit: challenges and 15-day timeline | **Met** | |

**Optional, not done (by instruction):** second seed, dedicated ambiguous/mixed experiment, new mutant operators.
**Environment-blocked:** compiling `report.tex` (no TeX), Docker (no daemon access), video recording (no asciinema/video tool; a `script` recording was made).
**Not in the plan's Day 4, still open for later:** scenario T (tool faults), `git_log/show/blame` tools.

## 1. Demo
`bash scripts/demo.sh` (about 2 s): rebuilds dev-03's CI state, shows the 2 real failing tests, replays the saved FULL decision/patch/Guard, the
saved B1/A3 runs, a no-model baseline and the rebuilt results table. Offline; nothing written under `results/`. Commands, expected output, an
actual transcript and a terminal recording: `docs/demo/` (README, `demo_transcript.txt`, `demo.typescript`, `demo.timing`).

## 2. LaTeX
`pdflatex`, `xelatex`, `lualatex`, `latexmk`, `latex` and `tectonic` are all absent (checked with `which`). `report.tex` has therefore **not been
compiled**. It was checked only for balanced braces, escaped special characters and ASCII-only text. `report.pdf` is a reportlab rendering of
the same content. No packages were installed and the report was not changed to force a compile. Compile on a machine with TeX:
`cd docs/report && pdflatex report.tex && pdflatex report.tex`.

## 3. Report read-through: corrections made (all demonstrable)
- Correct-repair denominators were stated as "3 to 7"; the saved tables give 1 to 7.
- The test-split summary said the direction ordering was driven by test-01 and test-03. From the logs, A1 is also wrong on test-04; reworded.
- test-04 wording "decoy swayed a hypothesis" was causal; changed to what the log shows (A1's hypothesis cited the decoy and was wrong).
- Section 10 gave `run_episode.py --condition FULL`, which needs `--memory` and a live API call; replaced with the offline demo (live option noted separately).
- Section 3 statuses said "dev" for runs that were also done on test; corrected. "Gold repair turns it green" now says "where a gold repair exists" (S5 has none).
- Dataset label counts were printed as a Python dict; now plain text.

Checked and found consistent with the saved files: every table number (generated from `runs.jsonl`), the 400k-token and 74 s figures (day3-human3 §7:
399,013 tokens, 55 min / 45 runs), the 7 of 19 Day-2 `api_failure` runs (day3-human1 §5), 5 of 13 episodes with gold evidence, A3's weakening rate 0 by
construction, and each per-episode statement in the test-split block against `review_runs.py`.

## 4. Unresolved decisions and missing metrics (not invented)
- **Decisions:** how A1 is scored (16/16 dev and 4/4 test escalations); how "ESCALATE after a correct, probe-less repair" is scored; whether to relax the
  literal-only test-patch rule or the FIX_CODE citation rule (left unchanged); model for the final study (still `gemini-3.5-flash-lite` here; the
  proposal's Qwen3 Coder / Devstral choice is untested).
- **Not computed:** citation support, stale-evidence rate, memory over-generalisation on decoys, tool-call accuracy, confidence intervals, seed
  variance, mutant-kill retention as a headline metric (only pooled raw counts from the audit, undefined for FULL and for all test-split conditions).
- **Rounding note:** the generated table prints 0.562 where `docs/day3-human3.md` writes 0.563 for FULL's final-action accuracy (9/16 = 0.5625;
  Python rounds half to even). Not an error, but the two documents differ in the last digit.
- **Known limitations to disclose:** one seed, one model, one library, 13 episodes; dev-01 provisional; Docker untested; scenario T unwired.

## 5. Changed files in this step (uncommitted)
`scripts/demo.sh` (new), `docs/demo/*` (new), `docs/day4_final_review.md` (new), `docs/report/build_report.py`, `report.tex`, `report.pdf`.
