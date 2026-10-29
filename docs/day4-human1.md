# Day 4, Human 1 (agent system): review, failure triage and architecture notes

Local `main`, on top of `12ae1f8`. Feature freeze applies (5-day plan, Day 4): only bug fixes and documentation.
**Verification labels** as in earlier days: UNIT, OFFLINE INTEGRATION, LIVE (a saved real-model run).

| Day-4 slot (Human 1) | Status |
|---|---|
| Fix Full/B1 failures | **Reviewed, no code change needed.** The remaining failures are the strict safety rule and the decision policy working as designed (§1) |
| Run Full and B1 on the frozen pilot | **Done**: `results/day4/pilot-test-5cond` (all 5 conditions, 4 test episodes, 0 failures) |
| Test ambiguous and mixed cases | **From saved logs only** (dev-01 S5, dev-08 S3): no new run yet (§2) |
| Document architecture and decisions | **Done**: report §2 and §3 (`docs/report/`) |
| Help integrate report and demo | **Done for the report**; demo command in report §10 |

## 1. Failures from the saved Day-3 logs (`results/day3/pilot-dev-5cond`)

| Episode, condition | What happened | Verdict |
|---|---|---|
| dev-05, FULL/A2/A3 | Correct hypothesis (outdated test). Both repair attempts changed the dict literals **and** replaced a non-literal expected expression and deleted a comment. `patch_content` refused both (`change is not limited to expected literal values ...`). ESCALATE, gold UPDATE_TEST | **Not a bug.** The rule is a safety boundary (plan: a prompt is not one). Cost of the rule, reported as such |
| dev-04, FULL | ESCALATE for "no cited evidence that the change was intended", gold FIX_CODE. A3 on the same episode accepted the source patch after citing a source comment | Policy working; single run, may be sampling noise |
| dev-06, FULL | Guard ESCALATE: updated test kills 0 of 2 mutants. The developer's own reference test also kills none | Mutant set too small (Human 2 §2) |
| A1, all episodes | Escalated 16/16 | Faithful "no intent evidence" ablation; only memory is citable |

No code was changed for these. Relaxing the patch rule or the citation requirement would be a new feature and would change the
condition semantics after the Day-3 results.

## 2. Decisions the team still has to take (recorded, not made)

- **A1 scoring.** Direction is scored from the hypothesis; final-action accuracy collapses to the escalation rate. The report prints both
  columns side by side (new "Final-action acc." column) and does not rank A1 on either alone.
- **"ESCALATE after a correct, probe-less repair"** is counted in final-action accuracy as wrong and not in correct-repair rate.
- Ambiguous/mixed cases: dev-08 (S3, 4 failing tests) was handled per test by FULL/A2; dev-01 (S5) is provisional because
  `git_log/show/blame` are stubs.

## 3. Architecture summary used in the report
One deterministic orchestrator (`orchestrator.py`), profile-driven conditions (`conditions.py`), one entry point `run_episode`, one
model client, shared result schema. Details: `docs/interfaces.md`, `docs/day2-human1.readme.md`, `docs/day3-human1.md`.
