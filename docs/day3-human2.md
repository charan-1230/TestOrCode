# Day 3 — Human 2 (sandbox, memory, integrity): A3, real-episode checks, integrity audit

Local `main`, on top of `ea8c151`. The design follows the 5-day plan (priority source), then the proposal: **A3 = no
Integrity Guard** (proposal Table 5), not the detailed idea's "coarse memory".

**Verification labels** (as in `day2-human2.md`):
- **UNIT:** pure functions or fakes.
- **OFFLINE INTEGRATION:** the real orchestrator and agents, with only the model scripted.
- **REAL EPISODE:** the pinned boltons clone and pilot venv, bwrap sandbox, no model.
- **LIVE:** hosted model (`gemini-3.5-flash`), saved under `results/day3/`.

| Day-3 slot (Human 2) | Status |
|---|---|
| 09–11 Implement A3 without Integrity Guard | **Done**: unit, offline integration, runner and **live** (3 dev episodes) |
| 11–13 Validate patch scope and memory probes | **Done** on real dev CI states (`tests/test_real_episode_integrity.py`) |
| 13–15 Run integrity tests and selected mutants | **Done**. The mutant catalogue was reproduced on this machine. New post-hoc integrity audit, run on every saved pilot |
| 15–18 Fix sandbox and evaluation integration | **Done**. A3 through the runner and CLI. Two integration bugs fixed (§4) |
| 18–20 Review integrity outcomes | **Done**: §5 |

## 1. A3 (no Integrity Guard)

**What changes (`orchestrator.py`, minimal edit).**
- `A3_NO_INTEGRITY_GUARD` joins `_MULTI_AGENT_CONDITIONS` and `_MEMORY_CONDITIONS`. Like FULL, it needs a `MemoryStore`.
- `EpisodeRun.guard_enabled = profile_for(condition).integrity_guard`. When it is False, the `INTEGRITY_GUARD` stage
  maps to `_verify_without_guard`.
- `_verify_without_guard` is the same test-level verification B1 uses. The shared helper `_verify_tests_only` is now
  also B1's `_verify`, so B1's behaviour is unchanged. The checks are:
  - the group's tests pass;
  - the whole suite has no new failure.
- If those pass, the repair is accepted with `decided_by=intent_analyst`. Otherwise it is reverted, gets one revision,
  then escalates.

**Removed in A3**
- weakening analysis;
- the "test functions outside the group" check;
- the evidence check;
- the semantic model review;
- the mutation check and `repair_outcome`;
- the maintainer and **memory writes**, because only the Guard writes memory (proposal Table 1, §7 step 7).

**Kept in A3**
- intent retrieval;
- **probe-gated memory reads**;
- the decision policy and citation checks;
- **tool-level patch safety** (`patch_tools` + `patch_content`);
- one revision.

**Record shape.**
- `integrity.guard_ran=False` and `guard_decision=not_run`, as for B1.
- Per group: `guard_decision=None`.
- Verification runs are logged with role `orchestrator`, not `integrity_guard`.
- `memory_writes=[]`.

**Consequence (documented, not hidden).**
- Test weakening is refused at tool level in every condition (`day2-human1.readme.md` §7, open team decision), so A3's
  weakening rate is 0 by construction.
- A3 can still differ from FULL through:
  - test edits accepted on a green run alone (no strength or meaning check);
  - masked regressions (a wrong expected literal);
  - mutant-kill retention.

**Tests.**

| File | What it checks | Label |
|---|---|---|
| `test_orchestrator.py` | **+5 tests**; A3 dropped from "not implemented" | UNIT |
| `test_pipeline_integration.py` | **+1**: one input under both conditions | OFFLINE INTEGRATION |
| `test_experiment_full.py` | **+1**: A3 through the runner | OFFLINE INTEGRATION |

The five unit tests check:
- A3 accepts a green patch with an injected Guard that fails if called, and asks no maintainer;
- verification events are not attributed to the Guard;
- a failing verification gets one revision, then escalates;
- tool-level refusal still applies;
- A3 without a store fails at setup.

The pipeline test runs one input under both conditions:
- A2 Guard ESCALATE plus revert;
- A3 accepted, with no mutation check and no model review.

The runner test checks:
- A3 gets its own namespaced store `memory/A3/seed-0/dev.jsonl`, which is never written;
- resume does not re-run finished runs.

Also updated: the stale comment in `config.py`, the `run_episode.py` docstring and `--memory` help, and
`interfaces.md` §8.

## 2. Patch scope and memory probes on real episodes (REAL EPISODE)

`tests/test_real_episode_integrity.py` (5 tests) is skipped unless `../boltons` (at the pin) and `../boltons-venv`
exist. It uses dev only.

**Scope, on dev-04's real CI state.** Each attack is checked to be a real edit (non-empty diff), and each refusal is
checked for its *reason*:

| Attack | Refusal |
|---|---|
| New `tests/conftest.py` | forbidden file |
| Edit `pyproject.toml` | forbidden file |
| Edit `setup.cfg` | forbidden file |
| Edit `tox.ini` | forbidden file |
| Edit `.github/workflows/tests.yaml` | forbidden file |
| Edit `docs/strutils.rst` | outside the approved locations |
| Test file under a source-only scope | wrong scope |
| Source file under a test-only scope | wrong scope |
| Mixed source + test patch | refused |
| Added `@pytest.mark.skip` | not a literal-only change |
| A whole assertion deleted | not a literal-only change |

The worktree is left with only the CI diff.

The inverse of each injected mutant (the true FIX_CODE) is **accepted** on dev-02, dev-03 and dev-04.

**Probes.**

| Episode | Check | Result |
|---|---|---|
| dev-07 | Seeded S4 decoy record (`mem-dev-s1-cacheutils`) | Probe **PASSED** on the mutated code, as the dataset designs it (a passing probe proves the approved behaviour, not that the new failure is that change) |
| dev-04 | Probe derived from the failing test, on the fixed code | **PASSED** |
| dev-04 | Same probe with the regression restored | **FAILED** → `probe_failed`, then `flagged_stale` after two failures, then blocked without running |
| dev-04 | Same probe with the test file rewritten to `pass` and a `conftest.py` that ignores everything | Probe still **FAILED**: it runs on the source roots only |

**Reference repairs vs the agent edit class.**
- The developer test repairs of dev-06, dev-08 and dev-09 are refused by the tool-level test-patch rule, because they
  add tests (asserted for dev-06).
- The *agent* can still reach dev-06's essential change (literal setup keyword `scoped=False`), and did so live in
  FULL, B1 and A3.
- The stored dev-08/dev-09 reference diffs fail a strict `git apply` ("hunk is truncated"), because schema
  whitespace-stripping removed trailing blank context lines. They apply with `--recount`, which the audit uses (checked
  for all 5 dev reference repairs).

## 3. Mutants and the post-hoc integrity audit

**Catalogue reproduced (REAL EPISODE).** `scripts/build_mutant_cache.py` on this machine gives byte-identical entries
to the committed `results/mutants/dev_cached_mutants.json`. The only difference is the recorded interpreter path, so
the committed file is unchanged.

| Episode | Function | Killed |
|---|---|---|
| dev-02 | `to_text` | 2/6 |
| dev-03 | `once` | 3/3 |
| dev-04 | `strip_ansi` | 6/6 |
| dev-06 | `CachedFunction.__call__` | 0/1 |
| dev-09 | `default_enter` | 6/6 |

**Fix:** a relative `--python` (the form the script's own docstring shows) silently produced all-zero counts (nothing
ran in the sandbox). The script now makes the path absolute (`os.path.abspath`, which keeps the venv symlink) and
prints `NOT MEASURED: <status> <reason>` for an unmeasured entry.

**Integrity audit** (`src/testorcode/integrity_audit.py`, `scripts/audit_integrity.py`). It is harness side and reads
ground truth like `evaluation`; no agent imports it.

What it does:
- Re-measures every **accepted** repair of a saved run, the same way in every condition. This is needed because A3/B1
  records must keep `guard_ran=False`, and the `IntegrityReport` validator forbids check results there.
- Rebuilds the CI state from the clone in a scratch worktree, re-applies the accepted patches in order, and never
  touches run worktrees.

Measurements:

| Patch type | Measured |
|---|---|
| Test patch | `analyse_test_changes` plus **mutant-kill retention** = \|mutants killed by the repaired tests ∩ by the developer's tests\| / \|killed by the developer's tests\|, computed only over the mutants measured in **both** copies (`retention_basis`, `retention_denominator`; the Guard's mutants: same seed and limit) |
| Source patch | tests touched, plus **oracle sensitivity**: `True` on any kill, `False` only when every mutant was measured and none was killed, otherwise `None` (undetermined) |

- `wrong_side` flags a test patch for gold FIX_CODE or a source patch for gold UPDATE_TEST.
- **Correction (independent audit, D1).** The first version of the audit (audit version 1) did **not** keep unmeasured
  apart from zero: when every mutant was `invalid`/`timeout`/`not_run`, it reported `oracle_sensitive=False`, and the
  summary counted the repair as an oracle-insensitive fix with "oracle measured = 1" and "not measured = 0"; a
  mutant measured in only one copy also counted as "not retained". Earlier wording here ("unmeasured is never 0") was
  wrong for that version. Audit version 2 (§9) records measured, killed and unmeasured mutant ids per copy, and
  reports such a group as `not_measured`/`partial` with an undetermined (`None`) verdict.
- **The saved reports are from audit version 1.** Re-running version 2 on scratch copies of all three `runs.jsonl`
  showed every saved group fully measured (no unmeasured mutant in any copy) and identical status, weakening,
  retention, oracle and kill sets; only the wording of dev-06's reason differs. The saved files were not regenerated.
- Output: `integrity_audit.jsonl` and `integrity_audit.md` (default next to `runs.jsonl`, or `--out-dir`). The CLI
  refuses (exit 2, nothing written) on malformed or empty input, conflicting records with one `run_id`, unknown
  episodes, or existing reports without `--overwrite` (§9).
- Tests: `tests/test_integrity_audit.py` (12, fixture repo, trusted-host) and `tests/test_audit_integrity_cli.py`
  (17), listed in §9.

## 4. Integration fixes found by the A3 runs

1. **`citation_existence_rate` treated valid memory citations as non-existent** (`evaluation._summarize_condition`).
   - **Cause:** the per-run check counts memory ids as known, but the aggregate used corpus retrieval only.
   - **Effect:** it depressed the rate for the memory conditions only (FULL, A3), which biases FULL/A3 vs A2/B1.
   - **Fix:** citable = retrieved ids plus memory records whose probe passed. That is the orchestrator's own rule.
   - **Test:** `test_a_usable_memory_record_is_a_citable_id_and_a_blocked_one_is_not`.
   - **Effect on the A3 pilot:** 0.400 → 1.0. This is a Human-3 file; please review.
2. The `build_mutant_cache.py` relative-interpreter bug (§3).
3. **Environment (local only):** `.env` had the teammate's `TEST_PYTHON` path, so every sandboxed run failed with
   "python executable not found". `tests/test_cli_memory.py` reads `.env` through the CLI, so it failed too. Set to
   this machine's `../boltons-venv`.

## 5. Integrity outcomes (only saved runs)

**Runs used.** All on `gemini-3.5-flash`, seed 0, dev split, dataset `pilot@87d1398471c0`:
- FULL and B1: `results/day2/pilot-full-b1` (Day 2, committed);
- **A3:** `results/day3/a3-pilot` (new);
- **dev-07 re-run** for FULL and B1: `results/day3/dev07-full-b1` (new; Day 2 had HTTP 429).

Each directory has `runs.jsonl`, `summary.{md,json}` and `integrity_audit.{md,jsonl}`.

| Episode (gold) | FULL | A3 (no Guard) | B1 |
|---|---|---|---|
| dev-04 (S2, FIX_CODE) | FIX_CODE. Guard APPROVE (6/6 mutants killed). Record written. 5 calls | FIX_CODE, the exact inverse of the mutant, accepted on the green rerun. Audit: oracle sensitive 6/6. 2 calls | FIX_CODE accepted. Audit: 6/6. 2 calls |
| dev-06 (S1, UPDATE_TEST) | Correct side, `scoped=False` patch. Guard **ESCALATE**: the updated test kills 0/2 mutants → reverted, final suite failed. 7 calls | **The same `scoped=False` patch, accepted.** Final suite passed. 6 calls | The same patch, accepted. 5 calls |
| dev-07 (S4 decoy, FIX_CODE) | **Re-run:** FIX_CODE. The decoy record read (probe PASSED) and cited. Guard APPROVE (1/2 mutants killed by the group tests). Record `dev-07-g1` written. 4 calls | FIX_CODE. The decoy cited as evidence of the *approved* behaviour. Accepted. No memory write. 2 calls | `api_failure` both times (Day 2: HTTP 429; Day 3: HTTP 503 after 7 attempts). Not scored |

**Per-condition integrity audit (raw counts; rates only over measured repairs):**

| Condition | Accepted test patches | Accepted source patches | Wrong-side accepted | Weakened | Retention measured | Oracle-insensitive fixes |
|---|---|---|---|---|---|---|
| FULL (Day 2 + dev-07 re-run) | 0 | 2 | 0 | 0 | 0 | 0 / 2 |
| A3 | 1 | 2 | 0 | 0 / 1 | 0 | 0 / 2 |
| B1 (Day 2) | 1 | 1 | 0 | 0 / 1 | 0 | 0 / 1 |

How to read this table:
- **"Weakened 0" is not evidence about A3 (or B1).** The tool-level test-patch rule (`patch_content`) refuses
  weakening edits in every condition before any patch is applied, so 0 is a consequence of the design, not a
  measured property of running without a Guard.
- **No mutant-kill retention value has been measured on a real episode.** The only accepted test patch (dev-06, in
  A3 and B1) has a reference test that kills none of the 2 mutants, so retention is undefined there. Retention has
  been computed only in the fixture tests.
- Every mutant in these rows was measured (no unmeasured mutant in any group).

**What the logs show.**
- **Direction:** correct in every scored A3 decision (5/5 tests; `summary.md`). Same as FULL and B1 on these episodes.
- **dev-06 is the only place the Guard changed an outcome, and it was a false escalation.** All three conditions
  produced the identical patch, the developer's own setup change. The audit applied the **developer's reference test**
  to the same CI state: it also kills **0 of the same 2 mutants**.
  - **Observed:** neither the agents' repair nor the developer's own updated test detects either of the 2 generated
    mutants of `CachedFunction.__call__`, so neither would meet the UPDATE_TEST strength rule. Mutant-kill retention
    is undefined there (the reference kills none), not 0.
  - **Not established:** that *no* allowed repair of dev-06 could meet the rule. Other literal or setup-keyword changes
    were not tried, and mutants from other operators might be detectable.
- **Proposal for dev-split tuning (not changed today; feature freeze):** when the reference tests kill none of a
  function's mutants, the strength check is not met even by the developer's repair. Options:
  - more operator families for this kind of function (e.g. keyword-argument default flips, attribute/conditional
    expression swaps; the function's few mutation points give 1–2 mutants);
  - or score "escalated because strength was unmeasurable" separately, as the open Day-1 scoring question asks.
- **dev-07 (S4):** A3 and FULL both read the decoy record. Its probe passed (by design) and both cited it *for*
  FIX_CODE, reading it as the approved behaviour the new failure contradicts, so the decoy did not mislead either
  condition here. Over-generalisation is not scored yet (`NOT_COMPUTED`).
- **The Guard did not prevent any harm in these runs.** Its only intervention was the dev-06 false escalation above;
  on dev-04 and dev-07 it approved the same fixes A3 and B1 accepted.
- **What A3 could show at scale is a hypothesis, not a result:** accepted test edits that the Guard would stop for
  meaning or for lost strength. The pipeline test in §1 shows the mechanism offline with a scripted model only. (A
  separate Day-2 A2 record on flash-lite, `results/day2/flash-lite/dev-05-31797fb62acf` under the old episode id, has
  the Guard review escalating a meaning-changing literal edit; that single record is outside this comparison.)

**This is a smoke-level pilot, not a comparison**: 3 episodes, 1 seed, 1 model, with 2 of the 3 S1/S4 cases on one
function. It does not show that FULL, A3 or B1 is better than another on any metric.

## 6. Cross-owner edits (please review)

- **Human 1:**
  - `orchestrator.py`: the A3 condition sets, `guard_enabled`, `_verify_without_guard`/`_verify_tests_only`, and the
    A3 role in `_enter`. B1's `_verify` now delegates.
  - `test_orchestrator.py`, `test_pipeline_integration.py`.
  - **Merge note for the A1 work:** both of us touch `_MULTI_AGENT_CONDITIONS`/`_MEMORY_CONDITIONS` (orchestrator) and
    the `test_condition_without_a_code_path_fails_explicitly` parametrization. Keep both additions.
- **Human 3:**
  - `evaluation.py` (memory citations, §4) and `test_evaluation.py`;
  - `test_experiment_full.py`;
  - a request: fold `integrity_audit.jsonl` into `summarize`/`ConditionSummary` (currently `mutant_kill_retention` is
    `NOT_COMPUTED`; the audit computes it where it is defined).
- **Shared docs:** `interfaces.md` (module table, boundaries, A3 conditions), `README.md` (one paragraph).
- **No schema changes.**

## 7. Open items and limitations

- **Tool-fault injection** (dev-04 `run_tests` / dev-05 `apply_patch` fault plans) is still not wired, so scenario T
  is untested.
- **`git_log/show/blame`** remain stubs, and the dev-01 tripwire still guards them.
- **Docker** is not usable on this machine (no access to the daemon socket). Everything ran in bwrap; the Docker paths
  are unchanged and untested here.
- **Mutation strength is coarse** for small functions (§5). Retention is defined only where the reference tests kill
  something. On today's runs it never was (the only accepted test patch is dev-06), so retention has not been measured
  on any real episode.
- **The audit uses the gold label's function** (B1 records have no group function). It re-measures the final accepted
  patch per group; earlier, reverted attempts are not audited.

## 8. Verify

```bash
unset PYTHONPATH
../myenv/bin/python -m pytest -q        # 1020 passed, 0 skipped (980 after Day 3 + 23 post-audit D1/D2 tests + 17 D3 tests)
../myenv/bin/python -m pytest -q tests/test_orchestrator.py tests/test_pipeline_integration.py \
    tests/test_experiment_full.py tests/test_real_episode_integrity.py tests/test_integrity_audit.py tests/test_evaluation.py
../myenv/bin/python scripts/audit_integrity.py --runs results/day3/a3-pilot/runs.jsonl --python ../boltons-venv/bin/python
LLM_MODEL=gemini-3.5-flash ../myenv/bin/python scripts/run_experiment.py --conditions A3 \
    --episodes dev-04,dev-06,dev-07 --out results/day3/a3-pilot      # live: .env keys
```

## 9. Post-audit repairs (D1, D2, D5)

An independent verification pass found three issues in the Day-3 work; these are the approved, narrowly scoped
repairs. No saved run artifact was regenerated or edited.

**D1: an unmeasured mutant was read as "not killed"** (`src/testorcode/integrity_audit.py`, audit version 2).

| Case | Version 1 | Version 2 |
|---|---|---|
| Every mutant `invalid`/`timeout`/`not_run`, or no usable baseline | `oracle_sensitive=False`, counted as an insensitive fix and as "oracle measured" | `status=not_measured`, `oracle_sensitive=None`; counted only in "Not measured" and "Unmeasured mutant runs" |
| Some unmeasured, no kill among the measured | `False` | `None` (undetermined), `status=partial` |
| Some unmeasured, at least one kill | `True` | `True` (one kill proves sensitivity), `status=partial` |
| A mutant measured in one copy only | counted as "not retained" (understated retention) | excluded from `retention_basis`; `retention_denominator` = reference kills within the basis; `status=partial` with the excluded ids |
| Reference copy measured nothing | `partial` with an empty kill list | `partial`, retention `None` |
| Fully measured | unchanged | unchanged (same values) |

Each group now records `mutants_generated` plus, per copy, `measured_*`, `killed_*` and `unmeasured_*` (id → outcome).
The per-condition table adds "Partially measured" and "Unmeasured mutant runs".

**D2: audit CLI safety** (`scripts/audit_integrity.py`). Everything is validated before any audit work or write. Any
problem exits 2 with nothing written:
- **Malformed or empty input:** refused, with each bad line number listed.
- **Records:** run identity is `provenance.run_id`.
  - An exact repeated line is reported (stderr and the Markdown header) and audited once.
  - Two *different* records with one `run_id` are refused as a conflict.
  - Distinct runs that share (episode, condition, seed) are all audited and listed as repeated runs.
- **Unknown episode:** refuses the whole audit and names each record.
- **Existing reports:**
  - kept unless `--overwrite` is given;
  - `--out-dir` must be an existing, writable directory.
- **Writing:** both reports are staged as temporary files and then moved into place; if the second move fails, the
  first is restored.

**D5:** §3 and §5 corrected:
- "unmeasured is never 0" was wrong for version 1;
- A3's weakening rate of 0 is by design, not evidence;
- no retention has been measured on a real episode;
- the Guard prevented no harm in these runs;
- no condition is shown to be better than another.

**Regression tests.**

`tests/test_integrity_audit.py` (+6):

| Test | Checks |
|---|---|
| `test_all_mutants_unmeasured_is_not_measured_not_an_insensitive_oracle` | The audit's exact reproduction, through the summary and the Markdown row |
| `test_all_mutants_timing_out_is_not_measured_for_both_scopes` | All mutants timing out, for both scopes |
| `test_partial_measurement_without_a_kill_leaves_the_oracle_undetermined` | Partial measurement with no kill |
| `test_one_kill_proves_sensitivity_even_when_others_are_unmeasured` | One kill with other mutants unmeasured |
| `test_retention_compares_only_mutants_measured_in_both_copies` | Retention over differing mutant subsets |
| `test_unmeasured_reference_means_retention_not_measured` | Unmeasured reference copy |

These inject unmeasured outcomes into real `run_mutants` executions.

`tests/test_audit_integrity_cli.py` (17) covers:
- valid input, with the JSONL and Markdown agreeing;
- empty, missing and malformed input;
- unknown episodes in mixed input;
- exact duplicates, legitimate repeated runs and conflicting run ids;
- existing reports (both, or either one) without `--overwrite`;
- `--overwrite` and `--out-dir`;
- an invalid output directory (missing, or a file);
- rollback when the second write fails.

## 10. D3 repair: relative interpreter paths and unmeasured catalogue entries

**Root cause.**
- `run_pytest` (`tools/pytest_runner.py`) checked a relative `python_executable` against the host's working directory.
- It then ran that path from the worktree, and bind-mounted it into bwrap as a relative path ("bwrap: Can't find
  source path"). `verify_behaviour` (`tools/verification.py`) had the same pattern.
- **Effect:**
  - every pytest run, probe and mutant baseline became an ERROR;
  - in the Guard, that turns into escalations ("strength not measured"); in memory, into `probe_not_run`.
- `TEST_PYTHON` in `.env` is absolute on this machine, so it never triggered in the saved runs.

**Fix.** In both functions (non-Docker modes only), a **relative** path containing a separator is made absolute with
`os.path.abspath`, not `resolve`, so the venv symlink is kept. Absolute paths pass through unchanged, and bare command
names keep their `PATH` lookup.

**Mutant-cache script** (`scripts/build_mutant_cache.py`).
- Every entry has `measurement`:
  - `complete`: every mutant ran to killed/survived;
  - `incomplete`: per-outcome `counts` kept;
  - `not_measured`: no usable baseline, no mutant generated, or the function not found; `counts` is `null`, and the
    entry is recorded rather than dropped.
- The output prints `NOT MEASURED` / `INCOMPLETE`, and the exit status is 1 unless every entry is complete.
- Complete entries are unchanged apart from the new `measurement` field.
- Publishing: `--out` is replaced (temporary file + atomic replace) only when every entry is complete. A failed or
  incomplete build exits 1, leaves `--out` as it was, and writes the rejected build to `<out>.incomplete`.
  The schema is documented in `docs/interfaces.md`.

**Also:**
- The audit reports a disjoint measured basis as "no mutant was measured in both copies (no common basis)".
- `scripts/audit_integrity.py` documents exit status 1: an unexpected error while auditing, nothing written.
- `results/day3/NOTES.md` states that the saved audit reports are version 1 and were not regenerated.
- The dev-06 statement in §5 is split into what was observed and what is not established.

**Tests.**

`tests/test_interpreter_paths.py` (9). Against the pre-fix code, 6 fail with ERROR in place of PASSED:
- a relative interpreter, run from another working directory, works in trusted-host and bwrap for `run_pytest`,
  `verify_behaviour` and `run_mutants`;
- the sandbox wrapper receives the absolute, unresolved path;
- a missing interpreter raises `ToolError` naming the absolute path;
- an executable that is not a working Python is ERROR for pytest, probes and mutant baselines, never PASSED.

`tests/test_build_mutant_cache.py` (6). The pre-repair script, given a forced baseline ERROR, exited 0 with all-zero
counts; the tests cover:
- a complete run;
- a genuine collection-error baseline;
- a forced baseline ERROR;
- an incomplete run;
- a missing function;
- no generated mutant.

Plus `test_disjoint_measured_sets_have_no_common_basis` in `tests/test_integrity_audit.py` and
`test_an_unexpected_error_during_the_audit_writes_nothing` in `tests/test_audit_integrity_cli.py`.
