# Day 2 — Human 2 (sandbox, memory, integrity): what was built and how to verify it

Branch `feat/sandbox`, on top of `main` at `b0cd20be`. The design follows the submitted proposal and the 5-day plan.

**Verification labels:**
- **UNIT:** pure functions or fakes.
- **SANDBOXED:** real pytest, probes and mutants in the bubblewrap sandbox on fixture repositories.
- **OFFLINE INTEGRATION:** the real orchestrator and agents, with only the model's replies scripted.
- **REAL EPISODE:** boltons pilot code, rebuilt from the pinned clone; no model call.

Nothing below was run against a hosted model.

| Day-2 slot (Human 2) | Status |
|---|---|
| Finish patch restrictions and test runner | **Done**. Unchanged and still passing (194 patch-tool cases, runner, sandbox). |
| Assertion/skip checks | **Done**. `tools/weakening.py`, wired into the Integrity Guard. |
| Memory records and probe execution | **Done**. `memory.JsonlMemoryStore`, `tools/probes.py`, `build_memory_record`, the `--memory` CLI option. |
| Mutation-test runner for selected functions | **Done**. `tools/mutation.py`, the Guard's UPDATE_TEST check, `scripts/build_mutant_cache.py`. |
| Test unsafe patches and memory invalidation | **Done**. 96 new tests; 858 in total. |

## 1. Components

### Weakening analysis (`tools/weakening.py`)

`analyse_test_changes(before, after)` is AST-based and judged **per test function**. It fills every
`TestWeakeningFindings` field:
- **removed assertions:** `assert`, `self.assert*`/`assert_*` helpers, `pytest.raises`/`warns`; a deleted test loses all of its assertions;
- **added skip/xfail:** decorators (including through `from pytest import mark`), `pytest.skip/xfail/importorskip`,
  `unittest.skip*`/`expectedFailure`, `self.skipTest`, `raise SkipTest`, `pytestmark`, `__test__ = False`;
- **loosened comparisons:** an assertion at the same position got weaker — `==` → `!=`/`in`/`<`, a comparison
  → truthiness, a tautology, `assertEqual` → `assertTrue`, exact → `approx`, or a wider `approx` tolerance;
- **swallowed exceptions and broad expectations:** a broad `except` that doesn't re-raise, `contextlib.suppress`,
  `pytest.raises(Exception)`;
- **deleted or disabled tests.**

An expected-value update (`== 'a'` → `== 'A'`) is **not** weakening. `changed_tests` feeds the Guard's "test
functions outside this failure group" rejection. The Guard's former line counting and skip regex were removed.

### Executable probes (`tools/probes.py`) and trusted probes (`tools/verification.py`)

- **Where a probe's code comes from.** A probe is pytest code held by the harness, never read from the worktree.
  `probe_from_tests` builds a self-contained probe from a test module: its imports and helpers, plus the selected tests.
  `support_code` completes a bare dataset probe (dev-07 and test-04 ship a test function without imports) from the
  pristine CI-state module, at seeding time.
- **How it runs.** `run_executable_probe` copies **only the source roots** of the current worktree into a scratch
  directory, adds the probe as the only test module (no repository conftest or config), and runs it with
  `run_pytest` in the sandbox. Bent tests or an added conftest can't make a probe pass (tested).
- **How results map.**
  - `PASSED` / `FAILED` is the probe's verdict on the behaviour.
  - A probe that fails only with `NameError`/`ImportError` is `ERROR`: the probe is broken, so it doesn't count as a
    behaviour failure.
  - Collection errors, all-skipped runs, timeouts and malformed code are also `ERROR`. They are never counted as
    verified.
- **Bug fixed in `verify_behaviour`.** It put the source root itself on `sys.path`. So for `boltons/` the module
  `boltons.cacheutils` was refused, and only a top-level `cacheutils` (which breaks relative imports) was accepted.
  `import_base` now imports a package root by its package name and a plain root (`src/`) as top-level modules.

### Memory (`memory.py`)

`JsonlMemoryStore` implements the existing `MemoryStore` protocol unchanged.
- **Scope.** Records are scoped to `(repo_id, function_qualname)`; behaviour signature and direction travel with them.
- **Reads.** Every read re-runs each record's probe on the current code. **A record is never trusted because it is
  in the file:** a record injected without approval is still only as good as its probe (tested).
- **Blocked reasons.**
  - `superseded`: blocked, not run.
  - `probe_failed`: the probe failed. Two consecutive failures flag the record `flagged_stale`, persistently, and a
    stale record is never re-run or used.
  - `probe_not_run`: execution error; not counted towards staleness.
  - `out_of_scope`: approved at a commit that is not an ancestor of the episode's commit, so no memory from the future.
- **Writes.**
  - Need a `MemoryApproval` with the matching record id and commit, a Guard APPROVE and maintainer approval;
    otherwise `MemoryAuthorizationError` and nothing is written.
  - A record whose probe isn't runnable code is refused (`ToolError`).
  - Writes are atomic (temp file, fsync, rename, under an advisory lock) and idempotent.
  - Only a newer approved **UPDATE_TEST** record (a behaviour change) supersedes the function's older records. A
    FIX_CODE record restores behaviour and supersedes nothing; seeded records never supersede anything.
  - **Isolation contract:** one store per condition, seed and stream. `memory_path(base, condition, seed, stream)`
    gives the file; `namespace=memory_namespace(condition, seed, stream)` is recorded beside it, and opening it under
    another namespace raises `ConfigError`. A replayed stream needs a fresh store; a conflicting record id is refused.
  - Probes run without holding the file lock; only the failure counts are merged back under it.
  - Malformed lines are kept verbatim and never used.
- **Seeding.** `seed(records, worktree_path=...)` is the harness-only path for `visible_memory`; call it before any
  agent acts.

`record_from_repair` is what `integrity_guard.build_memory_record` now delegates to. It derives the probe from the
group's tests after the approved repair: the updated tests for UPDATE_TEST, the untouched verifying tests for FIX_CODE.
The record carries function, signature, direction, evidence ids (analysis plus patch) and the approval commit.

**Decoys (S4).** A passing probe proves the *approved* behaviour still holds, not that a new failure is the same
change. dev-07's decoy record passes on the mutated code by design. Relevance is left to the Analyst, which sees the
behaviour signature.

### Cached mutants (`tools/mutation.py`)

- **Generation.** `generate_mutants` produces deterministic one-line AST mutants of one function: comparison,
  arithmetic and boolean swaps, constants, `not`, `return None`. These are the operator families of
  `scripts/screening/mutation_probe.py`. Mutants are generated from the current source only, never from gold data, and
  memoised per source, function, limit and seed.
- **Execution.** `run_mutants` works in a scratch copy of the worktree; the worktree is never written (tested).
  - It runs a baseline first. If the baseline can't execute, nothing is measured.
  - Outcomes are `killed`, `survived`, `timeout`, `error`, `invalid` and `not_run` (overall deadline). Only a newly
    failing, previously passing test is a kill.
- **The Guard's check.** `mutation_check` runs for both directions, bounded by an overall deadline (300 s).
  - **UPDATE_TEST:** its baseline is **the same test files before this repair**, rebuilt by the Guard; the developer's
    reference repair stays harness-only. Mutants killed before but not after are **measured weakening** (REJECT). No
    executable mutant, unmeasured mutants, or updated tests that kill none mean **strength not demonstrated**
    (ESCALATE).
  - **FIX_CODE (oracle sensitivity):** the untouched group tests must kill at least one executable mutant of the
    repaired function, otherwise ESCALATE. No mutant available means no gap: the untouched tests remain the oracle.
  - The result fills `IntegrityReport.mutation`: `mutants_total` = measured cleanly; `killed_by_repaired_tests` =
    killed by **the failure group's own tests**; `killed_by_reference_tests=None` (reference retention is an evaluation
    metric). Kills by the whole test files are diagnostics in `MutationCheck`.
  - It is logged as one `run_mutants` tool event (operational budget).
- **Cached catalogues.** `load_cached_mutants(path)` reads and validates the mutant specifications of
  `results/mutants/dev_cached_mutants.json`; stored outcomes are never reused. The Guard takes them through the
  optional `cached_mutants={function: [...]}` keyword (e.g. `partial(verify_repair, cached_mutants=...)` in `Agents`),
  and they are executed like generated ones.

### Approval rule (`verification.repair_outcome`)

This implements the proposal-aligned design Human 1 handed over in `docs/day2-human1.readme.md` §7.
`trusted_outcome` is kept unchanged.

| Repair | APPROVE when | Otherwise |
|---|---|---|
| FIX_CODE | No test file touched, the group's tests pass, the whole suite has no new failure, the deterministic checks pass (all already in `verify_repair`), the source patch adds no test-run detection (`patch_content`: `sys.modules` and aliases, pytest-named strings, stack/frame/trace introspection), and the group tests detect mutants of the repaired function; a supplied `TrustedProbe` must also pass. `basis="original_tests"`/`"probe"` | failing run or failed probe: REJECT; insensitive oracle or unverifiable probe: ESCALATE |
| UPDATE_TEST | All deterministic checks including weakening, cited evidence, **and measured mutant-kill strength**. `basis="mutation"` | measured weakening: REJECT; strength not measured or not demonstrated: ESCALATE. **Never on a green run alone** |

The maintainer is still asked only after a Guard APPROVE, and memory is still written only after the maintainer
approves.

## 2. Cross-owner edits (please review)

**Human 1's files:**
- **`agents/integrity_guard.py`:**
  - the weakening analyser replaces the assertion count and skip regex (dead helpers removed);
  - `mutation_check` runs for UPDATE_TEST;
  - `repair_outcome` replaces `trusted_outcome` as the final step;
  - `IntegrityReport.mutation` is filled;
  - `build_memory_record` delegates to `memory.record_from_repair`.

  Signatures are unchanged.
- **`agents/integrity_guard.py` (final round):** `mutation_check` now runs for FIX_CODE too (oracle sensitivity); new
  optional keyword `cached_mutants` (backward compatible).
- **`scripts/run_episode.py`:** `--memory PATH` builds a namespaced `JsonlMemoryStore` with
  `TEST_PYTHON`/`TEST_TIMEOUT_SECONDS`, but only for conditions with memory, and seeds `visible_memory` before the
  run (FULL becomes runnable).
- **Tests that pinned "no probe ⇒ ESCALATE":**
  - In `test_pipeline_integration.py` and `test_guard_sequences.py`, FIX_CODE / measured UPDATE_TEST expectations are
    now APPROVE.
  - The tests whose point is "a green run never approves a test edit" keep their assertions, with an explicitly
    unmeasured mutation check.
  - The fixture harnesses run the mutation check on `trusted-host`, like their `run_tests`.
  - The budget test counts one more operational call (`run_mutants`).
  - `test_agents.py` stubs the mutation check in its Guard unit tests.
- **`docs/interfaces.md`:** module table, Guard description, deferred list.

**Human 3's files:** none. `datasets/` is untouched.

**No schema changes.** `TrustedVerification` (in `tools/verification.py`, Human 2) gained an optional `basis` field.

## 3. Real-episode checks (offline, no model)

- **Setup.** CI states were rebuilt in scratch directories from the pinned clone (`git archive` of the parent plus the
  episode's source diffs) and run with `../boltons-venv` (pytest 8.4.2) in bwrap.
- **dev-04 (S2, FIX_CODE)**, repair = inverse of the injected mutant:
  - Guard **APPROVE** (targeted and regression runs passed, tests untouched, 2.1 s).
  - Record written; its derived probe **PASSED** on the repaired code.
  - With the regression restored, the probe **FAILED** and the record was blocked (`probe_failed`).
- **dev-06 (S1, UPDATE_TEST)**, repair = the developer's setup-keyword hunk (`cached(lru, scoped=False)`):
  - Guard **ESCALATE**: 2 executable mutants of `CachedFunction.__call__`; the updated test itself kills none
    (`killed_by_repaired_tests=0`; another test of the file kills 1), so its strength isn't demonstrated.
  - This is the conservative outcome, not an approval.
- **dev-04 and dev-07 through the real orchestrator (FULL).** A scratch clone of the pinned repo (full history) was set
  to each episode's CI state. Triage, retrieval, memory, probes, the Guard (including the FIX_CODE oracle check) and
  the maintainer were all real; the analyst's decision and the Repairer's edit were scripted.
  - Both completed, approved and wrote memory.
  - dev-07's seeded decoy record was read with its probe **PASSED** (usable, as the dataset designs it).
  - The FIX_CODE record no longer supersedes it.
  - Correction to an earlier note: the existing `../episode-worktrees/dev-07` directory holds a stale state from a
    pre-revision episode id (HEAD `6e0ab7b`), so it is not dev-07's CI state. The clone above is the valid check.
- **`results/mutants/dev_cached_mutants.json`** (`scripts/build_mutant_cache.py`): generated mutants executed against
  each function's test file at the episode's green parent commit.

  | Episode | Function | Killed |
  |---|---|---|
  | dev-02 | `to_text` | 2/6 |
  | dev-03 | `once` | 3/3 |
  | dev-04 | `strip_ansi` | 6/6 |
  | dev-06 | `CachedFunction.__call__` | 0/1 |
  | dev-09 | `default_enter` | 6/6 |

  No timeouts, errors or invalid mutants.

## 4. Verify

```bash
unset PYTHONPATH
myvenv/bin/python -m pytest -q                                       # full suite
myvenv/bin/python -m pytest -q tests/test_weakening.py tests/test_probes.py tests/test_memory_store.py \
    tests/test_mutation.py tests/test_memory_integration.py tests/test_source_gaming.py tests/test_cli_memory.py
                                                                     # the Human 2 tests (bwrap needed)
myvenv/bin/python scripts/build_mutant_cache.py --python ../boltons-venv/bin/python   # needs ../boltons
myvenv/bin/python scripts/run_episode.py --episode dev-04 --condition FULL --memory results/memory/dev.jsonl
                                                                     # live: needs .env API keys
```

## 5. Limitations and open items

- **No live (hosted-model) run** of FULL yet. The FULL path is verified offline only (`test_memory_integration.py`).
- **Sandbox.** Docker is not usable on this machine (permission denied), so everything ran in bwrap. Docker paths are
  unchanged and untested here.
- **Mutation strength is coarse.** Few mutation points in small functions (dev-06: 1–2) can leave legitimate updates
  unverified, which means ESCALATE. A larger limit or more operators are a tuning decision for the dev split.
- **Literal edits to `!=` assertions** are allowed by the tool-level patch rule but rejected by the Guard's weakening
  analysis. A literal setup keyword that routes a test around the logic is caught only by the mutation check (tested
  end to end with cached mutants).
- **Overfitting a FIX_CODE to the tests' exact inputs** (`if s == 'a': return 'a'`) is still approved when the group
  tests also kill some mutant. No syntactic or mutation check can see it. Only an independent behavioural probe (a
  harness `TrustedProbe`, or a usable memory probe of the function) can, and supplying one is a harness decision.
- **The Guard approves strength, not direction:** a test edited to accept a regression with strong assertions passes
  the Guard. Direction rests on the cited evidence; the gold-based simulated maintainer then refuses the memory write.
  Masked regressions are an evaluation metric.

## 6. Final corrections (after the audit)

| Audit finding | Correction | Tests |
|---|---|---|
| High: FIX_CODE patches that detect the test run were approved | `patch_content._source_constructs`: newly added `sys.modules` (also via aliases, `from sys import modules`, `getattr(sys, "modules")`), pytest-named string literals (docstrings excepted), `sys._getframe`/`gettrace`/`settrace`, `inspect.stack`/`currentframe`, `traceback.extract_stack`, frame attributes → rejected; plus the FIX_CODE oracle-sensitivity mutation check | `test_source_gaming.py` (16 attacks rejected, 6 legitimate fixes accepted), `test_memory_integration.py::test_an_unsafe_fix_code_patch_is_never_approved` |
| FIX_CODE record superseded an approved behaviour change | Supersede only by an approved UPDATE_TEST; seeded records never supersede | `test_memory_store.py` (3 tests); real dev-07 run |
| `killed_by_repaired_tests` counted whole files | Group-level kills; file kills as diagnostics | `test_mutation.py::test_group_kills_and_file_kills_are_reported_separately`; real dev-06: 0 |
| No mutation deadline; cache unused | `deadline_seconds` (300 s) with `not_run`; `load_cached_mutants` + Guard `cached_mutants` | `test_mutation.py` (deadline, catalogue validation, stale catalogue measured `invalid`) |
| Analyser missed patterns | Reachable assertions, `except AssertionError`, content-keyed skip/xfail markers, inequality expectation changes | `test_weakening.py` (+18); 0/116 false positives on boltons |
| Record-id collisions between runs | Namespaced stores (`memory_namespace`, `memory_path`), CLI binds `<condition>/seed=<RANDOM_SEED>/<split>` | `test_memory_store.py`, `test_cli_memory.py` |
| Missing integration evidence | Cited memory reuse across episodes, CLI `--memory`, unsafe FIX_CODE, mutation-only weakening | `test_memory_integration.py`, `test_cli_memory.py` |
| Low | Probes run outside the store lock; typed sandbox failure; bounded mutant memo; CLI seeds only for memory conditions | existing and new tests |

**Decisions for Human 1 / Human 3:**
- **Human 1:** the FIX_CODE approval path (oracle-sensitivity escalation) and the optional `cached_mutants`
  keyword are in your Guard; please review.
- **Human 3:** please confirm the store-isolation contract (one store per condition/seed/stream) for the experiment
  runner, the meaning of `killed_by_repaired_tests` (group-level) for mutant-kill metrics, and how "escalated
  although the repair was correct" is scored.
- **Memory-write authority** is a convention checked by the store, not a security boundary (unchanged; see
  `interfaces.md` §5).
- **`git_log/show/blame`** remain stubs: they are not needed for today's components. The dev-01 S5 tripwire test still
  guards them.
- **A3 (Day 3).** `IntegrityReport` stays `guard_ran=False` when the Guard is skipped. `tools.weakening` and
  `tools.mutation` can be run by the evaluation harness on A3 runs to measure weakening without a Guard.
