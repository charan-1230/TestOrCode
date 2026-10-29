# Day 4, Human 2 (sandbox, memory, integrity): hygiene fixes and limitations

Local `main`, on top of `12ae1f8`. Only bug fixes and documentation (feature freeze).

| Day-4 slot (Human 2) | Status |
|---|---|
| Fix sandbox and integrity bugs | **Done for hygiene** (§1). No new bug found in the sandbox or audit. The dev-06 strength gap is **deferred** (§2) |
| Run integrity checks and A3 | **Done** on the test runs (`results/day4/pilot-test-5cond/integrity_audit.md`): no weakened or wrong-side accepted repair |
| Test unsafe patches and probe failures | **Already covered** on real episode states (`tests/test_real_episode_integrity.py`, Day 3 §2); not repeated |
| Document safety design and limitations | **Done**: §3 and report §2, §3, §7 |
| Validate setup instructions, clean Git state | **Done offline** (§1); a from-scratch venv install was not repeated |

## 1. Fixes
- `tests/test_cli_memory.py` read the developer's `.env` through the CLI (Day-3 limitation). The fixture now points `cli.ROOT` at the
  test's temp directory. Checked by running it with `TEST_PYTHON` set to a nonexistent path in `.env`: the 3 tests still pass.
- `.gitignore`: `results/day4/` is now tracked like `day1` to `day3`. `.env` stays ignored; `.env.example` is the only env file in Git.
  A search of the tree for key-like strings found none.

## 2. Deferred (not a freeze-day bug fix)
- **dev-06 strength gap.** The reference test kills 0 of the 2 cached mutants of `CachedFunction.__call__`, so FULL's Guard escalates
  an otherwise correct UPDATE_TEST. Options (more mutant operators, or a separate "strength unmeasurable" class) are listed for the
  15-day plan. Changing `tools/mutation.py` now would change which mutants exist and invalidate the saved comparison.
- Saved integrity audits from before Day 3's audit v2 were not regenerated (see `results/day3/NOTES.md`); `pilot-dev-5cond` already
  carries `audit_version: 2`.

## 3. Safety design and limitations (for the report)
- Host-side enforcement: allowed paths, scope (source or test), forbidden files (`conftest.py`, config, CI), expected-literal-only
  test patches, no network, bubblewrap sandbox. Probes run on the source roots only.
- Tool-level weakening refusal applies in every condition, so A3's weakening rate is 0 by construction.
- Limits: Docker untested (no daemon access); scenario T (tool faults) not wired; `git_log/show/blame` stubs; coarse mutants for small
  functions; mutant-kill retention is defined only where the reference tests kill something.

## 4. Fresh-environment check (README steps)
In a new Python 3.12.15 venv outside the repository: `pip install --upgrade pip`, `pip install -r requirements-lock.txt`,
`pip install --no-deps -e .`. Results: `pip check` reports no broken requirements; `import pydantic, pandas, testorcode` works;
`ruff 0.17.2`; the full offline suite gives **1097 passed, 0 failed** (396 s). The tests that use the real boltons clone ran because
`../boltons` and `../boltons-venv` exist on this machine, so a machine without them would skip those. Findings:
- `reportlab` (report PDF) is not in `requirements-lock.txt`; it is now in the optional `report` extra in `pyproject.toml` (lock file not
  regenerated). `matplotlib` was already in the lock.
- The README's "Conditions that run" line was stale (omitted A1); fixed, and a results/report section was added.
Not checked: Docker, and an install on a machine without the boltons clone.
