# Repository selection: toolz vs boltons

Date: 2026-10-27 (pilot configuration superseded on 2026-10-28, see the notice below). Status: **screening complete, decision conditional** (see section 1).
Raw numbers: `docs/screening_evidence/screening_summary.json`. Prototype episodes: `docs/screening_evidence/proto_episodes_boltons.json`.
Scripts (re-runnable, one-off tooling): `scripts/screening/`. Candidate clones were kept outside this repo (`../../screening/`).

> **Supersession notice (2026-10-28).** The pilot size and episode plan written in this record (**"6-12 episodes"** in sections 1 and 10 and the
> manifest plan in section 10, item 1) are **superseded by the approved pilot: 13 episodes, 9 dev + 4 frozen test**
> (dev: S1 x3, S2 x3, S3, S4, S5; test: S2 x2, S3, S4). 13 exceeds the 5-day plan's 6-12 target; the project team accepted that deviation.
> The authoritative description is `datasets/README.md`, with the revision-2 changes summarised there (the per-field audit file is not in the repository). Of the 7 verified S1
> bases below, 6 are used (`353f6bef` as the S5, `7b577b0f`, `8408ab57`, `6270b5e3`, `f387b64c` as the S3, and `1c2db466` as the test S3);
> `289edfcd` (partial) is unused. **What still stands:** the screening evidence, the scoring, and the decision to use boltons.
> Nothing else in this record has been re-assessed.

## 1. Decision

**`BOLTONS PRIMARY / TOOLZ BACKUP`, with a strong qualification: toolz is not an approved backup.**

- **boltons is approved for the pilot** (6-12 episodes *in this record; superseded by the 13-episode pilot, see the notice above*) and for S1-S5 construction. It is **not yet approved for the full 24-episodes-per-library scale**: it has 7 verified S1 bases against about 17 needed.
- **toolz fails an essential requirement.** It has **0 verified S1 bases** (S1 is the foundation of S1, S3, S4 and S5, about 20 of every 24 episodes per library). Its aggregate score (59%) hides that blocker, so it is listed as backup only because the task format needs a second name. It can serve S2 alone.
- **The proposal needs 3 evaluation libraries plus 1 dev library**, so no single repository is enough. The cheapest next step is to run the same harness (`scripts/screening/`) on 1-2 further candidates (the proposal names `tabulate`, `humanize`) to find a real backup and the other libraries.

Decisive evidence (all measured, Python 3.12.15):

| Evidence | toolz | boltons |
|---|---|---|
| Verified S1 bases (old tests fail under source-only change, pass at parent and at commit, developer test diff exists, stable over 5 repeats) | **0** | **7** (6 valid, 1 partial) |
| History runnable on Python 3.12 | about 5 recent + 9 trivial 2013 commits | 2023-2026 plus part of 2015 (pytest 8) |
| Schema validation of real S1-S5 episodes | not attempted (no S1) | **all 5 validate** (section 4) |

## 2. Source documents and conflicts

Read: `docs/ProjectIdeaProposal.pdf` (authoritative), `DetailedProjectIdea.pdf`, `TestOrCode_5-day_emergency_execution_plan.pdf`, `LMA_Group_Project_v1.pdf`, `src/testorcode/schemas/{episode,result}_schemas.py` and their tests, README, `docs/interfaces.md`. **No `response1`/`response2` files exist in the project**, so no earlier recommendation was used.

Requirements taken from the proposal: pure-Python, permissive licence, fast pytest suite, no compiled deps; 3 held-out libraries x 24 episodes (S1 30, S2 12, S3 12, S4 9, S5 9 in total, so about 10/4/4/3/3 per library) plus a dev stream; 5-run stability filter; Docker, no network, fixed Python and dependencies; real history is the base and only the mutants are synthetic. The plan simplifies to **one repository and 6-12 pilot episodes first**.

Inconsistencies found (not resolved here):

1. The proposal says "3 evaluation libraries"; the prompt screens 2 candidates. Neither alone covers the proposal.
2. The detailed document says to **prefer lesser-known libraries** (contamination). Both toolz and boltons are popular; boltons' 2026 commits are recent, which helps.
3. The proposal's S1 relies on "a real commit that changed source **and** tests", and S4 on a function "whose change was approved earlier". Both depend on the S1 supply, which is the scarce resource.

## 3. Reproducibility record

| Item | toolz | boltons |
|---|---|---|
| URL | https://github.com/pytoolz/toolz | https://github.com/mahmoud/boltons |
| Branch, pinned commit | `master`, `451af60dec590a6010e2babdbf391ea8f815122f` (2026-09-18) | `master`, `d31d821d56c8f7756af9c30a864077f2e5855321` (2026-10-25) |
| Commits / first commit / latest tag | 1,236 / 2013-09-10 / 1.2.0 | 1,716 / 2013-02-19 / 26.2.0 |
| Licence | BSD-3-Clause (`LICENSE.txt`) | BSD-3-Clause (`LICENSE`) |
| Python / deps | `>=3.9`; **no runtime deps**; test deps pytest (+coverage, pycodestyle) | `>=3.7`; **no runtime deps**; dev dep pytest>=8 |
| Environment | `python3.12 -m venv`; `pip install -e . pytest coverage pycodestyle` -> exit 0, `pip check` clean | `pip install -e . "pytest>=8"` -> exit 0, `pip check` clean |
| Test command | `python -m pytest` (config `testpaths=["toolz"]`) | `python -m pytest --doctest-modules boltons tests` (tox command); unit tests only: `python -m pytest tests` |
| Result, 5 repeats | 192 passed, 1 skipped, exit 0, every run; 0.2-0.6s | 775 passed + 22 subtests, exit 0, every run; 2.2-2.8s (621 tests without doctests) |
| Network-isolated run | `unshare -rn`: 192 passed | `unshare -rn` with loopback up: 621 passed (hangs with loopback down: socket tests) |
| pytest versions | 9.1.1 (primary), 8.4.2 (sensitivity) | 9.1.1 (primary), 8.4.2 (sensitivity) |

Limits: **Docker was not tested** (daemon permission denied in this session). Python 3.9 could not be downloaded; Python 3.10 was used as the only other interpreter, for toolz. Test-order randomisation and parallel runs were not tested. "Stable" means 5 identical runs of HEAD, nothing broader.

## 4. Scoring (suggested weights unchanged)

| Criterion | Weight | toolz | boltons | Rationale (confidence) |
|---|---:|:-:|:-:|---|
| Credible yield across S1-S5 | 25% | 1 | 3 | toolz: 0 S1, so S1/S3/S4/S5 blocked; boltons: 7 S1 bases, S2 plentiful, S3/S4 prototyped, but short of 17 (high) |
| Git history and intent evidence | 20% | 2 | 4 | toolz in maintenance mode (README "Project Status"), about 10 code-changing commits since 2022 (mostly Python-version upkeep), old history unrunnable; boltons dense, issue-linked messages, per-module docs, CHANGELOG (medium) |
| Baseline reliability and runtime | 15% | 5 | 4 | both 5/5 identical; boltons needs loopback for socket tests (high) |
| Integrity and mutation feasibility | 15% | 4 | 3 | kill rates in section 6; toolz tests stronger (medium; small samples) |
| Python 3.12 + schema compatibility | 10% | 2 | 4 | most history of both fails on 3.12; boltons S1-S5 validate against the real schema (high) |
| Install simplicity | 5% | 5 | 5 | zero runtime deps each (high) |
| Size and repairability | 5% | 5 | 3 | 3.9k vs 17.7k production lines (high) |
| Licence and execution suitability | 5% | 5 | 4 | both BSD-3-Clause; boltons tests use sockets, files, sleeps (medium) |
| **Weighted total** | | **2.95/5 (59%)** | **3.60/5 (72%)** | |

toolz's non-trivial total does not rescue it: **S1 = 0 is an essential-requirement failure.** boltons is **not fully approved**: its yield criterion is only met for the pilot.

## 5. Scenario feasibility (boltons unless noted; needs per library: S1 10, S2 4, S3 4, S4 3, S5 3)

| Scenario | toolz | boltons |
|---|---|---|
| S1 outdated test | **Not supported** (0 bases; 2013 renames are collection errors, not assertion failures) | **Directly, from real history: 7 verified bases** (need 10). Plausible extra bases only if older eras become runnable (unverified) |
| S2 injected regression | Weak: about 5 recent green bases with code changes (pytest 9), plus 9 from the first week of 2013 | Real bases: **68 (2026) and up to 110 across years** with a green baseline and an executable-code change; killed mutant found in prototype (`strip_ansi`) |
| S3 mixed | Not supported | Controlled construction from an S1 base + mutant in another function: prototype validates (`rotate_file` + `iter_find_files`); capped by 7 bases |
| S4 decoy | Not supported | Controlled construction validates (mutant in `rotate_file` after its S1); memory record and probe are **synthetic by design** |
| S5 intent stripped | Not supported | Variant of an S1 base with the commit message replaced; validates, no leak. Competes with S1/S3 for the same 7 bases |

**Base reuse is the key open decision:** S1 + S3 + S5 need about 17 distinct bases per library but only 7 exist. Reusing one base commit across several scenario variants breaks the one-commit-one-episode assumption of a chronological stream.

## 6. Historical audit (method and limits)

`scripts/screening/classify_commits.py` classifies all non-merge commits; `verify_s1.py` runs, in disposable git worktrees, tests at the parent (R_p), at the commit (R_c), and at the parent with only the commit's source files applied (R_so). **S1_VALID** = test passes at R_p and R_c, fails at R_so, is in a test file the developer changed, and no other test fails. Full `both` set verified (no sampling) for both repositories; `src`-only commits ≤150 lines also checked for S2 bases. Harness bug fixed during the audit: `FORCE_COLOR=3` made every per-test parse silently empty, so earlier "0 S1" numbers were discarded and all runs repeated.

| | toolz (Python 3.12 / 3.10) | boltons (Python 3.12, pytest 9 / pytest 8) |
|---|---|---|
| Non-merge commits: source+tests / source only | 196 / 358 | 269 / 835 |
| `both` commits with green baseline | 11 + 2 failing outside changed tests / (3.10) 31 + 7 | 70 / 88 |
| **S1 verified** | **0 / 0** | **6 valid + 1 partial / same set** |
| Not runnable on 3.12 (baseline not green) | 183 | 191 (pytest 9), 173 (pytest 8) |
| Why | `inspect.getargspec`, `collections.Sequence` removed; pytest import-mode clashes | `collections` ABC imports (2013-2022); pytest 9 removed a `conftest` hook argument |

Note: the pytest 8 environment for toolz lacked an installed toolz (its newer tests read package metadata), so the toolz pytest 8 counts are a lower bound; the pytest 9 run is authoritative for toolz.

### boltons S1 bases (all stable over 5 repeats, parent and commit green)

| Commit (full hash) | Date | Function / failing test under source-only change | Evidence at that commit |
|---|---|---|---|
| `1c2db4664d24cea0f3f466772c730a5f78919b36` | 2026-10-25 | `fileutils.rotate_file`: `test_rotate_file_full_rotation`, `..._no_ext` (old tests expected `keep=5` to keep 4 files) | message + #483; docstring; **commit trailer indicates tool-assisted authoring** |
| `6270b5e35ed902859df6188970eec5bee5ab73d1` | 2026-10-25 | `iterutils.remap`: `TestRemap::test_namedtuple` (old test pinned a known TypeError) | message lists 8 issues; one commit bundles several fixes |
| `f387b64c4b7c4267ea94bcfcd229923ab2f598ea` | 2026-10-25 | `strutils.bytes2human`: `test_bytes2human` | bundled message, 8 issues |
| `8408ab57e6c92b4ddd37db6220628ea41ee8bb5b` | 2026-10-25 | `cacheutils.cached`: `test_unscoped_cached_dec` | bundled message, 7 issues |
| `7b577b0f3cc7adc2418733ce5c7dcdf6af82187d` | 2026-07-17 | `tbutils.ParsedException`: `test_parsed_exc_with_anchor` | message + #333 |
| `289edfcd20474b92850ec0f982032c4e79e72432` | 2026-07-17 | `funcutils.wraps`: partial (13 failures, 5 of them parametrised ids renamed by the developer) | 12-line message + #343 |
| `353f6befb4739f1d3583a8ce0ab5db761babd411` | 2015-04-20 | `tbutils.ParsedException.from_string`: `test_normal_tb`, `test_multiline_exception_message` | one-line message only; ran only because 2015 era is green |

Intent-evidence caveats: the CHANGELOG entries for the 2026-10 fixes were added in **later** commits (`405a6b0`, `3620150`), so they are not visible at the simulated CI commit; the in-commit message and docstrings are. About 8 of 95 non-merge commits since 2026-06 carry the tool-session trailer. Most S1 bases sit within a few days of each other (clustered), so a chronological stream is lumpy. Contamination: commits after mid-2026 are likely newer than any evaluated model's training data (not verified); 2015 commits are not.

### toolz history

About 41 non-merge commits since 2022-01, of which about 10 change code (mostly Python-version upkeep); the README now carries a "Project Status" section. Everything before 2025 fails on Python 3.12 (and mostly on 3.10). Recent behaviour commits (`a1e25cbd`, `d2eba037`, `5a7e078c`) only add tests, so the old tests still pass (not S1).

## 7. Test integrity and mutation feasibility

- **Static checks are feasible in both**: plain `assert` statements (toolz about 927 `assert` lines in 15 test files; boltons about 1,335 in 30), pytest-style tests, parametrisation. boltons uses subtests and class-based tests; both are visible to AST checks.
- **Mutation probe** (`mutation_probe.py`; 30 AST mutants per module, fixed seed 7, kill = tests-only failure, 60s cap; this is a feasibility probe, not a campaign; equivalent mutants are not removed):

| Module | Kill rate (excl. timeouts) | Mean seconds per mutant |
|---|---:|---:|
| toolz `itertoolz` | 0.96 (2 timeouts) | 4.55 (timeouts included) |
| toolz `dicttoolz` | 0.93 | 0.37 |
| toolz `functoolz` | 0.73 | 0.47 |
| boltons `iterutils` | 0.67 | 0.48 |
| boltons `strutils` | 0.40 | 0.36 |
| boltons `dictutils` | 0.77 | 0.40 |

  Per-mutant cost is under 0.5s, so cached mutants are cheap. boltons' lower kill rates mean **mutant-kill retention** is measurable but must use only mutants a baseline test kills (the proposal's S2 requirement already does).
- Known gap: mutation adequacy for the S1 reference tests (kill retention against the developer's updated test) was **not** measured.

## 8. Schema compatibility

Real data from boltons was loaded into the **finalized** `Episode` model (`scripts/screening/proto_episodes.py`, schema imported read-only): all five scenarios validate, with `find_leaked_ground_truth` clean on each agent payload.

| Schema requirement | Source |
|---|---|
| repo, pinned base commit, current/parent commit, message, source diff | directly from git |
| failing test ids, tracebacks, assertion text | direct: pytest junit output at the source-only state |
| function identity (`function_qualname`), group per function | derived (diff hunk context / AST); needs a small mapper |
| `reference_repairs` (developer test diff), `gold_direction` for S1 | direct (real commit) |
| `gold_evidence_ids` | **manual curation** (which message sentences state intent) |
| mutants, S2/S3/S4 labels | controlled construction, verified by running tests |
| S4 memory record and probe, S5 stripped evidence, tool-fault plan | synthetic/constructed by design |
| 5-run stability (`stability_runs`) | direct, measured |

Prototype shortcuts (shape-only): the S4 episode lists one changed file path with a combined diff; `evidence_availability`, corpus items and memory content are minimal.

## 9. Risks and limitations

1. **S1 supply is the binding constraint**: 7 verified bases (boltons) vs about 17 needed per library; toolz 0. Mitigations to decide: allow base reuse across S3/S5 variants (flagged above), relax S1 (e.g. count commits where only part of the failing tests are in changed files), make older eras runnable with per-episode Python/dependency pins (unverified, cannot be estimated reliably), or add libraries.
2. **Historical runnability**: under fixed Python 3.12, most history of both libraries is unusable; `EnvironmentSpec` allows a per-episode `python_version`, but the proposal assumes fixed versions.
3. **Bug fixes vs behaviour changes**: boltons S1 bases are mostly bug fixes whose old test encoded the bug; whether that counts as "intended behaviour change" needs a labelling rule agreed by the team.
4. **Bundled commits and clustering**: several S1 bases fix many issues in one commit; 26 commits landed on 2026-10-25.
5. **Possible tool-assisted commit text** in the 2026 sweeps; evidence quality may differ from human-written messages.
6. **Not verified**: Docker isolation; boltons tests under `--network none` in a container; runs with randomised order or parallelism; mutation adequacy for reference tests; contamination of boltons/toolz in model training data; legal review of redistributing patches (BSD-3-Clause requires keeping notices; not legal advice).
7. **Selection bias in the audit**: S1 was verified exhaustively for `both` commits, but S2/S3/S4 base counts are upper-bound style (green baseline + code change, not each with a killed mutant).

## 10. Next actions (priority order, for the remaining days)

> *Historical, written on 2026-10-27.* Item 1 (the 6-12 episode manifest from 7 S1 bases plus 3-4 S2 bases) is **superseded** by the delivered
> 13-episode pilot (see the notice at the top and `datasets/README.md`). Items 2-5 were **not re-assessed** in this update and stand as written.

1. **Today:** pin `boltons` at `d31d821d56c8f7756af9c30a864077f2e5855321`; build the pilot manifest from the 7 verified S1 bases plus 3-4 S2 bases (e.g. `bba729bf` and others from `docs/screening_evidence`), targeting 6-12 episodes, 5-run stability per episode.
2. **Decide the base-reuse and relaxation policy** (section 9.1) before scaling.
3. **Screen 1-2 further candidates** with `scripts/screening/` (about one hour each) to supply a real backup and the other libraries; check Docker access (`docker info`) at the same time.
4. Write the function-identity mapper and gold-evidence curation rule; fill `EpisodeInput`/`Episode` from the verification output.
5. Run one real episode end to end in a no-network sandbox with the 3.12 environment.
