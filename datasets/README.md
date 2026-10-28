# Pilot dataset (boltons): 9 dev + 4 frozen test episodes (13 in total)

Real commits, diffs and tests from [boltons](https://github.com/mahmoud/boltons) pinned at
`d31d821d56c8f7756af9c30a864077f2e5855321` (BSD-3-Clause). Only the injected regressions (S2/S3/S4), the S4 decoy memory and the
S5 evidence stripping are constructed. Chosen by `docs/repository_selection.md`; built by `scripts/pilot/build_dataset.py`.
Corpus version `pilot-2` (revision of 2026-10-28, see "Revision 2" below).

**Read this first:** 13 episodes are a pilot for building and debugging the pipeline. **This exceeds the 5-day plan's 6-12 target**; the
count is reported as it is. The set cannot support statistical claims, and the 4 test episodes are a smoke-level check, not an
evaluation. **The S5 episode (dev-01) is provisional** (see Limitations).

## Files

| File | Contains | Who may read it |
|---|---|---|
| `pilot_dev_agent_view.jsonl` | 9 `EpisodeInput` records (what the system sees) | **your system, for development** |
| `pilot_test_agent_view.jsonl` | 4 `EpisodeInput` records | your system, only after the config is frozen |
| `pilot_dev.jsonl`, `pilot_test.jsonl` | full `Episode` records: gold labels, reference repairs, mutants, fault plans, notes | evaluation harness / scoring only |
| `pilot_manifest.json` | pins, dependency hash, split membership, corpus policy, SHA-256 of every jsonl file | humans, tests |
| `pilot_verification.json` | per-episode evidence (stable runs, round trip, leak audit, source-comment provenance) | humans, tests |

Agent-view files contain no ground truth by construction (`Episode.agent_payload()`); `load_agent_inputs` refuses a file that does.

## Use it

```python
from testorcode.dataset import agent_view_path, load_agent_inputs

for episode in load_agent_inputs(agent_view_path("dev")):      # stream order: memory carries over
    print(episode.episode_id, episode.ci_event.failing_tests[0].test_id)
```

Evaluation side only: `load_episodes(episode_path("dev"))` returns `Episode` objects with `gold_actions()`, `ground_truth`, `fault_plan`.
`DATASET_PATH` (see `.env.example`) defaults to `datasets/pilot_dev.jsonl`: the harness reads the full record and passes only
`episode.agent_input` to agents.

To run the real tests on an episode's CI state (needs a clone of boltons and the pilot venv, see "Pilot environment" below; `--dest` must be a
new directory outside the clone and this project):

```bash
git clone https://github.com/mahmoud/boltons ../boltons && git -C ../boltons checkout d31d821d56c8f7756af9c30a864077f2e5855321
python scripts/pilot/materialize_episode.py --repo ../boltons --dataset datasets/pilot_dev_agent_view.jsonl \
    --episode dev-06 --dest /tmp/dev-06
cd /tmp/dev-06 && PYTHONPATH=. /path/to/venv/bin/python -m pytest tests --import-mode=importlib -q
```

The CI state is `parent_commit` plus the `ci_event.changed_files` diffs (source only; the developer's test changes are hidden).

## Episodes

Ids are opaque and follow the commit-chronological stream order of each split. Scenarios are evaluation metadata (not in the agent view).
"Comments" is the number of source-comment evidence items in the corpus (policy below).

| Id | Scenario | Difficulty | CI commit | Failing tests | Function(s) | Comments |
|---|---|---|---|---|---|---|
| dev-01 | S5 intent stripped (**provisional**) | medium | `353f6bef` | 2 | `tbutils.ParsedException.from_string` | 4 |
| dev-02 | S2 injected regression | easy | `0c88f253` | 1 | `tableutils.to_text` | 1 |
| dev-03 | S2 injected regression | medium | `c0d580df` | 2 | `funcutils.once` | 0 |
| dev-04 | S2 injected regression | easy | `bba729bf` | 1 | `strutils.strip_ansi` | 4 |
| dev-05 | S1 outdated test | easy | `7b577b0f` | 1 | `tbutils.ParsedException.from_string` | 5 |
| dev-06 | S1 outdated test | easy | `8408ab57` | 1 | `cacheutils.CachedFunction.__call__` | 0 |
| dev-07 | S4 decoy memory | hard | `20619069` | 3 | `cacheutils.CachedFunction.__call__` (memory from dev-06) | 0 |
| dev-08 | S3 mixed | hard | `f387b64c` | 4 | `strutils.bytes2human` (update) + `listutils.BarrelList._cur_size_limit` (fix) | 9 |
| dev-09 | S1 outdated test | easy | `6270b5e3` | 1 | `iterutils.default_enter` | 10 |
| test-01 | S2 injected regression | easy | `827f1937` | 1 | `timeutils.daterange` | 0 |
| test-02 | S2 injected regression | easy | `ee3c660a` | 1 | `fileutils.FilePerms._FilePermProperty.__set__` | 0 |
| test-03 | S3 mixed | hard | `1c2db466` | 3 | `fileutils.rotate_file` (update) + `pathutils.augpath` (fix) | 6 |
| test-04 | S4 decoy memory | hard | `bd46e5ab` | 1 | `fileutils.rotate_file` (memory from test-03) | 2 |

Two dev episodes carry a tool-fault plan (dev-05: the first `apply_patch` call fails; dev-04: the second `run_tests` call gets an invalid test selector). The plan
exists only in the full records. The full old-to-new id mapping is in `docs/dataset_revision_audit.md`.

## Split rules (checked by `tests/test_dataset.py` and the builder)

- **Dev and test are separate streams** with disjoint CI commits, failing tests, functions and modules (module pools in
  `pilot_manifest.json`). Decoy memory in an S4 episode comes from an earlier episode of the same stream only. Within dev, dev-01 and
  dev-05 change the same function (`ParsedException.from_string`); dev-01 is an S5 and writes no memory, so dev-05 has no memory coverage.
- **The test split is frozen.** Do not tune prompts, retrieval or thresholds on it, and do not look at its gold labels while
  developing. `files_sha256` in `pilot_manifest.json` is checked by a test, so an accidental edit fails CI; regenerate only on purpose.
  (Revision 2 regenerated it on purpose: see below.)
- Tuning belongs on dev. When more libraries are added, keep the 3-held-out-libraries + 1-dev-library design from the proposal.

## What an agent sees

`EpisodeInput`: repo and pinned environment (Python 3.12, pytest 8.4.2, no network), the CI event (commit, parent, message, per-file source
diffs, failing test ids with pytest message and traceback; failure groups are **not** pre-computed, Triage must group), a retrieval
corpus, visible memory (`[]` normally, one record in S4), the edit policy (only `boltons/` and `tests/`; `conftest.py`, config and CI
files forbidden) and a placeholder budget.

Corpus, bounded and ending at the CI commit: last 100 non-merge commit messages, README, last 3 CHANGELOG sections as of that commit,
`docs/<module>.rst` and docstrings of the affected modules, **plus source comments (policy below)**. Evidence ids are deterministic so a
retriever can reuse them: `msg:<sha12>`, `doc:<path>#<n>`, `changelog:<n>`, `docstring:<module>.<qualname>`,
`comment:<module>.<qualname>#L<line>`.

### Source-comment evidence policy (corpus `pilot-2`)

- **Eligible:** every `#` comment inside a function or method of `boltons/*.py` whose AST differs between the parent commit and the CI
  state (the CI diff, including injected mutants). Consecutive comment-only lines are one item; an inline comment is its own item.
  Applied to **all 13 episodes** with the same rule; no relevance filtering, no rewriting (so TODOs and commented-out code are included).
- **Excluded:** pragma/directive comments (`noqa`, `type:`, `pragma`, `fmt:`), shebang and encoding lines, comments outside functions,
  and **every comment, TODO and docstring in test files**. Tests describe the behaviour being replaced and are treated as possibly stale
  evidence, so they stay out of the corpus even though an agent can still read them with `read_file` (dev-09's old test carries a 2015
  TODO that states the intent; it is not a corpus item).
- **Item shape:** `source_type: source_file`, `source_ref: <path>::<qualname>` (same convention as docstring items), `line_start/line_end` in
  the CI-state file, `text` = the verbatim comment lines, `commit_sha` = the CI commit for lines added by the CI diff, otherwise the commit
  that introduced the line (git blame). Origin (added by the CI diff or pre-existing), diff-hunk header and blame commits are recorded
  per item in `pilot_verification.json` under `notes.source_comments`.
- **Consequence for metrics:** gold evidence ids are now `msg:` or `comment:` ids. A comment added by the CI diff is also visible verbatim in
  the diff the agent already receives, like the commit message, so gold-evidence Recall@k should be reported per gold type. Comment-only
  gold is hard for keyword retrieval (see `docs/retrieval.md`).

## How it was built and verified

For every episode, by running the code (not by assumption): parent commit green; the CI state fails the same tests in **5 of 5 runs**;
the developer's test diff (S1/S3/S5) or reverting the mutant (S2/S4) restores a green suite, and S3 is split correctly (the developer
tests fix exactly the outdated half); S4's probe passes on the mutated code while other tests fail; the CI state is rebuilt from the agent
view alone and reproduces the failures; no ground-truth key, future commit, mutant text or reference-repair line appears in the agent
view; the S5's real commit message appears nowhere in its agent view; every gold-evidence selector resolved to exactly one agent-visible
item inside the labelled function; two independent builds are byte-identical. Memory addresses and pytest temp paths in tracebacks are
normalised for that.

### Authorised verification of the withheld S5 message

dev-01 (S5) shows the agent the generic message `tbutils: update`; the real commit message is withheld. `pilot_verification.json` does **not**
store the real message: it stores `notes.withheld_message_sha256` (SHA-256 of the exact message, stripped, UTF-8) and
`notes.withheld_message_chars`. The builder keeps the text only in memory, to check that it appears nowhere in the agent view.

An authorised reviewer (anyone with the pinned boltons clone, i.e. the public history) can confirm which message was withheld:

```bash
git -C ../boltons log -1 --format=%B 353f6befb4739f1d3583a8ce0ab5db761babd411 \
  | python3 -c "import sys,hashlib; m=sys.stdin.read().strip(); print(hashlib.sha256(m.encode()).hexdigest(), len(m))"
# expected: d6a334fb701d85ebdf318f16784bf9ac135adff457268d9948b02b06ea94d4c1 50
```

This is **not a secrecy mechanism**: the message is short and public on GitHub, so a hash only avoids storing the text in the dataset files
and lets a reviewer confirm the withheld text. The safeguard is that agents never receive `pilot_verification.json` or the full `Episode`
records (`load_agent_inputs` refuses them). `tests/test_dataset.py` checks that the file holds only the hash and that the text is absent
from the agent view. For that check the text is still written in `tests/test_dataset.py`, and it also appears in
`docs/screening_evidence/screening_summary.json` and the untracked `final_review.md`; none of these is agent-facing.

## Revision 2 (2026-10-28)

Built outside `datasets/`, validated, audited against the previous version, then promoted (hashes refreshed only at that point).

- **dev-09 (`6270b5e3`) was S5, now S1.** After review the S5 label was not defensible (the old test's TODO and the diff comments state the
  intent). The real 8-issue message is restored; gold = the message (clause "fix remap namedtuples") + the diff comment
  `# namedtuple, rebuilt in default_exit`.
- **dev-08 (`f387b64c`, S3):** the UPDATE_TEST half now has comment-only gold (`# pick the unit after rounding, so e.g. 1023.99K displays
  as 1M`), because the commit message only names the function. A note records the remaining ambiguity.
- **dev-01 (`353f6bef`) is a new S5**, replacing the removed one.
- **dev-06 (`8408ab57`) is unchanged** (label, gold evidence, corpus: it has no eligible comments).
- **All other episodes keep labels, failing tests, diffs, reference repairs, mutants and memory identical**; only corpus items were
  appended (including two frozen test episodes, test-03 and test-04) and the corpus version became `pilot-2`. Dev ids shifted by one
  because the 2015 episode sorts first. `docs/dataset_revision_audit.md` lists every id and every changed field.

## Limitations (needs a human pass before relying on it)

- **dev-01 (S5) is provisional.** `ESCALATE` is a judgement: nothing in the agent view states whether the leading space in the exception
  message was intended, but the parent commit (`c4a489d`, 52 seconds earlier, same author) had just added the test that pins it. More
  importantly the history-leak check cannot be run yet: `git_log`/`git_show`/`git_blame` are unimplemented stubs, and their contract
  ("reachable from the cutoff commit", cutoff = the CI commit) would expose the withheld message and the developer's test diff. A
  tripwire test (`test_history_access_tripwire_for_the_provisional_s5`) fails when those tools are implemented, to force a real
  regression test with the cutoff at the parent commit. Until then do not report dev-01 as a validated S5. It is also a 2015 commit in a
  stream of 2026 commits (higher chance of being in model training data) and has no changelog items.
- **dev-08's UPDATE_TEST half is weakly evidenced:** the only intent statement is a source comment added by the CI diff; an old test comment
  and an earlier commit point the other way, so `ESCALATE` is also arguable there. dev-09's message bundles 8 issues.
- `gold_evidence_ids` come from manifest selectors resolved by the builder and checked for presence and function; whether each item
  *states* the intent is a human judgement and still needs a final human pass.
- The S4 memory record and its probe (the developer's updated test function) are synthetic by construction. Mutants are single-operator
  swaps. S1 evidence is bug-fix flavoured: the old test encoded the bug.
- Only 6 clean S1 bases existed (`289edfcd` stays unused, partial), so the pilot cannot grow in S1 without new repositories. Docker was not
  available; `docker_image` is unset. Budgets in the episodes are placeholders to tune on dev.

## Pilot environment and safe worktrees

- **Environment.** `scripts/pilot/requirements-pilot.txt` pins the pilot environment exactly, transitive dependencies included
  (`iniconfig==2.3.1`, `packaging==26.3`, `pluggy==1.7.0`, `Pygments==2.21.0`, `pytest==8.4.2`). Create it with
  `python3.12 -m venv <venv> && <venv>/bin/python -m pip install -r scripts/pilot/requirements-pilot.txt`. A venv built that way reproduces the
  `dependency_lock_hash` recorded in `pilot_manifest.json`, and the builder **refuses to run** if `pip freeze` differs from the file.
  (This is separate from the main project's `requirements-lock.txt`, which pins pytest 9.x for TestOrCode itself.)
- **Worktrees.** `build_dataset.py` and `materialize_episode.py` reset their worktree with `git checkout -f` and `git clean -fdx`, which destroy
  uncommitted work, so the target is validated first. A new directory is created as a linked worktree and marked disposable. **Always refused,
  even with `--force`:** the main clone, any standalone repository, another repository's worktree, a non-empty plain directory, and any path
  equal to, containing or inside the clone or this project. **Needs `--force`:** an existing linked worktree of the right repository that
  these scripts did not create (for example one made by an earlier version); `--force` adopts and marks it, so later runs need no flag.
  Use `--force` only for a worktree you are happy to have reset.

Regenerate (≈10 min; needs the pinned clone and the pilot venv above):
`python scripts/pilot/build_dataset.py --repo ../boltons --python <venv>/bin/python --work ../wt-pilot --out <new dir>`, then compare with
`python scripts/pilot/audit_revision.py --new <new dir>` before replacing anything in `datasets/`. The first run on a worktree created by an
earlier version of the scripts (such as an existing `wt-pilot`) needs `--force`.
