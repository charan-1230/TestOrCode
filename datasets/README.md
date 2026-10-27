# Pilot dataset (boltons): 8 dev + 4 frozen test episodes

Real commits, diffs and tests from [boltons](https://github.com/mahmoud/boltons) pinned at
`d31d821d56c8f7756af9c30a864077f2e5855321` (BSD-3-Clause). Only the injected regressions (S2/S3/S4), the S4 decoy memory and the
S5 evidence stripping are constructed. Chosen by `docs/repository_selection.md`; built by `scripts/pilot/build_dataset.py`.

**Read this first:** 12 episodes are a pilot for building and debugging the pipeline. They cannot support statistical claims, and the
4 test episodes are a smoke-level check, not an evaluation.

## Files

| File | Contains | Who may read it |
|---|---|---|
| `pilot_dev_agent_view.jsonl` | 8 `EpisodeInput` records (what the system sees) | **your system, for development** |
| `pilot_test_agent_view.jsonl` | 4 `EpisodeInput` records | your system, only after the config is frozen |
| `pilot_dev.jsonl`, `pilot_test.jsonl` | full `Episode` records: gold labels, reference repairs, mutants, fault plans | evaluation harness / scoring only |
| `pilot_manifest.json` | pins, dependency hash, split membership, SHA-256 of every file | humans, tests |
| `pilot_verification.json` | per-episode evidence (stable runs, round trip, leak audit) | humans, tests |

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

To run the real tests on an episode's CI state (needs a clone of boltons and a venv with `pytest==8.4.2`):

```bash
git clone https://github.com/mahmoud/boltons ../boltons && git -C ../boltons checkout d31d821d56c8f7756af9c30a864077f2e5855321
python scripts/pilot/materialize_episode.py --repo ../boltons --dataset datasets/pilot_dev_agent_view.jsonl \
    --episode dev-05 --dest /tmp/dev-05
cd /tmp/dev-05 && PYTHONPATH=. /path/to/venv/bin/python -m pytest tests --import-mode=importlib -q
```

The CI state is `parent_commit` plus the `ci_event.changed_files` diffs (source only; the developer's test changes are hidden).

## Episodes

Ids are opaque and follow the commit-chronological stream order of each split. Scenarios are evaluation metadata (not in the agent view).

| Id | Scenario | Difficulty | CI commit | Failing tests | Function(s) |
|---|---|---|---|---|---|
| dev-01 | S2 injected regression | easy | `0c88f253` | 1 | `tableutils.to_text` |
| dev-02 | S2 injected regression | medium | `c0d580df` | 2 | `funcutils.once` |
| dev-03 | S2 injected regression | easy | `bba729bf` | 1 | `strutils.strip_ansi` |
| dev-04 | S1 outdated test | easy | `7b577b0f` | 1 | `tbutils.ParsedException.from_string` |
| dev-05 | S1 outdated test | easy | `8408ab57` | 1 | `cacheutils.CachedFunction.__call__` |
| dev-06 | S4 decoy memory | hard | `20619069` | 3 | `cacheutils.CachedFunction.__call__` (memory from dev-05) |
| dev-07 | S3 mixed | hard | `f387b64c` | 4 | `strutils.bytes2human` (update) + `listutils.BarrelList._cur_size_limit` (fix) |
| dev-08 | S5 intent stripped | medium | `6270b5e3` | 1 | `iterutils.default_enter` |
| test-01 | S2 injected regression | easy | `827f1937` | 1 | `timeutils.daterange` |
| test-02 | S2 injected regression | easy | `ee3c660a` | 1 | `fileutils.FilePerms._FilePermProperty.__set__` |
| test-03 | S3 mixed | hard | `1c2db466` | 3 | `fileutils.rotate_file` (update) + `pathutils.augpath` (fix) |
| test-04 | S4 decoy memory | hard | `bd46e5ab` | 1 | `fileutils.rotate_file` (memory from test-03) |

Two dev episodes carry a tool-fault plan (dev-04 first `apply_patch` fails; dev-02 second `run_tests` selector is invalid). The plan
exists only in the full records.

## Split rules (checked by `tests/test_dataset.py` and the builder)

- **Dev and test are separate streams** with disjoint CI commits, failing tests, functions and modules (module pools in
  `pilot_manifest.json`). Decoy memory in an S4 episode comes from an earlier episode of the same stream only.
- **The test split is frozen.** Do not tune prompts, retrieval or thresholds on it, and do not look at its gold labels while
  developing. `files_sha256` in `pilot_manifest.json` is checked by a test, so an accidental edit fails CI; regenerate only on purpose.
- Tuning belongs on dev. When more libraries are added, keep the 3-held-out-libraries + 1-dev-library design from the proposal.

## What an agent sees

`EpisodeInput`: repo and pinned environment (Python 3.12, pytest 8.4.2, no network), the CI event (commit, parent, message, per-file source
diffs, failing test ids with pytest message and traceback; failure groups are **not** pre-computed, Triage must group), a retrieval
corpus, visible memory (`[]` normally, one record in S4), the edit policy (only `boltons/` and `tests/`; `conftest.py`, config and CI
files forbidden) and a placeholder budget.

Corpus, bounded and ending at the CI commit: last 100 non-merge commit messages, README, last 3 CHANGELOG sections as of that commit,
`docs/<module>.rst` and docstrings of the affected modules. Evidence ids are deterministic so a retriever can reuse them:
`msg:<sha12>`, `doc:<path>#<n>`, `changelog:<n>`, `docstring:<module>.<qualname>`.

## How it was built and verified

For every episode, by running the code (not by assumption): parent commit green; the CI state fails the same tests in **5 of 5 runs**;
the developer's test diff (S1/S3/S5) or reverting the mutant (S2/S4) restores a green suite, and S3 is split correctly (the developer
tests fix exactly the outdated half); S4's probe passes on the mutated code while other tests fail; the CI state is rebuilt from the agent
view alone and reproduces the failures; no ground-truth key, future commit, mutant text or reference-repair line appears in the agent
view; two independent builds are byte-identical. Memory addresses and pytest temp paths in tracebacks are normalised for that.

## Limitations (needs a human pass before relying on it)

- `gold_evidence_ids` are proposed by the builder: the CI commit message of each outdated test. Several commits fix many issues at once,
  so that message is only partly about the failing test. Please review them.
- S5 hides the commit message and the changed `docs/*.rst`, but not comments or docstrings inside the source diff (dev-08's diff has one
  small explanatory comment). The S4 memory record and its probe (the developer's updated test function) are synthetic by construction.
- Mutants are single-operator swaps. For 4 of 6 S2 episodes the killed mutant is in the changed file but not in the commit's changed
  function. S1 evidence is bug-fix flavoured: the old test encoded the bug.
- Only 6 clean S1 bases existed, all used once or kept as spares (`353f6bef`, `289edfcd`), so this pilot cannot grow in S1 without new
  repositories. Docker was not available; `docker_image` is unset.
- Budgets in the episodes are placeholders to tune on dev.

Regenerate (≈7 min; needs the pinned clone and a venv with `pytest==8.4.2`):
`python scripts/pilot/build_dataset.py --repo ../boltons --python <venv>/bin/python --work ../wt-pilot`
