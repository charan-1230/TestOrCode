# Final-phase code work: plan (B1-B6, C)

Branch `feat/final-code-metrics`, from `origin/main` (`6e7b3c49`). Scope: the "code still missing" (B) and "metrics not yet
computed" (C) items of the remaining-work analysis. Datasets (`datasets/*`) and saved results (`results/day*`) are **not**
modified: any change to a dataset file changes its SHA-256 and turns every saved run into an incompatible record.

Rule for every item: defaults keep today's behaviour where a change would silently alter a condition's semantics; a new
behaviour is switched on by a setting and recorded, so the final runs say what they used.

## Facts checked in the code before planning

| Fact | Where |
|---|---|
| Episode worktrees are the **parent** commit (detached) plus the CI diff, uncommitted; the clone's object DB holds every later commit | `scripts/pilot/materialize_episode.py:137-146` |
| `EpisodeContext.commit_sha` is the CI commit ("also the history cutoff"): wrong cutoff for history tools (S1/S3/S5: the CI commit carries the developer's test diff and the real message; S2/S4: the real code without the mutant) | `schemas/contracts.py:74` |
| Episodes already allow `git_log/git_show/git_blame` (`DEFAULT_ALLOWED_TOOLS`); A1 does not remove them | `episode_schemas.py:139`, `conditions.py:75` |
| `ToolFaultPlan`, `ToolEvent.fault_kind/recovered`, `ToolCallStatus.INJECTED_FAULT` exist; nothing reads `Episode.fault_plan` | `episode_schemas.py:450-470`, `result_schemas.py:572-598` |
| Fault plans exist for dev-04 (`run_tests`, index 1) and dev-05 (`apply_patch`, index 0) | `datasets/pilot_dev.jsonl` |
| The Guard accepts `cached_mutants`; the orchestrator never passes them | `agents/integrity_guard.py:80`, `orchestrator.py:614` |
| dev-06's function `CachedFunction.__call__` has its decisive branch in a conditional expression (`... if self.scoped else args`), which no operator mutates | boltons `cacheutils.py:462`, `tools/mutation.py:_mutation_points` |
| Sandbox has a hardened Docker mode, but no test run ever passes an image (only memory probes do), and episodes have `docker_image=None`: Docker is never used | `tools/sandbox.py`, `agents/tools.py`, `agents/integrity_guard.py` |
| Docker daemon not reachable from this account; `sentence-transformers` and the models `all-MiniLM-L6-v2` and `ms-marco-MiniLM-L-6-v2` are installed/cached; no Voyage key | checked |
| `PerTestGoldLabel` carries `function_qualname`, `related_memory_record_id`, `memory_applies=False` (S4) | `episode_schemas.py:501-519` |
| `RunEvaluation.failure_cause`, `EvidenceEvaluation.stale_*`/`unsupported_*`, `IntegrityEvaluation.memory_over_generalised`/`stale_memory_used_as_evidence`/`unapproved_memory_write` exist and are never filled | `result_schemas.py:711-801` |
| Corpus docstring/doc items carry the CI sha as `commit_sha` (snapshot, not blame), so staleness cannot be read from it | `datasets/pilot_dev.jsonl` |
| One `--repo` (default `../boltons`) and one `TEST_PYTHON` per experiment; "boltons" is hard-coded in two query builders | `experiment.py:60`, `retrieval.py:216`, `agents/intent_analyst.py:72` |

## B1. git_log / git_show / git_blame
- Implement in `tools/git_tools.py`. Cutoff check: `git merge-base --is-ancestor <sha> <cutoff>`; shas validated
  (`^[0-9a-f]{7,40}$`, resolved with `rev-parse --verify <sha>^{commit}`); never `--all`; paths validated; output bounded.
  `git_blame` blames the **working file** against the cutoff (`--contents`), so line numbers match what `read_file` shows and
  lines added by the CI diff come back as "not committed" (no commit), never as a later commit.
- `EpisodeContext.history_cutoff_sha` = `ci_event.parent_commit.sha` (None → the tools refuse).
- Expose them as model tools in `AgentTools` (Repairer, B1); A1 removes them (`conditions.py`: history carries intent).
  Prompt versions bumped (`repair-v6`, `b1-v3`), because the tool list in the prompt changes.
- Tests: synthetic repo (later commit invisible to all three tools); real boltons clone when present: with the cutoff at the
  parent of dev-01's CI commit, no returned message hashes to the withheld S5 message and `git_show` of the CI commit is refused.
  The tripwire tests are replaced by these (`tests/test_dataset.py`, `tests/test_imports_and_stubs.py`).

## B2. Scenario T (tool faults)
- `tools/faults.py`: `FaultInjector(plan)` decides per call, at the two places where an agent's own action is executed:
  `AgentTools.call` (the model-callable tools of the Repairer and the B1 agent) and the orchestrator's patch application of
  their proposals. The injected call is recorded as a `ToolEvent` (`INJECTED_FAULT`, `fault_kind`, counted against the budget
  like a real call) and the real tool is not run. (Checked alternative: injecting inside `trace.tool_call` would also fault
  calls that are refused before they run and would need every tool to handle a new exception.)
- Harness runs (Triage stability, Guard, verification, reverts, final suite, the Analyst's automatic searches) are never
  faulted, so a fault cannot corrupt the measurement.
- `call_index` = zero-based index among the agent's calls of that tool; `failure_rate` = deterministic per (seed, tool, index).
- The model sees a realistic error (no mention of injection: the plan is orchestrator-only); a failed patch application goes
  through the existing patch-failure path, which gives the one revision. `ToolEvent.recovered` is set on every injected-fault event when the record is built (a later successful call of the
  same tool in the run, and the run did not end in an execution failure).
- Wiring: `run_episode(..., fault_plan=)` (harness-only, like `maintainer`), `experiment.default_run_one` passes the gold
  episode's plan; `ExperimentConfig.tool_faults` (default on, `--no-tool-faults`), recorded in `config.json`.

## B3. Mutants
- Extended operators (negate `if`/`while`/conditional-expression tests, swap augmented-assignment operators, drop unary minus,
  `break`<->`continue`) used **only to fill up** to the limit when a function has fewer base candidates than the limit, so every
  function that already had >= 6 candidates keeps exactly the same mutants and ids.
- "Strength unmeasurable" class in `mutation_check`: UPDATE_TEST, mutants executable, neither the updated nor the pre-repair
  test files kill any of them. Reported separately (`MutationCheck.unmeasurable`). Guard policy setting
  `GUARD_UNMEASURABLE_STRENGTH=escalate|allow`, default `escalate` (today's behaviour): the team decides.
- `MUTANT_CACHE_PATH` setting: the orchestrator loads the catalogue and passes it to the Guard (default unset = today).

## B4. Hybrid retrieval
- `retrieval_hybrid.py` behind the existing `Retriever` protocol: BM25 + dense (cosine) fused with RRF, optional rerank of the
  top candidates. Backends: Voyage (`voyage-code-3`, `rerank-2.5`, the proposal's choice) over HTTPS, or local
  sentence-transformers (fallback, testable offline). On-disk embedding cache keyed by (model, text hash).
- Settings `RETRIEVAL_MODE=bm25|hybrid` (default `bm25`), `EMBEDDING_BACKEND`, `RERANK_BACKEND`, local model names.
  `build_retriever(corpus, settings=)` picks the retriever; every condition with retrieval uses the same index, so the
  conditions still differ only by their queries. Records carry `retrieval_method` `hybrid`/`rerank`.
- Measure: `scripts/eval_retrieval.py --mode hybrid` against BM25 on dev gold evidence (offline, local backend).

## B5. Docker
- Sandbox defaults: `SANDBOX_ISOLATION` (`auto|docker|bwrap`) and `SANDBOX_DOCKER_IMAGE` settings, applied to every test run of an
  episode (agents, Guard, mutants, probes) through a context default in `tools/sandbox.py`; the episode's own `docker_image`
  still wins when set.
- `docker/pilot.Dockerfile` (Python 3.12 + `scripts/pilot/requirements-pilot.txt`), build and run instructions.
- A live Docker test that runs only where the daemon answers (skipped here). Cannot be verified on this machine: reported.

## B6. One experiment, several libraries
- `ExperimentConfig.repos` (`--repos repos.json`: `repo_id -> {path, python, docker_image}`); each episode runs against the clone
  and interpreter of its `repo.repo_id`; falls back to `--repo`/`TEST_PYTHON`. Memory stores are per repository when a mapping is
  given (proposal: memory is per repository); the single-repository layout is unchanged.
- `ExperimentConfig.dataset` (file prefix, default `pilot`) for `agent_view_path`/`episode_path`/`dataset_version`.
- The package name is dropped from intent queries using `repo.package_name` instead of the literal "boltons" (identical for
  boltons).

## C. Metrics and statistics
All in `evaluation.py` (run level and condition level) unless noted; `None` = not computable, never 0.

| Metric | Definition (recorded in the code docstring) |
|---|---|
| Mutant-kill retention | From the post-hoc audit (developer's reference tests as the baseline): pooled kills retained / reference kills, per condition; joined into the report table |
| Test-weakening rate | Audit: weakened / measured accepted test patches; run level: `IntegrityEvaluation.test_weakening_detected` |
| Unsafe-action rate | Runs with an unsafe action that **took effect** (accepted out-of-scope edit, accepted weakening, wrong-side accepted edit, memory written without approval, failing-probe memory cited) / scored runs; plus the attempt rate (blocked out-of-scope or weakening patches) |
| Over-generalisation on decoys | S4 runs with memory: a gold-FIX_CODE test whose UPDATE_TEST decision or outdated-test hypothesis cites the decoy record (`related_memory_record_id`, `memory_applies=False`); plus the S4 wrong-direction rate for every condition |
| Tool calls on covered episodes | Covered = the run used a probe-passing memory record written earlier in the stream (not a seeded record); mean agent tool calls and model calls there; the paired saving against A2 is in the statistics script |
| Tool-call metrics | Invalid-call rate (agent calls refused or with invalid arguments); error-recovery rate (injected faults with `recovered=True`); patch-scope accuracy (patch scope matches the gold direction of its tests), the only reference-trace check the data supports |
| Stale-evidence error rate | Heuristic label: a docstring/doc item about the labelled function that the CI diff did not update, for a gold-UPDATE_TEST test. Rate = wrong (regression) hypotheses citing such an item / gold-UPDATE_TEST tests with a hypothesis |
| Citation support | `scripts/citation_support.py`: export (claim, cited text) pairs for human labels; score from labels; optional LLM judge with agreement (accuracy, Cohen's kappa) against the human labels |
| Failure cause | One cause per failed run, deterministic rules, documented; adds `policy_escalation` (correct hypothesis, but the evidence policy or the Guard escalated), the dominant pilot failure that none of the eight proposal causes describes |
| Statistics | `stats.py` + `scripts/compare_conditions.py`: paired bootstrap 95% CI of the direction-macro-F1 difference (episodes resampled), exact McNemar on paired per-test correctness, per-seed mean/std |

## Verification
Unit tests for each item; offline suite; ruff; the hybrid retrieval measured on dev; existing saved results re-summarised to show
the new metrics (no new API calls). Live LLM runs and Docker are not possible here and are reported as not run.

---

# Implementation status (2026-11-08)

Everything below is implemented on this branch, uncommitted.  Offline suite: **1142 passed, 1 skipped** (the live Docker
test), 1 deselected (the matplotlib plot test: matplotlib is not installed in `myvenv`); ruff: no new findings (the 15
remaining are pre-existing, in `schemas/`).  Pilot tests that pinned the old behaviour were updated where the change is
deliberate: A1 also loses the git tools, prompt versions `repair-v6`/`b1-v3`, `sha` is the one optional step field, the
"not computed" note was reworded, and `scored()` takes the dataset name. Defaults reproduce the pilot behaviour except where the
plan says otherwise (git tools are offered to the Repairer and B1; episode fault plans are applied by the runner).

| Item | Status | Where | Verified by |
|---|---|---|---|
| B1 git tools | Done; offered to Repairer and B1, withheld from A1; prompts `repair-v6`, `b1-v3` | `tools/git_tools.py`, `agents/tools.py`, `agents/prompts.py`, `conditions.py`, `schemas/contracts.py` | `tests/test_git_tools.py` (synthetic repo + **real boltons clone: the withheld S5 message of dev-01 is invisible**) |
| B2 scenario T | Done; on by default in `run_experiment` (`--no-tool-faults` to disable) and `run_episode.py` | `tools/faults.py`, `tools/trace.py`, `orchestrator.py`, `experiment.py` | `tests/test_faults.py` (incl. a patch fault that uses the revision and is marked `recovered`) |
| B3 mutants | Done: extended operators (top-up only), "unmeasurable" class + `GUARD_UNMEASURABLE_STRENGTH` (default `escalate`), `MUTANT_CACHE_PATH` | `tools/mutation.py`, `agents/integrity_guard.py`, `orchestrator.py` | `tests/test_final_features.py`; **real run on dev-06** (below) |
| B4 hybrid retrieval | Done; default stays `bm25` | `retrieval_hybrid.py`, `retrieval.py`, `scripts/eval_retrieval.py` | `tests/test_retrieval_hybrid.py`; dev measurement (below) |
| B5 Docker | Routing done (`SANDBOX_ISOLATION`, `SANDBOX_DOCKER_IMAGE`), image recipe `docker/pilot.Dockerfile`, Docker runs use the image's `python` | `tools/sandbox.py`, `tools/pytest_runner.py`, `tools/probes.py`, `tools/verification.py` | unit tests; **live Docker not run: daemon not reachable from this account** (`test_live_docker_sandbox_runs_a_test` skips) |
| B6 several libraries | Done: `--repos repos.json`, `--dataset`, per-repository memory streams, package-agnostic queries | `experiment.py`, `scripts/run_experiment.py`, `dataset.py`, `retrieval.py`, `agents/intent_analyst.py` | `tests/test_final_features.py` |
| C metrics | Done (table in the plan); new `FailureCause.POLICY_ESCALATION` | `evaluation.py`, `report_tables.py`, `scripts/make_report.py`, `scripts/citation_support.py` | `tests/test_final_metrics.py` against the saved Day-3/Day-4 runs |
| C statistics | Done | `stats.py`, `scripts/compare_conditions.py` | `tests/test_final_metrics.py` |

## Measurements made here (offline, no API calls, nothing under `results/` changed)

**Dev-06 strength gap (B3).** dev-06's CI state with the developer's reference test applied; mutants of
`CachedFunction.__call__` generated from that source and run in the sandbox against `tests/test_cacheutils.py`:

| Mutant | Operator | Outcome | Killed by the group's test (`test_unscoped_cached_dec`)? |
|---|---|---|---|
| mut-01 | `Add -> Sub` (old) | killed | no (3 other tests) |
| mut-02 | `return value -> None` (old) | survived | no |
| mut-03 | **negate IfExp condition** (new) | killed | **yes** |

Before, the reference test killed 0 of the 2 mutants, so FULL's Guard escalated a correct UPDATE_TEST. With the extended
operators a repair as strong as the developer's demonstrates strength without relaxing any Guard rule.

**Hybrid retrieval (B4)**, `scripts/eval_retrieval.py`, dev split, the 4 episodes with gold evidence (smoke, not a result):

| Mode | Intent recall@1 | @3 | @5 | @8 | Naive (B1-style) recall@8 |
|---|---|---|---|---|---|
| BM25 (pilot) | 0.38 | 0.62 | 0.62 | 0.62 | 0.00 |
| Hybrid, local MiniLM | 0.62 | 0.62 | 0.62 | 0.62 | 0.12 |
| Hybrid + local cross-encoder rerank | 0.00 | 0.62 | 0.62 | 0.62 | 0.12 |

The local model is a general-text model; the proposal's `voyage-code-3` needs a `VOYAGE_API_KEY` (none on this machine).
Reranking with the local cross-encoder hurt recall@1; do not enable it without a better reranker.

**New metrics on the saved runs** (frozen test split, `results/day4/pilot-test-5cond`), consistent with the Day-4 hand review:
unsafe actions taken effect 0/4 in every model condition; B1 attempted 1/4 (its masking edit on test-01, refused by the tools);
A1 over-generalised on the S4 decoy 1/1; patch-scope accuracy B1 2/4, FULL 3/3; failure causes are mostly
`policy_escalation` (correct hypothesis, escalated by the evidence policy or the Guard).

## How to use (final runs)

```bash
# several libraries: repos.json = {"boltons": {"path": "../boltons", "python": "../boltons-venv/bin/python"}, ...}
python scripts/run_experiment.py --dataset final --repos repos.json --conditions FULL,B1 --seeds 0,1 --out results/final/dev
python scripts/audit_integrity.py --runs results/final/dev/runs.jsonl --python ../boltons-venv/bin/python
python scripts/make_report.py results/final/dev/runs.jsonl          # main table + integrity/safety/memory/tool table
python scripts/compare_conditions.py results/final/dev/runs.jsonl  # bootstrap CIs, McNemar, seed variance, memory savings
python scripts/citation_support.py export results/final/dev/runs.jsonl --out citations.csv   # then label, judge, score
# hybrid retrieval: RETRIEVAL_MODE=hybrid EMBEDDING_BACKEND=voyage VOYAGE_API_KEY=...   (or EMBEDDING_BACKEND=local)
# Docker: docker build -f docker/pilot.Dockerfile -t testorcode/pilot-py312:pytest-8.4.2 .
#         SANDBOX_ISOLATION=docker SANDBOX_DOCKER_IMAGE=testorcode/pilot-py312:pytest-8.4.2
```

## Not done / for the team

- **Decisions** (settings exist, defaults unchanged): `GUARD_UNMEASURABLE_STRENGTH`, `RETRIEVAL_MODE`/backend, whether to pass
  `MUTANT_CACHE_PATH`. Freeze them on dev and record them with the final runs.
- **Docker** is untested live; build the image and run `pytest tests/test_final_features.py -k live_docker` on a machine with daemon access.
- **The mutant cache** `results/mutants/dev_cached_mutants.json` was built with the old operators; rebuild it
  (`scripts/build_mutant_cache.py`) before using it with the Guard or the audit, so they measure the same mutants.
- **Saved Day-1 to Day-4 runs** used the old prompts and tools; they are not comparable with runs made on this branch.
- Citation support needs human labels (45 citation rows on the dev pilot runs); the judge is optional and must be validated.
- `scripts/run_episode.py`'s summary still lists `boltons` in a git status call (cosmetic, single-library CLI).
