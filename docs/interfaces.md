# TestOrCode shared interfaces

This is the contract for feature branches. Signatures below are what the stub modules in
`src/testorcode/` declare today. Change a signature only through a coordinated edit to this file,
the stub and its tests.

## 1. Layout

`schemas` is already a **package** with substantial reviewed models, so it stays one. Everything else is a
single module or a small package, as in the README.

```text
src/testorcode/
├── config.py            Settings.from_env()                       implemented
├── logging_config.py    configure_logging / get_logger            implemented
├── errors.py            exception hierarchy                       implemented
├── schemas/
│   ├── episode_schemas.py   dataset side (existing)               implemented
│   ├── result_schemas.py    run-log side (existing)               implemented
│   ├── contracts.py         agent/tool messages (new)             implemented
│   └── schemas.py           alias of episode_schemas (existing)
├── llm_client.py        LLMClient, OpenAICompatibleClient,Fake   implemented
├── orchestrator.py      run_episode (state machine, conditions)   implemented (see below)
├── retrieval.py         Retriever, build_retriever                implemented (BM25)
├── memory.py            MemoryStore, JsonlMemoryStore,
│                        record_from_repair                        implemented (Day 2, Human 2)
├── conditions.py        ConditionProfile, episode_for_condition    implemented
├── evaluation.py        evaluate_run, summarize                   implemented (Human 3)
├── integrity_audit.py   audit_run, summarize_audits (harness)     implemented (Day 3, Human 2)
├── baselines.py         trivial always-UPDATE_TEST/FIX_CODE      implemented (Human 3)
├── experiment.py        ExperimentConfig, run_experiment          implemented (Human 3; FULL memory wiring Day 2)
├── agents/              triage, intent_analyst, repairer,
│                        integrity_guard                           implemented (see below)
│   ├── prompts.py       four role prompts + B1 prompt, reply models implemented
│   ├── tools.py         model-callable search_repo/read_file/run_tests implemented
│   ├── tool_loop.py     bounded JSON-step tool loop               implemented
│   └── single_agent.py  B1 baseline agent                         implemented
└── tools/               pytest_runner, patch_tools, sandbox,
                         patch_content, verification, trace        implemented
                         weakening, probes, mutation               implemented (Day 2, Human 2)
                         git_tools (run_git helpers only;
                         git_log/show/blame)                       stub
```

`memory` imports `tools.probes`/`tools.git_tools`; no tool imports `memory`.
`integrity_audit` (harness side, reads ground truth like `evaluation`) imports `tools.mutation`/`tools.weakening`;
no agent, tool or the orchestrator imports it.

Import graph (no cycles): `schemas.*` → `errors` → `config` → `llm_client`, `retrieval`, `memory`, `tools.*` →
`agents.*` → `orchestrator`. `evaluation` depends only on `schemas`. `logging_config` depends on nothing in the package.

## 2. Schemas

Use the existing names; do not redefine them elsewhere.

| Planned name | Actual type | Module |
|---|---|---|
| `Decision` | `DecisionLabel` (`UPDATE_TEST`, `FIX_CODE`, `ESCALATE`) | `episode_schemas` |
| (which side is wrong) | `RepairDirection` (`UPDATE_TEST`, `FIX_CODE`; ESCALATE is an action, never a direction) | `episode_schemas` |
| `Hypothesis` kind | `HypothesisKind` (`outdated_test`, `regression`, `unknown`) | `episode_schemas` |
| `Hypothesis` | `Hypothesis(kind, explanation)` | `contracts` |
| `FailureGroup` | `FailureGroup(group_id, function_qualname, test_ids)` | `episode_schemas` |
| failure details | `FailingTest` (traceback, assertion diff, stability) | `episode_schemas` |
| group + context | `FailureGroupDetail(group, failing_tests, affected_files, context_commits, diff_context, mapped_by)` | `contracts` |
| `Evidence` (corpus) | `EvidenceItem(evidence_id, source_type, source_ref, commit_sha, line range, text)` | `episode_schemas` |
| `Evidence` (retrieved) | `RetrievedEvidence` (adds query, method, rank, score, support (the agent's **claimed** reading, not verified), freshness) | `result_schemas` |
| `TestRunResult` (executed) | `PytestExecution` (selectors, command, status, exit_code, timed_out, counts, failures) | `contracts` |
| `TestRunResult` (logged) | `TestRunResult` (written via `PytestExecution.to_record`) | `result_schemas` |
| `IntentAnalysis` | `IntentAnalysis` | `contracts` |
| `PatchProposal` | `PatchProposal`; refusal is `EscalationRequest`; union `RepairerOutput` | `contracts` |
| `PatchValidation` | `PatchValidation(valid, allowed, changed_paths, rejection_reasons)` | `contracts` |
| `GuardResult` | `GuardResult(decision, reasons, post_repair_run, integrity)` | `contracts` |
| `MemoryRecord` | `MemoryRecord` (function, behaviour signature, direction, probe, evidence ids, approved-at commit, `flagged_stale`, `supersedes`) | `episode_schemas` |
| `MemoryReadResult` | `MemoryReadResult(checks: MemoryRecordCheck[])`; `.usable` lists only valid records | `contracts` |
| write approval | `MemoryApproval` | `contracts` |
| `EpisodeContext` | `EpisodeContext` (built from `EpisodeInput`; `stability_runs` from `environment.stability_runs`) | `contracts` |
| `EpisodeResult` | `ExperimentRunRecord` (one JSONL line per run; additive `failure_groups: FailureGroupRecord[]`, `unstable_test_ids`) | `result_schemas` |
| `EvaluationResult` | `RunEvaluation` (per run), `ConditionSummary` (aggregate, provisional) | `result_schemas`, `evaluation` |

Rules enforced by validators (see the tests):

- **Agents never receive gold labels.** Pass `EpisodeInput`, built from `Episode.agent_payload()`; never `Episode`.
- **Evidence is traceable.** Models can write any string as an id. `IntentAnalysis` and `PatchProposal` carry ids only;
  the orchestrator must call `unknown_evidence_ids(cited, retrieved_ids)` and treat a non-empty result as a failed analysis.
- **UPDATE_TEST needs cited evidence** (`IntentAnalysis`, `PatchProposal`). A hypothesis restricts the decision:
  `outdated_test` → `UPDATE_TEST`/`ESCALATE`; `regression` → `FIX_CODE`/`ESCALATE`; `unknown` → `ESCALATE`.
- **Patch scope matches direction:** `UPDATE_TEST` ⇒ `PatchScope.TEST`, `FIX_CODE` ⇒ `PatchScope.SRC`.
- **Test outcomes are distinct:** `FAILED`, `ERROR`, `COLLECTION_ERROR`, `TIMEOUT`; unknown counts are `None`, never 0.
- **`GuardResult` is explicit:** `APPROVE`/`REJECT`/`ESCALATE` (never `NOT_RUN`); `APPROVE` requires a passing post-repair run,
  which is necessary but not sufficient.
- **Stale memory is not evidence:** a record counts only if its probe passed on the current code, it is not
  `flagged_stale`, and it has no blocking reason.

Pydantic cannot enforce sandboxing, path policy, memory-write authority or history cutoffs. Those are runtime duties of the
tools and orchestrator (`RUNTIME_ENFORCEMENT` in `episode_schemas.py` lists them).

## 3. Modules, signatures and ownership

All agents are plain functions; dependencies are passed in explicitly (no framework, no globals).

| Module | Signature | Owner branch |
|---|---|---|| `tools.patch_tools` | `validate_patch`, `apply_patch`, `revert_patch` | sandbox |
| `orchestrator` | `run_episode(episode: EpisodeInput, *, worktree_path, settings, llm, memory=None, retriever=None, agents=None, patch_tools=None, maintainer=None) -> ExperimentRunRecord` (`retriever`/`agents`/`patch_tools` are injections for tests; `maintainer(group, decision, guard_result) -> bool` is the harness's simulated maintainer). The single entry point for **every** condition: the trivial baselines return `baselines.trivial_record` (no model, worktree or memory; `llm` may be `None`) | agent-core |
| `llm_client` | `LLMClient.complete_structured(*, role, system, user, response_model) -> LLMResponse[T]`; `create_llm_client(settings)` | agent-core |
| `agents.triage` | `run_triage(context, ci_event, *, llm, settings) -> TriageResult` | agent-core |
| `agents.intent_analyst` | `analyse_intent(context, group, *, retriever, memory, llm) -> IntentAnalysis` (`retriever=None` = no intent retrieval, A1: no search, the prompt says retrieval is disabled; `memory=None` = no memory) | agent-core |
| `agents.repairer` | `propose_repair(context, group, analysis, *, llm, previous_rejection=None, rejection_reasons=(), tools=None) -> RepairerOutput` (`previous_rejection` = a Guard rejection; `rejection_reasons` = a validation/apply/direction rejection; `tools: AgentTools` enables the tool loop, at most 4 model calls per attempt) | agent-core |
| `agents.single_agent` | `run_single_agent(context, ci_event, tests, *, llm, tools, rejection_reasons=()) -> B1Step \| None` (B1; used by `orchestrator.SingleAgentRun`) | agent-core |
| `agents.tools` | `AgentTools(context, *, retriever, timeout_seconds, python_executable=None, top_k=8, code_only=False)`: `search_repo(query)`, `read_file(path, start_line, end_line)`, `run_tests(selectors)` → text for the model; permission/argument checks; every call is a `ToolEvent`. `code_only=True` (set by the orchestrator for conditions with no retrieval, i.e. A1) limits `read_file` to Python source under the source roots and files under the test roots; documentation/change notes (README, CHANGELOG, `.md`/`.rst`, `docs/`, anything outside the roots) are `NOT_PERMITTED`, checked on the path and on the symlink target | agent-core |
| `agents.integrity_guard` | `verify_repair(context, group, analysis, proposal, *, llm=None, python_executable=None, timeout_seconds=120, probe=None, known_failures=frozenset(), cached_mutants=None) -> GuardResult` (`cached_mutants`: function -> `MutantSpec`s from `tools.mutation.load_cached_mutants`, executed instead of generated ones); `verify_group(context, group, known_failures, *, timeout_seconds, python_executable=None) -> (targeted, regression, problems)` (shared with B1); `suite_selectors(context)`; `build_memory_record(context, group, analysis, proposal, result) -> MemoryRecord` (delegates to `memory.record_from_repair`: probe derived from the group's tests after the approved repair); `authorize_memory_write(result, record, *, maintainer_approved) -> MemoryApproval` | agent-core |
| `tools.pytest_runner` | `run_tests(worktree_path, selectors, *, timeout_seconds, python_executable=None, isolation="auto", docker_image=None) -> PytestExecution`; `run_pytest(...)` also returns bounded stdout/stderr and the isolation used. `auto` = Docker (image + reachable daemon) else bubblewrap else `ToolError`; `"trusted-host"` only for trusted fixtures | sandbox |
| `tools.git_tools` | `git_log`, `git_show`, `git_blame` (every call takes `cutoff_commit`) | sandbox |
| `tools.patch_tools` | `validate_patch`, `apply_patch(worktree_path, diff, *, policy, scope) -> PatchValidation`, `revert_patch(worktree_path, diff, *, policy=None, scope=None)` (only undoes the exact diff `apply_patch` applied in that worktree, with the recorded scope); `scope_for_decision(decision)`; `is_protected_path(path)` (config, CI, dependency, generated and backup files at any depth, rejected on top of `EditPolicy`; the Integrity Guard should reuse it). `tools.patch_content`: a test-only patch may change **only (a) expected literal values in existing `assert <expr> ==/!= <literal>` statements and (b) literal keyword arguments of setup calls in test-function bodies (added or changed, never removed; not inside asserts, decorators or pytest/mock/assert/skip-style helpers)** (AST comparison; no new/deleted/renamed/non-Python test files, no changed positional inputs); anything else is rejected and should be escalated. The Integrity Guard additionally rejects a test patch that changes any test function outside the failure group. A source patch may not newly add pytest imports, hooks, skips, hard exits, or test-run detection (`sys.modules` incl. aliases and `getattr`, pytest-named string literals except docstrings, `sys._getframe`/`gettrace`/`settrace`, `inspect.stack`/`currentframe`, `traceback.extract_stack`, frame attributes) | sandbox |
| `tools.verification` | `verify_behaviour(worktree_path, probe: TrustedProbe | None, *, timeout_seconds, policy=None, isolation="auto", docker_image=None) -> TrustedVerification`; `trusted_outcome(post_repair_run, verification) -> TrustedVerification`. A raw `PytestExecution.PASSED` is never approval: only `TrustedVerification.verified` (`ProbeStatus.PASSED` from a harness-supplied probe run in a fresh sandbox) may recommend `GuardDecision.APPROVE`; no/failed/unverifiable probe → `ESCALATE`/`REJECT`. `verify_behaviour` imports a package source root (`boltons/`) by its package name and a plain root (`src/`) as top-level modules (`import_base`). `repair_outcome(direction, post_repair_run, *, probe_verification=None, mutation=None, mutation_problems=(), mutation_gaps=())` is the Guard's last step: FIX_CODE is verified by the untouched original tests (`basis="original_tests"`, or `"probe"`); UPDATE_TEST needs measured mutant-kill strength (`basis="mutation"`); measured weakening → REJECT, unmeasured → ESCALATE; a supplied probe must pass. `trusted_outcome` is unchanged (probe-only) | sandbox |
| `tools.weakening` | `analyse_test_changes(before, after) -> WeakeningAnalysis(findings: TestWeakeningFindings, problems, changed_tests)`: AST, per test function (removed assertions, skip/xfail/importorskip/unittest skips, loosened comparisons, swallowed exceptions and broad expectations, deleted/disabled tests); an expected-value update is not weakening | sandbox |
| `tools.probes` | `run_executable_probe(worktree_path, probe: ExecutableProbe, *, policy, timeout_seconds, python_executable=None, isolation="auto", docker_image=None) -> TrustedVerification` (harness-held probe code against a sandboxed scratch copy of the **source roots only**; a missing name/import, collection error, skip-only or timeout is `ERROR`, never PASSED or FAILED); `probe_from_tests(module_text, qualnames)`, `support_code(module_text)` | sandbox |
| `tools.mutation` | `generate_mutants(source, file_path, function, *, module=None, limit=6, seed=7) -> list[MutantSpec]` (deterministic, memoised); `run_mutants(worktree_path, mutants, selectors, *, timeout_seconds, python_executable=None, isolation="auto", docker_image=None, overrides=None, deadline=None) -> MutationRun` (scratch copy; outcomes `killed`/`survived`/`timeout`/`error`/`invalid`/`not_run`, only `killed` is a kill); `mutation_check(context, group, proposal, pre_patch, *, timeout_seconds, ..., mutants=None, deadline_seconds=300) -> MutationCheck(result: MutationResult, problems, gaps, killed_by_file_after, killed_by_file_before)` (UPDATE_TEST: baseline = the same tests before the repair, never the reference repair, lost kills → problem, unproven strength → gap; FIX_CODE: oracle sensitivity, no group kill → gap; `result.killed_by_repaired_tests` = kills by the failure group's own tests; one `run_mutants` tool event); `load_cached_mutants(path) -> {function: [MutantSpec]}` (validated specifications only, never stored outcomes) | sandbox |
| `retrieval` | `build_retriever(corpus, worktree_path=None) -> Retriever`; `Retriever.search(query, *, top_k)`; implemented (BM25), see `docs/retrieval.md` | agent-core |
| `memory` | `MemoryStore.read(context, function_qualname) -> MemoryReadResult`; `MemoryStore.write(record, approval)`. `JsonlMemoryStore(path, *, python_executable=None, timeout_seconds=120, isolation="auto", docker_image=None, stale_after_failures=2, check_history=True, namespace=None)`: scoped to `(repo_id, function_qualname)`; every read re-runs each record's probe outside the store lock (PASSED → usable; FAILED → `probe_failed`, flagged stale after 2 consecutive failures; ERROR → `probe_not_run`); superseded, stale and not-yet-approved-at-this-commit (`out_of_scope`) records are blocked without running; writes need a matching `MemoryApproval` (record id and commit) and are atomic; only an approved UPDATE_TEST record supersedes the function's older records (FIX_CODE and seeded records supersede nothing); `seed(records, *, worktree_path)` is the harness-only path for `visible_memory` (before any agent acts). **Isolation contract:** one store per condition, seed and stream: `memory_path(base, condition, seed, stream)`, `namespace=memory_namespace(condition, seed, stream)` (a mismatching namespace raises `ConfigError`) | sandbox (5-day plan: Human 2) |
| `evaluation` | `evaluate_run(record, episode) -> RunEvaluation`; `summarize(records, episodes, *, dataset_version=None) -> list[ConditionSummary]`: records from another dataset revision (or naming tests the episode lacks) are set aside (`n_incompatible_records`, `incompatible_run_ids`), never scored; `direction_macro_f1`/`direction_accuracy` score the side blamed per failing test (an escalation by its hypothesis; gold-ESCALATE tests excluded; undecided = miss), `macro_f1`/`accuracy` the final action | evaluation |
| `integrity_audit` | `audit_run(record, episode, *, repo, materialize, python, timeout_seconds=120, isolation="auto", cached_mutants=None, max_mutants=6, scratch=None) -> RunAudit`: re-measures every **accepted** repair of a saved run after the run, the same way in every condition (A3/B1 have no Guard report): test patches → `analyse_test_changes` and mutant-kill retention (mutants of the labelled function killed by the repaired tests ∩ by the developer's `reference_repairs` tests / killed by the reference; `None` when the reference kills none); source patches → tests touched and oracle sensitivity. The CI state is rebuilt from the clone (`materialize`), run worktrees are never touched. Audit version 2: only killed/survived mutants are measured; per copy `measured_*`/`killed_*`/`unmeasured_*`; retention only over mutants measured in both copies (`retention_basis`, `retention_denominator`); `oracle_sensitive` is `False` only when every mutant was measured and none killed, else `None` if undetermined; `status` `measured`/`partial`/`not_measured`. `summarize_audits`, `markdown_table`; CLI `scripts/audit_integrity.py [--out-dir DIR] [--overwrite]` writes `integrity_audit.{jsonl,md}` (default next to `runs.jsonl`) and refuses with exit 2, writing nothing, on malformed/empty input, conflicting `run_id`s, unknown episodes or existing reports; exact duplicate lines are audited once and reported | sandbox (Human 2) |
| `report_tables` | `condition_rows(records, episodes, *, dataset_version, conditions=ORDER)`: one row per condition (episodes run, direction macro-F1, direction and final-action accuracy, correct-repair rate with raw counts, masked regressions; a condition with no scored run is `evaluated=False`, "not yet evaluated", never 0); `markdown_table`; `integrity_lines(audit_jsonl)` (per-condition audit counts and pooled mutant-kill retention as raw counts, "not measurable" where the reference kills no mutant); `write_plots(rows, out_dir, split="dev")` (PNG, evaluated conditions only). CLI `scripts/make_report.py <runs.jsonl>` writes `report.md` and PNGs next to the runs. `docs/report/build_report.py` builds `report.tex` and `report.pdf` from these tables | evaluation (Human 3) |
| `scripts/build_mutant_cache.py` | `python scripts/build_mutant_cache.py --python PY [--repo R] [--episodes ...] [--limit N] [--timeout S] [--isolation M] [--out FILE]` builds the mutant catalogue read by `tools.mutation.load_cached_mutants` (default `results/mutants/dev_cached_mutants.json`). Top level: `generator` (`seed`, `limit`, `tool`), `kills_measured_against`, `python` (absolute), `isolation`, `entries`. Each entry: `episode_id`, `function_qualname`, `commit`, `test_selector`, `measurement`, `run_status` (`null` if the function was not found), `run_reason`, `baseline_failing`, `counts`, `seconds`, `mutation_fixture` (`null` if not found), `outcomes` (`mutant_id`, `outcome`, `killed_by`, `detail`). `measurement`: `complete` = every mutant ran to killed/survived (a `counts` of zero kills here is a real measurement); `incomplete` = some mutants timed out, errored, were invalid or not run (`counts` still tally each outcome); `not_measured` = no usable baseline (run status not `passed`), no mutant generated, or function not found: `counts` is `null`, never zeros, and `run_reason` says why. A `null` count means "not measured", not 0. Exit 0 only if every entry is `complete`; then `--out` is replaced atomically (temporary file in the same directory) and `<out>.incomplete` removed. Otherwise exit 1, `--out` is left exactly as it was (or absent), and the rejected build is written to `<out>.incomplete` for inspection. Only the mutant specifications are ever reused; stored outcomes are not | sandbox (Human 2) |
| `experiment` | `run_experiment(config, *, run_one=None)`; `default_run_one(config, *, settings=None, llm_factory=None, materializer=None)`: episodes in stream order; FULL gets one `JsonlMemoryStore` per condition/seed/split under `<out>/memory/` (`memory_path`/`memory_namespace`), seeded with `visible_memory` before each episode; other conditions `memory=None`; `ExperimentConfig.retry_failed` (CLI `--retry-failed`, default off) re-runs `api_failure` runs and replaces their line atomically (`ExperimentResult.retried`), and is refused with `ValueError` for a memory condition when a later episode of the same condition/seed already wrote or seeded memory or the failed attempt wrote memory (`unsafe_retries`) | evaluation |

Boundaries:

- **LLM:** one `LLMClient`, built once by the orchestrator from `Settings`, shared by every agent. No agent imports a provider SDK.
  Budgets are enforced by the orchestrator from the usage each `LLMResponse` reports.
- **Tools** take the episode worktree, never the user's checkout. Git operations and retrieval must not return anything newer than the
  episode's current commit (`HistoryCutoffError`). Patch scope is decided by `EditPolicy.classify_path`, the single path rule shared by
  `patch_tools` and the Integrity Guard.
- **Repairer proposes, tools apply.** A `PatchProposal` never touches the worktree.
- **Evaluation** (and the harness-side `integrity_audit`) are the only modules that read `Episode.ground_truth`.

## 4. Agent communication flow

```text
EpisodeInput ──► orchestrator builds EpisodeContext, one LLMClient, Retriever, MemoryStore?
  Triage ──► TriageResult (FailureGroupDetail per stable group)
  for each group:
    Intent Analyst ──► IntentAnalysis ──(ESCALATE)──► EscalationSummary
          │ validate cited ids against retrieved/memory ids
    Repairer ──► PatchProposal | EscalationRequest
          │ patch_tools.validate_patch ──► PatchValidation (reject → treat as rejection)
          │ patch_tools.apply_patch, pytest_runner.run_tests
    Integrity Guard ──► GuardResult
          ├─ APPROVE ──► maintainer approval ──► authorize_memory_write ──► MemoryStore.write
          ├─ REJECT ───► one revision (Repairer with previous_rejection) ──► Guard again
          └─ REJECT again / ESCALATE ──► ESCALATE
orchestrator ──► ExperimentRunRecord (predictions, patches, test_runs, integrity, memory events, usage, errors)
```

The revision limit is `min(settings.max_repair_revisions, context.budget.max_revisions)`, at most 1.

## 5. Memory read/write boundary

- Reads return every stored record for a function with its probe status on the current code (`MemoryReadResult`).
  Use `.usable` as evidence; `.unusable` explains why a record was blocked.
- A write requires a `MemoryApproval`, which `MemoryApproval` itself refuses to construct unless the Guard decision is `APPROVE`
  and the maintainer approved. `MemoryStore.write` must raise `MemoryAuthorizationError` without a matching one.
- **Limitation:** this is a convention checked by the store, not a security boundary. Any Python code can construct a
  `MemoryApproval`. Keep the write path out of reach of every agent except the Integrity Guard's workflow, and log every attempt
  (`MemoryWriteAttempt`).

## 6. Errors

Exceptions derive from `TestOrCodeError` and carry the `ErrorCategory`/`RunStatus` the run record needs
(`error.to_error_record(error_id)`).

| Exception | Meaning |
|---|---|
| `ConfigError` | invalid or missing setting |
| `LLMError`, `LLMStructuredOutputError` | provider failure, unparsable reply |
| `LLMRateLimitError`, `LLMProviderOutageError` | logged, **not** scored against the agent |
| `ToolError`, `ToolTimeoutError`, `PatchError`, `ScopeViolationError`, `HistoryCutoffError` | tool failures |
| `BudgetExceededError` | a budget ran out (`RunStatus.BUDGET_EXHAUSTED`) |
| `MemoryAuthorizationError` | write without valid approval |

- **Escalation is a decision, not an error.** Return `ESCALATE`; do not raise.
- A failing test is a *result* (`PytestExecution.status`), not an exception. A patch that is bad is a `PatchValidation`, not a raise,
  until someone tries to `apply_patch` it.
- Never report an execution failure (tool, API, budget) as an escalation.

## 7. Configuration and logging

```python
from testorcode.config import Settings
from testorcode.logging_config import configure_logging, episode_logging, get_logger

settings = Settings.from_env(env_file=".env")        # explicit; nothing loads at import time
configure_logging(settings.log_level, log_file=settings.log_file, secrets=settings.secret_values())

logger = get_logger(__name__)                         # in every module
with episode_logging(episode.episode_id):             # orchestrator, per episode
    logger.info("triage found %d groups", n)
```

- `Settings` reads the variables listed in `.env.example`; real environment variables beat `.env`; blank means unset.
  No API key is needed to build it; `settings.require_llm()` checks before a live call.
- `configure_logging` configures only the `testorcode` logger, is safe to call repeatedly, writes to stderr (plus a file only if given),
  redacts secrets and truncates long messages. Never log prompts, source files, diffs or patches: log ids, sizes and counts.

## 8. Stable vs deferred

**Stable (branch against these):** all of `schemas/`, `errors.py`, `config.py`, `logging_config.py`, the function signatures in
section 3, and the `LLMClient`, `Retriever` and `MemoryStore` protocols.

**Deferred (raise `NotImplementedError`):** `tools.git_tools.git_log/git_show/git_blame`, and building a
retrieval corpus from a worktree; `ConditionSummary` is provisional.  (`memory.JsonlMemoryStore` and
`integrity_guard.build_memory_record` are implemented since Day 2, Human 2: see `docs/day2-human2.md`.)

**Agents.** Signatures are as in section 3.
- **Triage** runs the CI's failing tests `context.stability_runs` times (5 for the pilot; `triage.stable_failures`, shared
  with B1). PASSED and FAILED are the only test outcomes: a test is kept only if it failed in every run (`stable=True`),
  one that passes in some run is unstable. A run ending in ERROR / COLLECTION_ERROR / TIMEOUT raises `ToolError` /
  `ToolTimeoutError` (run status `tool_failure`), never "unstable". Tests are mapped to the changed function they exercise:
  `target_function`, else the deepest traceback frame inside a changed function, else one model call (`TriageGrouping`);
  a test with no function found gets its own group (`mapped_by="unmapped"`), never a shared one by test file.
- **Intent Analyst** makes one model call per group (`IntentAnalysis`, incl. `alternative_explanation`). Decision policy
  (proposal section 7), enforced by `contracts.decision_policy_problem` in the orchestrator (an escalation, not a parse
  error): UPDATE_TEST must cite positive intent evidence (contract validator); **FIX_CODE must cite the evidence of the
  expected behaviour the change breaks** (a docstring, doc, changelog entry, an unrelated CI commit message, a memory
  record) — the mere absence of intent evidence escalates; an id cited both for and against escalates as contradictory.
  Citations are checked to exist (`unknown_evidence_ids`), not to support the claim.
- **Repairer**: inside `run_episode` it gets `tools` and runs a bounded tool loop (`agents.tool_loop`, at most 4 model
  calls per attempt; each reply is one JSON step, a tool call or the final `patch`/`escalate`); running out of steps is an
  escalation. It returns snippet edits; the module builds the diff and re-asks once if they cannot be placed.
- **Integrity Guard**: test-level verification by `verify_group` — (1) the group's own failing tests must pass, (2) a
  whole-suite regression run (the policy's test roots) may fail only on `known_failures` (other groups' still-unresolved
  CI failures; groups already accepted are excluded, so they must keep passing). Then the deterministic checks (scope,
  `tools.weakening` findings, test functions outside the group, FIX_CODE touching tests, UPDATE_TEST without evidence) →
  REJECT; then an optional model review that can only lower the verdict to ESCALATE; then
  `tools.mutation.mutation_check`; then `verification.repair_outcome` (the design of `docs/day2-human1.readme.md` §7,
  implemented by Human 2): FIX_CODE → APPROVE on the untouched original tests when they detect mutants of the repaired
  function (otherwise ESCALATE), and source patches that detect the test run never reach the Guard
  (`patch_content`); UPDATE_TEST → APPROVE only with measured mutant-kill strength, REJECT on measured weakening,
  otherwise ESCALATE; a supplied `TrustedProbe` must also pass.
  `GuardResult.post_repair_run` is the targeted run, `GuardResult.regression_run` the suite run.  The content checks
  judge **only the current patch**: the pre-patch state is rebuilt by reversing this patch's diff on the worktree
  (`git apply -R --recount` in a scratch copy), so repairs of earlier, already accepted groups are part of both states
  and never blamed on a later group.  A later patch that edits an earlier group's test is still rejected (test function
  outside the group).  The orchestrator passes `python_executable=settings.test_python` and
  `timeout_seconds=settings.test_timeout_seconds` to every Guard call (default or injected).

**Prompts** (`agents/prompts.py`). Each role has `*_SYSTEM`, `build_*_user` and a version; `PROMPT_VERSION`
(`triage-v1+intent-v4+repair-v5+guard-v1`) goes into `RunProvenance.prompt_template_version` for the multi-agent
pipeline; B1 runs record `b1-v2` (`B1_SYSTEM`). All share `COMMON_RULES` (green build is not the goal, cite only listed
ids, proposal 11.1 unsafe actions). Every field of a reply model is required. Bump a role's version whenever its prompt
changes. Condition-dependent text (A1, Day 3) is not a version change: `tool_instructions(allowed_tools)` drops the
`search_repo` line only when the episode does not allow it, and `build_intent_user(..., retrieval=False)` says retrieval
is disabled; FULL/A2/B1 prompts are byte-identical to before.

**Orchestrator.**
- *Conditions:* `settings.experiment_condition` selects the episode view (`conditions.episode_for_condition`) and the
  run reads its `ConditionProfile` (`EpisodeRun.profile`): `memory` (otherwise `memory` is ignored; a profile with
  memory needs a `MemoryStore`) and `intent_retrieval`. Each run class executes only its own conditions:
  `EpisodeRun` → FULL, A1 and A3 (all need a `MemoryStore`) and A2 (the `Settings` default); `SingleAgentRun` → B1;
  `run_episode` itself answers the trivial baselines (`TRIVIAL_UPDATE_TEST`, `TRIVIAL_FIX_CODE`). Conditions without a
  code path (the non-proposal B0, B2, A4, A5), or a condition sent to the wrong class, fail explicitly as
  `ORCHESTRATOR_ERROR` at stage `setup`.
- *A3 (no Integrity Guard, Day 3):* `profile.integrity_guard=False` maps the `INTEGRITY_GUARD` stage to
  `EpisodeRun._verify_without_guard`: the same `verify_group` rerun as B1 (group tests pass, no new suite failures)
  decides, `decided_by=intent_analyst`. No weakening/scope-of-group/evidence/semantic review, no mutation check, no
  maintainer and no memory write (only the Guard writes memory; A3 still reads probe-gated memory).
  `integrity.guard_ran=False`; the verification runs are logged with role `orchestrator`. Tool-level patch safety
  (`patch_tools` + `patch_content`) is unchanged, so test weakening stays blocked in every condition; the post-hoc
  integrity audit (`integrity_audit`) measures A3/B1 repairs.
- *A1 (no intent retrieval, Day 3):* FULL minus every route to change-intent text. The episode view withholds the CI
  commit message, the corpus (`items=[]`) and `search_repo` (the pilot corpora contain the CI commit's own message as
  `msg:<sha>`, so withholding only the message would leak it); the Analyst is called with `retriever=None`; the
  Repairer's tool list omits `search_repo` (a call anyway is refused, `not_permitted`), and its `read_file` is
  `code_only` (README/CHANGELOG/docs are refused at runtime; source and test files stay readable). Memory, Triage, the Guard and
  the decision policy are unchanged, so A1 can decide UPDATE_TEST/FIX_CODE only when it can cite a usable memory
  record and otherwise escalates; its *direction* is still scored from `hypothesis`. The record's
  `retrieved_evidence` is empty (evaluation reports no recall for it).
- *B1* (`SingleAgentRun`): shared `stable_failures` baseline, one agent with standard RAG (`naive_query` through
  `search_repo`) and the same three tools, at most `TOOL_STEPS_PER_TEST` (6) model calls per failing test it is asked
  about; its own grouping (`B1Decision` items); the same evidence and decision-policy checks; one placement re-ask for
  edits that cannot be placed (like the Repairer); the same patch validation/application, `verify_group` verification
  and revision limit; no Guard (`integrity.guard_ran=False`, `decided_by=single_agent`).
- *Revisions:* a rejection before the Guard (unsupported evidence id, direction mismatch, validation, apply) reaches the
  Repairer as `rejection_reasons`; a Guard rejection as `previous_rejection`.
- *Evidence:* the analysis's cited/contradicting ids and the patch's `evidence_ids` must all be ids that were retrieved
  in this episode or usable memory records (`unknown_evidence_ids`); an unsupported analysis id escalates, an
  unsupported patch id rejects the patch.
- *Tools:* `tools.trace` records every tool call of the episode (`run_tests`, `read_file`, `apply_patch` with
  `operation=validate|apply|revert`, `search_repo`, `memory.read`, `memory.write`) as a `ToolEvent` with the acting role
  (refused calls as `not_permitted`, bad arguments as `invalid_arguments`; in B1 the agent's own calls are its
  searches/reads/test runs and its patch handling, while stability runs and verification reruns are the orchestrator's)
  and its budget category in
  `arguments["budget"]`: `agent` (calls made while Intent Analysis, the Repairer or B1's decision loop act; limited by
  `budget.max_tool_calls`), `operational` (Triage stability runs, patch validate/apply, Guard or B1 verification runs,
  memory write; limited by a finite safety cap, `OPERATIONAL_CAP_FACTOR` (3) × `max_tool_calls`, or 200 without a
  budget) and `exempt` (reverts and the one final suite run; never blocked). Either limit raises
  `BudgetExceededError` (`budget_exhausted`). `read_file` refuses `..`,
  absolute paths, `.git`, `.env*`, and symlinks whose target is one of those or lies outside the worktree.
- *Final state:* after every group, `PatchTools.run_suite` runs the whole suite once (`TestRunPhase.FINAL`,
  `run_id="final-suite"`); `final_test_status` is that run's status.  If it cannot start (NOT_RUN) or cannot execute the
  tests (ERROR / COLLECTION_ERROR / TIMEOUT, kept as the final status) the run is `tool_failure` with a non-fatal
  `final_suite` error (category `tool_error`/`timeout`).  If an accepted group's tests fail in it, that group's
  `disposition` is `failed_final_verification` with `final_failing_test_ids`, a `final_suite` `validation_error` names
  the groups and tests, and the run is `incomplete`; decisions and verification history are kept.  Group verifications
  stay in `test_runs` as `<patch>-post` (targeted) and `<patch>-regression`. `run_suite=None` (fake tools in unit
  tests) skips it.
- *Approval:* after a Guard APPROVE, `maintainer` decides; on approval the Guard's record is written through
  `authorize_memory_write`. Every attempt is logged in `memory_writes`; `outcome.maintainer_approved` is set.
- *Escalation summary:* `evidence_missing` and `competing_hypotheses` (the analyst's and its `alternative_explanation`,
  or a generic opposite) are filled.
- *Per-group record (for Human 3):* each `failure_groups[]` entry (`FailureGroupRecord`, additive, all new fields
  optional so older records load) has `decision`, `guard_decision` (last verdict), `guard_reasons` (all verdicts, in
  order), `integrity` (last `IntegrityReport`), `patch_ids` (`<group>-r<n>`), `verification_run_ids` (that group's
  `-post`/`-regression` runs), `disposition` (`accepted`, `failed_final_verification`, `reverted`, `not_applied`;
  `null` if the group did not finish) and `final_failing_test_ids`.  `ExperimentRunRecord.integrity` still holds only
  the last Guard verdict of the run.
- *Triage/Analyst outputs in the record:* `failure_groups`, `unstable_test_ids`; `retrieved_evidence[].support` is the
  agent's **claimed** reading (`supports` for cited ids, `contradicts` for contradicting ids; run-level, an id cited by
  one group and contradicted by another stays `unknown`). It is not evidence that the item supports the claim; citation
  faithfulness belongs to the evaluation harness.
- *Provenance:* `prompt_template_version` and `code_revision` (git sha, `-dirty` with uncommitted changes); the CLI adds
  `split` and `dataset_version`.

**Configuration.** `TEST_PYTHON` and `TEST_TIMEOUT_SECONDS` reach every test run of an episode (Triage, the Repairer's
`run_tests`, the Guard, the final suite), whether it is started by `run_episode` or the CLI. `LLM_API_KEY_1..9` are
used in order (then `LLM_API_KEY`): a 429 naming a per-day quota, or rate limits that outlast the retries, switch to
the next key; with every key exhausted the call is an `LLMRateLimitError` (API failure). A key whose daily quota ran out
is remembered process-wide for an hour (by hash), so the next client (the next episode of an experiment) starts at the
first key not known to be exhausted. Logs name key positions only; `LLM_STRUCTURED_MODE`,
`LLM_MAX_RETRIES`, `LLM_TIMEOUT_SECONDS`. `patch_tools` restores
the final newline that contract stripping removes, and the orchestrator reverts with the episode's `EditPolicy`.

## 9. Open decisions for the team

1. ~~`EXPERIMENT_CONDITION` default~~ Resolved: `A2` in `Settings` and `.env.example` (A2 and B1 run; FULL needs memory).
2. ~~Where agent prompts and the tool-call loop live~~ Resolved: all four role prompts are in `agents/prompts.py`;
   shared file helpers are in `agents/_common.py`; the tool loop is `agents/tool_loop.py` (JSON steps validated by
   Pydantic, no provider function calling). The Repairer and B1 use it; the Intent Analyst still retrieves in code.
3. Whether `Settings.log_file` should default to a per-run file under `results/`.
4. `ConditionSummary` shape (evaluation branch).
5. `requirements-lock.txt` is both tracked and listed in `.gitignore`; the README says to commit it.

## 10. Pilot dataset

`datasets/pilot_{dev,test}_agent_view.jsonl` hold `EpisodeInput` records for the system under test; `pilot_{dev,test}.jsonl` hold full
`Episode` records for the evaluation harness only. Load with `testorcode.dataset` (`load_agent_inputs`, `load_episodes`). Splits,
rules, the frozen test split and limitations: `datasets/README.md`.

`retrieval` and `conditions` are implemented: BM25 retrieval, the A1/A2/B1 switches, the `search_repo` tool wrapper and metric helpers (`docs/retrieval.md`). A1 runs end to end since Day 3 (`docs/day3-human1.md`).
