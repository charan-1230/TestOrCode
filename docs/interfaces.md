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
├── orchestrator.py      run_episode (state-machine scaffolding)   scaffolding
├── retrieval.py         Retriever, build_retriever                implemented (BM25)
├── memory.py            MemoryStore                               stub
├── conditions.py        ConditionProfile, episode_for_condition    implemented
├── evaluation.py        evaluate_run, summarize                   stub
├── agents/              triage, intent_analyst, repairer,
│                        integrity_guard                           stub
└── tools/               pytest_runner, patch_tools, sandbox,
                         patch_content, verification               implemented
                         git_tools (run_git helpers only;
                         git_log/show/blame)                       stub
```

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
| group + context | `FailureGroupDetail(group, failing_tests, affected_files, context_commits, diff_context)` | `contracts` |
| `Evidence` (corpus) | `EvidenceItem(evidence_id, source_type, source_ref, commit_sha, line range, text)` | `episode_schemas` |
| `Evidence` (retrieved) | `RetrievedEvidence` (adds query, method, rank, score, support, freshness) | `result_schemas` |
| `TestRunResult` (executed) | `PytestExecution` (selectors, command, status, exit_code, timed_out, counts, failures) | `contracts` |
| `TestRunResult` (logged) | `TestRunResult` (written via `PytestExecution.to_record`) | `result_schemas` |
| `IntentAnalysis` | `IntentAnalysis` | `contracts` |
| `PatchProposal` | `PatchProposal`; refusal is `EscalationRequest`; union `RepairerOutput` | `contracts` |
| `PatchValidation` | `PatchValidation(valid, allowed, changed_paths, rejection_reasons)` | `contracts` |
| `GuardResult` | `GuardResult(decision, reasons, post_repair_run, integrity)` | `contracts` |
| `MemoryRecord` | `MemoryRecord` (function, behaviour signature, direction, probe, evidence ids, approved-at commit, `flagged_stale`, `supersedes`) | `episode_schemas` |
| `MemoryReadResult` | `MemoryReadResult(checks: MemoryRecordCheck[])`; `.usable` lists only valid records | `contracts` |
| write approval | `MemoryApproval` | `contracts` |
| `EpisodeContext` | `EpisodeContext` (built from `EpisodeInput`) | `contracts` |
| `EpisodeResult` | `ExperimentRunRecord` (one JSONL line per run) | `result_schemas` |
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
| `orchestrator` | `run_episode(episode: EpisodeInput, *, worktree_path, settings, llm, memory=None, retriever=None, agents=None, patch_tools=None) -> ExperimentRunRecord` (the last three are optional injections for tests) | agent-core |
| `llm_client` | `LLMClient.complete_structured(*, role, system, user, response_model) -> LLMResponse[T]`; `create_llm_client(settings)` | agent-core |
| `agents.triage` | `run_triage(context, ci_event, *, llm, settings) -> TriageResult` | agent-core |
| `agents.intent_analyst` | `analyse_intent(context, group, *, retriever, memory, llm) -> IntentAnalysis` | agent-core |
| `agents.repairer` | `propose_repair(context, group, analysis, *, llm, previous_rejection=None) -> RepairerOutput` | agent-core |
| `agents.integrity_guard` | `verify_repair(context, group, analysis, proposal, *, llm=None) -> GuardResult`; `authorize_memory_write(result, record, *, maintainer_approved) -> MemoryApproval` | agent-core |
| `tools.pytest_runner` | `run_tests(worktree_path, selectors, *, timeout_seconds, python_executable=None, isolation="auto", docker_image=None) -> PytestExecution`; `run_pytest(...)` also returns bounded stdout/stderr and the isolation used. `auto` = Docker (image + reachable daemon) else bubblewrap else `ToolError`; `"trusted-host"` only for trusted fixtures | sandbox |
| `tools.git_tools` | `git_log`, `git_show`, `git_blame` (every call takes `cutoff_commit`) | sandbox |
| `tools.patch_tools` | `validate_patch`, `apply_patch(worktree_path, diff, *, policy, scope) -> PatchValidation`, `revert_patch(worktree_path, diff, *, policy=None, scope=None)` (only undoes the exact diff `apply_patch` applied in that worktree, with the recorded scope); `scope_for_decision(decision)`; `is_protected_path(path)` (config, CI, dependency, generated and backup files at any depth, rejected on top of `EditPolicy`; the Integrity Guard should reuse it). `tools.patch_content`: a test-only patch may change **only expected literal values in existing `assert <expr> ==/!= <literal>` statements** (AST comparison; no new/deleted/renamed/non-Python test files); anything else is rejected and should be escalated | sandbox |
| `tools.verification` | `verify_behaviour(worktree_path, probe: TrustedProbe | None, *, timeout_seconds, policy=None, isolation="auto", docker_image=None) -> TrustedVerification`; `trusted_outcome(post_repair_run, verification) -> TrustedVerification`. A raw `PytestExecution.PASSED` is never approval: only `TrustedVerification.verified` (`ProbeStatus.PASSED` from a harness-supplied probe run in a fresh sandbox) may recommend `GuardDecision.APPROVE`; no/failed/unverifiable probe → `ESCALATE`/`REJECT`. The Integrity Guard itself is not implemented | sandbox |
| `retrieval` | `build_retriever(corpus, worktree_path=None) -> Retriever`; `Retriever.search(query, *, top_k)`; implemented (BM25), see `docs/retrieval.md` | agent-core |
| `memory` | `MemoryStore.read(context, function_qualname) -> MemoryReadResult`; `MemoryStore.write(record, approval)` | agent-core |
| `evaluation` | `evaluate_run(record, episode) -> RunEvaluation`; `summarize(records, episodes) -> list[ConditionSummary]` | evaluation |

Boundaries:

- **LLM:** one `LLMClient`, built once by the orchestrator from `Settings`, shared by every agent. No agent imports a provider SDK.
  Budgets are enforced by the orchestrator from the usage each `LLMResponse` reports.
- **Tools** take the episode worktree, never the user's checkout. Git operations and retrieval must not return anything newer than the
  episode's current commit (`HistoryCutoffError`). Patch scope is decided by `EditPolicy.classify_path`, the single path rule shared by
  `patch_tools` and the Integrity Guard.
- **Repairer proposes, tools apply.** A `PatchProposal` never touches the worktree.
- **Evaluation** is the only module that reads `Episode.ground_truth`.

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

**Deferred (raise `NotImplementedError`):** every body in `llm_client`, `orchestrator`, `memory`, `evaluation`,
`agents/*` and `tools/*`; `ConditionSummary` is provisional.

## 9. Open decisions for the team

1. `EXPERIMENT_CONDITION` defaults to `B1` to match `.env.example`; confirm that is the intended default.
2. Where agent prompts and the tool-call loop live (inside each agent, or a shared helper in `llm_client`).
3. Whether `Settings.log_file` should default to a per-run file under `results/`.
4. `ConditionSummary` shape (evaluation branch).
5. `requirements-lock.txt` is both tracked and listed in `.gitignore`; the README says to commit it.

## 10. Pilot dataset

`datasets/pilot_{dev,test}_agent_view.jsonl` hold `EpisodeInput` records for the system under test; `pilot_{dev,test}.jsonl` hold full
`Episode` records for the evaluation harness only. Load with `testorcode.dataset` (`load_agent_inputs`, `load_episodes`). Splits,
rules, the frozen test split and limitations: `datasets/README.md`.

`retrieval` and `conditions` are implemented: BM25 retrieval, the A1/A2/B1 switches, the `search_repo` tool wrapper and metric helpers (`docs/retrieval.md`).
