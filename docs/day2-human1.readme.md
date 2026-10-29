# Day 2 — Human 1 (agent core): what was built, how it works, and how to verify it

> **Historical (Day 2).** Statements below that A1 or A3 are not implemented were true at the time. Since then A1 (Human 1) and A3
> (Human 2) are implemented and all five conditions ran live on the dev split; see `docs/day3-human1.md`, `docs/day3-human2.md`,
> `docs/day3-human3.md`. The condition table (§ "Conditions") and the "Not done" row for A1/A3 are updated below.

This document covers **all of Human 1's Day-2 work** from the 5-day plan (`docs/TestOrCode_5-day_emergency_execution_plan.pdf`).
The design follows the final proposal (`docs/ProjectIdeaProposal.pdf`), with `docs/DetailedProjectIdea.pdf` as a
supplement; when they differ, the plan wins, then the proposal.

**Verification labels used throughout:**
- **UNIT:** fake components.
- **OFFLINE INTEGRATION:** the real orchestrator, real agents, real git, real pytest, real patch tools; **only the model's
  replies are scripted**.
- **LIVE:** a real Gemini call, saved under `results/`.

Nothing here is claimed live-verified unless a saved run shows it. **The current code has been run live** on dev-03
and dev-05, in both A2 and B1, on `gemini-3.5-flash` and `gemini-3.5-flash-lite`: 8 runs with a decision, recorded in
`results/day2/` (§10).

**Branch:** `agent-core`. All Day-2 work sits on top of `f84cc290` and is committed together after this review (§10).

| Slot | Plan task (Human 1) | Status |
|---|---|---|
| 09:00–11:00 | Complete orchestration and role prompts | **Done in code + offline** |
| 11:00–13:00 | Implement Triage and Analyst outputs | **Done in code + offline** (stability error handling corrected) |
| 13:00–15:00 | Connect Repairer and API tools | **Done for the Repairer** (code + offline). The Intent Analyst still retrieves in code; git tools are Human 2's |
| 15:00–18:00 | Build B1 single-agent baseline | **Done in code + offline + live** (dev-03, dev-05) |
| 18:00–20:00 | Integrate and fix failures | **Done offline and live**, including mixed failures, final status, audit corrections (§6.1) and multi-key live runs (§6.2) |

**Day-2 exit criteria involving Human 1:**

| Criterion | Status | Why |
|---|---|---|
| Full processes ≥1 real episode end to end | **Not met for FULL** | Needs Human 2's `MemoryStore`. A2 (Full minus memory) runs end to end **live** on real dev-03 and dev-05 with the current code |
| B1 runs on the same input | **Met (live)** | dev-03 and dev-05 with the same model as A2; on flash-lite, all four runs completed |
| BM25 retrieves real repository evidence | **Met** | Human 3's retriever; live citations of real `msg:*` ids |
| ≥6 labelled pilot episodes | **Met** | Human 3: 8 dev + 4 test |
| Results machine-readable | **Met** | `ExperimentRunRecord` JSON + `summary.json` |

---

## 1. How to verify (no API key needed)

```bash
cd TestOrCode
unset PYTHONPATH                                    # a ROS PYTHONPATH on this machine breaks direct pytest calls
myvenv/bin/python -m pytest -q                      # expect: 730 passed
myvenv/bin/python -m pytest -q tests/test_mixed_integration.py tests/test_pipeline_integration.py \
    tests/test_decision_policy.py tests/test_guard_sequences.py   # 52 real-orchestrator / real-tool scenarios
myvenv/bin/python -m pytest -q tests/test_prompt_leaks.py        # 23 reference-repair leak checks
myvenv/bin/python -m pytest -q tests/test_llm_client.py -k KeyPool   # API-key switching (no network)
myvenv/bin/python -m ruff check src/testorcode/agents src/testorcode/orchestrator.py src/testorcode/tools \
    src/testorcode/config.py src/testorcode/llm_client.py src/testorcode/schemas/contracts.py scripts tests
                                                    # expect: All checks passed
```

- **Repo-wide ruff** reports 15 findings, all in `src/testorcode/schemas/`. They existed before Day 2: the 5 in
  `result_schemas.py` are identical on `f84cc290`.
- **Real-episode offline check** (dev-03, A2 and B1): Appendix A. It needs `../boltons` and `../boltons-venv`
  (README §8a).
- **Live runs:** Appendix B (needs API keys in `.env`, §6.2).

---

## 2. 09:00–11:00 — Orchestration and role prompts

### 2.1 Prompts: `src/testorcode/agents/prompts.py`

| Role | Version | What it makes the model do | When it is called |
|---|---|---|---|
| Triage | `triage-v1` | Map each failing test to one of the listed *changed functions*, or "unknown" | Only for tests the rules can't map (§3.1) |
| Intent Analyst | `intent-v4` | Decision policy (§2.3), the stale-text warning, an explicit `alternative_explanation` | Once per group |
| Repairer | `repair-v5` | Edit only the allowed side. Test edits limited to expected literals and literal setup-call keywords. "Keep what the test is about", else escalate. Use rejection reasons. May call tools first | Tool loop, ≤4 model calls per attempt |
| Integrity Guard | `guard-v1` | Semantic review: does a changed test still check what its name and setup say? Can only *lower* the verdict | Only after every deterministic check passed, and only with a model |
| B1 single agent | `b1-v2` | Decide and repair per test by itself, with the same evidence rules (§5) | B1 condition only |

- **Shared rules** (`COMMON_RULES`): a passing build is not the goal; cite only listed ids; the proposal's §11.1 unsafe
  actions are forbidden.
- **Reply models:** every field is required, because Gemini dropped optional arrays.
- **Provenance:** `PROMPT_VERSION = triage-v1+intent-v4+repair-v5+guard-v1` (multi-agent) or `b1-v2` (B1) is written to
  `provenance.prompt_template_version`.
- **repair-v5** only replaces the examples of v2–v4, which came from pilot dev-05's reference repair (§6.1, M4).
  The examples are now synthetic. `tests/test_prompt_leaks.py` keeps every static prompt free of pilot repairs.

### 2.2 Orchestrator: `src/testorcode/orchestrator.py`

| Feature | Where | Behaviour |
|---|---|---|
| State machine | `EpisodeRun.execute`, `_handlers` | `TRIAGE → per group: INTENT_ANALYSIS → REPAIR → PATCH_VALIDATION → PATCH_APPLICATION → INTEGRITY_GUARD → DECIDED/ESCALATED`, then the final suite run, then `DONE`. Exceptions become an `ErrorRecord` with the matching `RunStatus`; API, tool and budget failures are never escalations |
| Conditions | `_condition_problem`, class attribute `CONDITIONS` | Default **A2**. `EpisodeRun` runs A2 and FULL (FULL needs a `MemoryStore`); `SingleAgentRun` runs B1 (Day 2; since Day 3 also A1 and A3 on `EpisodeRun`, and the trivial baselines in `run_episode`). Everything else (A1, A3; non-proposal B0/B2/A4/A5/trivial), or a condition sent to the wrong class, fails at stage `setup` |
| Evidence ids | `_intent_analysis`, `_patch_validation`, B1 `_decide` | Every cited id (analysis or patch) must have been retrieved this episode, or be a usable memory record |
| Decision policy | `contracts.decision_policy_problem` | §2.3 |
| Revision | `_rejected` | One revision. Pre-Guard rejections reach the Repairer as `rejection_reasons`, Guard rejections as `previous_rejection`. A second rejection escalates |
| Tool log and budget | `tools/trace.py` | Every tool call becomes a `ToolEvent` with the acting role and its budget category (`agent` / `operational` / `exempt`, §6.1 M1). Agent calls use `budget.max_tool_calls`; harness work uses a finite safety cap; reverts and the final suite run are exempt |
| Maintainer and memory hook | `_after_approval` | After a Guard **APPROVE** only: the harness maintainer decides, then `build_memory_record` (a **stub**, Human 2) → `authorize_memory_write` → `store.write`. Every attempt is logged; nothing is faked |
| Escalation summary | `_escalation_summary` | Evidence found and missing; the analyst's hypothesis and its `alternative_explanation` |
| Provenance | `_build_record`, `_code_revision` | Prompt version, git sha (`-dirty`), split, dataset version |

### 2.3 Decision policy (proposal §7)

The model is told the policy, and the orchestrator enforces it. A violation **escalates**; it is not a parse error.

| Decision | Requirement | Enforced by |
|---|---|---|
| UPDATE_TEST | Hypothesis `outdated_test`, plus ≥1 cited id of **positive intent evidence** | `IntentAnalysis` validator (existing) |
| FIX_CODE | Hypothesis `regression`, plus ≥1 cited id of the **expected behaviour the change breaks**: a docstring, doc, changelog entry, a CI commit message about something else, or a memory record whose probe passed. "Nothing says the change was intended" is **not** enough | `decision_policy_problem` (new) → ESCALATE `insufficient_evidence` |
| Conflict | The same id cited both for and against | `decision_policy_problem` → ESCALATE `contradictory_evidence` |
| Memory | A record counts only if its probe passed on the current code | `MemoryStore.read(...).usable` (the store itself is Human 2's) |

- **Stale text:** the prompt says neither the existing test nor a docstring is automatically right, and that old
  descriptions count as FIX_CODE evidence only when nothing in or after the change says the behaviour was meant to change.
- **Limit:** a cited id is checked to **exist**, not to support the claim (§9).

---

## 3. 11:00–13:00 — Triage and Analyst outputs

### 3.1 Triage: `agents/triage.py`

**Stability** (`stable_failures`, shared with B1). The CI failures run `context.stability_runs` times; 5 for the pilot,
from `environment.stability_runs`.

| Outcome over the runs | Result |
|---|---|
| FAILED in every run | stable (`stable=True`), grouped |
| PASSED in any run | unstable (`unstable_test_ids`), dropped |
| **ERROR / COLLECTION_ERROR in any run, including the baseline** | **`ToolError`** → run status `tool_failure`, error at `triage` |
| **TIMEOUT in any run** | **`ToolTimeoutError`** → run status `tool_failure` |

**Mapping tests to functions:**
1. `target_function` (`None` in all pilot data);
2. else the deepest traceback frame inside a function the CI diff changed;
3. else one model call (`TriageGrouping`).

A test with no function found gets **its own group** (`mapped_by="unmapped"`). Tests are no longer merged by test file,
because one file can hold an outdated test *and* a regression.

**Recorded:** `ExperimentRunRecord.failure_groups` (group, function, tests, `mapped_by`, stability runs) and
`unstable_test_ids`. These are additive schema fields; all 10 earlier saved records still load.

### 3.2 Analyst outputs

- `IntentAnalysis.alternative_explanation` (optional), used in the escalation summary.
- `retrieved_evidence[].support` records the **agent's claimed reading**: `supports` for cited ids, `contradicts` for
  contradicting ids, `unknown` otherwise. It is set at run level, so an id cited by one group and contradicted by another
  stays `unknown`. **It is not proof that an item supports a claim.** Citation faithfulness is Human 3's evaluation.

---

## 4. 13:00–15:00 — Repairer and API tools

**Model-callable tools** (`agents/tools.py`, `AgentTools`). Every call is checked against `allowed_tools`, recorded,
and counted against the budget.

| Tool | Behaviour | Safety |
|---|---|---|
| `search_repo(query)` | BM25 over the episode corpus, which ends at the CI commit; results are **citable ids** | Non-empty query |
| `read_file(path, start, end)` | Numbered lines, ≤200 per call | Relative path; no `..`, `.git`, `.env*`; **symlinks are checked by target** (no `.git`/`.env*` target, nothing outside the worktree) |
| `run_tests(selectors)` | Sandboxed pytest on the current, *unpatched* code | Selectors under the test roots only |

- **Refused or invalid calls** return text to the model and are logged as `not_permitted` / `invalid_arguments`.
- **Nothing here writes to the worktree.** Applying a patch is the orchestrator's job.

**Tool loop** (`agents/tool_loop.py`):
- **Steps:** each reply is one validated JSON step, either a tool call or a final action.
- **Bounded:** the last step must be final; a tool call on the last step ends the loop and the caller escalates.
- **Malformed replies:** the client re-asks once, then the run is `incomplete` with a validation error. That is not an
  escalation, and nothing is applied.

**Repairer:** `propose_repair(..., tools=)`. The orchestrator always passes the tools; the Repairer gets ≤4 model calls
per attempt, plus one re-ask if its edits can't be placed.

**Not done here:** `git_log/show/blame` (`tools/git_tools.py`, Human 2) and an Intent-Analyst tool loop (§8).

---

## 5. 15:00–18:00 — B1 single-agent baseline

**Proposal §10:** "a single tool-using agent with the same tools and standard top-k RAG over the repository, no
persistent memory", with the same model and budgets.

| | A2 (multi-agent, no memory) | B1 |
|---|---|---|
| Model, client, episode budgets | Same `create_llm_client`; same `_MeteredLLM` (LLM calls, tokens) and `ToolRecorder` (tool calls) | Same |
| Retrieval | 3 intent queries (function, changed identifiers, commit subject) | **Standard RAG:** `naive_query` (failing-test text) → `search_repo` |
| Tools | Repairer: `search_repo`, `read_file`, `run_tests` | Same three |
| Model-call caps | Per group: Intent 1 + Repairer ≤4 (+1 placement re-ask) + Guard review ≤1 | **6 per failing test** asked about (`TOOL_STEPS_PER_TEST`) + 1 placement re-ask per group and revision |
| Decision rules | §2.3 | **The same checks** (`IntentAnalysis` contract + `decision_policy_problem`) |
| Grouping | Triage | The agent's own `B1Decision` items; uncovered tests escalate |
| Patch rules | Shared `validate_patch` / `patch_content` | Same |
| Verification | `verify_group` + Guard checks + probe | `verify_group` only (**no Guard**: `guard_ran=False`) |
| Revisions | 1 | 1 |
| Final state | Final suite run | Final suite run |

**Fairness work done:**
- **Tool budget.** Both conditions charge only the agent's own calls to `max_tool_calls`. Harness verification is
  operational, and A2's is larger (Guard regression runs), so it no longer takes A2's agent budget (§6.1 M1).
- **Step cap.** B1's flat 6-call cap for the whole episode was raised to 6 per failing test.
- **Placement re-ask.** B1 now gets the same re-ask as the Repairer instead of losing its revision.
- **Budget accounting.** Both are checked to count every model call against the shared budget. Budget exhaustion is
  `budget_exhausted`, never a decision.

**Known unfairness that remains (§9):**
- A2 cannot reach APPROVE, because there are no trusted probes, while B1 accepts a verified rerun. A2-vs-B1 **outcome**
  metrics are not comparable until the Guard design (§7) is completed.
- Compare **direction** instead, using `predictions[].hypothesis`, which is kept even when A2's final action is ESCALATE.

---

## 6. 18:00–20:00 — Integration, mixed failures, final status

**Group verification** (`integrity_guard.verify_group`, used by both the Guard and B1):
1. **Targeted:** the group's own failing tests must pass.
2. **Regression:** the whole test suite (the policy's test roots) is run. Every failure must be one of the group's own
   tests or a `known_failures` test: another group's CI failure that is still unresolved. Groups whose repair was already
   accepted are **not** known failures, so a later repair that breaks them is rejected.

This replaced a confirmed defect: verification used to rerun whole test files, so two correct repairs whose tests shared
a file rejected each other.

**Final state:**
- After all groups, the orchestrator runs the whole suite once (`TestRunPhase.FINAL`, `run_id="final-suite"`). It is not
  counted against the tool budget, like reverts.
- `final_test_status` is **that run's** status, not the last group's rerun.
- ERROR / COLLECTION_ERROR / TIMEOUT stay themselves, and the run becomes `tool_failure` with a `final_suite` error. A run
  that cannot start is NOT_RUN, also `tool_failure` (§6.1 M2).
- If an accepted group's tests fail in the final suite, the group is marked `failed_final_verification` and the run is
  `incomplete` (§6.1 M3).
- Each group's verification stays in `test_runs` as `<patch>-post` and `<patch>-regression`.

**Offline integration scenarios:**

| File | Scenario | Checked outcome |
|---|---|---|
| `test_mixed_integration.py` (×2: same file / different files) | B1 repairs a FIX_CODE and an UPDATE_TEST group | Both decided per test; both patches applied; final suite PASSED |
| | A2: probed FIX group while the other group stays failing | FIX **approved** despite the other failure; UPDATE group ESCALATE (no probe) and rolled back; last group passed but **final FAILED** (the earlier failure remains); direction kept in `hypothesis` |
| | A repair that breaks a previously passing test | Rejected ("previously passing tests now fail: …test_shout"), reverted, the reason reaches the revision, the correct repair completes |
| | A later repair that breaks an accepted group | Rejected; the accepted state is kept; final FAILED |
| | B1 step cap with 2 tests | 7 tool calls + finish within 12 steps; `usage.llm_calls == 8` |
| | B1 unplaceable edit | One re-ask; revision not used |
| | B1 FIX_CODE without evidence | ESCALATE |
| | Episode LLM budget (A2 and B1) | `budget_exhausted` after exactly 2 calls; no prediction; worktree clean |
| | Malformed tool-loop reply | `incomplete` + validation error; no escalation; nothing applied |
| `test_pipeline_integration.py` (10) | Single-test A2/B1 paths | APPROVE with a probe; Guard REJECT → revision → ESCALATE; weakening blocked; tool use; refused tool; B1 paths; A2 vs B1 record shape |
| `test_decision_policy.py` (10) | Real pilot corpora (dev-03, dev-08) with the real Analyst and orchestrator | dev-03 FIX_CODE citing its retrieved commit message reaches repair; dev-08 FIX_CODE without evidence escalates; unsupported and contradictory citations escalate |

`test_guard_sequences.py` (12) covers several groups judged one after another by the **real Guard**; see §6.1.

**Real-episode offline run** (dev-03, real boltons, bwrap sandbox, pytest 8.4.2, scripted model; Appendix A):

| | A2 | B1 |
|---|---|---|
| Stability / grouping | 5 runs; `strutils.strip_ansi` via traceback | 5 runs; the agent's item |
| Group verification | targeted 1/1 passed; regression 441 passed | targeted 1/1; regression 441 passed |
| Decision | Guard ESCALATE (no trusted probe) → repair reverted | FIX_CODE, repair kept |
| **Final suite** | **FAILED (440 passed / 1 failed)**: the true end state after the revert | **PASSED (441 / 0)** |
| Model calls / tool events | 3 / 14 | 1 / 11 |

### 6.1 Final corrections (audit findings H1, H2, M1–M4, M6)

All are verified offline with real git, pytest and patch tools; only the model replies are scripted.

| Id | Defect | Fix | Tests |
|---|---|---|---|
| **H1** | The Guard compared the test files with **HEAD**. A repair already accepted for an earlier group therefore looked like part of the current patch. A correct later FIX_CODE was rejected ("modified test files"), and a later UPDATE_TEST in the same file was rejected ("outside this failure group"). | `verify_repair` rebuilds the state **just before this patch** by reversing only this patch's diff in a scratch copy (`git apply -R --recount`). Assertion counts, changed test functions and the FIX_CODE "tests untouched" check compare that state with the current one. Unchanged: the approval policy, a later patch that edits an earlier group's test is still rejected, and a rollback undoes only the current patch. | `test_guard_sequences.py`: update→fix, fix→update, two updates in one file, groups in different files (both orders); a later patch editing an approved group's test is rejected; a rejected later fix rolls back only itself. Reproduced on the old code first: 3 of the 5 orders were rejected. |
| **H2** | The default Guard ran the target tests with the harness interpreter and a 120 s timeout. Only the CLI wired `TEST_PYTHON` in, so direct `run_episode` calls differed from the CLI. | The orchestrator passes `python_executable=settings.test_python` and `timeout_seconds=settings.test_timeout_seconds` to every Guard call, default or injected. The CLI no longer builds its own Guard. | `test_configured_interpreter_and_timeout_reach_every_run` (default and injected Guard). A non-default interpreter wrapper and a 37 s timeout reach every `run_tests` call in Triage, the Guard and the final suite. |
| **M1** | One counter served the agents and the harness. Fixed verification overhead (5 stability runs, plus validate, apply, targeted and regression runs per group) used up `max_tool_calls`, so a multi-group A2 episode starved its agents sooner than B1 did. | `ToolEvent.arguments["budget"]`: `agent` (Intent Analysis, Repairer and B1-decision stages; limited by `max_tool_calls`), `operational` (Triage, patch handling, verification, memory write; finite cap = 3 × `max_tool_calls`, or 200), `exempt` (reverts, final suite). Every real call is still logged. The budget numbers are unchanged. | A 2-group A2 episode needs 8 agent + 13 operational + 1 exempt calls and completes with `max_tool_calls=8`; the old counter ran out at call 9. Agent exhaustion still gives `budget_exhausted` with a clean worktree. The operational cap is enforced. B1 has the same categories. |
| **M2** | A final suite with ERROR / COLLECTION_ERROR / TIMEOUT, or one that could not start, still left the run `completed`. | The status is kept; a `final_suite` error record (`tool_error` / `timeout`) is added; the run becomes `tool_failure`. Decisions are kept. | `test_final_suite_that_does_not_execute_the_tests_is_a_tool_failure` (×3), `test_final_suite_that_cannot_start_is_not_run_with_an_error` |
| **M3** | An accepted group whose tests failed again in the final suite was still reported as success. | The group gets `disposition="failed_final_verification"` and `final_failing_test_ids`. A `final_suite` `validation_error` names the groups and tests, and the run becomes `incomplete`. History is kept. | `test_accepted_group_failing_in_the_final_suite_is_flagged_not_completed` (2 groups: one accepted, one escalated) |
| **M4** | `repair-v2`…`v4` contained dev-05's reference repair (`cached(lru)` → `cached(lru, scoped=False)`) and a hint naming its test ('unscoped'). | `repair-v5` uses synthetic examples (`connect(host)` → `connect(host, retries=0)`; a test named 'case_insensitive_match'). Affected run records are marked in `results/day2/NOTES.md` and kept. Reference repairs are unchanged. | `test_prompt_leaks.py` (23): no static prompt shares a 20-character substring with any of the 12 pilot reference repairs, names a pilot failing test, or contains `unscoped` / `scoped=False` / `cached(lru`. No agent view contains a reference repair's added test lines (154 checked). The detectors flag the old v4 text. |
| **M6** | Only the last Guard verdict of the run was recorded. | `FailureGroupRecord` gained optional fields: `decision`, `guard_decision`, `guard_reasons` (every verdict), `integrity`, `patch_ids`, `verification_run_ids`, `disposition` (`accepted` / `failed_final_verification` / `reverted` / `not_applied`), `final_failing_test_ids`. All 10 earlier records still load. | `test_failure_group_records_keep_every_guard_verdict`, plus the H1 and M3 tests |

**For Human 3:** per-group results are in `failure_groups[]`; field definitions are in `docs/interfaces.md` §8. The
shared `ExperimentRunRecord.integrity` is unchanged and still holds only the run's last verdict.

### 6.2 Live runs with several API keys (`config.py`, `llm_client.py`)

Free-tier Gemini keys allow 20 requests per day per Google Cloud project for `gemini-3.5-flash`, and about 500 per day
(15 per minute) for `gemini-3.5-flash-lite`. To finish live runs within one day, the client can use several keys.

- **Configuration:** `.env` rows `LLM_API_KEY_1` … `LLM_API_KEY_9` (`.env.example` shows 1–3, blank). They are used in
  that order, then the old `LLM_API_KEY` as a last fallback; duplicates and blank rows are ignored. Each key must come
  from a **different Google Cloud project**, because the quota belongs to the project.
- **Switching:** a 429 whose body names a per-day quota (Gemini quota id `...PerDay...`) switches to the next key at
  once. Rate limits that outlast the retries also switch. A per-minute 429 is retried on the same key with backoff. The
  new key gets the full retry allowance.
- **All keys used up:** the call fails as `LLMRateLimitError`, so the run is `api_failure` (not counted against the
  agent), never a decision.
- **Secrets:** logs name key positions only ("API key 1/4 is out of quota; switching to key 2/4"); every key is
  `SecretStr` and in `secret_values()` for log redaction; the provider's error body is inspected but never logged.
- **Limitation:** each CLI process starts again at key 1, so once key 1 is used up, each run spends one rejected request
  before switching.
- **Tests (offline):** `TestKeyPool` in `tests/test_llm_client.py` (5) and `test_numbered_keys_are_used_in_order_then_the_single_key`
  in `tests/test_config.py`.
- **Seen live:** in the dev-05 B1 runs on 2026-10-28, keys 1 and 2 ran out and the client moved on to keys 2 and 3.

---

## 7. Guard approval: design handoff to Human 2 / Human 3 (not implemented by Human 1)

**Current behaviour, kept on purpose:**
- `verify_repair` runs `verify_group` and the deterministic checks, then `verification.trusted_outcome` (Human 2's).
- `trusted_outcome` returns APPROVE **only** with a passing harness `TrustedProbe`. No pilot episode supplies one: all
  `mutation_fixtures` are empty, and dev-06's memory `ExecutableProbe` is a different type.
- So **no real repair is approved today.** The maintainer and memory-write path runs only in tests where the harness
  supplies a probe.
- Nothing approves a test repair on a green build alone.

**The conflict:**
- The proposal's order is **Repair → independent integrity verification → simulated maintainer approval → memory
  record with an executable probe derived from the approved patch**.
- Requiring a probe *before* approval is circular for memory-derived probes, and is not in the proposal's Guard
  (§7 step 5).

**Recommended, proposal-aligned design (needs Human 2's agreement):**

| Repair | Approve when | Missing piece |
|---|---|---|
| FIX_CODE | Test files untouched (already checked) + `verify_group` passes (the group's tests pass; the whole suite has no new failures; the tests are the original oracle) + deterministic checks pass; a trusted probe, if supplied, must also pass | Only the change to the rule in `trusted_outcome` / `verify_repair` |
| UPDATE_TEST | All of the above + cited positive intent evidence + **mutation-based test strength**: the updated tests kill at least as many cached mutants as the reference tests (proposal Table 4, mutant-kill retention) | Human 2's cached-mutant runner, plus Human 3's mutation fixtures and reference tests. **Until then: ESCALATE, never APPROVE** |

- **After APPROVE:** the simulated maintainer runs (harness, gold-based, already wired). Then Human 2's
  `build_memory_record` derives the probe from the approved patch's assertions; `authorize_memory_write`, then
  `MemoryStore.write`.

**What the orchestrator needs from Human 2:**
1. A decision on the approval rule above, and the matching change to `verification.trusted_outcome` (their module).
2. A mutant runner callable, e.g. `run_mutants(worktree, group, *, reference_tests) -> MutationResult`. It fills
   `IntegrityReport.mutation`, and Human 1's Guard will call it for UPDATE_TEST.
3. `MemoryStore` (read/write) and `build_memory_record(context, group, analysis, proposal, result) -> MemoryRecord`
   with an `ExecutableProbe`.

**Human 3:** mutation fixtures / reference tests per episode, and how metrics score "escalated, although the repair
was correct".

**Missing fixtures or probes:** UPDATE_TEST stays ESCALATE, with the reason recorded in `evidence_missing`. No APPROVE
is ever fabricated.

**Tool-safety rules vs Guard checks (affects A3):**
- **Tool safety** (`patch_tools` + `patch_content`, every condition, including B1 and a future A3): path scope, protected
  files, no mixed test/source patches, and test patches limited to expected literals and literal setup keywords.
- **Guard checks** (Full/A2 only): assertion count, skip/xfail markers, edits outside the group, test-meaning review,
  and (planned) mutant-kill retention.
- **Consequence:** test *weakening* (removed assertions, skips, loosened comparisons) is impossible in **every**
  condition, so the test-weakening rate is 0 by construction. A3 ("no Integrity Guard") can still differ through
  masked regressions (a wrong expected literal), mutant-kill retention, and meaning changes, but not through weakening.
- **Open team decision (Humans 1, 2, 3):** keep the content rule at tool level (safer) and redefine A3's expected effect,
  **or** move it into the Guard so B1/A3 can weaken and be measured. This was **not changed**; safety was not weakened.

---

## 8. Remaining work and ownership

| Item | Status | Reason | Owner | Blocks Human 1 Day 2? |
|---|---|---|---|---|
| Live runs of the current code | **Done** for dev-03 and dev-05, A2 + B1, on two models (§10). dev-07 not run live | — | Human 1 | No |
| Team choice of the experiment model | Open | flash: 20/day, overloaded (503) during the runs; flash-lite: 500/day, ran cleanly. Every condition must use the same model | Team | No |
| FULL condition end to end | Not possible | `MemoryStore`, `build_memory_record` missing | Human 2 | No; blocks the FULL exit criterion |
| Guard APPROVE on real episodes | Not possible | §7: approval-rule decision + mutant runner | Human 2 (+ Human 1 wiring after) | No; blocks outcome metrics |
| `git_log/show/blame` with history cutoff | Not done | `tools/git_tools.py` belongs to Human 2 | Human 2, then Human 1 exposes them as tools | No |
| Intent Analyst tool loop (proposal Table 2) | Not done | Retrieval is code-driven; the 13:00–15:00 task names only the Repairer | Human 1 (later) | No |
| CLI option to pass a `MemoryStore` | Not done | No store exists yet | Human 1, after Human 2 | No |
| Review of the widened test-patch rule (`patch_content.py`) | Pending | Changes Human 2's module | Human 2 | No |
| Metrics (direction F1 from `hypothesis`, correct-repair rate, citation faithfulness) | Not done | `evaluation.py` stub | Human 3 | No |
| A1 / A3 switches, sanity baselines | **Done on Day 3** | A1 Human 1, A3 Human 2, baselines Human 3 | Human 1 / 2 / 3, Day 3 | No |
| Response caching (proposal §13) | Not done | Not scheduled | Team | No |

---

## 9. Known limitations

1. **Approval.** A2 always ends in ESCALATE for repairs (§7). Outcome metrics vs B1 are not comparable; compare
   direction (`hypothesis`).
1a. **A2/B1 fairness, remaining limits.** The budgets are now split the same way for both conditions (M1). Differences
   that remain by design: A2 pays for a Guard (verification is operational, its review is a model call), B1 doesn't, and
   B1's step cap is per failing test while A2's is per group and role.  The operational cap (3 × `max_tool_calls`) is a
   safety stop, not a fairness lever; neither condition comes near it on the pilot episodes.
1b. **Contaminated live record.** `results/day2/dev-05-e310b38a0015` ran with `repair-v2`, whose hint named dev-05's
   test. It is kept, but marked (`results/day2/NOTES.md`) and must not count. dev-03/dev-07 Day-2 records used the same
   prompt version but stopped (HTTP 429) before the Repairer. Day-1 records are clean.
2. **Evidence semantics.** Ids are checked to exist, not to support the claim. A model could cite an irrelevant but
   retrieved docstring for FIX_CODE. The prompts forbid it; only evaluation can measure it.
3. **Regression scope.** The whole test suite runs per group verification, plus once at the end. That is cheap for
   boltons (~441 tests), but heavier repositories will cost more time per episode.
4. **Unstable tests count as known failures** during verification. They failed in CI, so a later failure of theirs is
   not blamed on the repair.
5. **Escalation trigger labels** are a fixed contract set. A Guard escalation is recorded as `insufficient_evidence`; a
   rejected revision, B1's included, as `guard_rejection_after_revision`.
6. **`naive_query` uses the first 3 failing tests** for B1's initial search; B1 can search more itself.
7. **Guard pre-patch state** is rebuilt by reversing the current diff. If that fails (the worktree no longer holds
   the patch as applied), the Guard raises a `ToolError` (`tool_failure`); it never guesses.
7a. **Worktrees share the full boltons history**, including future commits. No model-reachable path reads it: `read_file`
   refuses `.git` (also through symlinks), and there are no git tools yet.
8. **`trusted-host` isolation** is used only in tests on our own fixture repositories; real episodes run in bwrap.
9. **Live evidence is small.** Two episodes × two conditions × two models. It shows the pipeline working live; it is
   not a comparison of conditions or models.
10. **API-key switching** restarts at key 1 in every CLI process (§6.2).

---

## 10. Files changed (branch `agent-core`, on top of `f84cc290`)

**Modified (22):**
- `.env.example` (blank `LLM_API_KEY_1..3` rows), `.gitignore`
- `docs/interfaces.md`
- `scripts/{llm_smoke,run_episode}.py`
- `src/testorcode/agents/{integrity_guard,intent_analyst,repairer,triage}.py`
- `src/testorcode/config.py`, `src/testorcode/llm_client.py` (API-key pool, §6.2), `src/testorcode/orchestrator.py`
- `src/testorcode/schemas/{contracts,result_schemas}.py` (**additive only**)
- `src/testorcode/tools/{patch_content,patch_tools,pytest_runner}.py` (**Human 2's files**; changed in earlier Day-2
  rounds, not in the final corrections; the Guard has its own reverse-apply helper)
- `tests/{test_agents,test_config,test_llm_client,test_orchestrator,test_patch_tools}.py`

**New (13 paths):**
- `src/testorcode/agents/{prompts,tools,tool_loop,single_agent}.py`, `src/testorcode/tools/trace.py`
- `tests/{test_prompts,test_tools,test_pipeline_integration,test_mixed_integration,test_decision_policy}.py`
- `tests/{test_guard_sequences,test_prompt_leaks}.py` (final corrections)
- `docs/day2_orchestration.md` (historical log), `docs/day2-human1.readme.md` (this file)
- `results/day2/`: the live records (older `repair-v2` runs and the current-code `gemini-3.5-flash` runs), `NOTES.md`
  (the contamination notice), and `flash-lite/` (the `gemini-3.5-flash-lite` runs)

**Review needed before merging:**
- **Human 2:** the three tool files, especially the `patch_content` policy.
- **Human 3:** the schema additions:
  - `failure_groups` (including the per-group Guard fields of §6.1 M6), `unstable_test_ids`, `TestRunPhase.FINAL`;
  - `IntentAnalysis.alternative_explanation`, `GuardResult.regression_run`;
  - `EpisodeContext.stability_runs`, `mapped_by` values;
  - the A2 default.

**Local only:** `.env` (git-ignored) holds the API keys. No key is in any tracked file or run record.

**Test count:**
- **Over the day:** 551 (start of Day 2) → 643 (after 11:00–20:00) → 684 (first corrections) → 724 (final
  corrections, §6.1) → **730** (API-key pool, §6.2).
- **Per file:**

| File | Tests |
|---|---|
| `test_patch_tools.py` | 194 |
| `test_orchestrator.py` | 66 |
| `test_agents.py` | 33 |
| `test_mixed_integration.py` | 20 |
| `test_tools.py` | 17 |
| `test_prompts.py` | 16 |
| `test_pipeline_integration.py` | 10 |
| `test_decision_policy.py` | 10 |
| `test_guard_sequences.py` | 12 |
| `test_prompt_leaks.py` | 23 |
| `test_llm_client.py` | 39 (5 new: key pool) |
| `test_config.py` | 14 (1 new: key pool) |

**Live records:**

| Code | Records |
|---|---|
| Day 1 (prompts v1) | `results/day1/*`: dev-01/03/05 complete |
| Day 2, intent-v2/repair-v2 | `results/day2/*`: dev-05 complete but **contaminated** (`NOTES.md`); dev-03 and dev-07 `api_failure` (429) |
| **Day 2, current code** (`triage-v1+intent-v4+repair-v5+guard-v1` / `b1-v2`), 2026-10-28 | see "Live runs on the current code" below |

**Live runs on the current code** (gemini-3.5-flash, real boltons, bwrap; `results/day2/`):

| Episode | Cond. | Record | Status | Decision (gold) | Repair | Final suite | Model calls |
|---|---|---|---|---|---|---|---|
| dev-03 (S2) | A2 | `dev-03-193e8f2b9296` | escalated | ESCALATE, direction FIX_CODE (FIX_CODE) | the correct fix; Guard ESCALATE (no trusted probe) → reverted | failed 440/1 | 3 |
| dev-03 (S2) | B1 | `dev-03-356b4ebc96be` | completed | FIX_CODE (FIX_CODE) | the correct fix, kept | **passed 441/0** | 3 |
| dev-05 (S1) | A2 | `dev-05-9704ba9e49aa` | escalated | ESCALATE, direction UPDATE_TEST (UPDATE_TEST) | `cached(lru)` → `cached(lru, scoped=False)` (the developer's own fix, found **without** the old hint); Guard ESCALATE → reverted | failed 546/1 | 7 |
| dev-05 (S1) | B1 | `dev-05-86e06edaa242`, `-422c122c92a3`, `-0419c79d8012` | api_failure ×3 | — | — | — | Gemini HTTP 503 (overloaded) / connection errors outlasted 6 retries each time; recorded as provider outage, never as a decision. Kept as history; done on flash-lite instead |

- Before the runs, one test call per key confirmed that all three keys worked.
- Both A2 runs chose the right **direction** and produced the reference-equivalent repair. A2 ends in ESCALATE only
  because no trusted probe exists (§7). This is the known approval gap, not a reasoning error.
- The clean dev-05 A2 run replaces the contaminated `dev-05-e310b38a0015` (`NOTES.md`).
- Key switching worked live: keys 1 and 2 ran out of quota mid-run and the client moved on to the next key.
- dev-05 B1 on gemini-3.5-flash was not completed (overload); the full set was re-run on flash-lite instead (below).

**Live runs on `gemini-3.5-flash-lite`** (all four conditions on one model, so A2 and B1 are comparable; 15 RPM /
500 per day; `results/day2/flash-lite/`; started with `LLM_MODEL=gemini-3.5-flash-lite` set in the environment, so
`.env` is unchanged):

| Episode | Cond. | Record | Status | Decision (gold) | Repair | Final suite | Calls |
|---|---|---|---|---|---|---|---|
| dev-03 (S2) | A2 | `dev-03-344ca3dee28d` | escalated | ESCALATE, direction FIX_CODE (FIX_CODE) | the correct fix; Guard ESCALATE (no trusted probe) → reverted | failed 440/1 | 3 |
| dev-03 (S2) | B1 | `dev-03-34bd5d845f26` | escalated | ESCALATE, direction FIX_CODE (FIX_CODE) | none: chose FIX_CODE but cited no evidence → the decision policy escalated (`insufficient_evidence`) | failed 440/1 | 1 |
| dev-05 (S1) | A2 | `dev-05-31797fb62acf` | escalated | ESCALATE, direction UPDATE_TEST (UPDATE_TEST) | **meaning-changing**: `call_count == 0` → `== 1` in `test_unscoped_cached_dec`. **The Guard's semantic review flagged it** ("contradicting the test name and setup") → ESCALATE → reverted | failed 546/1 | 4 |
| dev-05 (S1) | B1 | `dev-05-911a25ecc5e6` | completed | UPDATE_TEST (UPDATE_TEST) | the developer's fix (`scoped=False`), kept | **passed 547/0** | 1 |

- **Direction:** correct in all 8 live runs on the current code, on both models and in both conditions.
- **Guard value, seen live:** flash-lite's A2 Repairer took the cheap literal change that turns the "unscoped" test into a
  scoped one. The Guard's review caught it, and nothing reached the worktree. B1 has no Guard, so it would have
  accepted such a change if it had made one; here it happened to choose the right setup change.
- **Decision-policy value, seen live:** flash-lite's B1 on dev-03 claimed FIX_CODE without citing evidence. The policy
  escalated instead of letting it patch.
- **Too few runs to compare conditions.** This shows the pipeline working live, not a result.

---

## Appendix A — real-episode offline check (dev-03, A2 and B1, scripted model)

Run from `TestOrCode/` with `unset PYTHONPATH`. It needs `../boltons` at `d31d821d` and `../boltons-venv`
(pytest 8.4.2). It writes nothing to `results/`, and the worktree is reset at the end.

```python
import importlib.util, os
from pathlib import Path
from testorcode.config import Settings
from testorcode.dataset import agent_view_path, load_agent_inputs
from testorcode.llm_client import FakeLLMClient
from testorcode.orchestrator import run_episode
from testorcode.schemas.result_schemas import AgentRole

spec = importlib.util.spec_from_file_location("m", "scripts/pilot/materialize_episode.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
ep = next(e for e in load_agent_inputs(agent_view_path("dev")) if e.episode_id == "dev-03")
wt = Path("../episode-worktrees/dev-03").absolute()
NO = {"query": "", "path": "", "start_line": 1, "end_line": 0, "selectors": []}
fix = {"path": "boltons/strutils.py", "old": "if target_type or target_type != type(cleaned):",
       "new": "if target_type and target_type != type(cleaned):"}
base = {"LLM_MODEL": "fake-model-1", "TEST_PYTHON": os.path.abspath("../boltons-venv/bin/python")}
for cond in ("A2", "B1"):
    m.materialize("../boltons", wt, ep.model_dump(mode="json"))
    s = Settings.from_env({**base, "EXPERIMENT_CONDITION": cond})
    if cond == "A2":  # FIX_CODE cites the retrieved CI commit message, which is about MultiReplace, not strip_ansi
        llm = FakeLLMClient({
            AgentRole.INTENT_ANALYST: [{"group_id": "g1", "hypothesis": {"kind": "regression", "explanation": "x"},
                "expected_behaviour": "x", "alternative_explanation": "x", "cited_evidence_ids": ["msg:bba729bf4539"],
                "contradicting_evidence_ids": [], "confidence": 0.8, "recommended_decision": "FIX_CODE"}],
            AgentRole.REPAIRER: [{"action": "patch", "edits": [fix], "rationale": "x", "evidence_ids": [], **NO}],
            AgentRole.INTEGRITY_GUARD: [{"verdict": "consistent", "test_meaning_preserved": True, "reasons": []}]})
    else:  # B1's standard-RAG search returns strip_ansi's docstring, which it cites
        tid = ep.ci_event.failing_tests[0].test_id
        llm = FakeLLMClient({AgentRole.SINGLE_AGENT: [{"action": "finish", "rationale": "x", **NO, "decisions": [
            {"test_ids": [tid], "decision": "FIX_CODE", "hypothesis": "regression", "explanation": "x",
             "cited_evidence_ids": ["docstring:strutils.strip_ansi"], "confidence": 0.8, "edits": [fix]}]}]})
    r = run_episode(ep, worktree_path=wt, settings=s, llm=llm)  # TEST_PYTHON reaches every run, the Guard's too
    print(cond, r.outcome.status.value, r.provenance.prompt_template_version,
          [(t.run_id.split("-", 1)[-1], t.phase.value, t.status.value, t.passed, t.failed) for t in r.test_runs],
          r.integrity.guard_decision.value, r.final_test_status.value)
m.materialize("../boltons", wt, ep.model_dump(mode="json"))
```

Expected:
- **A2:** `escalated`, `triage-v1+intent-v4+repair-v5+guard-v1`, runs `baseline failed` → `r0-post passed 1` →
  `r0-regression passed 441` → `suite final failed 440/1`, guard `escalate`, final `failed`. (The group was not
  accepted, so M3 does not apply; the outcome stays `escalated`.)
- **B1:** `completed`, `b1-v2`, the same runs but `suite final passed 441`, guard `not_run`, final `passed`.

## Appendix B — live runs (as run on 2026-10-28)

Fill the `LLM_API_KEY_n` rows in `.env` first (§6.2). Each run prepares a fresh worktree, and writes a record and a
`.summary.json` (failure groups, tool events, gold vs predicted).

```bash
unset PYTHONPATH
# gemini-3.5-flash (the .env model): 3-7 model calls per run, 20 requests/day per key
for e in dev-03 dev-05; do for c in A2 B1; do
  myvenv/bin/python scripts/run_episode.py --episode $e --repo ../boltons --condition $c
done; done
# gemini-3.5-flash-lite (500 requests/day): override the model for these runs only; .env stays unchanged
for e in dev-03 dev-05; do for c in A2 B1; do
  LLM_MODEL=gemini-3.5-flash-lite myvenv/bin/python scripts/run_episode.py --episode $e --repo ../boltons \
      --condition $c --out results/day2/flash-lite
done; done
```

Model calls on flash-lite: 1–4 per run, 9 for all four.
