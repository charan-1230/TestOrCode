# Day 2, 09:00–11:00 (Human 1): orchestration and role prompts

> **Historical log, superseded.** This file records the 09:00–11:00 slot as it was built: prompt versions
> (intent-v2/repair-v2, then repair-v3), test counts (592, then 611) and the dev-05 live run belong to that moment.
> The current code differs: intent-v4, repair-v4, b1-v2, stability errors as tool failures, the FIX_CODE evidence
> rule, group verification with a whole-suite regression run, and a final suite run. The current, verified state
> is in [`docs/day2-human1.readme.md`](day2-human1.readme.md).


**Slot task (5-day plan):** "Complete orchestration and role prompts". Human 1's role list adds "four role-specific
prompts", and the plan's rules add "configuration flags to switch components off".

**Out of scope here:**
- 11:00–13:00: Triage/Analyst *outputs*.
- 13:00–15:00: Repairer/API tool calling.
- 15:00–18:00: B1.
- Human 2's work: memory store, probes, mutants.
- Day 3: the A1/A3 switches.

Numbers below come only from [`results/day2/`](../results/day2/). The code ran with uncommitted changes on top of
`f84cc290`.

## 1. What was done

### Role prompts (all four), in [`agents/prompts.py`](../src/testorcode/agents/prompts.py)

| Role | Version | What the prompt does | When it runs |
|---|---|---|---|
| Triage | `triage-v1` (new) | Maps each failing test to the changed function it exercises, or "unknown". | Only for tests the rules can't map: no `target_function`, and no traceback frame inside a changed function. |
| Intent Analyst | `intent-v2` | States the proposal's decision policy: UPDATE_TEST needs cited *positive* evidence of intent; memory counts only with a passing probe; FIX_CODE when behaviour contradicts the evidence; otherwise ESCALATE. Warns that docstrings and old tests describe the old behaviour. | One call per group. |
| Repairer | `repair-v2` | Allowed scope only; names the forbidden weakening edits. For UPDATE_TEST, **keep what the test is about**: if only a literal change would pass but it changes the test's meaning, escalate. Uses rejection reasons on a revision. | One call per group (plus one re-ask if an edit can't be placed). |
| Integrity Guard | `guard-v1` (new) | Semantic review of the applied repair: is the test's meaning preserved, and is the repair legitimate? It can **only lower** the verdict, to ESCALATE; it never approves. | Only after every deterministic check passed, and only if `llm` is given. |

**Shared by all four** (`COMMON_RULES`):
- A green build is not the goal.
- Cite only listed ids.
- The proposal's §11.1 unsafe actions are forbidden.

**Reply models.** Every field of every reply model is required (Gemini tended to drop optional arrays).

**Version tracking.** `PROMPT_VERSION = triage-v1+intent-v2+repair-v2+guard-v1` is written into every run record.

### Orchestration, in [`orchestrator.py`](../src/testorcode/orchestrator.py)

1. **Conditions are honoured.**
   - `settings.experiment_condition` selects the episode view and whether memory is used.
   - A2 runs. FULL/A4 need a `MemoryStore`.
   - B0, B1, B2, A1, A3, A5 and the trivial baselines fail explicitly with "not implemented" at stage `setup`. They are
     never silently run as the multi-agent pipeline.
   - The CLI has `--condition` (default A2).
2. **Revision feedback.** A rejection before the Guard runs now reaches the Repairer through the new keyword
   `rejection_reasons`. That covers patch validation, `git apply` failure and a direction mismatch. Before, the
   revision repeated the same patch blind (Day 1, `dev-05-6eb8d8bd5212`).
3. **Tool trajectory and budget** ([`tools/trace.py`](../src/testorcode/tools/trace.py)).
   - Every `run_tests`, `apply_patch` (validate, apply or revert), `search_repo`, `memory.read` and `memory.write` call
     becomes a `ToolEvent` recording the acting role.
   - `budget.max_tool_calls` is enforced. Reverting a patch is never blocked by the budget.
4. **Simulated maintainer → memory write** (proposal workflow step 7).
   - `run_episode(..., maintainer=...)`. The CLI's policy is "approve when the direction matches gold and the tests
     pass" (proposal §9). It is harness-side only.
   - Every write attempt goes into `memory_writes`, and `maintainer_approved` is recorded.
   - The Guard's `build_memory_record` is a stub (probe derivation is memory work), so an approved write is logged as
     `write_error`.
5. **Escalation summaries** now list the evidence that is missing, plus the analyst's hypothesis and the opposite one,
   as the decision policy asks.
6. **Provenance** records `prompt_template_version`, `code_revision` (git sha, with `-dirty` when there are uncommitted
   changes), `split` and `dataset_version` (`pilot@<sha12 of the agent-view file>`).

**Checks.** `python -m pytest`: **592 passed**. That's 551 before, plus 41 new tests across `test_prompts.py`, the
additions to `test_orchestrator.py` and `test_agents.py`. Ruff is clean on every changed file.

### Rule-based Triage mapping checked offline on real dev worktrees (no model calls)

| Episode | Mapped by rules | Left for the model |
|---|---|---|
| dev-01 | `test_table_dicts` → `tableutils.to_text` | – |
| dev-03 | `test_strip_ansi` → `strutils.strip_ansi` | – |
| dev-06 | 3 tests → `cacheutils.CachedFunction.__call__` | – |
| dev-07 | 3 listutils/queueutils tests → `listutils.BarrelList._cur_size_limit` | `test_bytes2human` (assertion-only failure) |
| dev-05 | – | `test_unscoped_cached_dec` (assertion-only failure) |

These match the functions `datasets/README.md` documents for each episode.

## 2. Live check

Model `gemini-3.5-flash`, condition A2, dev split. One run each.

| Run | Result |
|---|---|
| [`dev-05-e310b38a0015`](../results/day2/dev-05-e310b38a0015.summary.json) | **Escalated, and correctly.** Triage needed its model fallback. The analyst again chose UPDATE_TEST, the correct direction. The refined **Repairer refused the literal-only patch**: it explained that the test needs `scoped=False` in its setup (the developer's real fix) and that changing `== 0` → `== 1` would change the test's meaning. On Day 1 the semantically wrong patch was applied and only caught by escalation; now it is never proposed. 3 model calls, 12,976 tokens, 4 tool events (1 `run_tests` by Triage, 3 `search_repo` by the analyst). The run predates the "Repairer escalation → `evidence_missing`" addition, so that list is empty in this record. |
| [`dev-03-06b08bb69be4`](../results/day2/dev-03-06b08bb69be4.summary.json) | `api_failure`: HTTP 429, free-tier **daily** limit (20 requests per day). Triage had already mapped the test by rules with no model call. Not counted against the agent. |
| [`dev-07-f495c2a68fd6`](../results/day2/dev-07-f495c2a68fd6.summary.json) | `api_failure`: same daily limit, hit at Triage's model fallback for `test_bytes2human`. |

**Not live-verified today** because of the quota:
- the Guard's semantic review on a real patch;
- Triage's model fallback on a mixed episode (dev-07);
- the regression path (dev-03) under the new prompts.

All three are covered by offline tests with `FakeLLMClient`. To run them once the quota resets (about 13 h):
`scripts/run_episode.py --episode dev-03` and `--episode dev-07`.

## 3. Left for later slots and for human review

- **11:00–13:00 (Triage/Analyst outputs).**
  - The run record has no field for each group's mapped function, so Triage's model mapping can't be audited from
    the record (only from logs).
  - Triage reruns failures once; the proposal's 5-run stability filter is not done.
  - The analyst's single hypothesis plus a generic "alternative" is what the escalation summary can show today.
- **13:00–15:00.** The Repairer and Analyst use code-driven tools. Model tool calling (`search_repo`, `git_*`,
  `read_file` chosen by the model) is not wired, and `git_log`/`git_show`/`git_blame` are still stubs.
- **Human 2.**
  - `MemoryStore`, `build_memory_record` (probe derivation) and trusted probes. Until they exist, FULL can't run and
    no repair can be approved.
  - Mutation checks for the Guard.
- **Team decisions.**
  - `EXPERIMENT_CONDITION` still defaults to `B1` in `.env.example`, and B1 now fails explicitly as not implemented.
    Pick A2 (or FULL once memory exists) as the default, or build B1 (15:00–18:00).
  - The free tier's 20 requests per day per key stopped this check after 3 model calls on the live path. The plan
    needs a paid tier, or more keys, before Day 3's experiments.
- **Human review.** dev-05 now ends in a *Repairer* escalation that names the real fix (`scoped=False`). Should a
  setup-argument change be allowed for UPDATE_TEST, under Guard checks? Or is escalation the correct gold outcome for
  this kind of S1 case? (This is the same question as `day1_overview.md` §6.1, now with the model asking for it.)

## 4. Finalization (same slot, after the read-only audit)

All of this is offline work; no model quota was used. The dev-05 live run above used `repair-v2` and predates these
changes.

| Audit finding | Change | Tests |
|---|---|---|
| A patch's `evidence_ids` were not checked | `orchestrator._patch_validation` rejects a patch citing ids that were never retrieved (or aren't usable memory); the reason reaches the revision | `test_patch_citing_unretrieved_evidence_*` (2) |
| The test-patch rule rejected every non-literal fix, including dev-05's real fix | `patch_content`: literal keyword arguments of *setup calls in test functions* may be added or changed (never removed). Positional inputs, asserts, decorators, pytest/mock/assert/skip-style helpers, other functions and added/removed statements stay locked. The Guard rejects edits to test functions outside the failure group. Repairer prompt is now `repair-v3`. | 10 new cases in `test_patch_tools.py` (3 accepted, 7 rejected); all 184 earlier cases still pass, including every attack case; `test_test_patch_touching_another_test_function_is_rejected` |
| No offline test ran the real agents through the real orchestrator | `tests/test_pipeline_integration.py`: real orchestrator, four real agents, real git worktree, real `run_tests`/patch tools/probe; only the model is scripted | 3 scenarios (FIX_CODE → APPROVE with a trusted probe; Guard REJECT → revision → ESCALATE; weakening patch blocked twice → ESCALATE), plus 2 ad-hoc negative controls (not in the suite) |
| Default condition B1 failed; A4 silently ran as FULL | `Settings` default and `.env.example` → `A2`; A4 now fails explicitly | `test_config.py`; condition parametrization extended to A4, B2, TRIVIAL_UPDATE_TEST |

**dev-05 re-examined offline** (harness-side, with the reference repair; never shown to an agent):
- **Full developer test diff:** still rejected, because it adds 7 new tests.
- **Its first hunk** (`cached(lru)` → `cached(lru, scoped=False)`, twice): now **accepted**.
- **Applying that hunk on the real dev-05 worktree** and running the real Guard (bwrap sandbox, pytest 8.4.2): 23/23 tests pass, assertions 101 → 101, verdict **ESCALATE** (no trusted probe).
- **Day-1 model patch** (`== 0` → `== 1`): still structurally valid. No static rule can see that it changes the test's meaning. That is handled by the Repairer prompt (it refused this patch live on Day 2), by the Guard's model review (unit-tested only), and by the rule that nothing is approved without a probe.

**`python -m pytest`: 611 passed** (592 + 19). Ruff is clean on all changed files.
