# Day 1 overview: first hosted-model call and first vertical slice

**Scope (5-day plan, Day 1, 13:00–18:00).**
- Make one hosted-model call and get a valid structured decision.
- Connect the stages: *test execution → model decision → restricted patch or escalation → test rerun*, and record the outcome.

**Status.** Both are done and work on real episodes. Every number below comes from a file in
[`results/day1/`](../results/day1/). These are three dev episodes, one run each: a smoke check, not an evaluation.

## 1. Setup used

| Item | Value |
|---|---|
| Model | `gemini-3.5-flash` through Gemini's OpenAI-compatible endpoint (`LLM_PROVIDER=gemini`), temperature 0. Google's quota error lists requests to this id under `gemini-3.6-flash`. |
| Structured output | `response_format=json_schema` (non-strict). Every reply is validated client-side against the existing Pydantic contracts (`IntentAnalysis`; the Repairer's `RepairReply` → `PatchProposal`). |
| Client | Existing `llm_client.OpenAICompatibleClient`: 6 retries, 90 s timeout, prompts and keys never logged. |
| Condition | **A2** (no memory). `MemoryStore` is still a stub, and A2 is the multi-agent pipeline with intent retrieval and the Integrity Guard. |
| Repository | boltons at `d31d821d`, pilot **dev** split only (the test split stays frozen). |
| Tests | Run in the bubblewrap sandbox with `../boltons-venv` (Python 3.12, pytest 8.4.2, the dataset's pin). |
| Code check | `python -m pytest`: **551 passed** (535 before this work, plus 16 new agent tests). Ruff is clean on every changed file. The pre-existing ruff findings in `schemas/` were left untouched. |

## 2. Hosted call ([`llm_smoke.json`](../results/day1/llm_smoke.json))

There was one live call: the real Intent Analyst prompt for dev-05, with the response model `IntentAnalysis`.
- **Result.** It validated against the contract: hypothesis `outdated_test` → `UPDATE_TEST`, confidence 0.95.
- **Citation.** It cited `msg:8408ab57e6c9`. That id was among the 8 retrieved items (0 invented ids) and is also the gold evidence id.
- **Cost.** 3,208 input tokens, 198 output tokens, 5,683 total (the gap is the model's internal reasoning), 36.8 s including retries.

## 3. End-to-end runs (final code)

| | dev-05 | dev-03 | dev-01 |
|---|---|---|---|
| Scenario (evaluation metadata) | S1 outdated test | S2 injected regression | S2 injected regression |
| Record | [`dev-05-d1cdd5d4ed14`](../results/day1/dev-05-d1cdd5d4ed14.summary.json) | [`dev-03-ddeb3157dfa8`](../results/day1/dev-03-ddeb3157dfa8.summary.json) | [`dev-01-ffb6d0393b18`](../results/day1/dev-01-ffb6d0393b18.summary.json) |
| 1. Real pytest failure (baseline) | FAILED, 0 passed / 1 failed (`assert 1 == 0`) | FAILED, 0 / 1 | FAILED, 0 / 1 |
| 2. Model hypothesis → direction | `outdated_test` → UPDATE_TEST | `regression` → FIX_CODE | `regression` → FIX_CODE |
| Cited evidence (all actually retrieved) | `msg:8408ab57e6c9` | `msg:bba729bf4539` | `msg:0c88f25323de`, `docstring:tableutils.Table.to_text` |
| Gold direction / direction correct | UPDATE_TEST / **yes** | FIX_CODE / **yes** | FIX_CODE / **yes** |
| 3. Restricted patch | test scope, `tests/test_cacheutils.py`: `== 0` → `== 1` | src scope, `boltons/strutils.py`: `or` → `and` | src scope, `boltons/tableutils.py`: `or` → `and` |
| Patch validation (scope, protected paths, test-literal rule) | passed, applied | passed, applied | passed, applied |
| Matches reference repair? | **No**, see §6.1 | Yes: exact inverse of the injected mutant | Yes: exact inverse of the injected mutant |
| 4. Rerun (group's test files) | PASSED 23 / 0 | PASSED 17 / 0 | PASSED 3 / 0 |
| Integrity Guard | ESCALATE: no trusted probe | ESCALATE: no trusted probe | ESCALATE: no trusted probe |
| Final action / run status | ESCALATE / `escalated` (patch reverted) | ESCALATE / `escalated` (patch reverted) | ESCALATE / `escalated` (patch reverted) |
| Model calls / tokens | 2 / 10,413 | 2 / 7,478 | 2 / 6,992 |

**How to read this.**
- **Direction:** correct in **3 of 3** runs, judged by the side the model blamed.
- **Final action:** ESCALATE in **3 of 3** runs. That is by design: per `docs/interfaces.md` and `tools/verification.py`,
  a passing rerun is never approval on its own. Without a harness-supplied `TrustedProbe`, the Guard escalates and the
  orchestrator reverts the patch. The patch, the rerun and the verdict all stay in the run record.
- **Scoring:** the final action therefore does not match the gold *action* in any run (0 of 3). That is not an error in
  the direction decision. `summary.json` scores direction and final action separately.
- **Worktrees:** after each run, each worktree held only the CI state (the source diff); every repair was reverted.

## 4. Other recorded runs

These are kept as evidence, not results.

| Run | What happened |
|---|---|
| [`dev-05-6eb8d8bd5212`](../results/day1/dev-05-6eb8d8bd5212.summary.json) | The model chose the right patch, but `git apply` rejected it as corrupt, twice. This exposed the whitespace-stripping bug (§5.1). |
| [`dev-05-af4bb40c293c`](../results/day1/dev-05-af4bb40c293c.summary.json) | First complete pass after that fix. Same outcome as the final run; the Repairer needed a second attempt. |
| [`dev-03-5ca5fe845ea4`](../results/day1/dev-03-5ca5fe845ea4.summary.json) | The free-tier daily quota ran out (HTTP 429), so the run is `api_failure`. It is correctly *not* counted as an escalation or against the agent. |
| [`dev-03-35d28101bd01`](../results/day1/dev-03-35d28101bd01.summary.json) | Direction correct, but the Repairer twice returned `action: patch` with no edits, so the run escalated (§5.3). |

## 5. What was built or fixed

**What was added:**
- **The four agents** (`src/testorcode/agents/`), which were stubs before:
  - **Triage** is deterministic: it reruns the failing tests and groups them.
  - **Intent Analyst** runs BM25 retrieval, makes one model call and returns an `IntentAnalysis`.
  - **Repairer** asks the model for exact snippet edits, builds the diff with `difflib`, and re-asks once if an edit cannot be placed.
  - **Integrity Guard** is deterministic: it reruns the tests, checks scope and protected paths, counts assertions and
    skip markers, checks that FIX_CODE left tests untouched and that UPDATE_TEST cited evidence, then calls `trusted_outcome`.
- **Scripts:** `scripts/llm_smoke.py` and `scripts/run_episode.py` (the CLI).
- **Settings:** `TEST_PYTHON`, `LLM_STRUCTURED_MODE`, `LLM_MAX_RETRIES`, `LLM_TIMEOUT_SECONDS`.

**Integration bugs that only the real path exposed** (the existing tests use fake patch tools):
1. **Diffs could never apply.** Contracts strip edge whitespace, so `PatchProposal.unified_diff` lost its final newline
   and `git apply` called it corrupt. Fixes:
   - `patch_tools` restores the final newline.
   - The Repairer drops whitespace-only trailing context lines, which stripping would otherwise delete, and fixes the hunk counts.
2. **Reverts would have been refused for boltons.** The orchestrator reverted without the episode's `EditPolicy`, so the
   default policy (`src/`) would have rejected reverting any `boltons/` patch.
3. **Gemini dropped optional arrays.** When `edits` was optional in the JSON schema, Gemini sometimes omitted it. Every
   `RepairReply` field is now required.

**Model choice.** `gemini-2.5-flash` (the plan's choice) is unavailable to new keys. `gemini-3.8-flash` returned 503s
and timeouts on every attempt today, so Day 1 runs on `gemini-3.5-flash`.

## 6. Escalated to human verification

These items need a person. None of them can be settled by the system or by a coding agent.

### 6.1 Repairs the Guard escalated (one per run)

All three were reverted by design. The diffs are in the run records.

- **dev-05 (`tests/test_cacheutils.py::test_unscoped_cached_dec`). Most important.**
  - **Model's repair:** change the expected value `other_inner_func.call_count == 0` → `== 1`. Tests pass.
  - **Developer's reference repair** (commit `8408ab57`): keep `== 0` and construct the cache with `cached(lru, scoped=False)`.
  - **Why it matters:** the model's patch makes a test named *unscoped* assert *scoped* behaviour. It is green but changes
    what the test means.
  - **Why the model did this:** the current test-patch policy (`tools/patch_content.py`: only expected literals in
    existing `assert ==/!=` may change) would **reject the developer's real fix**. So the model could only pick a literal
    change. The direction is right; the repair is semantically wrong.
  - **Questions for the reviewer:**
    1. Should UPDATE_TEST allow changing call arguments in test setup, under stricter Guard checks?
    2. Or should such cases be escalated by design, and scored as correct escalations?
- **dev-03 (`boltons/strutils.py`, `strip_ansi`)** and **dev-01 (`boltons/tableutils.py`, `to_text`).**
  - The patches restore `and` where the injected mutant put `or`, and match the reference exactly.
  - **Question:** confirm that this is approval-worthy. If so, the harness should supply a `TrustedProbe` (provenance
    `dataset`) for these functions so the Guard can return APPROVE.

### 6.2 Decisions the team must make

1. **Trusted probes (blocks any APPROVE).** Every repair escalates until probes exist.
   - **Decide:** who writes them (Human 2 / sandbox branch), from which source (maintainer, dataset, approved memory),
     and for which functions.
   - **Until then:** "correct-repair rate" cannot be measured, only direction accuracy and patch validity.
2. **API budget (blocks the experiment matrix).**
   - **Free-tier limit:** about **20 requests per day per model and per Google Cloud project**.
   - **Day-1 cost:** one episode = 2–4 calls, roughly 7–12k tokens. The 72-episode × 5-condition matrix needs hundreds
     to thousands of calls.
   - **Already happened:** one key was exhausted today, and a second key from another project was used. That works for
     Day 1 but is not a plan.
   - **Decide:** enable billing (paid tier) or change the model, and **freeze the model id** before the test split is run.
     Re-check whether `gemini-3.5-flash` is the model actually served (the quota error names `gemini-3.6-flash`).
3. **Scoring definition.** Decide whether "ESCALATE after a correct, test-passing repair that lacked a probe" counts as
   correct, incorrect or a separate category in the metrics. Today it counts as an action mismatch.
4. **Condition wiring.** `run_episode` does not read `conditions.PROFILES` yet; the CLI pins A2.
   - **Needed before Day 3:** FULL needs `MemoryStore`, B1 needs the single-agent path, A3 needs a Guard-off switch.

### 6.3 Dataset items for Human 3

1. **Gold evidence ids.** They are builder-proposed (`datasets/README.md`, "Limitations"). dev-05's gold id is the one
   the model cited; the other episodes in this run (S2) have no gold evidence ids. They still need the human pass.
2. **Tool-fault plans.** `datasets/README.md` says dev-02 and dev-04 carry tool-fault plans. The data enables them on
   **dev-03** (`run_tests` invalid selector) and **dev-04** (`apply_patch` fails). The runner does not inject faults
   yet, so today's dev-03 runs were unaffected. Fix the README or the data.

### 6.4 Environment notes

- **Interpreter path.** `TEST_PYTHON` must be the venv path itself. A resolved symlink (`/usr/bin/python3.12`) loses the
  venv inside the sandbox and picks up system pytest.
- **Shell `PYTHONPATH`.** A shell-wide `PYTHONPATH` (ROS here) breaks direct `pytest` calls. Run `unset PYTHONPATH`;
  sandboxed runs are unaffected.
- **Old venv.** The old broken `myvenv` was moved to `myvenv.broken/`. Delete it once you're happy.

## 7. Reproduce

```bash
unset PYTHONPATH
myvenv/bin/python -m pytest
myvenv/bin/python scripts/llm_smoke.py --episode dev-05 --worktree ../episode-worktrees/dev-05
myvenv/bin/python scripts/run_episode.py --episode dev-05 --repo ../boltons   # also dev-03, dev-01
```

Live runs need `.env` filled in as in README §8, plus the boltons clone and venv from README §8a.
