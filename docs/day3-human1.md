# Day 3 — Human 1 (agent core): what was done, what is blocked, how to verify

Branch `worktree-day3-human1`, on top of `ea8c1510`. Code commit `a6563189`.

**Verification labels:**
- **UNIT:** fake components.
- **OFFLINE INTEGRATION:** the real orchestrator, agents, git, pytest and patch tools; only the model is scripted.
- **LIVE:** a saved real-model run.

**Nothing from Day 3 was run LIVE.** No API quota was spent. `python -m pytest`: **984 passed, 1 skipped** (961 + 23
new). Ruff is clean on every changed file.

| Slot | Plan task (Human 1) | Status |
|---|---|---|
| 09:00–11:00 | Stabilize Full/B1 shared interfaces | **Done.** Conditions are profile-driven. `run_episode` is the one entry point for every condition. B1 tool attribution fixed (§1, §4) |
| 11:00–13:00 | Implement A1 switch (with Human 3) | **Done in code + offline integration.** One A1 semantics question goes to Human 3 (§2.3). Live A1 runs not done (quota) |
| 13:00–15:00 | Deterministic sanity baselines | **Done.** Implemented by Human 3 on Day 2. Now also reachable through `run_episode` and `run_episode.py`. First table on the dev split (§3) |
| 15:00–18:00 | Fix common agent failures | **Done for the failures inside Human 1's code** (§4). The rest are dependencies (§6) |
| 18:00–20:00 | Review decisions and logs | **Done for the saved live records** (§5). New tool: `scripts/review_runs.py` |

## 1. Shared interfaces (Full / B1 / A1 / trivial)

- **Profile-driven conditions** (`orchestrator.py`):
  - `EpisodeRun.profile = profile_for(condition)`. Memory and intent retrieval are now read from the profile, not
    from hard-coded sets.
  - A profile with memory needs a `MemoryStore`: FULL and A1.
  - Which run class runs which condition is still explicit:
    - `EpisodeRun`: FULL, A1, A2.
    - `SingleAgentRun`: B1.
    - `run_episode` itself: the trivial baselines.
  - A4's profile equals FULL's, so a purely profile-based dispatch would silently run A4 as FULL. It stays refused, as
    do A3, A5, B0 and B2.
- **One entry point:**
  - `run_episode(...)` answers `TRIVIAL_UPDATE_TEST` / `TRIVIAL_FIX_CODE` with `baselines.trivial_record`. It uses no
    model, worktree or memory, and `llm` may be `None`.
  - `scripts/run_episode.py --condition TRIVIAL_FIX_CODE --episode dev-07` needs no repository or API key.
  - The CLI now reports direction exactly as the metrics score it (`evaluation.predicted_direction`). Before, a
    trivial FIX_CODE was printed as `direction=None`.
- **Unchanged:**
  - every result field (`ExperimentRunRecord`)
  - `PROMPT_VERSION` and `b1-v2`. The FULL/A2/B1 prompt text is byte-identical; a test checks this.
  - all Human 2 / Human 3 modules except `conditions.py` (§2.1)

## 2. A1 — no intent retrieval

### 2.1 What A1 is in code

The proposal's Table: "Analyst sees only failing output and diff". A1 is FULL minus every route to change-intent text:

| Route | FULL | A1 | Where |
|---|---|---|---|
| CI commit message in Analyst / Guard prompts | shown | withheld (`WITHHELD`, no `context_commits`) | `conditions.episode_for_condition`, `triage.py` (existing) |
| Analyst's BM25 intent retrieval | 3 fused queries, top 8 | **none** (`retriever=None`; prompt: "evidence retrieval is disabled") | `orchestrator._intent_analysis`, `intent_analyst.analyse_intent` |
| Corpus in the episode view | 100+ items (100 commit messages, docs, docstrings, changelog) | **`items=[]`** | `conditions.episode_for_condition` |
| Repairer `search_repo` tool | offered | **not offered**; a call anyway is refused (`not_permitted`) | `allowed_tools`, `prompts.tool_instructions` |
| Repairer `read_file` on README / CHANGELOG / `docs/` / other non-code files (added after the Day-3 audit) | readable | **refused at runtime** (`not_permitted`; path and symlink target); Python source under the source roots and files under the test roots stay readable | `AgentTools(code_only=True)` from `orchestrator._agent_tools`; `agents/tools._code_only_problem` |
| Memory | probe-gated | **same** (citable) | unchanged |
| Triage, Repairer `read_file`/`run_tests`, Guard, decision policy, budgets | — | **same** | unchanged |

**Why the corpus and the tool are withheld too.** Every pilot corpus contains the CI commit's own message as
`msg:<ci sha>`. All 9 dev episodes do; `test_a1_view_withholds_every_retrieval_route` checks it. Withholding
`ci_event.commit_message` alone, which is all the Day-2 profile did, would have left it one search away.

**Coordination with Human 3.** `conditions.py` is Human 3's module. The change there is additive and is the module's
stated purpose ("the view a condition may see"). It affects only profiles with neither `intent_retrieval` nor
`standard_rag`: A1, and B0, which is not runnable.

### 2.2 Verified (offline)

| Test | Checks |
|---|---|
| `test_conditions.py::test_a1_view_withholds_every_retrieval_route` (real dev data) | `msg:<sha>` is in the original corpus; the A1 view has no corpus and no `search_repo`; memory is kept; the original episode is untouched |
| `test_conditions.py::test_conditions_with_retrieval_keep_the_corpus_and_search` | FULL/A2/A3/B1 views unchanged |
| `test_agents.py::test_intent_analyst_without_retriever_searches_nothing_and_says_so` | no corpus id reaches the prompt |
| `test_orchestrator.py::test_a1_*` (3) | runs with memory, no `search_repo` event, empty `retrieved_evidence`; a corpus-id citation escalates (`insufficient_evidence`); no store → explicit setup error |
| `test_pipeline_integration.py::test_a1_real_pipeline_...` | **real** Triage / Repairer / Guard (mutation check + probe) / git / pytest: the Analyst cites a probe-passing memory record → FIX_CODE → Guard APPROVE. The Repairer's `search_repo` attempt is refused and logged. Neither the commit message nor any corpus id appears in the Analyst prompt; `search_repo` is absent from the Repairer system prompt |

### 2.2a Remaining A1 exposure, by design

- **Memory** stays in A1 (proposal: A1 removes intent retrieval only). Seeded `visible_memory` records are dataset-made;
  e.g. dev-07's `mem-dev-s1-cacheutils` has an earlier commit's subject line as its `behaviour_signature`. This is
  memory, not retrieval, and was **not** changed. Flag for Human 3 / team.
- **Source code** (docstrings, comments) stays readable, as the Analyst's "diff" and the Repairer's repair need it.
- **Test output** from `run_tests` (existing tests only, on unpatched code) stays available.

### 2.3 Open question for Human 3 / team: A1 and the decision policy

The policy is unchanged: UPDATE_TEST needs cited positive intent evidence, and FIX_CODE needs cited expected-behaviour
evidence. Without retrieval the only citable evidence in A1 is a usable memory record. So **A1 will mostly escalate**,
except on episodes whose seeded or earlier memory covers the function.

That is a faithful "no intent evidence" ablation, and A1's **direction** is still scored from `hypothesis` (as for
FULL escalations). But it means A1's final-action accuracy measures the policy as much as the reasoning.

Alternatives, if the team prefers:
- (a) score A1 on direction only;
- (b) relax the citation requirement for A1 only. This would be a new condition behaviour, and I have **not** done it.

## 3. Sanity baselines — first metrics table (dev split, no LLM)

`results/day3/trivial-dev/` was generated with `scripts/run_experiment.py --conditions TRIVIAL_UPDATE_TEST,TRIVIAL_FIX_CODE`
then `scripts/summarize_results.py`. It contains `runs.jsonl` (18 runs), `summary.md`, `summary.json` and `config.json`
(`code_revision` `a6563189`).

| condition | runs | tests | direction macro-F1 | direction acc. | accuracy | masked regression | reverted intent | acc. S1 / S2 / S3 / S4 / S5 |
|---|---|---|---|---|---|---|---|---|
| always FIX_CODE | 9 | 16 | 0.417 | 0.714 | 0.625 | 0 | 1 | 0 / 1 / 0.75 / 1 / 0 |
| always UPDATE_TEST | 9 | 16 | 0.222 | 0.286 | 0.250 | 1 | 0 | 1 / 0 / 0.25 / 0 / 0 |

- The dev split is FIX_CODE-heavy: 10 of 14 directional tests.
- **Always-FIX_CODE has direction accuracy 0.714.** That is the floor FULL/B1 must beat on accuracy; macro-F1 is the
  fairer comparison.

## 4. Agent workflow failures fixed

| Failure (evidence) | Fix | Test |
|---|---|---|
| **B1 logs misattribute harness work to the agent.** Every B1 tool call, including 5 stability runs and 2 verification runs, was `called_by=single_agent`. The live dev-04/dev-06 B1 records show `run_tests/single_agent=7` although the agent ran 0 tests itself. This distorted per-role tool counts and trace review | `SingleAgentRun._enter`: the agent owns its decision loop and its patch handling; stability and verification runs are `orchestrator` (as Triage/Guard are in FULL). Budget categories unchanged | `test_mixed_integration.py::test_b1_repairs_both_directions_and_records_each_test` |
| **Quota exhaustion repeated per episode.** `experiment.default_run_one` builds one client per run, and each restarted at API key 1. Over a Day-3 experiment, every run after key 1's daily quota ran out spent a rejected request on it (doc §6.2 limitation 10) | The client remembers daily-quota-exhausted keys process-wide: sha256-prefix only, never the key; 1 h TTL, because quotas reset. New clients start at the first key not known to be exhausted. Per-minute 429s are not remembered | `test_llm_client.py::TestKeyPool` (+4) |
| A1 would have leaked the CI commit message through retrieval | §2 | §2.2 |
| `run_episode.py` could not run the sanity baselines (it required a worktree and an API key) | §1 | `test_run_episode_answers_trivial_baselines_without_model_or_worktree` |

**Checked and not a bug.** B1's prompt already states the FIX_CODE evidence rule. flash-lite's B1 escalation on dev-03
(FIX_CODE without a citation) is the policy working, not a prompt defect.

## 5. Review of decisions and logs (saved live records)

Command: `python scripts/review_runs.py results/day2/pilot-full-b1/runs.jsonl results/day2/flash-lite/*.json results/day2/dev-0*.json`.

The output is from the Day-2 records; the B1 attribution there predates the §4 fix.

| What | Seen |
|---|---|
| Direction | Correct in every run that reached a decision (FULL, A2, B1; both models) |
| **Successful repair** | dev-04 FULL: FIX_CODE → Guard APPROVE (tests untouched, suite green, 6/6 mutants killed by the group test) → maintainer approved → memory record `dev-04-g1` written → final suite passed |
| **Failure 1: UPDATE_TEST never approved in FULL** | dev-06 FULL: correct hypothesis and a reasonable test edit, but the Guard escalated because "the updated tests detect none of the 2 executable mutants" of `CachedFunction.__call__`. The patch was reverted and the final suite failed. B1 kept the same kind of edit. Cause: the Guard's approval rule, which needs measured mutant kills, plus the small mutant set for that function. Human 2's design (§6) |
| Failure 2: meaning-changing literal edit | dev-05 A2 (flash-lite): the Repairer changed `call_count == 0` → `== 1`; the Guard's semantic review flagged it → ESCALATE → reverted. The Guard working as intended |
| Failure 3: execution | 7 of 19 Day-2 records are `api_failure` (HTTP 429 daily quota, HTTP 503 overload). All are correctly recorded as API failures, never as decisions |
| A2 Day-2 escalations "no trusted probe" | Pre-Human-2 Guard rule. Superseded: FIX_CODE is now approved on untouched tests (dev-04) |

## 6. Dependencies and not done (not faked)

| Item | Blocked on | Owner | Note |
|---|---|---|---|
| A3 (no Integrity Guard) | Guard-off code path | **Human 2** | `profile.integrity_guard` is in place and A3 is refused explicitly. Adding A3 to `_MULTI_AGENT_CONDITIONS` with a `verify_group`-only branch (as B1's `_verify`) is the expected change; Human 1 can wire it once the semantics (what replaces the Guard verdict) are agreed |
| UPDATE_TEST approval in FULL | Mutation strength / mutant selection (`tools.mutation`, `verification.repair_outcome`) | **Human 2** | dev-06. Optionally, the orchestrator could pass `cached_mutants` (`results/mutants/dev_cached_mutants.json`) to the Guard, which is never done today; that needs a settings key agreed with Human 2 |
| Memory citations scored as invalid | `evaluation._known_ids` / `citation_existence_rate` ignore in-stream memory ids | **Human 3** | FULL citing a memory record written earlier in the stream (or A1 citing memory) lowers `citation_existence_rate`. Fix: treat `record.memory_reads[*].record_id` with a passed probe as known |
| Tests with no prediction skipped in `summarize` | `evaluation.py:370` | **Human 3** | Contradicts "a missing prediction counts as wrong" |
| Resuming an experiment after `api_failure` | `experiment.run_experiment` treats an `api_failure` record as done | **Human 3** | Not patched here: for FULL/A1 a retry is only valid if the memory stream is re-run in order (a partial run may already have written memory) |
| Live A1 / FULL / B1 runs (Day-3 exit) | API quota | Team | Commands below |
| always-ESCALATE baseline | New `ExperimentCondition` value | Human 3 / team | Not in the plan; optional |
| `git_log/show/blame` tools | `tools/git_tools.py` stubs | Human 2 | Not needed by A1 |

## 7. Commands

```bash
unset PYTHONPATH
myvenv/bin/python -m pytest -q                                   # offline suite
# sanity baselines (no key, no repository)
myvenv/bin/python scripts/run_experiment.py --conditions TRIVIAL_UPDATE_TEST,TRIVIAL_FIX_CODE --out results/day3/trivial-dev
myvenv/bin/python scripts/summarize_results.py results/day3/trivial-dev/runs.jsonl
# live pilot on 2-3 episodes, all runnable conditions (needs .env keys, ../boltons, ../boltons-venv)
myvenv/bin/python scripts/run_experiment.py --conditions FULL,A1,A2,B1 --episodes dev-03,dev-04,dev-06 --out results/day3/pilot
myvenv/bin/python scripts/summarize_results.py results/day3/pilot/runs.jsonl
myvenv/bin/python scripts/review_runs.py results/day3/pilot/runs.jsonl            # decisions, traces, failure causes
```
