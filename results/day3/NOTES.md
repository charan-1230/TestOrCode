# Day-3 live runs (Human 2)

These are **live hosted-model runs**. The experiment runner made them on 2026-10-29, with the same setup as
`results/day2/pilot-full-b1`:

| Setting | Value |
|---|---|
| Model | `gemini-3.5-flash`, set as `LLM_MODEL` for these runs; `.env` says flash-lite |
| Seed | 0 |
| Split | dev |
| Dataset | `pilot@87d1398471c0` |
| Repository and sandbox | boltons at the pin, `../boltons-venv` (pytest 8.4.2), bubblewrap |
| Code | `ea8c151` plus the uncommitted Day-3 changes, recorded as `-dirty` |

| Directory | Contents |
|---|---|
| `a3-pilot/` | **A3 (no Integrity Guard)** on dev-04, dev-06 and dev-07. All 3 completed with the correct direction and accepted repairs. A3's own memory store holds only the seeded dev-07 decoy, and nothing was written. 10 model calls in total |
| `dev07-full-b1/` | dev-07 re-run for FULL and B1, because both hit HTTP 429 on Day 2. FULL completed: Guard APPROVE, and record `dev-07-g1` written. B1 ended as `api_failure` (HTTP 503 after 7 attempts), which is not scored |

Each directory has these files:
- `summary.{md,json}`, from `scripts/summarize_results.py`;
- `integrity_audit.{md,jsonl}`, from `scripts/audit_integrity.py`. This is a post-hoc audit of every accepted repair, done the same way for every condition.

**Audit version.** The saved `integrity_audit.{md,jsonl}` files here (and in `results/day2/pilot-full-b1/`) were made by **audit version 1** and were **not regenerated** after the audit was revised (version 2: explicit measured/unmeasured mutant sets, retention over mutants measured in both copies). You can tell them apart because the version-1 JSONL lines have no `audit_version` field. Re-running version 2 on scratch copies found every group in these files fully measured, with identical verdicts; only the wording of dev-06's reason differs (`docs/day3-human2.md` §3 and §9).

`results/day2/pilot-full-b1/integrity_audit.*` was added for the Day-2 FULL/B1 runs; their records are unchanged.

**What this is:** a smoke-level pilot of 3 episodes and 1 seed, not a comparison. The interpretation is in `docs/day3-human2.md` §5.
