# Day-2 run records: contamination notice

The records in this folder are kept unchanged.  Read them with this notice.

## What was wrong

Repairer prompts `repair-v2` to `repair-v4` contained text taken from pilot episode **dev-05**:

- from `repair-v2`: a hint naming the failing test, "a test named 'unscoped' that would then assert scoped
  behaviour";
- from `repair-v3`: also the developer's actual repair, used as an example of an allowed setup change:
  `cached(lru)` -> `cached(lru, scoped=False)`.

The failing test of dev-05 is `tests/test_cacheutils.py::test_unscoped_cached_dec`.  Its reference repair adds
`scoped=False` to the `cached(lru)` call.  A dev-05 run with `repair-v2` was told what to look for, and with `repair-v3`/`v4` it would have been given the
answer outright.

Fixed in `repair-v5` (`src/testorcode/agents/prompts.py`).  The example is now synthetic
(`connect(host)` -> `connect(host, retries=0)`; a test named 'case_insensitive_match').
`tests/test_prompt_leaks.py` now fails if any static prompt shares a 20-character substring with any pilot
reference repair, names a pilot failing test, or contains `unscoped`, `scoped=False` or `cached(lru`.
It also checks that no agent-view episode contains a reference repair's added test lines.

## Affected records

| Record | Prompt version | Reached the Repairer? | Status |
|---|---|---|---|
| `dev-05-e310b38a0015` | `triage-v1+intent-v2+repair-v2+guard-v1` | yes | **Contaminated** (hint only, not the full example). The Repairer's escalation summary follows the hint to the fix ("explicitly pass `scoped=False` when calling `cached(lru)`"). Do not use it as evidence of agent ability. |
| `dev-03-06b08bb69be4` | same | no (`api_failure`, HTTP 429 before the Repairer) | Same prompt version, contaminated text not exercised.  No decision recorded; nothing to score. |
| `dev-07-f495c2a68fd6` | same | no (`api_failure`, HTTP 429 before the Repairer) | Same as dev-03. |

## Not affected

- `results/day1/*`: the Day-1 prompts (committed at `f84cc290`) contain neither the example nor the hint.
  Those records mention `scoped=False` only in the model's own reasoning, which it derived from the visible CI
  diff (the commit adds the `scoped` parameter).
- `repair-v3` and `repair-v4` (which also carried the example) were used only in offline tests with scripted model
  replies.  No hosted-model run used them.

## Consequence

Every live dev-05 run must be re-done with `repair-v5` or later before it counts for anything.  Reference repairs
in the dataset are unchanged; they stay harness-only.

## Clean replacements (current code, `repair-v5`)

- `dev-05-9704ba9e49aa` (A2, gemini-3.5-flash): found the `scoped=False` setup change without any hint.
- `results/day2/flash-lite/dev-05-31797fb62acf` (A2) and `dev-05-911a25ecc5e6` (B1), gemini-3.5-flash-lite.

Details in `docs/day2-human1.readme.md` §10.
