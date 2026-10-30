# Demo (offline, saved setup)

Shortest reliable end-to-end demonstration. **No API call, no experiment run, nothing written under `results/`** (checked: the SHA-256 of
`results/day3/pilot-dev-5cond/runs.jsonl` and `results/day4/pilot-test-5cond/runs.jsonl` is identical before and after). It takes about 2 seconds.

```bash
cd TestOrCode               # project root; needs ../boltons at the pin, ../boltons-venv and ../myenv (override with PYTHON=...)
bash scripts/demo.sh
```

| Step | What it shows | Expected |
|---|---|---|
| 1 | Rebuilds dev-03's CI state from the agent view (`scripts/pilot/materialize_episode.py`, temp worktree, removed on exit) and runs its failing tests with the pinned pytest | `2 failed` (`test_once_executes_only_once`, `test_once_thread_safety`) |
| 2 | Saved live FULL run on dev-03: groups, cited evidence (`msg:42586df3dd67`, `docstring:funcutils.once`), decision FIX_CODE vs gold, accepted patch, tool calls, Guard approval, memory write | `status=completed final_suite=passed`, `guard=approve`; the patch flips `if result is _UNSET:` to `if result is not _UNSET:` |
| 3 | Saved B1 and A3 runs on the same episode | both FIX_CODE with an accepted source patch (B1: 1 model call, A3: 3 calls, FULL: 4 calls) |
| 4 | A no-model baseline through the real entry point `scripts/run_episode.py --condition TRIVIAL_FIX_CODE` (record written to a temp dir) | `FIX_CODE ... action_match=True`, `run status: completed tokens: 0` |
| 5 | The results table rebuilt by `scripts/make_report.py` from a temp copy of the saved records | the 9-episode dev table (FULL 14/14 direction, final-action 0.562, etc.) |

**Actual output:** `docs/demo/demo_transcript.txt` (captured from a real run of the script; temp-dir paths in step 4 differ per run).
**Recording:** `docs/demo/demo.typescript` + `demo.timing` (util-linux `script`; no asciinema/video tool is installed):

```bash
scriptreplay --log-timing docs/demo/demo.timing --log-out docs/demo/demo.typescript      # add -m 0.3 to cap pauses
```

The run being recorded is fast, so the replay is quick; use the transcript to read it.

## Optional live procedure (not used for this demo; spends API quota)
1. `.env` with a valid key (`cp .env.example .env`), `unset PYTHONPATH`.
2. `python scripts/run_episode.py --condition A2 --episode dev-03 --out /tmp/demo-live` prints decision, evidence, patch and verification for one
   fresh model run. FULL and A3 additionally need `--memory <path>.jsonl`. The frozen test split is refused unless `--allow-test-split`.
