#!/usr/bin/env bash
# Offline demo of TestOrCode on saved data.  No API call, no experiment run, no file under results/ is written.
#   bash scripts/demo.sh            (from the project root; needs ../boltons at the pin and ../boltons-venv)
set -euo pipefail
cd "$(dirname "$0")/.."
unset PYTHONPATH
PY="${PYTHON:-../myenv/bin/python}"
TMP="$(mktemp -d)"
WT="$TMP/dev-03-worktree"
trap 'git -C ../boltons worktree remove --force "$WT" >/dev/null 2>&1 || true; rm -rf "$TMP"' EXIT
RUNS=results/day3/pilot-dev-5cond/runs.jsonl

echo "=== 1. A real failure: rebuild dev-03's CI state from the agent view and run its failing tests ==="
"$PY" scripts/pilot/materialize_episode.py --repo ../boltons --dataset datasets/pilot_dev_agent_view.jsonl \
    --episode dev-03 --dest "$WT" >/dev/null
TESTS=$("$PY" - <<'P'
import json
for l in open("datasets/pilot_dev_agent_view.jsonl"):
    r = json.loads(l)
    if r["episode_id"] == "dev-03":
        print(" ".join(t["test_id"] for t in r["ci_event"]["failing_tests"]))
P
)
BV="$(cd ../boltons-venv/bin && pwd)/python"
(cd "$WT" && PYTHONPATH=. "$BV" -m pytest $TESTS --import-mode=importlib -q --color=no -p no:cacheprovider 2>&1 | tail -6) || true

echo; echo "=== 2. What FULL did on it (saved live run, results/day3/pilot-dev-5cond): evidence, decision, patch, Guard ==="
"$PY" scripts/review_runs.py "$RUNS" --condition FULL --episode dev-03 | sed -n 1,8p

echo "--- the accepted source patch (saved record) ---"
"$PY" - <<'P'
import json
for l in open("results/day3/pilot-dev-5cond/runs.jsonl"):
    r = json.loads(l)
    if r["provenance"]["episode_id"] == "dev-03" and r["provenance"]["condition"] == "FULL":
        for p in r["patches"]:
            print(p["unified_diff"].rstrip())
P

echo; echo "=== 3. The same episode under B1 and A3 (saved runs) ==="
"$PY" scripts/review_runs.py "$RUNS" --condition B1,A3 --episode dev-03 | grep -E "^===|status=|decision tests|patch "

echo; echo "=== 4. A no-model sanity baseline through the real entry point (written to a temp dir) ==="
"$PY" scripts/run_episode.py --condition TRIVIAL_FIX_CODE --episode dev-03 --out "$TMP/out" | tail -8

echo; echo "=== 5. The results table, rebuilt from the saved records (copy in a temp dir) ==="
cp "$RUNS" results/day3/pilot-dev-5cond/integrity_audit.jsonl "$TMP/"
"$PY" scripts/make_report.py "$TMP/runs.jsonl" --no-plots | sed -n 5,14p
