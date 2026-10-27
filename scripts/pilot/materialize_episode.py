"""Rebuild the exact CI state of a pilot episode from its AGENT VIEW alone.

CI state = ``ci_event.parent_commit`` checked out in a git worktree, with every
``ci_event.changed_files[*].unified_diff`` applied (uncommitted).  Run the tests
there to see the failures recorded in the episode.  Needs a clone of the pinned
boltons repository (see ``datasets/pilot_manifest.json``); never touches it.

    python scripts/pilot/materialize_episode.py --repo ../../screening/boltons \
        --dataset datasets/pilot_dev_agent_view.jsonl --episode dev-01 --dest /tmp/dev-01
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def _git(cwd: str | Path, *args: str, stdin: str | None = None) -> str:
    done = subprocess.run(
        ["git", "-C", str(cwd), *args], input=stdin, capture_output=True, text=True, check=False
    )
    if done.returncode != 0:
        msg = f"git {' '.join(args)} failed: {done.stderr.strip()[:500]}"
        raise RuntimeError(msg)
    return done.stdout


def materialize(repo: str | Path, dest: str | Path, agent_input: dict) -> Path:
    """Create or reset the worktree ``dest`` to the CI state of ``agent_input`` (an agent payload)."""
    dest = Path(dest)
    ci = agent_input["ci_event"]
    parent = ci["parent_commit"]["sha"]
    if (dest / ".git").exists():
        _git(dest, "checkout", "-q", "-f", "--detach", parent)
        _git(dest, "clean", "-fdxq")
    else:
        _git(repo, "worktree", "add", "-q", "--detach", str(dest), parent)
    for change in ci["changed_files"]:
        diff = change.get("unified_diff")
        if diff:
            # The schema strips edge whitespace from strings, which drops the final newline (and possibly a blank
            # trailing context line); restore the newline and let git recount the hunk lengths.
            _git(dest, "apply", "--whitespace=nowarn", "--recount", "-", stdin=diff.rstrip("\n") + "\n")
    return dest


def load_agent_view(path: str | Path, episode_id: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            if payload["episode_id"] == episode_id:
                return payload
    msg = f"episode {episode_id!r} not found in {path}"
    raise KeyError(msg)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--repo", required=True, help="clone of the pinned library repository")
    parser.add_argument("--dataset", required=True, help="an *_agent_view.jsonl file")
    parser.add_argument("--episode", required=True)
    parser.add_argument("--dest", required=True, help="worktree directory to create or reset")
    args = parser.parse_args()
    dest = materialize(args.repo, args.dest, load_agent_view(args.dataset, args.episode))
    sys.stdout.write(f"{dest}\n")
    return 0


if __name__ == "__main__":
    os.environ.pop("GIT_DIR", None)
    raise SystemExit(main())
