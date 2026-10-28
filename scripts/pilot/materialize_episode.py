"""Rebuild the exact CI state of a pilot episode from its AGENT VIEW alone.

CI state = ``ci_event.parent_commit`` checked out in a git worktree, with every
``ci_event.changed_files[*].unified_diff`` applied (uncommitted).  Run the tests
there to see the failures recorded in the episode.  Needs a clone of the pinned
boltons repository (see ``datasets/pilot_manifest.json``); never touches it.

    python scripts/pilot/materialize_episode.py --repo ../../screening/boltons \
        --dataset datasets/pilot_dev_agent_view.jsonl --episode dev-01 --dest /tmp/dev-01

Safety: the destination is reset with ``git checkout -f`` and ``git clean -fdx``, which destroy uncommitted work.
``prepare_worktree`` therefore validates the destination first (see its docstring): a fresh worktree is created and marked
disposable; an existing linked worktree this tool did not create needs ``--force``; the main clone, a standalone repository,
another repository's worktree, or any path containing/inside the clone or this project is always refused.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MARKER = "testorcode-disposable-worktree"  # written inside the worktree's private git dir, never in the work tree


class WorktreeSafetyError(RuntimeError):
    """The target directory is not a safe place for a destructive reset."""


def _git_try(cwd: str | Path, *args: str) -> str | None:
    done = subprocess.run(["git", "-C", str(cwd), *args], capture_output=True, text=True, check=False)
    return done.stdout.strip() if done.returncode == 0 else None


def _related(a: Path, b: Path) -> bool:
    return a == b or a in b.parents or b in a.parents


def _common_dir(path: Path) -> Path | None:
    out = _git_try(path, "rev-parse", "--git-common-dir")
    if not out:
        return None
    found = Path(out)
    return (found if found.is_absolute() else path / found).resolve()


def prepare_worktree(repo: str | Path, dest: str | Path, ref: str, *, force: bool = False) -> Path:
    """Validate ``dest`` as a disposable linked worktree of ``repo``, creating it at ``ref`` if it does not exist.

    Always refused, even with ``force``: ``repo`` that is not a main clone; ``dest`` equal to, containing, or inside ``repo`` or this
    project; a standalone repository or main clone (``dest/.git`` is a directory); a worktree of another repository; a non-empty
    directory that is not a worktree; a path that is not the worktree root.

    Refused unless ``force``: an existing linked worktree of ``repo`` that this tool did not create (it may hold someone's uncommitted
    work).  With ``force`` it is adopted and marked.  Worktrees created here carry a marker and need no flag afterwards.

    Nothing is deleted or reset by this function.

    Raises:
        WorktreeSafetyError: the destination is not safe.
    """
    repo, dest = Path(repo).resolve(), Path(dest).resolve()
    if not (repo / ".git").is_dir():
        msg = f"--repo must be the root of a main clone (a directory containing .git/): {repo}"
        raise WorktreeSafetyError(msg)
    for protected, what in ((repo, "the repository clone"), (PROJECT_ROOT, "this project")):
        if _related(dest, protected):
            msg = (
                f"refusing {dest}: it is, contains or lies inside {what} ({protected}). "
                "This is never allowed, even with --force."
            )
            raise WorktreeSafetyError(msg)
    if dest.exists():
        if not dest.is_dir():
            msg = f"refusing {dest}: not a directory"
            raise WorktreeSafetyError(msg)
        entry = dest / ".git"
        if entry.is_dir():
            msg = (
                f"refusing {dest}: it is a standalone repository or a main clone (.git is a directory). "
                "Resetting it would destroy uncommitted work; --force does not override this."
            )
            raise WorktreeSafetyError(msg)
        if entry.is_file():
            if _common_dir(dest) != (repo / ".git").resolve():
                msg = f"refusing {dest}: it is not a worktree of {repo}; --force does not override this"
                raise WorktreeSafetyError(msg)
            top = _git_try(dest, "rev-parse", "--show-toplevel")
            gitdir = _git_try(dest, "rev-parse", "--absolute-git-dir")
            if top is None or gitdir is None or Path(top).resolve() != dest:
                msg = f"refusing {dest}: it is not the root of a git worktree"
                raise WorktreeSafetyError(msg)
            marker = Path(gitdir) / MARKER
            if not marker.exists():
                if not force:
                    msg = (
                        f"refusing {dest}: an existing worktree of {repo} that this tool did not create. It will be reset with "
                        "'git checkout -f' and 'git clean -fdx', discarding uncommitted work. Re-run with --force only if it "
                        "is disposable."
                    )
                    raise WorktreeSafetyError(msg)
                marker.write_text("adopted with --force; safe to reset\n")
            return dest
        if any(dest.iterdir()):
            msg = f"refusing {dest}: a non-empty directory that is not a git worktree"
            raise WorktreeSafetyError(msg)
    _git(repo, "worktree", "add", "-q", "--detach", str(dest), ref)
    (Path(_git(dest, "rev-parse", "--absolute-git-dir").strip()) / MARKER).write_text(
        "created by scripts/pilot; safe to reset\n"
    )
    return dest


def _git(cwd: str | Path, *args: str, stdin: str | None = None) -> str:
    done = subprocess.run(
        ["git", "-C", str(cwd), *args], input=stdin, capture_output=True, text=True, check=False
    )
    if done.returncode != 0:
        msg = f"git {' '.join(args)} failed: {done.stderr.strip()[:500]}"
        raise RuntimeError(msg)
    return done.stdout


def materialize(repo: str | Path, dest: str | Path, agent_input: dict, *, force: bool = False) -> Path:
    """Create or reset the worktree ``dest`` to the CI state of ``agent_input`` (an agent payload).

    ``dest`` is validated by ``prepare_worktree`` before anything is reset.

    Raises:
        WorktreeSafetyError: ``dest`` is not a safe place for a destructive reset.
    """
    ci = agent_input["ci_event"]
    parent = ci["parent_commit"]["sha"]
    dest = prepare_worktree(repo, dest, parent, force=force)
    _git(dest, "checkout", "-q", "-f", "--detach", parent)
    _git(dest, "clean", "-fdxq")
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
    parser.add_argument(
        "--force",
        action="store_true",
        help="adopt and reset an existing linked worktree of --repo that this tool did not create "
        "(never overrides the refusal for the main clone, standalone repositories or other repositories)",
    )
    args = parser.parse_args()
    try:
        dest = materialize(args.repo, args.dest, load_agent_view(args.dataset, args.episode), force=args.force)
    except WorktreeSafetyError as exc:
        sys.stderr.write(f"{exc}\n")
        return 2
    sys.stdout.write(f"{dest}\n")
    return 0


if __name__ == "__main__":
    os.environ.pop("GIT_DIR", None)
    raise SystemExit(main())
