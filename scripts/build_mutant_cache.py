"""Build the cached-mutant catalogue for selected pilot functions and record measured kills.

For each selected dev episode: the episode's parent commit (a green suite) is exported from the pinned clone with
``git archive`` into a scratch directory (the clone is only read), the labelled function's mutants are generated with
``tools.mutation.generate_mutants`` (the same deterministic seed and limit the Integrity Guard uses, so the Guard
regenerates exactly these mutants from the same source), and each mutant is executed against the function's test file
in the sandbox with ``tools.mutation.run_mutants``.  Every outcome in the output comes from an actual run.

Kills are by the tests **at the parent commit**: a test-strength baseline for the function, not the developer's
reference tests (``reference_killed_mutant_ids`` is therefore left unset).  Only dev episodes are read; the frozen test
split is never touched.

Each entry's ``measurement`` is ``complete`` (every mutant ran to killed/survived), ``incomplete`` (some mutants timed
out, errored, were invalid or not run; ``counts`` still show each outcome) or ``not_measured`` (no usable baseline, no mutant
generated, or the function was not found; ``counts`` is ``null``, never zeros).  ``--out`` is published (temporary
file + atomic replace) only when every entry is complete, exit 0.  Otherwise exit 1, ``--out`` is left exactly as it
was, and the rejected build is written to ``<out>.incomplete`` for inspection (removed by the next complete build).

    python scripts/build_mutant_cache.py --repo ../boltons --python ../boltons-venv/bin/python
"""

from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

from testorcode.dataset import episode_path, load_episodes
from testorcode.schemas.contracts import EpisodeContext
from testorcode.schemas.episode_schemas import MutationFixture
from testorcode.tools.mutation import (
    DEFAULT_MAX_MUTANTS,
    DEFAULT_SEED,
    generate_mutants,
    locate_function,
    run_mutants,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EPISODES = ("dev-02", "dev-03", "dev-04", "dev-06", "dev-09")


def export(repo: Path, sha: str, dest: Path) -> None:
    data = subprocess.run(["git", "-C", str(repo), "archive", sha], capture_output=True, check=True).stdout
    with tarfile.open(fileobj=io.BytesIO(data)) as archive:
        archive.extractall(dest, filter="data")


def write_atomic(target: Path, text: str) -> None:
    """Write ``text`` to ``target`` through a temporary file in the same directory and an atomic replace."""
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(tmp, target)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--repo", default=str(ROOT.parent / "boltons"))
    parser.add_argument("--python", required=True, help="interpreter of the pinned pilot environment")
    parser.add_argument("--episodes", nargs="*", default=list(DEFAULT_EPISODES))
    parser.add_argument("--limit", type=int, default=DEFAULT_MAX_MUTANTS)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--isolation", default="auto")
    parser.add_argument("--out", default=str(ROOT / "results" / "mutants" / "dev_cached_mutants.json"))
    args = parser.parse_args(argv)
    # The sandbox runs from a scratch directory, so a relative interpreter path would not be found there and every
    # mutant would go unmeasured.  abspath (not resolve): a venv's python is a symlink that must stay the venv's.
    args.python = os.path.abspath(args.python)

    episodes = {e.episode_id: e for e in load_episodes(episode_path("dev"))}
    entries = []
    for episode_id in args.episodes:
        episode = episodes[episode_id]
        function = episode.ground_truth.per_test_labels[0].function_qualname
        parent = episode.agent_input.ci_event.parent_commit.sha
        test_file = episode.agent_input.ci_event.failing_tests[0].test_file
        with tempfile.TemporaryDirectory(prefix="mutant-cache-") as tmp:
            work = Path(tmp) / "repo"
            export(Path(args.repo), parent, work)
            context = EpisodeContext.from_episode_input(episode.agent_input, work)
            located = locate_function(context, function)
            if located is None:
                reason = f"{function} not found at {parent[:10]}"
                print(f"{episode_id}: NOT MEASURED: {reason}", file=sys.stderr)
                entries.append({
                    "episode_id": episode_id, "function_qualname": function, "commit": parent,
                    "test_selector": test_file, "measurement": "not_measured", "run_status": None,
                    "run_reason": reason, "baseline_failing": [], "counts": None, "seconds": None,
                    "mutation_fixture": None, "outcomes": [],
                })
                continue
            path, module, inner = located
            source = (work / path).read_text(encoding="utf-8")
            mutants = generate_mutants(source, path, inner, module=module, limit=args.limit)
            started = time.monotonic()
            run = run_mutants(work, mutants, [test_file], timeout_seconds=args.timeout, python_executable=args.python,
                              isolation=args.isolation)
            seconds = round(time.monotonic() - started, 1)
        outcomes = [{"mutant_id": o.mutant_id, "outcome": o.outcome, "killed_by": list(o.killed_by),
                     "detail": o.detail} for o in run.outcomes]
        reason = run.reason or (None if mutants else "no mutants generated")
        if run.status.value != "passed" or not mutants:
            measurement, counts = "not_measured", None  # no usable baseline or no mutant: not "0 kills"
        else:
            counts = {k: sum(o["outcome"] == k for o in outcomes)
                      for k in ("killed", "survived", "timeout", "error", "invalid")}
            measured = sum(o["outcome"] in ("killed", "survived") for o in outcomes)
            measurement = "complete" if measured == len(mutants) else "incomplete"
        entries.append({
            "episode_id": episode_id, "function_qualname": function, "commit": parent, "test_selector": test_file,
            "measurement": measurement, "run_status": run.status.value, "run_reason": reason,
            "baseline_failing": sorted(run.baseline_failing),
            "counts": counts, "seconds": seconds,
            "mutation_fixture": MutationFixture(function_qualname=function, mutants=mutants).model_dump(mode="json"),
            "outcomes": outcomes,
        })
        if measurement == "not_measured":
            note = f" NOT MEASURED: {run.status.value} {reason or ''}".rstrip()
        elif measurement == "incomplete":
            note = f" INCOMPLETE: {len(mutants) - counts['killed'] - counts['survived']} mutant(s) not measured"
        else:
            note = ""
        print(f"{episode_id} {function}: {len(mutants)} mutants -> {counts} ({seconds}s){note}")

    out = Path(args.out)
    text = json.dumps({
        "generator": {"seed": DEFAULT_SEED, "limit": args.limit, "tool": "testorcode.tools.mutation.generate_mutants"},
        "kills_measured_against": "the function's test file at the episode's parent commit (not the reference repair)",
        "python": args.python, "isolation": args.isolation, "entries": entries,
    }, indent=2) + "\n"
    incomplete = [e["episode_id"] for e in entries if e["measurement"] != "complete"]
    rejected = out.with_name(out.name + ".incomplete")
    if incomplete:
        # Validated before publishing: a failed build never replaces --out (a valid catalogue stays as it was).
        write_atomic(rejected, text)
        print(f"measurement incomplete or failed for: {', '.join(incomplete)}", file=sys.stderr)
        print(f"{out} left unchanged; the rejected build is in {rejected}", file=sys.stderr)
        return 1
    write_atomic(out, text)
    rejected.unlink(missing_ok=True)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
