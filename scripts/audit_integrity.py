"""Post-hoc integrity audit of an experiment's saved runs (harness side; see ``testorcode.integrity_audit``).

Every accepted repair of every run in ``runs.jsonl`` is re-measured the same way whatever its condition: weakening
analysis and mutant-kill retention against the developer's reference repair for test patches, oracle sensitivity for
source patches.  The CI state of each episode is rebuilt from the pinned clone in a scratch directory; run worktrees
are never touched.  Writes ``integrity_audit.jsonl`` (one line per audited run) and ``integrity_audit.md``
(per-condition table) into ``--out-dir`` (default: the directory of ``runs.jsonl``).

Everything is checked **before** any audit work or write, and any problem refuses the whole audit (exit 2, nothing
written):
  * the input exists and every line is a valid run record (malformed lines are listed by line number);
  * run identity is ``provenance.run_id``: an exact repeat of a line is a duplicate, reported and audited once; two
    *different* records with one ``run_id`` are a conflict.  Distinct runs that share (episode, condition, seed) are
    all audited and listed as repeated runs;
  * every record's episode exists in its split (an unknown episode is never skipped silently);
  * the output directory exists, and neither report exists unless ``--overwrite`` is given.
Both reports are written to temporary files first and then moved into place together; if the second move fails, the
first is rolled back, so the two files always come from the same audit.

Exit status: 0 = audited and written; 2 = refused (see stderr), nothing written; 1 = an unexpected error while
auditing (e.g. the clone cannot be materialised or no sandbox is available; Python traceback on stderr), nothing
written.

    python scripts/audit_integrity.py --runs results/day2/pilot-full-b1/runs.jsonl \\
        --repo ../boltons --python ../boltons-venv/bin/python [--out-dir DIR] [--overwrite]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path

from pydantic import ValidationError

from testorcode.dataset import episode_path, load_episodes
from testorcode.integrity_audit import AUDIT_VERSION, audit_run, markdown_table, summarize_audits
from testorcode.schemas.result_schemas import ExperimentRunRecord
from testorcode.tools.mutation import load_cached_mutants

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ("integrity_audit.jsonl", "integrity_audit.md")
REFUSED = 2


class Refused(Exception):
    """The audit cannot run as asked; nothing has been written."""


def _materializer():
    spec = importlib.util.spec_from_file_location("materialize_episode", ROOT / "scripts/pilot/materialize_episode.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.materialize


def _load_episodes(splits: set[str]) -> dict:
    return {e.episode_id: e for split in sorted(splits) for e in load_episodes(episode_path(split))}


def _read_records(runs: Path) -> tuple[list[ExperimentRunRecord], list[tuple[int, str]], list[str]]:
    """``(unique records, duplicates as (line, run_id), repeated (episode, condition, seed) keys)``."""
    if not runs.is_file():
        raise Refused(f"input not found or not a file: {runs}")
    lines = runs.read_text(encoding="utf-8").splitlines()
    parsed: list[tuple[int, ExperimentRunRecord]] = []
    malformed = []
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            parsed.append((number, ExperimentRunRecord.model_validate_json(line)))
        except ValidationError as exc:
            malformed.append(f"line {number}: {exc.errors()[0].get('msg', 'invalid')} "
                             f"({exc.error_count()} error(s))")
    if malformed:
        raise Refused("malformed run record(s), nothing audited:\n  " + "\n  ".join(malformed))
    if not parsed:
        raise Refused(f"no run records in {runs}")

    first: dict[str, tuple[int, str]] = {}
    unique: list[ExperimentRunRecord] = []
    duplicates: list[tuple[int, str]] = []
    conflicts = []
    for number, record in parsed:
        run_id, canonical = record.provenance.run_id, record.model_dump_json()
        if run_id not in first:
            first[run_id] = (number, canonical)
            unique.append(record)
        elif first[run_id][1] == canonical:
            duplicates.append((number, run_id))
        else:
            conflicts.append(f"run_id {run_id}: line {first[run_id][0]} and line {number} differ")
    if conflicts:
        raise Refused("conflicting records share a run_id, nothing audited:\n  " + "\n  ".join(conflicts))
    keys = Counter((r.episode_id, r.condition.value, r.provenance.seed) for r in unique)
    repeated = [f"{e} {c} seed={s} ({n} runs)" for (e, c, s), n in sorted(keys.items(), key=str) if n > 1]
    return unique, duplicates, repeated


def _check_outputs(out_dir: Path, overwrite: bool) -> list[Path]:
    if not out_dir.is_dir():
        raise Refused(f"output directory does not exist or is not a directory: {out_dir}")
    if not os.access(out_dir, os.W_OK):
        raise Refused(f"output directory is not writable: {out_dir}")
    targets = [out_dir / name for name in REPORTS]
    for target in targets:
        if target.exists() and not target.is_file():
            raise Refused(f"output path exists and is not a file: {target}")
    existing = [str(t) for t in targets if t.exists()]
    if existing and not overwrite:
        raise Refused("refusing to overwrite existing report(s) (pass --overwrite): " + ", ".join(existing))
    return targets


def _write_together(outputs: list[tuple[Path, str]]) -> None:
    """Write every file or none: temporaries first, then move into place, rolling back on failure."""
    staged: list[tuple[Path, Path]] = []
    try:
        for target, text in outputs:
            fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.", suffix=".tmp")
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(text)
            staged.append((target, Path(tmp)))
    except BaseException:
        for _, tmp in staged:
            tmp.unlink(missing_ok=True)
        raise
    moved: list[tuple[Path, Path | None]] = []  # (target, backup of its previous content)
    try:
        for target, tmp in staged:
            backup = None
            if target.exists():
                backup = target.with_name(f".{target.name}.bak")
                os.replace(target, backup)
            moved.append((target, backup))
            os.replace(tmp, target)
    except BaseException:
        for target, backup in reversed(moved):
            if backup is not None:
                os.replace(backup, target)
            else:
                target.unlink(missing_ok=True)
        for _, tmp in staged:
            tmp.unlink(missing_ok=True)
        raise
    for _, backup in moved:
        if backup is not None:
            backup.unlink(missing_ok=True)


def _markdown(runs: Path, audits: list, duplicates: list[tuple[int, str]],
              repeated: list[str]) -> str:
    notes = [f"- Input: `{runs}`; audit version {AUDIT_VERSION}; {len(audits)} unique run(s) audited."]
    if duplicates:
        notes.append(f"- {len(duplicates)} exact duplicate line(s) audited once: "
                     + ", ".join(f"line {n} ({rid})" for n, rid in duplicates) + ".")
    if repeated:
        notes.append("- Repeated runs of one (episode, condition, seed), each audited: " + "; ".join(repeated) + ".")
    return (
        f"# Post-hoc integrity audit of `{runs.name}`\n\n"
        "Produced by `scripts/audit_integrity.py` (harness side, after the runs). Every accepted repair is re-measured "
        "the same way in every condition. Only mutants that ran to killed/survived count as measured; rates are over "
        "measured repairs only, and `–` means nothing was measured, not 0. Retention = mutants killed by both the "
        "repaired and the developer's tests / mutants the developer's tests kill, over the mutants measured in both. "
        "Per-repair detail, including unmeasured mutants: `integrity_audit.jsonl`.\n\n"
        + "\n".join(notes) + "\n\n" + markdown_table(summarize_audits(audits))
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--runs", required=True, help="an experiment's runs.jsonl")
    parser.add_argument("--repo", default=str(ROOT.parent / "boltons"), help="the pinned clone")
    parser.add_argument("--python", required=True, help="interpreter of the pinned pilot environment")
    parser.add_argument("--cached-mutants", default=None,
                        help="mutant catalogue (default: generate the Guard's mutants from the CI-state source)")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--isolation", default="auto")
    parser.add_argument("--out-dir", default=None, help="where to write the reports (default: next to --runs)")
    parser.add_argument("--overwrite", action="store_true", help="replace existing integrity_audit.{jsonl,md}")
    args = parser.parse_args(argv)
    python = os.path.abspath(args.python)  # the sandbox runs elsewhere; keep the venv symlink (no resolve)
    runs = Path(args.runs)

    try:
        targets = _check_outputs(Path(args.out_dir) if args.out_dir else runs.parent, args.overwrite)
        records, duplicates, repeated = _read_records(runs)
        splits = {r.provenance.split or "dev" for r in records}
        episodes = _load_episodes(splits)
        unknown = sorted({f"{r.episode_id} (run {r.provenance.run_id})" for r in records
                          if r.episode_id not in episodes})
        if unknown:
            raise Refused(f"episode(s) not in split(s) {sorted(splits)}, nothing audited: " + ", ".join(unknown))
    except Refused as exc:
        print(f"audit refused: {exc}", file=sys.stderr)
        return REFUSED
    for number, run_id in duplicates:
        print(f"duplicate: line {number} repeats run {run_id}; audited once", file=sys.stderr)
    for key in repeated:
        print(f"repeated runs (each audited): {key}", file=sys.stderr)

    cached = load_cached_mutants(args.cached_mutants) if args.cached_mutants else None
    materialize = _materializer()
    audits = []
    for record in records:
        audit = audit_run(record, episodes[record.episode_id], repo=Path(args.repo).resolve(),
                          materialize=materialize, python=python, timeout_seconds=args.timeout,
                          isolation=args.isolation, cached_mutants=cached)
        audits.append(audit)
        detail = "; ".join(
            f"{g.group_id} {g.scope} {g.status} weakened={g.weakened} retention={g.retention} "
            f"measured={len(g.measured_after) if g.measured_after is not None else '-'}"
            f"/{g.mutants_generated if g.mutants_generated is not None else '-'} "
            f"oracle={g.oracle_sensitive}" for g in audit.groups) or audit.reason
        print(f"{record.episode_id} {record.condition.value} seed={record.provenance.seed}: {detail}")

    jsonl = "".join(json.dumps(a.to_dict(), sort_keys=True) + "\n" for a in audits)
    _write_together([(targets[0], jsonl),
                     (targets[1], _markdown(runs, audits, duplicates, repeated))])
    print(f"wrote {targets[0]} and {targets[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
