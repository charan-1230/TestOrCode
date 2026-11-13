"""Merge the three held-out libraries into the files the runners expect: ``heldout_test.jsonl`` and
``heldout_test_agent_view.jsonl`` (``--dataset heldout --split test``).

The merge is a byte-for-byte concatenation of the per-library files (more_itertools, attrs, packaging, in that order); no
record is edited, so gold labels and reference repairs are untouched.  The per-library files stay the source of truth: this
script is deterministic, ``--check`` verifies the merged files and the manifest against them, and the manifest
(``heldout_manifest.json``) gets ``files_sha256`` for the merged files, which ``experiment.dataset_version`` reads.

    python scripts/heldout/merge_heldout.py            # write the merged files and update the manifest
    python scripts/heldout/merge_heldout.py --check    # exit 1 if they are missing or out of date
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

DATASETS = Path(__file__).resolve().parents[2] / "datasets"
LIBRARIES = ("more_itertools", "attrs", "packaging")
SUFFIXES = {"heldout_test.jsonl": "", "heldout_test_agent_view.jsonl": "_agent_view"}


def _expected() -> dict[str, bytes]:
    out = {}
    for name, suffix in SUFFIXES.items():
        parts = []
        for lib in LIBRARIES:
            data = (DATASETS / f"heldout_{lib}{suffix}.jsonl").read_bytes()
            parts.append(data if data.endswith(b"\n") else data + b"\n")
        out[name] = b"".join(parts)
    return out


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    expected = _expected()
    manifest_path = DATASETS / "heldout_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    hashes = {name: _sha(data) for name, data in expected.items()}
    if args.check:
        bad = [n for n, d in expected.items() if not (DATASETS / n).is_file() or (DATASETS / n).read_bytes() != d]
        if manifest.get("files_sha256") != hashes:
            bad.append("heldout_manifest.json files_sha256")
        if bad:
            sys.stderr.write("out of date: " + ", ".join(bad) + "\n")
            return 1
        sys.stdout.write("merged held-out files and manifest are up to date\n")
        return 0
    for name, data in expected.items():
        (DATASETS / name).write_bytes(data)
    manifest["files_sha256"] = hashes
    manifest["merged_from"] = [f"heldout_{lib}[_agent_view].jsonl" for lib in LIBRARIES]
    manifest_path.write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    sys.stdout.write(f"wrote {', '.join(SUFFIXES)} and updated heldout_manifest.json\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
