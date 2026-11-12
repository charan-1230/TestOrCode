"""Build one held-out evaluation stream (proposal section 9: 3 libraries x 24 episodes) with the pilot builder.

Each library has ``scripts/heldout/<lib>/manifest.json`` (same format as ``scripts/pilot/manifest.json`` plus a
``library`` block) and ``requirements.txt`` (exact ``pip freeze`` of the test interpreter).  The pilot builder does all
the executed checks; the held-out rules it switches on are: test-body failures only (no collection/setup errors or
missing-API AttributeError/ImportError/NameError), S5 strips every indexed doc /
changelog / README file the CI commit changed, and unittest-method probes are reduced to their class.

After the build, every S4 decoy probe is additionally run through the runtime probe runner
(``testorcode.tools.probes.run_executable_probe``, after ``MemoryStore.seed``-style completion with the test module's
support code) on the materialised CI state: the decoy only tests what it should if the probe PASSES there.

    python scripts/heldout/build_heldout.py more_itertools --repo <clone> --python <venv>/bin/python \\
        --work <scratch>/wt-heldout-mit [--out datasets]
    python scripts/heldout/build_heldout.py --aggregate [--out datasets]   # writes heldout_manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "pilot"))

import build_dataset as bd
from materialize_episode import materialize

from testorcode.schemas.episode_schemas import Episode
from testorcode.tools.probes import run_executable_probe, support_code

LIBRARIES = (
    "more_itertools",
    "attrs",
    "cachetools",
    "click",
    "sortedcontainers",
    "validators",
    "packaging",
    "pyparsing",
    "slugify",
    "marshmallow",
    "dateutil",
    "tomlkit",
    "toolz",
    "humanize",
)
SCENARIO_TARGET = {"S1": 10, "S2": 4, "S3": 4, "S4": 3, "S5": 3}  # proposal Table 3 / 3 libraries


def configure(lib: str) -> dict:
    manifest = json.loads((HERE / lib / "manifest.json").read_text())
    cfg = dict(manifest["library"])
    abbr = cfg.pop("abbr")
    for key in ("pytest_args", "doc_files"):
        if key in cfg:
            cfg[key] = tuple(cfg[key])
    bd.LIB = bd.Library(
        **cfg,
        corpus_version="heldout-1",
        stream_prefix=f"heldout-{abbr}",
        id_prefix={"test": abbr},
        file_prefix=f"heldout_{lib}",
        split_in_filename=False,
        requirements=str(HERE / lib / "requirements.txt"),
        test_body_failures_only=True,
        strip_changed_docs=True,
        class_probes=True,
        memory_limit_mb=1536,
        hash_seed="0",
        normalize_ids=True,
    )
    return manifest


def check_counts(manifest: dict) -> None:
    got: dict[str, int] = {}
    for spec in manifest["episodes"]:
        if spec["split"] != "test":
            raise SystemExit(f"{spec['key']}: held-out episodes use split 'test'")
        got[spec["scenario"]] = got.get(spec["scenario"], 0) + 1
    if got != SCENARIO_TARGET:
        raise SystemExit(f"scenario counts {got} != proposal per-library target {SCENARIO_TARGET}")


def verify_s4_probes(lib: str, repo: Path, python: Path, work: Path, out: Path) -> dict[str, str]:
    """Runtime probe of each S4 decoy record on the materialised CI state; must PASS."""
    results = {}
    path = out / f"{bd.LIB.stem('test')}.jsonl"
    for line in path.read_text().splitlines():
        ep = Episode.model_validate_json(line)
        if ep.scenario.value != "S4":
            continue
        materialize(repo, work, ep.agent_payload())
        for rec in ep.agent_input.visible_memory:
            probe = rec.probe
            module = work / (probe.probe_selector or "").split("::", 1)[0]
            code = support_code(module.read_text()) + "\n" + probe.probe_code
            res = run_executable_probe(
                work,
                probe.model_copy(update={"probe_code": code}),
                policy=ep.agent_input.edit_policy,
                timeout_seconds=120,
                python_executable=python,
                isolation="bwrap",
            )
            results[f"{ep.episode_id}:{rec.record_id}"] = res.status.value
            if res.status.value != "passed":
                raise SystemExit(
                    f"{ep.episode_id}: runtime decoy probe {res.status.value}: {res.reasons}"
                )
    return results


def aggregate(out: Path) -> int:
    libs = []
    for lib in LIBRARIES:
        man = out / f"heldout_{lib}_manifest.json"
        if not man.exists():
            continue
        data = json.loads(man.read_text())
        eps = data["splits"]["test"]
        counts: dict[str, int] = {}
        for e in eps:
            counts[e["scenario"]] = counts.get(e["scenario"], 0) + 1
        libs.append(
            {
                "library": lib,
                "repo_url": data["pin"]["repo_url"],
                "head": data["pin"]["head"],
                "python": data["pin"]["python"],
                "pytest": data["pin"]["pytest"],
                "dependency_lock_hash": data["dependency_lock_hash"],
                "episodes": len(eps),
                "scenario_counts": dict(sorted(counts.items())),
                "files_sha256": data["files_sha256"],
                "manifest_sha256": hashlib.sha256(man.read_bytes()).hexdigest(),
                "verification_sha256": hashlib.sha256(
                    (out / f"heldout_{lib}_verification.json").read_bytes()
                ).hexdigest(),
            }
        )
    summary = {
        "description": "Held-out evaluation set (proposal section 9): 3 libraries x 24 episodes, one chronological "
        "stream per library. Gold labels live only in heldout_<lib>.jsonl; agents receive heldout_<lib>_agent_view.jsonl.",
        "scenario_target_per_library": SCENARIO_TARGET,
        "libraries": libs,
        "total_episodes": sum(x["episodes"] for x in libs),
    }
    (out / "heldout_manifest.json").write_text(
        json.dumps(summary, indent=1) + "\n", encoding="utf-8"
    )
    sys.stdout.write(
        f"heldout_manifest.json: {len(libs)} libraries, {summary['total_episodes']} episodes\n"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("lib", nargs="?", choices=LIBRARIES)
    parser.add_argument("--repo")
    parser.add_argument("--python")
    parser.add_argument("--work")
    parser.add_argument("--out", default=str(ROOT / "datasets"))
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    out = Path(args.out)
    if args.aggregate:
        return aggregate(out)
    if not (args.lib and args.repo and args.python and args.work):
        parser.error("lib, --repo, --python and --work are required")
    manifest = configure(args.lib)
    check_counts(manifest)
    argv = [
        "--repo",
        args.repo,
        "--python",
        args.python,
        "--work",
        args.work,
        "--manifest",
        str(HERE / args.lib / "manifest.json"),
        "--out",
        str(out),
    ]
    if args.force:
        argv.append("--force")
    rc = bd.main(argv)
    if rc:
        return rc
    probes = verify_s4_probes(
        args.lib, Path(args.repo), Path(args.python), Path(f"{args.work}-rt"), out
    )
    ver_path = out / f"{bd.LIB.file_prefix}_verification.json"
    ver = json.loads(ver_path.read_text())
    for key, status in probes.items():
        ver[key.split(":")[0]]["runtime_decoy_probe"] = status
    ver_path.write_text(json.dumps(ver, indent=1) + "\n", encoding="utf-8")
    sys.stdout.write(f"S4 runtime probes: {probes}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
