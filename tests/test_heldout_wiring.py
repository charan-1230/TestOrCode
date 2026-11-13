"""The 72 held-out episodes run through the ordinary runner: ``--dataset heldout --split test`` + ``--repos``.

The merged files are a byte-for-byte concatenation of the per-library files (``scripts/heldout/merge_heldout.py``); gold labels and
reference repairs are never edited.  Pinned clones and interpreters are checked when they are present on this machine.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest

from testorcode.dataset import agent_view_path, episode_path
from testorcode.experiment import ExperimentConfig, dataset_version, load_repos, load_split
from testorcode.schemas.result_schemas import ExperimentCondition

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ROOT / "datasets"
pytestmark = pytest.mark.skipif(not (DATASETS / "heldout_attrs.jsonl").is_file(), reason="held-out dataset missing")
LIBS = ("more_itertools", "attrs", "packaging")


def _config(**extra) -> ExperimentConfig:
    return ExperimentConfig(split="test", final=True, dataset="heldout", conditions=(ExperimentCondition.FULL,),
                            out=Path("unused"), **extra)


def test_merged_files_are_the_concatenation_of_the_library_files() -> None:
    for suffix, path in (("", episode_path("test", dataset="heldout")),
                         ("_agent_view", agent_view_path("test", dataset="heldout"))):
        expected = b"".join((DATASETS / f"heldout_{lib}{suffix}.jsonl").read_bytes() for lib in LIBS)
        assert path.read_bytes() == expected  # no record edited: gold labels and reference repairs untouched


def test_merge_script_check_passes_and_manifest_hashes_match() -> None:
    manifest = json.loads((DATASETS / "heldout_manifest.json").read_text())
    for name, digest in manifest["files_sha256"].items():
        assert hashlib.sha256((DATASETS / name).read_bytes()).hexdigest() == digest, name
    assert subprocess.run(["python3", str(ROOT / "scripts" / "heldout" / "merge_heldout.py"), "--check"]).returncode == 0
    # the per-library manifests still describe the unmerged files
    for lib in manifest["libraries"]:
        for name, digest in lib["files_sha256"].items():
            assert hashlib.sha256((DATASETS / name).read_bytes()).hexdigest() == digest, name


def test_runner_loads_all_72_episodes_in_stream_order() -> None:
    pairs = load_split(_config())
    assert len(pairs) == 72
    by_stream: dict[str, list[int]] = {}
    for _, gold in pairs:
        by_stream.setdefault(gold.stream_id, []).append(gold.stream_position)
    assert len(by_stream) == 3 and all(len(v) == 24 and v == sorted(v) for v in by_stream.values())
    assert {inp.repo.repo_id for inp, _ in pairs} == {"repo-b", "repo-c", "repo-d"}
    assert dataset_version("test", "heldout").startswith("heldout@")


def test_agent_view_file_has_no_ground_truth() -> None:
    from testorcode.dataset import load_agent_inputs

    assert len(load_agent_inputs(agent_view_path("test", dataset="heldout"))) == 72  # raises on any leaked key


def test_heldout_has_no_dev_split() -> None:
    config = ExperimentConfig(split="dev", dataset="heldout", conditions=(ExperimentCondition.FULL,), out=Path("x"))
    with pytest.raises(FileNotFoundError):  # only the frozen held-out set exists
        load_split(config)


# --- repos mapping ---------------------------------------------------------------------------------------------------


def test_load_repos_resolves_relative_paths_and_keeps_symlinks(tmp_path: Path) -> None:
    real = tmp_path / "real" / "bin"
    real.mkdir(parents=True)
    (real / "python3.12").write_text("")
    link = tmp_path / "venv" / "bin" / "python"
    link.parent.mkdir(parents=True)
    os.symlink(real / "python3.12", link)
    (tmp_path / "clone").mkdir()
    mapping = tmp_path / "cfg" / "repos.json"
    mapping.parent.mkdir()
    mapping.write_text(json.dumps({"repo-x": {"path": "../clone", "python": "../venv/bin/python"}}))
    env = load_repos(mapping)["repo-x"]
    assert env.path == tmp_path / "clone" and env.python == link and env.python.is_symlink()  # venv kept, not resolved


def test_load_repos_rejects_bad_files(tmp_path: Path) -> None:
    bad = tmp_path / "r.json"
    bad.write_text("[]")
    with pytest.raises(ValueError):
        load_repos(bad)
    bad.write_text(json.dumps({"repo-x": {"path": "p", "interpreter": "x"}}))
    with pytest.raises(ValueError):
        load_repos(bad)


def test_repos_file_maps_every_heldout_repo_id() -> None:
    repos = load_repos(ROOT / "configs" / "repos.heldout.json")
    assert set(repos) == {inp.repo.repo_id for inp, _ in load_split(_config())}


def test_pinned_clones_and_interpreters_match_the_dataset() -> None:
    repos = load_repos(ROOT / "configs" / "repos.heldout.json")
    if not all(env.path.is_dir() and env.python and env.python.exists() for env in repos.values()):
        pytest.skip("held-out clones or venvs are not on this machine")
    manifest = json.loads((DATASETS / "heldout_manifest.json").read_text())
    pins = {lib["library"]: lib for lib in manifest["libraries"]}
    seen = {}
    for inp, _ in load_split(_config()):
        seen[inp.repo.repo_id] = inp.environment.python_version
    for lib_name, repo_id in (("more_itertools", "repo-b"), ("attrs", "repo-d"), ("packaging", "repo-c")):
        env = repos[repo_id]
        head = subprocess.run(["git", "-C", str(env.path), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        assert head == pins[lib_name]["head"], lib_name
        version = subprocess.run([str(env.python), "-c", "import sys; print('%d.%d' % sys.version_info[:2])"],
                                 capture_output=True, text=True).stdout.strip()
        assert version == seen[repo_id], lib_name
