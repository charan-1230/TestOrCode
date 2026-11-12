"""The committed held-out evaluation set (proposal section 9): 3 libraries x 24 episodes (offline, no git needed)."""

from __future__ import annotations

import hashlib
import json
from collections import Counter

import pytest

from testorcode.dataset import DATASETS_DIR, load_agent_inputs, load_episodes
from testorcode.schemas.episode_schemas import Episode, ScenarioType, find_leaked_ground_truth

MANIFEST = DATASETS_DIR / "heldout_manifest.json"
TARGET = {
    "S1": 10,
    "S2": 4,
    "S3": 4,
    "S4": 3,
    "S5": 3,
}  # proposal Table 3 (30/12/12/9/9) split over 3 libraries

pytestmark = pytest.mark.skipif(not MANIFEST.exists(), reason="held-out set not built")


@pytest.fixture(scope="module")
def manifest() -> dict:
    return json.loads(MANIFEST.read_text())


@pytest.fixture(scope="module")
def streams(manifest) -> dict[str, list[Episode]]:
    return {
        lib["library"]: load_episodes(DATASETS_DIR / f"heldout_{lib['library']}.jsonl")
        for lib in manifest["libraries"]
    }


@pytest.fixture(scope="module")
def verification(manifest) -> dict[str, dict]:
    return {
        lib["library"]: json.loads(
            (DATASETS_DIR / f"heldout_{lib['library']}_verification.json").read_text()
        )
        for lib in manifest["libraries"]
    }


def test_three_libraries_with_24_episodes_each(manifest, streams) -> None:
    assert len(streams) == 3 and manifest["total_episodes"] == 72
    for lib, eps in streams.items():
        assert len(eps) == 24, lib
        assert Counter(e.scenario.value for e in eps) == TARGET, lib


def test_files_match_manifest_hashes(manifest) -> None:
    for lib in manifest["libraries"]:
        for name, digest in lib["files_sha256"].items():
            assert hashlib.sha256((DATASETS_DIR / name).read_bytes()).hexdigest() == digest, name


def test_agent_view_is_exactly_agent_input_and_leak_free(streams) -> None:
    for lib, eps in streams.items():
        views = load_agent_inputs(DATASETS_DIR / f"heldout_{lib}_agent_view.jsonl")
        assert [v.model_dump(mode="json") for v in views] == [e.agent_payload() for e in eps]
        for ep in eps:
            assert not find_leaked_ground_truth(ep.agent_payload()), ep.episode_id


def test_streams_are_chronological_and_isolated(streams) -> None:
    repo_ids = set()
    for lib, eps in streams.items():
        assert [e.stream_position for e in eps] == list(range(24)), lib
        assert len({e.stream_id for e in eps}) == 1
        # topological order: a later episode's CI commit is never an ancestor of an earlier one, i.e. it is in every
        # earlier episode's future (ancestor counts are not monotonic across merged branches, so stream_index is not
        # compared)
        for i, earlier in enumerate(eps):
            future = set(earlier.ground_truth.future_commit_shas)
            assert all(e.agent_input.ci_event.commit.sha in future for e in eps[i + 1 :]), (
                earlier.episode_id
            )
        assert len({e.agent_input.ci_event.commit.sha for e in eps}) == 24, lib
        ids = {e.agent_input.repo.repo_id for e in eps}
        assert (
            len(ids) == 1 and not ids & repo_ids and "repo-a" not in ids
        )  # repo-a is the boltons dev library
        repo_ids |= ids


def test_s4_decoys_come_from_earlier_approved_episodes(streams) -> None:
    for eps in streams.values():
        by_commit = {e.agent_input.ci_event.commit.sha: e for e in eps}
        for ep in eps:
            if ep.scenario is not ScenarioType.S4_DECOY_REGRESSION:
                continue
            (record,) = ep.agent_input.visible_memory
            source = by_commit[record.approved_at_commit]
            assert source.stream_position < ep.stream_position
            assert source.scenario in (ScenarioType.S1_OUTDATED_TEST, ScenarioType.S3_MIXED)


def test_verification_records_every_executed_check(streams, verification) -> None:
    for lib, eps in streams.items():
        for ep in eps:
            rec = verification[lib][ep.episode_id]
            assert rec["stable_runs"] == 5 and rec["round_trip_reproduces_failures"], ep.episode_id
            assert rec["leak_audit"]["clean"], ep.episode_id
            if ep.scenario is ScenarioType.S4_DECOY_REGRESSION:
                assert rec["runtime_decoy_probe"] == "passed", ep.episode_id
            if ep.scenario in (ScenarioType.S1_OUTDATED_TEST, ScenarioType.S5_INTENT_STRIPPED):
                assert rec["notes"]["reference_repair_green"], ep.episode_id
            if ep.scenario is ScenarioType.S5_INTENT_STRIPPED:
                assert rec["leak_audit"]["real_message_absent_from_agent_view"], ep.episode_id
