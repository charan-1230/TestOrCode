"""The committed pilot dataset: validity, split hygiene, leakage and freeze checks (offline, no git needed)."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import pytest

from testorcode.dataset import (
    DATASETS_DIR,
    agent_view_path,
    episode_path,
    load_agent_inputs,
    load_episodes,
)
from testorcode.schemas.episode_schemas import (
    DecisionLabel,
    Episode,
    EpisodeInput,
    RepairDirection,
    ScenarioType,
    find_leaked_ground_truth,
)

S = ScenarioType


@pytest.fixture(scope="module")
def episodes() -> dict[str, list[Episode]]:
    return {split: load_episodes(episode_path(split)) for split in ("dev", "test")}


@pytest.fixture(scope="module")
def views() -> dict[str, list[EpisodeInput]]:
    return {split: load_agent_inputs(agent_view_path(split)) for split in ("dev", "test")}


def scenarios(eps: list[Episode]) -> Counter:
    return Counter(e.scenario for e in eps)


def test_split_sizes_and_scenario_mix(episodes) -> None:
    assert len(episodes["dev"]) == 8 and len(episodes["test"]) == 4
    assert scenarios(episodes["dev"]) == {S.S1_OUTDATED_TEST: 2, S.S2_INJECTED_REGRESSION: 3, S.S3_MIXED: 1,
                                          S.S4_DECOY_REGRESSION: 1, S.S5_INTENT_STRIPPED: 1}
    assert scenarios(episodes["test"]) == {S.S2_INJECTED_REGRESSION: 2, S.S3_MIXED: 1, S.S4_DECOY_REGRESSION: 1}


def test_ids_are_opaque_unique_and_in_stream_order(episodes) -> None:
    for split, eps in episodes.items():
        ids = [e.episode_id for e in eps]
        assert ids == [f"{split}-{i:02d}" for i in range(1, len(eps) + 1)]
        assert [e.stream_position for e in eps] == list(range(len(eps)))
        assert {e.stream_id for e in eps} == {f"pilot-{split}"}
        indexes = [e.agent_input.ci_event.commit.stream_index for e in eps]
        assert indexes == sorted(indexes)  # chronological stream


def test_agent_views_match_full_records(episodes, views) -> None:
    for split in ("dev", "test"):
        assert [v.episode_id for v in views[split]] == [e.episode_id for e in episodes[split]]
        for view, full in zip(views[split], episodes[split], strict=True):
            assert view == full.agent_input


def test_agent_views_leak_no_ground_truth(episodes) -> None:
    for split in ("dev", "test"):
        for line in agent_view_path(split).read_text().splitlines():
            record = json.loads(line)
            assert find_leaked_ground_truth(record) == []
            assert "ground_truth" not in record and "scenario" not in record and "fault_plan" not in record
    for eps in episodes.values():
        for ep in eps:
            text = json.dumps(ep.agent_payload())
            for sha in ep.ground_truth.future_commit_shas:
                assert sha[:12] not in text, f"{ep.episode_id} exposes future commit {sha[:12]}"
            for mutant in ep.ground_truth.injected_mutants:
                assert mutant.mutant_id not in text
                assert (mutant.operator or "\0") not in text


def test_dev_and_test_are_disjoint(episodes) -> None:
    def facets(eps: list[Episode]) -> dict[str, set[str]]:
        return {
            "commits": {e.agent_input.ci_event.commit.sha for e in eps},
            "functions": {lbl.function_qualname for e in eps for lbl in e.ground_truth.per_test_labels},
            "modules": {lbl.function_qualname.split(".")[1] for e in eps for lbl in e.ground_truth.per_test_labels},
            "tests": {lbl.test_id for e in eps for lbl in e.ground_truth.per_test_labels},
        }

    dev, test = facets(episodes["dev"]), facets(episodes["test"])
    for kind in dev:
        assert not dev[kind] & test[kind], f"dev/test share {kind}: {dev[kind] & test[kind]}"


def test_environment_is_pinned_and_offline(views) -> None:
    pins = {(v.environment.python_version, v.environment.pytest_version, v.environment.stability_runs,
             v.environment.network_enabled, v.environment.dependency_lock_hash) for vs in views.values() for v in vs}
    assert len(pins) == 1
    python, pytest_version, runs, network, lock = next(iter(pins))
    assert (python, pytest_version, runs, network) == ("3.12", "8.4.2", 5, False)
    assert lock and re.fullmatch(r"[0-9a-f]{64}", lock)


def test_gold_labels_follow_scenarios(episodes) -> None:
    for ep in (e for eps in episodes.values() for e in eps):
        labels = ep.ground_truth.per_test_labels
        failing = {t.test_id for t in ep.agent_input.ci_event.failing_tests}
        assert {lbl.test_id for lbl in labels} == failing
        if ep.scenario is S.S5_INTENT_STRIPPED:
            assert {lbl.expected_action for lbl in labels} == {DecisionLabel.ESCALATE}
            assert {lbl.gold_direction for lbl in labels} == {RepairDirection.UPDATE_TEST}
        if ep.scenario in (S.S2_INJECTED_REGRESSION, S.S4_DECOY_REGRESSION):
            assert {lbl.expected_action for lbl in labels} == {DecisionLabel.FIX_CODE}
            assert not ep.ground_truth.reference_repairs  # tests stay untouched
        if ep.scenario is S.S3_MIXED:
            assert {lbl.expected_action for lbl in labels} == {DecisionLabel.UPDATE_TEST, DecisionLabel.FIX_CODE}
        if ep.scenario is S.S1_OUTDATED_TEST:
            assert all(lbl.gold_evidence_ids for lbl in labels)


def test_gold_evidence_exists_in_the_corpus(episodes) -> None:
    for ep in (e for eps in episodes.values() for e in eps):
        corpus = {i.evidence_id for i in ep.agent_input.corpus.items or []}
        for lbl in ep.ground_truth.per_test_labels:
            assert set(lbl.gold_evidence_ids or []) <= corpus


def test_s4_decoy_memory_comes_from_an_earlier_episode_in_the_same_stream(episodes) -> None:
    for eps in episodes.values():
        for ep in (e for e in eps if e.scenario is S.S4_DECOY_REGRESSION):
            (record,) = ep.agent_input.visible_memory or []
            assert record.record_id in ep.ground_truth.decoy_memory_record_ids
            earlier = [e for e in eps if e.stream_position < ep.stream_position]
            assert record.approved_at_commit in {e.agent_input.ci_event.commit.sha for e in earlier}
            assert {lbl.function_qualname for lbl in ep.ground_truth.per_test_labels} == {record.function_qualname}
            assert all(lbl.memory_applies is False for lbl in ep.ground_truth.per_test_labels)
            assert record.probe.probe_code and record.probe.probe_selector
            assert not {t.test_id for t in ep.agent_input.ci_event.failing_tests} & {record.probe.probe_selector}


def test_s5_hides_the_intent_evidence(episodes) -> None:
    (ep,) = [e for e in episodes["dev"] if e.scenario is S.S5_INTENT_STRIPPED]
    assert ep.agent_input.visible_memory == []
    assert len(ep.agent_input.ci_event.commit_message or "") < 40
    assert ep.ground_truth.stripped_evidence
    stripped_docs = {s.source_ref for s in ep.ground_truth.stripped_evidence if s.source_ref}
    assert not {i.source_ref for i in ep.agent_input.corpus.items or []} & {p for p in stripped_docs if p.startswith("docs/")}


def test_tool_fault_plans_exist_only_in_dev_and_outside_the_agent_view(episodes) -> None:
    assert any(e.fault_plan.enabled for e in episodes["dev"])
    assert not any(e.fault_plan.enabled for e in episodes["test"])
    for ep in (e for e in episodes["dev"] if e.fault_plan.enabled):
        text = json.dumps(ep.agent_payload())
        assert "fault_plan" not in text
        assert not any(i.fault_kind.value in text for i in ep.fault_plan.injections)


def test_frozen_files_match_their_recorded_hashes() -> None:
    manifest = json.loads((DATASETS_DIR / "pilot_manifest.json").read_text())
    for name, digest in manifest["files_sha256"].items():
        assert hashlib.sha256((DATASETS_DIR / name).read_bytes()).hexdigest() == digest, (
            f"{name} changed since it was generated; the test split is frozen (see datasets/README.md)"
        )


def test_verification_record_covers_every_episode(episodes) -> None:
    report = json.loads((DATASETS_DIR / "pilot_verification.json").read_text())
    assert set(report) == {e.episode_id for eps in episodes.values() for e in eps}
    for ep_id, rec in report.items():
        assert rec["stable_runs"] == 5 and rec["round_trip_reproduces_failures"] and rec["leak_audit"]["clean"], ep_id


def test_loaders_refuse_the_wrong_kind_of_file() -> None:
    with pytest.raises(ValueError, match="ground-truth keys"):
        load_agent_inputs(episode_path("dev"))
    with pytest.raises(ValueError, match="agent-view"):
        load_episodes(agent_view_path("dev"))
    with pytest.raises(ValueError, match="split"):
        agent_view_path("train")


def test_loader_types(views: dict[str, list[EpisodeInput]]) -> None:
    assert all(isinstance(v, EpisodeInput) for vs in views.values() for v in vs)
    assert Path(agent_view_path("dev")).name == "pilot_dev_agent_view.jsonl"
