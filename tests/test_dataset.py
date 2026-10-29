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
    EvidenceSourceType,
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
    # 13 episodes in total: this exceeds the 5-day plan's 6-12 target (documented in datasets/README.md).
    assert len(episodes["dev"]) == 9 and len(episodes["test"]) == 4
    assert scenarios(episodes["dev"]) == {S.S1_OUTDATED_TEST: 3, S.S2_INJECTED_REGRESSION: 3, S.S3_MIXED: 1,
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


# ---- revision 2: source-comment evidence policy, dev-09 as S1, S5 on 353f6bef -------------------------------------------

DEV_09_SHA = "6270b5e35ed902859df6188970eec5bee5ab73d1"  # iterutils remap namedtuples (was S5, now S1)
DEV_08_SHA = "f387b64c4b7c4267ea94bcfcd229923ab2f598ea"  # strutils bytes2human + listutils mutant (S3)
S5_SHA = "353f6befb4739f1d3583a8ce0ab5db761babd411"  # tbutils ParsedException.from_string
S5_WITHHELD_MESSAGE = "Exception message should not start with whitespace"


def by_commit(episodes, sha: str) -> Episode:
    (ep,) = [e for eps in episodes.values() for e in eps if e.agent_input.ci_event.commit.sha == sha]
    return ep


def test_corpus_version_and_source_comment_boundary(episodes) -> None:
    """Eligible: comments in production source (boltons/). Excluded: anything from test files."""
    for ep in (e for eps in episodes.values() for e in eps):
        corpus = ep.agent_input.corpus
        assert corpus.corpus_version == "pilot-2"
        ids = [i.evidence_id for i in corpus.items or []]
        assert len(ids) == len(set(ids))
        assert not any(i.source_type is EvidenceSourceType.TEST_FILE for i in corpus.items or [])
        for item in (i for i in corpus.items or [] if i.evidence_id.startswith("comment:")):
            assert item.source_type is EvidenceSourceType.SOURCE_FILE
            path, _, function = item.source_ref.partition("::")  # same convention as docstring items
            assert path.startswith("boltons/") and path.endswith(".py") and function
            assert item.text and item.text.lstrip().startswith("#")
            assert item.line_start is not None and item.line_end is not None and item.line_start <= item.line_end
            assert item.commit_sha and item.commit_index is not None and item.commit_index <= corpus.history_cutoff_index
            assert not item.source_ref.startswith("tests/")


def test_gold_evidence_resolves_and_sits_in_the_labelled_function(episodes) -> None:
    for ep in (e for eps in episodes.values() for e in eps):
        corpus = {i.evidence_id: i for i in ep.agent_input.corpus.items or []}
        for label in ep.ground_truth.per_test_labels:
            for gid in label.gold_evidence_ids or []:
                assert gid in corpus, f"{ep.episode_id}: gold {gid} is not an agent-visible corpus item"
                if gid.startswith("comment:"):
                    module_function = label.function_qualname.removeprefix("boltons.")
                    assert gid.startswith(f"comment:{module_function}#"), (ep.episode_id, gid)


def test_dev_08_update_half_is_supported_by_the_bytes2human_comment_only(episodes) -> None:
    ep = by_commit(episodes, DEV_08_SHA)
    corpus = {i.evidence_id: i for i in ep.agent_input.corpus.items or []}
    update = [lbl for lbl in ep.ground_truth.per_test_labels if lbl.expected_action is DecisionLabel.UPDATE_TEST]
    assert len(update) == 1 and update[0].function_qualname == "boltons.strutils.bytes2human"
    (gold,) = update[0].gold_evidence_ids or []
    assert gold.startswith("comment:strutils.bytes2human#")
    assert "pick the unit after rounding" in (corpus[gold].text or "")
    fix = [lbl for lbl in ep.ground_truth.per_test_labels if lbl.expected_action is DecisionLabel.FIX_CODE]
    assert fix and all(not lbl.gold_evidence_ids for lbl in fix)


def test_dev_09_is_a_clean_s1_with_agent_visible_gold_evidence(episodes) -> None:
    ep = by_commit(episodes, DEV_09_SHA)
    assert ep.scenario is S.S1_OUTDATED_TEST and ep.stream_id == "pilot-dev"
    gt, ci = ep.ground_truth, ep.agent_input.ci_event
    assert {lbl.expected_action for lbl in gt.per_test_labels} == {DecisionLabel.UPDATE_TEST}
    assert {lbl.gold_direction for lbl in gt.per_test_labels} == {RepairDirection.UPDATE_TEST}
    assert not gt.stripped_evidence and gt.expected_episode_outcome is None and gt.reference_repairs
    assert ci.commit_message and "fix remap namedtuples" in ci.commit_message  # the real, unstripped message
    corpus = {i.evidence_id: i for i in ep.agent_input.corpus.items or []}
    gold = set().union(*(set(lbl.gold_evidence_ids or []) for lbl in gt.per_test_labels))
    assert f"msg:{DEV_09_SHA[:12]}" in gold
    (comment,) = [g for g in gold if g.startswith("comment:")]
    assert "namedtuple, rebuilt in default_exit" in (corpus[comment].text or "")


def test_s5_on_353f6bef_is_provisional_and_hides_the_message(episodes) -> None:
    ep = by_commit(episodes, S5_SHA)
    assert ep.scenario is S.S5_INTENT_STRIPPED
    gt = ep.ground_truth
    assert {lbl.expected_action for lbl in gt.per_test_labels} == {DecisionLabel.ESCALATE}
    assert ep.agent_input.visible_memory == [] and ep.agent_input.ci_event.commit_message == "tbutils: update"
    assert {s.source_type for s in gt.stripped_evidence} == {EvidenceSourceType.COMMIT_MESSAGE}
    assert S5_WITHHELD_MESSAGE not in json.dumps(ep.agent_payload()), "the withheld message is visible to the agent"
    assert not any(S5_WITHHELD_MESSAGE in (i.text or "") for i in ep.agent_input.corpus.items or [])
    assert (gt.note or "").startswith("PROVISIONAL"), "S5 must stay provisional until history access is tested"


def test_verification_file_stores_only_a_hash_of_the_withheld_message() -> None:
    """The human-only verification file must not carry the withheld S5 message in clear text (authorised check: datasets/README.md)."""
    raw = (DATASETS_DIR / "pilot_verification.json").read_text()
    assert S5_WITHHELD_MESSAGE not in raw
    ((_, record),) = [(k, v) for k, v in json.loads(raw).items() if v["scenario"] == "S5"]
    notes = record["notes"]
    assert "real_message" not in notes
    assert notes["withheld_message_sha256"] == hashlib.sha256(S5_WITHHELD_MESSAGE.encode("utf-8")).hexdigest()
    assert notes["withheld_message_chars"] == len(S5_WITHHELD_MESSAGE)


def test_history_access_tripwire_for_the_provisional_s5() -> None:
    """The agent-facing history tools are stubs, so the S5 leak path (git_show of the CI commit) cannot be tested yet.

    When they are implemented this test fails ON PURPOSE: replace it with a regression test that, with the cutoff at the
    parent commit, git_log/git_show/git_blame cannot return the CI commit's message or test diff, then clear the
    PROVISIONAL note on the S5 episode.
    """
    from testorcode.tools import git_tools

    for call in (
        lambda: git_tools.git_log(Path("."), cutoff_commit="a1b2c3d4"),
        lambda: git_tools.git_show(Path("."), "a1b2c3d4", cutoff_commit="a1b2c3d4"),
        lambda: git_tools.git_blame(Path("."), "x.py", 1, 1, cutoff_commit="a1b2c3d4"),
    ):
        with pytest.raises(NotImplementedError):
            call()


def test_gold_evidence_coverage_is_the_documented_one(episodes) -> None:
    """datasets/README.md "Review status": which episodes carry gold evidence ids, and that only UPDATE_TEST labels do."""
    labelled = {}
    every = [ep for split in episodes.values() for ep in split]
    for ep in every:
        gold = [lbl for lbl in ep.ground_truth.per_test_labels if lbl.gold_evidence_ids]
        if gold:
            labelled[ep.episode_id] = {lbl.expected_action for lbl in gold}
    assert set(labelled) == {"dev-05", "dev-06", "dev-08", "dev-09", "test-03"}
    assert all(actions == {DecisionLabel.UPDATE_TEST} for actions in labelled.values())
    assert len(every) == 13
