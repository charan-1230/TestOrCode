"""Tests for ``testorcode.episode_schemas``.

All data is synthetic.  No claim is made about any real repository history.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from testorcode.episode_schemas import (
    Availability,
    CIEvent,
    CommitRef,
    DecisionLabel,
    EditPolicy,
    Episode,
    EpisodeGroundTruth,
    EvidenceItem,
    EvidenceSourceType,
    FailureGroup,
    LabelProvenance,
    MemoryRecord,
    MutantSpec,
    PatchScope,
    PathClass,
    PerTestGoldLabel,
    RepairDirection,
    RetrievalCorpusSpec,
    ScenarioType,
    ToolFaultInjection,
    ToolFaultKind,
    ToolFaultPlan,
    ToolName,
    find_leaked_ground_truth,
)

from testorcode.src.testorcode.schemas.tests.conftest import (  # noqa: E402  (pytest adds tests/ to sys.path)
    FUTURE,
    HEAD,
    ci_event,
    failing_test,
    intent_evidence,
    provenance_label,
)


# --------------------------------------------------------------------------- #
# 1-5. Valid episodes, one per scenario
# --------------------------------------------------------------------------- #


def test_s1_outdated_test_episode(s1_episode: Episode) -> None:
    assert s1_episode.scenario is ScenarioType.S1_OUTDATED_TEST
    labels = s1_episode.ground_truth.per_test_labels
    assert {lbl.gold_direction for lbl in labels} == {RepairDirection.UPDATE_TEST}
    assert any(
        r.kind == "developer_test_diff" for r in s1_episode.ground_truth.reference_repairs
    )
    assert not s1_episode.ground_truth.injected_mutants


def test_s2_injected_regression_episode(s2_episode: Episode) -> None:
    gt = s2_episode.ground_truth
    assert {lbl.gold_direction for lbl in gt.per_test_labels} == {
        RepairDirection.FIX_CODE
    }
    assert gt.injected_mutants and gt.injected_mutants[0].mutant_id == "mut-1"
    # tests must stay untouched: no developer test diff is recorded as reference
    assert all(r.kind != "developer_test_diff" for r in gt.reference_repairs)


def test_s3_mixed_episode_carries_both_directions(s3_episode: Episode) -> None:
    directions = {lbl.gold_direction for lbl in s3_episode.ground_truth.per_test_labels}
    assert directions == {RepairDirection.UPDATE_TEST, RepairDirection.FIX_CODE}
    actions = set(s3_episode.gold_actions().values())
    assert actions == {DecisionLabel.UPDATE_TEST, DecisionLabel.FIX_CODE}
    assert len(s3_episode.agent_input.ci_event.failure_groups) == 2


def test_s4_decoy_episode_has_prior_memory_context(s4_episode: Episode) -> None:
    gt = s4_episode.ground_truth
    assert gt.decoy_memory_record_ids == ["mem-approved-normalise"]
    assert gt.per_test_labels[0].memory_applies is False
    visible = {r.record_id for r in (s4_episode.agent_input.visible_memory or [])}
    assert "mem-approved-normalise" in visible
    # the prior record exists, but the gold direction is still FIX_CODE
    assert gt.per_test_labels[0].gold_direction is RepairDirection.FIX_CODE


def test_s5_intent_stripped_episode(s5_episode: Episode) -> None:
    gt = s5_episode.ground_truth
    assert gt.expected_episode_outcome is DecisionLabel.ESCALATE
    assert gt.per_test_labels[0].expected_action is DecisionLabel.ESCALATE
    # underlying direction is still recorded, and differs from the expected action
    assert gt.per_test_labels[0].gold_direction is RepairDirection.UPDATE_TEST
    assert gt.evidence_availability is Availability.WITHHELD
    assert len(gt.stripped_evidence) == 2
    assert s5_episode.agent_input.ci_event.commit_message is None


# --------------------------------------------------------------------------- #
# 6. Tool faults
# --------------------------------------------------------------------------- #


def test_tool_fault_plan_is_orchestrator_only(s1_episode: Episode) -> None:
    plan = ToolFaultPlan(
        enabled=True,
        seed=7,
        target_failure_rate=0.1,
        injections=[
            ToolFaultInjection(
                fault_id="f-1",
                tool=ToolName.APPLY_PATCH,
                fault_kind=ToolFaultKind.PATCH_DOES_NOT_APPLY,
                call_index=0,
            ),
            ToolFaultInjection(
                fault_id="f-2",
                tool=ToolName.RUN_TESTS,
                fault_kind=ToolFaultKind.INVALID_TEST_SELECTOR,
                trigger="probabilistic",
                probability=0.1,
            ),
        ],
    )
    episode = s1_episode.model_copy(update={"fault_plan": plan})
    assert episode.fault_plan.enabled
    # the fault plan is not reachable from the agent payload
    assert "fault_plan" not in episode.agent_payload()


def test_tool_faults_are_not_a_gold_decision_class() -> None:
    assert set(DecisionLabel) == {
        DecisionLabel.UPDATE_TEST,
        DecisionLabel.FIX_CODE,
        DecisionLabel.ESCALATE,
    }
    assert not any(member.value.lower().startswith("tool") for member in ScenarioType)


def test_fault_plan_rejects_inconsistent_configuration() -> None:
    with pytest.raises(ValidationError, match="disabled but defines injections"):
        ToolFaultPlan(enabled=False, target_failure_rate=0.1)
    with pytest.raises(ValidationError, match="must define injections"):
        ToolFaultPlan(enabled=True)
    with pytest.raises(ValidationError, match="requires call_index"):
        ToolFaultInjection(
            fault_id="f", tool=ToolName.RUN_TESTS, fault_kind=ToolFaultKind.TIMEOUT
        )


# --------------------------------------------------------------------------- #
# 7. Invalid action values
# --------------------------------------------------------------------------- #


def test_invalid_action_value_is_rejected() -> None:
    with pytest.raises(ValidationError):
        PerTestGoldLabel(
            test_id="t",
            gold_direction=RepairDirection.UPDATE_TEST,
            expected_action="DELETE_TEST",  # type: ignore[arg-type]
            cause="outdated_test",
        )


def test_escalate_is_not_a_repair_direction() -> None:
    with pytest.raises(ValidationError):
        PerTestGoldLabel(
            test_id="t",
            gold_direction="ESCALATE",  # type: ignore[arg-type]
            expected_action=DecisionLabel.ESCALATE,
            cause="outdated_test",
        )


def test_expected_action_must_match_direction_unless_escalate() -> None:
    with pytest.raises(ValidationError, match="contradicts gold_direction"):
        PerTestGoldLabel(
            test_id="t",
            gold_direction=RepairDirection.UPDATE_TEST,
            expected_action=DecisionLabel.FIX_CODE,
            cause="outdated_test",
        )


# --------------------------------------------------------------------------- #
# 8. Multiple per-test decisions in a mixed episode
# --------------------------------------------------------------------------- #


def test_mixed_episode_is_not_forced_to_one_global_action(s3_episode: Episode) -> None:
    gold = s3_episode.gold_actions()
    assert len(gold) == 2
    assert len(set(gold.values())) == 2


def test_s3_requires_regression_in_a_different_function(s3_episode: Episode) -> None:
    data = s3_episode.ground_truth.model_dump()
    for label in data["per_test_labels"]:
        label["function_qualname"] = "synthlib.core.normalise"
    with pytest.raises(ValidationError, match="different"):
        EpisodeGroundTruth.model_validate(data)


def test_s3_without_both_directions_is_rejected(s3_episode: Episode) -> None:
    data = s3_episode.ground_truth.model_dump()
    data["per_test_labels"] = data["per_test_labels"][:1]
    with pytest.raises(ValidationError, match="both UPDATE_TEST and FIX_CODE"):
        EpisodeGroundTruth.model_validate(data)


# --------------------------------------------------------------------------- #
# 9. No gold-label or reference-repair leakage
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "fixture_name", ["s1_episode", "s2_episode", "s3_episode", "s4_episode", "s5_episode"]
)
def test_agent_payload_contains_no_ground_truth(
    fixture_name: str, request: pytest.FixtureRequest
) -> None:
    episode: Episode = request.getfixturevalue(fixture_name)
    payload = episode.agent_payload()
    assert find_leaked_ground_truth(payload) == []
    blob = json.dumps(payload)
    for needle in ("gold_", "reference_repair", "expected_action", "injected_mutant"):
        assert needle not in blob
    # the developer's reference diff text itself must not appear
    for repair in episode.ground_truth.reference_repairs:
        assert repair.unified_diff not in blob


def test_episode_input_forbids_attaching_gold_fields(s1_episode: Episode) -> None:
    data = s1_episode.agent_input.model_dump()
    data["gold_direction"] = "UPDATE_TEST"
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        type(s1_episode.agent_input).model_validate(data)


def test_agent_input_is_immutable(s1_episode: Episode) -> None:
    with pytest.raises(ValidationError):
        s1_episode.agent_input.episode_id = "other"  # type: ignore[misc]


def test_leak_detector_finds_planted_ground_truth(s1_episode: Episode) -> None:
    payload = s1_episode.agent_payload()
    payload["ci_event"]["gold_evidence_ids"] = ["ev-1"]
    assert find_leaked_ground_truth(payload) == ["ci_event.gold_evidence_ids"]


# --------------------------------------------------------------------------- #
# 13. Missing and unavailable evidence
# --------------------------------------------------------------------------- #


def test_empty_corpus_differs_from_unmaterialised_corpus() -> None:
    empty = RetrievalCorpusSpec(
        corpus_id="c", history_cutoff_commit=HEAD, items=[]
    )
    deferred = RetrievalCorpusSpec(corpus_id="c", history_cutoff_commit=HEAD)
    assert empty.items == []
    assert deferred.items is None
    assert empty.model_dump()["items"] != deferred.model_dump()["items"]


def test_memory_disabled_differs_from_memory_empty(
    s5_episode: Episode, s2_episode: Episode
) -> None:
    assert s5_episode.agent_input.visible_memory == []  # enabled, nothing applies
    assert s2_episode.agent_input.visible_memory is None  # disabled for this episode


def test_gold_evidence_none_differs_from_empty_list(
    s5_episode: Episode, s2_episode: Episode
) -> None:
    assert s5_episode.ground_truth.gold_evidence_ids is None  # not annotated / stripped
    assert s2_episode.ground_truth.gold_evidence_ids == []  # annotated: none exists


def test_missing_commit_message_must_declare_availability() -> None:
    tests = [failing_test("tests/t.py::test_a", "pkg.f")]
    with pytest.raises(ValidationError, match="AVAILABLE but commit_message is None"):
        ci_event(tests, commit_message=None, availability=Availability.AVAILABLE)
    with pytest.raises(ValidationError, match="must be None unless"):
        ci_event(tests, commit_message="text", availability=Availability.WITHHELD)


def test_s5_requires_recorded_stripped_evidence(s5_episode: Episode) -> None:
    data = s5_episode.ground_truth.model_dump()
    data["stripped_evidence"] = []
    with pytest.raises(ValidationError, match="stripped_evidence"):
        EpisodeGroundTruth.model_validate(data)


# --------------------------------------------------------------------------- #
# 14. Malformed data rejection
# --------------------------------------------------------------------------- #


def test_bad_commit_sha_is_rejected() -> None:
    with pytest.raises(ValidationError):
        CommitRef(sha="not-a-sha")


def test_failure_group_referencing_unknown_test_is_rejected() -> None:
    tests = [failing_test("tests/t.py::test_a", "pkg.f")]
    with pytest.raises(ValidationError, match="unknown test ids"):
        CIEvent(
            event_id="ci",
            commit=CommitRef(sha=HEAD),
            commit_message="m",
            failing_tests=tests,
            failure_groups=[FailureGroup(group_id="g", test_ids=["tests/t.py::ghost"])],
        )


def test_unlabelled_failing_test_is_rejected(s1_episode: Episode) -> None:
    data = s1_episode.model_dump()
    data["agent_input"]["ci_event"]["failing_tests"].append(
        failing_test("tests/test_synthlib.py::test_extra", "synthlib.core.normalise").model_dump()
    )
    data["agent_input"]["ci_event"]["failure_groups"][0]["test_ids"].append(
        "tests/test_synthlib.py::test_extra"
    )
    with pytest.raises(ValidationError, match="failing tests without a gold label"):
        Episode.model_validate(data)


def test_gold_label_for_non_failing_test_is_rejected(s1_episode: Episode) -> None:
    data = s1_episode.model_dump()
    data["ground_truth"]["per_test_labels"][0]["test_id"] = "tests/t.py::ghost"
    with pytest.raises(ValidationError, match="not failing in the CI event"):
        Episode.model_validate(data)


def test_future_commit_in_corpus_is_rejected(s1_episode: Episode) -> None:
    data = s1_episode.model_dump()
    data["agent_input"]["corpus"]["items"].append(
        EvidenceItem(
            evidence_id="ev-future",
            source_type=EvidenceSourceType.COMMIT_MESSAGE,
            source_ref=FUTURE,
            commit_sha=FUTURE,
            commit_index=10,
            text="a later commit",
        ).model_dump()
    )
    with pytest.raises(ValidationError, match="future commits"):
        Episode.model_validate(data)


def test_corpus_cutoff_rejects_later_commit_index() -> None:
    with pytest.raises(ValidationError, match="future commits must never be indexed"):
        RetrievalCorpusSpec(
            corpus_id="c",
            history_cutoff_commit=HEAD,
            history_cutoff_index=5,
            items=[
                EvidenceItem(
                    evidence_id="ev",
                    source_type=EvidenceSourceType.COMMIT_MESSAGE,
                    source_ref=HEAD,
                    commit_index=6,
                )
            ],
        )


def test_corpus_cutoff_must_match_current_commit(s1_episode: Episode) -> None:
    data = s1_episode.agent_input.model_dump()
    data["corpus"]["history_cutoff_commit"] = "0000000"
    with pytest.raises(ValidationError, match="history_cutoff_commit"):
        type(s1_episode.agent_input).model_validate(data)


def test_scenario_rules_are_enforced(s1_episode: Episode) -> None:
    data = s1_episode.ground_truth.model_dump()
    data["injected_mutants"] = [
        MutantSpec(
            mutant_id="m", function_qualname="pkg.f", file_path="src/pkg.py"
        ).model_dump()
    ]
    with pytest.raises(ValidationError, match="S1 must not inject a mutant"):
        EpisodeGroundTruth.model_validate(data)


def test_memory_record_requires_approval() -> None:
    from testorcode.src.testorcode.schemas.tests.conftest import memory_record

    data = memory_record().model_dump()
    data["approved_by_maintainer"] = False
    with pytest.raises(ValidationError):
        MemoryRecord.model_validate(data)


# --------------------------------------------------------------------------- #
# Edit policy (declarative half of the safety boundary)
# --------------------------------------------------------------------------- #


def test_edit_policy_classifies_paths() -> None:
    policy = EditPolicy()
    assert policy.classify_path("src/pkg/core.py") is PathClass.SRC
    assert policy.classify_path("tests/test_core.py") is PathClass.TEST
    assert policy.classify_path("tests/conftest.py") is PathClass.FORBIDDEN
    assert policy.classify_path("conftest.py") is PathClass.FORBIDDEN
    assert policy.classify_path("pyproject.toml") is PathClass.FORBIDDEN
    assert policy.classify_path(".github/workflows/ci.yml") is PathClass.FORBIDDEN
    assert policy.classify_path("docs/index.md") is PathClass.OUTSIDE


def test_edit_policy_scope_restriction() -> None:
    policy = EditPolicy()
    assert policy.is_editable("src/pkg/core.py", PatchScope.SRC)
    assert not policy.is_editable("tests/test_core.py", PatchScope.SRC)
    assert policy.is_editable("tests/test_core.py", PatchScope.TEST)
    assert not policy.is_editable("tests/conftest.py", PatchScope.TEST)


# --------------------------------------------------------------------------- #
# JSONL round-trip
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "fixture_name", ["s1_episode", "s2_episode", "s3_episode", "s4_episode", "s5_episode"]
)
def test_episode_jsonl_round_trip(
    fixture_name: str, request: pytest.FixtureRequest
) -> None:
    episode: Episode = request.getfixturevalue(fixture_name)
    line = episode.model_dump_json()
    assert "\n" not in line
    restored = Episode.model_validate_json(line)
    assert restored == episode


def test_label_provenance_requires_known_method() -> None:
    with pytest.raises(ValidationError):
        LabelProvenance(method="vibes")  # type: ignore[arg-type]
    assert provenance_label().stability_runs == 5


def test_evidence_items_must_be_unique() -> None:
    item = intent_evidence()[0]
    with pytest.raises(ValidationError, match="duplicate evidence_id"):
        RetrievalCorpusSpec(
            corpus_id="c", history_cutoff_commit=HEAD, items=[item, item]
        )
