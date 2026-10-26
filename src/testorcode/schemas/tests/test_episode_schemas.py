"""Tests for ``testorcode.episode_schemas`` (synthetic data only)."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import pytest
from pydantic import ValidationError

from testorcode.schemas.tests.conftest import (
    FUTURE, HEAD, NORM, S5 as S5_ID, SLUG, T_NORM, T_SLUG, TEST_DIFF, intent_evidence,
    memory_record, make_s1, make_s2, make_s3, make_s4, make_s5,
)
from testorcode.schemas.episode_schemas import (
    AGENT_FORBIDDEN_KEYS, Availability, CIEvent, CommitRef, DecisionLabel, EditPolicy, Episode,
    EpisodeGroundTruth, EpisodeInput, EvidenceItem, EvidenceSourceType, FailureGroup, PatchScope,
    PathClass, PerTestGoldLabel, RepairDirection, RetrievalCorpusSpec, ScenarioType,
    ToolFaultInjection, ToolFaultKind, ToolFaultPlan, ToolName, find_leaked_ground_truth, same_commit,
)

MAKERS: dict[str, Callable[[], Episode]] = {
    "S1": make_s1, "S2": make_s2, "S3": make_s3, "S4": make_s4, "S5": make_s5}


def edit(ep: Episode, fn: Callable[[dict[str, Any]], None]) -> Episode:
    """Rebuild an episode from its dump after mutating the dict, to trigger validation."""
    data = ep.model_dump(mode="json")
    fn(data)
    return Episode.model_validate(data)


def gt_edit(ep: Episode, fn: Callable[[dict[str, Any]], None]) -> EpisodeGroundTruth:
    data = ep.ground_truth.model_dump(mode="json")
    fn(data)
    return EpisodeGroundTruth.model_validate(data)


# 1. valid S1-S5 ------------------------------------------------------------- #

def test_s1_valid() -> None:
    ep = make_s1()
    assert {l.gold_direction for l in ep.ground_truth.per_test_labels} == {RepairDirection.UPDATE_TEST}
    assert ep.ground_truth.reference_repairs and not ep.ground_truth.injected_mutants


def test_s2_valid() -> None:
    ep = make_s2()
    assert {l.gold_direction for l in ep.ground_truth.per_test_labels} == {RepairDirection.FIX_CODE}
    assert ep.ground_truth.injected_mutants and not ep.ground_truth.reference_repairs


def test_s3_valid_and_per_test_actions() -> None:
    ep = make_s3()
    assert set(ep.gold_actions().values()) == {DecisionLabel.UPDATE_TEST, DecisionLabel.FIX_CODE}


def test_s4_valid_with_prior_memory() -> None:
    ep = make_s4()
    assert ep.agent_input.visible_memory and ep.ground_truth.decoy_memory_record_ids == ["mem-approved"]
    assert ep.ground_truth.per_test_labels[0].memory_applies is False


def test_s5_valid_direction_differs_from_action() -> None:
    ep = make_s5()
    lbl = ep.ground_truth.per_test_labels[0]
    assert (lbl.gold_direction, lbl.expected_action) == (RepairDirection.UPDATE_TEST, DecisionLabel.ESCALATE)
    assert ep.ground_truth.expected_episode_outcome is DecisionLabel.ESCALATE


# 2-3. S5 rejections --------------------------------------------------------- #

def test_s5_rejects_wrong_gold_direction() -> None:
    def f(d: dict[str, Any]) -> None:
        d["per_test_labels"][0]["gold_direction"] = "FIX_CODE"
    with pytest.raises(ValidationError, match="gold_direction must be UPDATE_TEST"):
        gt_edit(make_s5(), f)


def test_s5_rejects_wrong_expected_action() -> None:
    def f(d: dict[str, Any]) -> None:
        d["per_test_labels"][0]["expected_action"] = "UPDATE_TEST"
    with pytest.raises(ValidationError, match="expected_action must be ESCALATE"):
        gt_edit(make_s5(), f)


def test_s5_rejects_missing_episode_outcome() -> None:
    with pytest.raises(ValidationError, match="expected_episode_outcome"):
        gt_edit(make_s5(), lambda d: d.update(expected_episode_outcome=None))


def test_s5_rejects_gold_evidence_present() -> None:
    def f(d: dict[str, Any]) -> None:
        d["per_test_labels"][0]["gold_evidence_ids"] = ["ev-msg"]
    with pytest.raises(ValidationError, match="gold_evidence_ids must be None or"):
        gt_edit(make_s5(), f)


def test_s5_rejects_available_evidence_and_missing_stripped_record() -> None:
    with pytest.raises(ValidationError, match="WITHHELD or ABSENT"):
        gt_edit(make_s5(), lambda d: d.update(evidence_availability="available"))
    with pytest.raises(ValidationError, match="stripped_evidence"):
        gt_edit(make_s5(), lambda d: d.update(stripped_evidence=[]))


def test_s5_rejects_corpus_that_still_exposes_stripped_evidence() -> None:
    def by_id(d: dict[str, Any]) -> None:
        d["agent_input"]["corpus"]["items"] = [i.model_dump(mode="json") for i in intent_evidence()[:1]]
    with pytest.raises(ValidationError, match="still exposes stripped evidence 'ev-msg'"):
        edit(make_s5(), by_id)

    def by_ref(d: dict[str, Any]) -> None:
        d["agent_input"]["corpus"]["items"] = [EvidenceItem(
            evidence_id="ev-doc", source_type=EvidenceSourceType.DOCUMENTATION,
            source_ref="docs/api.md").model_dump(mode="json")]
    with pytest.raises(ValidationError, match="still exposes stripped documentation"):
        edit(make_s5(), by_ref)


def test_s5_rejects_memory_coverage() -> None:
    def f(d: dict[str, Any]) -> None:
        d["agent_input"]["visible_memory"] = [memory_record("mem-x", NORM).model_dump(mode="json")]
    with pytest.raises(ValidationError, match="no memory coverage"):
        edit(make_s5(), f)


def test_s5_rejects_missing_function_qualname() -> None:
    def f_gt(d: dict[str, Any]) -> None:
        d["per_test_labels"][0]["function_qualname"] = None
    with pytest.raises(ValidationError, match="requires a non-empty function_qualname"):
        gt_edit(make_s5(), f_gt)

    def f_ep(d: dict[str, Any]) -> None:
        d["ground_truth"]["per_test_labels"][0]["function_qualname"] = None
    with pytest.raises(ValidationError, match="requires a non-empty function_qualname"):
        edit(make_s5(), f_ep)


def test_s5_allows_unrelated_memory_record() -> None:
    def f(d: dict[str, Any]) -> None:
        d["agent_input"]["visible_memory"] = [
            memory_record("mem-unrelated", "synthlib.core.unrelated").model_dump(mode="json")
        ]
    ep = edit(make_s5(), f)
    assert len(ep.agent_input.visible_memory) == 1


# 4-5. S4 ------------------------------------------------------------------- #

@pytest.mark.parametrize("memory, message", [
    (None, "S4 requires memory to be enabled"),      # disabled memory must not make S4 look valid
    ([], "decoy memory records not in visible_memory"),  # enabled but decoy missing
])
def test_s4_rejects_missing_or_disabled_decoy_memory(memory: list[Any] | None, message: str) -> None:
    def f(d: dict[str, Any]) -> None:
        d["agent_input"]["visible_memory"] = memory
    with pytest.raises(ValidationError, match=message):
        edit(make_s4(), f)


def test_s4_rejects_decoy_record_absent_from_visible_memory() -> None:
    def f(d: dict[str, Any]) -> None:
        d["agent_input"]["visible_memory"] = [memory_record("mem-other").model_dump(mode="json")]
    with pytest.raises(ValidationError, match="decoy memory records not in visible_memory"):
        edit(make_s4(), f)


def test_s4_rejects_non_fix_code_direction() -> None:
    def f(d: dict[str, Any]) -> None:
        d["per_test_labels"][0].update(gold_direction="UPDATE_TEST", expected_action="UPDATE_TEST",
                                      injected_mutant_id=None)
    with pytest.raises(ValidationError, match="S4 requires every gold_direction to be FIX_CODE"):
        gt_edit(make_s4(), f)


@pytest.mark.parametrize("value", [None, True])
def test_s4_requires_decoy_marked_not_applicable(value: bool | None) -> None:
    def f(d: dict[str, Any]) -> None:
        d["per_test_labels"][0]["memory_applies"] = value
    with pytest.raises(ValidationError, match="memory_applies=False"):
        gt_edit(make_s4(), f)


def test_s4_rejects_label_pointing_at_non_decoy_record() -> None:
    def f(d: dict[str, Any]) -> None:
        d["per_test_labels"][0]["related_memory_record_id"] = "mem-unknown"
    with pytest.raises(ValidationError, match="related_memory_record_id"):
        gt_edit(make_s4(), f)


def test_s4_rejects_function_mismatch_between_label_and_record() -> None:
    def f(d: dict[str, Any]) -> None:
        d["agent_input"]["visible_memory"][0]["function_qualname"] = "synthlib.core.other"
    with pytest.raises(ValidationError, match="differs from decoy record function"):
        edit(make_s4(), f)


def test_s4_rejects_missing_function_qualname() -> None:
    def f_gt(d: dict[str, Any]) -> None:
        d["per_test_labels"][0]["function_qualname"] = None
    with pytest.raises(ValidationError, match="requires a non-empty function_qualname"):
        gt_edit(make_s4(), f_gt)

    def f_ep(d: dict[str, Any]) -> None:
        d["ground_truth"]["per_test_labels"][0]["function_qualname"] = None
    with pytest.raises(ValidationError, match="requires a non-empty function_qualname"):
        edit(make_s4(), f_ep)


# 6-7. S3 ------------------------------------------------------------------- #

@pytest.mark.parametrize("drop", ["UPDATE_TEST", "FIX_CODE"])
def test_s3_rejects_missing_direction(drop: str) -> None:
    def f(d: dict[str, Any]) -> None:
        d["per_test_labels"] = [l for l in d["per_test_labels"] if l["gold_direction"] != drop]
    with pytest.raises(ValidationError, match="both UPDATE_TEST and FIX_CODE"):
        gt_edit(make_s3(), f)


def test_s3_rejects_missing_reference_repair() -> None:
    with pytest.raises(ValidationError, match="reference_repairs"):
        gt_edit(make_s3(), lambda d: d.update(reference_repairs=[]))


def test_s3_rejects_missing_mutant() -> None:
    def f(d: dict[str, Any]) -> None:
        d["injected_mutants"] = []
        for l in d["per_test_labels"]:
            l["injected_mutant_id"] = None
    with pytest.raises(ValidationError, match="injected mutant"):
        gt_edit(make_s3(), f)


def test_s3_rejects_same_function_for_both_directions() -> None:
    def f(d: dict[str, Any]) -> None:
        for l in d["per_test_labels"]:
            l["function_qualname"] = NORM
        d["injected_mutants"][0]["function_qualname"] = NORM
    with pytest.raises(ValidationError, match="different function"):
        gt_edit(make_s3(), f)


def test_s3_rejects_missing_function_identity_rather_than_assuming() -> None:
    def f(d: dict[str, Any]) -> None:
        d["per_test_labels"][0]["function_qualname"] = None
    with pytest.raises(ValidationError, match="requires function_qualname on every label"):
        gt_edit(make_s3(), f)


def test_s3_rejects_group_mixing_both_directions() -> None:
    def f(d: dict[str, Any]) -> None:
        d["agent_input"]["ci_event"]["failure_groups"] = [
            {"group_id": "g-n", "function_qualname": None, "test_ids": [T_NORM, T_SLUG]}]
        d["ground_truth"]["per_test_labels"][1]["group_id"] = "g-n"
    with pytest.raises(ValidationError, match="mixes UPDATE_TEST and FIX_CODE"):
        edit(make_s3(), f)


# other scenario rules ------------------------------------------------------- #

def test_s1_s2_scenario_rules() -> None:
    with pytest.raises(ValidationError, match="S1 must not inject a mutant"):
        gt_edit(make_s1(), lambda d: d.update(injected_mutants=[
            {"mutant_id": "m", "function_qualname": NORM, "file_path": "src/x.py"}]))
    with pytest.raises(ValidationError, match="reference_repairs"):
        gt_edit(make_s1(), lambda d: d.update(reference_repairs=[]))
    with pytest.raises(ValidationError, match="no reference test repair"):
        gt_edit(make_s2(), lambda d: d.update(reference_repairs=[{"unified_diff": TEST_DIFF}]))
    with pytest.raises(ValidationError, match="at least one injected mutant"):
        gt_edit(make_s2(), lambda d: (d.update(injected_mutants=[]),
                                      d["per_test_labels"][0].update(injected_mutant_id=None)))


def test_expected_episode_outcome_only_for_s5() -> None:
    with pytest.raises(ValidationError, match="expected_episode_outcome"):
        gt_edit(make_s1(), lambda d: d.update(expected_episode_outcome="ESCALATE"))


def test_decoy_ids_only_valid_for_s4() -> None:
    with pytest.raises(ValidationError, match="only valid for S4"):
        gt_edit(make_s1(), lambda d: d.update(decoy_memory_record_ids=["m"]))


# 8. id references ----------------------------------------------------------- #

def test_label_mutant_reference_must_exist_and_match_function() -> None:
    with pytest.raises(ValidationError, match="unknown injected mutant"):
        gt_edit(make_s2(), lambda d: d["per_test_labels"][0].update(injected_mutant_id="nope"))
    with pytest.raises(ValidationError, match="differs from its mutant's function"):
        gt_edit(make_s2(), lambda d: d["per_test_labels"][0].update(function_qualname=NORM))


def test_failing_tests_and_labels_must_match() -> None:
    def extra(d: dict[str, Any]) -> None:
        d["agent_input"]["ci_event"]["failing_tests"].append(
            {"test_id": "tests/t.py::extra", "test_file": "tests/t.py", "test_name": "extra"})
    with pytest.raises(ValidationError, match="failing tests without a gold label"):
        edit(make_s1(), extra)
    with pytest.raises(ValidationError, match="not failing in the CI event"):
        edit(make_s1(), lambda d: d["ground_truth"]["per_test_labels"][0].update(test_id="tests/t.py::ghost"))


def test_failure_group_must_reference_real_tests_and_labels_must_agree() -> None:
    with pytest.raises(ValidationError, match="unknown test ids"):
        CIEvent(event_id="c", commit=CommitRef(sha=HEAD), commit_message="m",
                failing_tests=make_s1().agent_input.ci_event.failing_tests,
                failure_groups=[FailureGroup(group_id="g", test_ids=["tests/t.py::ghost"])])
    with pytest.raises(ValidationError, match="does not contain it"):
        edit(make_s1(), lambda d: d["ground_truth"]["per_test_labels"][0].update(group_id="g-missing"))


def test_gold_evidence_must_exist_in_materialised_corpus() -> None:
    def f(d: dict[str, Any]) -> None:
        d["per_test_labels"][0]["gold_evidence_ids"] = ["ev-msg", "ev-ghost"]
    with pytest.raises(ValidationError, match="gold_evidence_ids not found"):
        edit(make_s1(), lambda d: f(d["ground_truth"]))


def test_unmaterialised_corpus_skips_gold_evidence_lookup() -> None:
    ep = edit(make_s1(), lambda d: d["agent_input"]["corpus"].update(items=None))
    assert ep.agent_input.corpus.items is None


def test_s1_requires_gold_evidence() -> None:
    with pytest.raises(ValidationError, match="needs non-empty gold_evidence_ids"):
        gt_edit(make_s1(), lambda d: d["per_test_labels"][0].update(gold_evidence_ids=[]))


def test_future_commit_leakage_is_rejected_including_abbreviated_shas() -> None:
    def f(d: dict[str, Any]) -> None:
        d["agent_input"]["corpus"]["items"].append(EvidenceItem(
            evidence_id="ev-future", source_type=EvidenceSourceType.COMMIT_MESSAGE, source_ref="x",
            commit_sha=FUTURE[:12], commit_index=10).model_dump(mode="json"))
    with pytest.raises(ValidationError, match="future commit"):
        edit(make_s1(), f)
    assert same_commit(FUTURE, FUTURE[:9]) and not same_commit(HEAD, FUTURE)


def test_corpus_cutoff_rules() -> None:
    item = EvidenceItem(evidence_id="e", source_type=EvidenceSourceType.COMMIT_MESSAGE, source_ref=HEAD, commit_index=6)
    with pytest.raises(ValidationError, match="future commits must never be indexed"):
        RetrievalCorpusSpec(corpus_id="c", history_cutoff_commit=HEAD, history_cutoff_index=5, items=[item])
    with pytest.raises(ValidationError, match="duplicate evidence_id"):
        RetrievalCorpusSpec(corpus_id="c", history_cutoff_commit=HEAD, items=[item, item])
    with pytest.raises(ValidationError, match="current commit"):
        edit(make_s1(), lambda d: d["agent_input"]["corpus"].update(history_cutoff_commit="0000000"))


# 9. ground-truth leakage ---------------------------------------------------- #

@pytest.mark.parametrize("scenario", sorted(MAKERS))
def test_agent_payload_has_no_ground_truth(scenario: str) -> None:
    ep = MAKERS[scenario]()
    payload = ep.agent_payload()
    assert find_leaked_ground_truth(payload) == []
    blob = json.dumps(payload)
    for repair in ep.ground_truth.reference_repairs:
        assert repair.unified_diff not in blob
    for m in ep.ground_truth.injected_mutants:
        assert m.mutant_id not in blob
    assert ep.episode_id in blob and f"S{scenario[-1]}" not in ep.episode_id  # opaque id


def _schema_property_names(node: Any) -> set[str]:
    names: set[str] = set()
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "properties" and isinstance(value, dict):
                names |= set(value)
            names |= _schema_property_names(value)
    elif isinstance(node, list):
        for item in node:
            names |= _schema_property_names(item)
    return names


def test_episode_input_schema_cannot_hold_any_forbidden_field() -> None:
    """Structural guarantee: no nested agent-visible model even declares a gold field."""
    assert _schema_property_names(EpisodeInput.model_json_schema()) & AGENT_FORBIDDEN_KEYS == set()


def test_episode_input_rejects_attached_gold_fields_and_is_frozen() -> None:
    data = make_s1().agent_payload()
    data["gold_direction"] = "UPDATE_TEST"
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        EpisodeInput.model_validate(data)
    with pytest.raises(ValidationError):
        make_s1().agent_input.episode_id = "x"  # type: ignore[misc]


def test_leak_detector_finds_planted_key_and_fault_plan_stays_outside() -> None:
    payload = make_s1().agent_payload()
    payload["ci_event"]["gold_evidence_ids"] = ["x"]
    assert find_leaked_ground_truth(payload) == ["ci_event.gold_evidence_ids"]
    plan = ToolFaultPlan(enabled=True, seed=0, injections=[ToolFaultInjection(
        tool=ToolName.APPLY_PATCH, fault_kind=ToolFaultKind.PATCH_DOES_NOT_APPLY, call_index=0)])
    ep = make_s1().model_copy(update={"fault_plan": plan})
    assert "fault_plan" not in ep.agent_payload()


def test_ground_truth_isolation_guarantee() -> None:
    for sc, maker in MAKERS.items():
        ep = maker()
        plan = ToolFaultPlan(enabled=True, seed=42, injections=[ToolFaultInjection(
            tool=ToolName.APPLY_PATCH, fault_kind=ToolFaultKind.PATCH_DOES_NOT_APPLY, call_index=0)])
        ep_with_fault = ep.model_copy(update={"fault_plan": plan})
        payload = ep_with_fault.agent_payload()
        assert find_leaked_ground_truth(payload) == []
        for key in ("ground_truth", "fault_plan", "scenario", "difficulty"):
            assert key not in payload
        blob = json.dumps(payload)
        for repair in ep.ground_truth.reference_repairs:
            assert repair.unified_diff not in blob
        for mutant in ep.ground_truth.injected_mutants:
            assert mutant.mutant_id not in blob
        for label in ep.ground_truth.per_test_labels:
            if label.injected_mutant_id:
                assert label.injected_mutant_id not in blob


# 13/14. absent vs empty vs malformed -------------------------------------- #

def test_none_and_empty_are_kept_distinct() -> None:
    assert make_s5().agent_input.visible_memory == [] and make_s2().agent_input.visible_memory is None
    assert make_s5().agent_input.corpus.items == []
    assert RetrievalCorpusSpec(corpus_id="c", history_cutoff_commit=HEAD).items is None
    assert make_s2().ground_truth.per_test_labels[0].gold_evidence_ids == []


def test_commit_message_availability_must_match_message() -> None:
    kw: dict[str, Any] = dict(event_id="c", commit=CommitRef(sha=HEAD),
                              failing_tests=make_s1().agent_input.ci_event.failing_tests)
    with pytest.raises(ValidationError, match="AVAILABLE but commit_message is None"):
        CIEvent(commit_message=None, **kw)
    with pytest.raises(ValidationError, match="must be None unless"):
        CIEvent(commit_message="m", commit_message_availability=Availability.WITHHELD, **kw)


def test_malformed_values_rejected() -> None:
    with pytest.raises(ValidationError):
        CommitRef(sha="not-a-sha")
    with pytest.raises(ValidationError):
        PerTestGoldLabel(test_id="t", gold_direction="ESCALATE", expected_action="ESCALATE")  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        PerTestGoldLabel(test_id="t", gold_direction="UPDATE_TEST", expected_action="DELETE_TEST")  # type: ignore[arg-type]
    with pytest.raises(ValidationError, match="contradicts gold_direction"):
        PerTestGoldLabel(test_id="t", gold_direction=RepairDirection.UPDATE_TEST,
                         expected_action=DecisionLabel.FIX_CODE)


def test_tool_fault_plan_validation() -> None:
    with pytest.raises(ValidationError, match="disabled but defines"):
        ToolFaultPlan(enabled=False, failure_rate=0.1)
    with pytest.raises(ValidationError, match="must define"):
        ToolFaultPlan(enabled=True)
    assert set(DecisionLabel) == {DecisionLabel.UPDATE_TEST, DecisionLabel.FIX_CODE, DecisionLabel.ESCALATE}
    assert not hasattr(ScenarioType, "T")


def test_edit_policy() -> None:
    p = EditPolicy()
    expected = {"src/pkg/a.py": PathClass.SRC, "tests/test_a.py": PathClass.TEST,
                "tests/conftest.py": PathClass.FORBIDDEN, "pyproject.toml": PathClass.FORBIDDEN,
                ".github/workflows/ci.yml": PathClass.FORBIDDEN, "docs/x.md": PathClass.OUTSIDE}
    assert {k: p.classify_path(k) for k in expected} == expected
    assert p.is_editable("src/pkg/a.py", PatchScope.SRC) and not p.is_editable("tests/test_a.py", PatchScope.SRC)
    assert not p.is_editable("tests/conftest.py", PatchScope.TEST)


# 10. JSONL round trip ------------------------------------------------------- #

@pytest.mark.parametrize("scenario", sorted(MAKERS))
def test_episode_jsonl_round_trip(scenario: str) -> None:
    ep = MAKERS[scenario]()
    line = ep.model_dump_json()
    assert "\n" not in line and Episode.model_validate_json(line) == ep
