"""Synthetic fixtures.  Every sha, test id, diff and snippet is invented; nothing here
describes a real repository, commit or experimental result."""

from __future__ import annotations

import pytest

from testorcode.schemas.episode_schemas import (
    Availability, CIEvent, CommitRef, DecisionLabel, Difficulty, Episode, EpisodeBudget,
    EpisodeGroundTruth, EpisodeInput, EnvironmentSpec, EvidenceItem, EvidenceSourceType,
    ExecutableProbe, FailingTest, FailureGroup, FileChange, LabelProvenance, MemoryRecord,
    MutantSpec, MutationFixture, PerTestGoldLabel, ReferenceRepair, RepairDirection, RepoRef,
    RetrievalCorpusSpec, ScenarioType, StrippedEvidence,
)

HEAD = "a1b2c3d4e5f60718293a4b5c6d7e8f9012345678"
PARENT = "0f1e2d3c4b5a69788796a5b4c3d2e1f098765432"
FUTURE = "ffffffffffffffffffffffffffffffffffffffff"
NORM, SLUG = "synthlib.core.normalise", "synthlib.core.slugify"
T_NORM = "tests/test_synthlib.py::test_normalise"
T_SLUG = "tests/test_synthlib.py::test_slugify"
TEST_DIFF = "@@\n-    assert normalise('ab') == 'ab'\n+    assert normalise('ab') == 'Ab'"


def failing_test(test_id: str, function: str) -> FailingTest:
    return FailingTest(
        test_id=test_id, test_file="tests/test_synthlib.py", test_name=test_id.split("::")[-1],
        target_function=function, failure_message="assert 'A' == 'a'",
        traceback="Traceback (most recent call last): ...", stable=True,
    )


def intent_evidence() -> list[EvidenceItem]:
    return [
        EvidenceItem(evidence_id="ev-msg", source_type=EvidenceSourceType.COMMIT_MESSAGE,
                     source_ref=HEAD, commit_sha=HEAD, commit_index=10,
                     text="Capitalise normalise() output on purpose."),
        EvidenceItem(evidence_id="ev-changelog", source_type=EvidenceSourceType.CHANGELOG,
                     source_ref="CHANGELOG.md", commit_index=10, line_start=3, line_end=4,
                     text="normalise() now returns a capitalised string."),
    ]


def memory_record(record_id: str = "mem-1", function: str = NORM) -> MemoryRecord:
    return MemoryRecord(
        record_id=record_id, repo_id="synthlib", function_qualname=function,
        behaviour_signature="normalise() returns a capitalised string",
        direction=RepairDirection.UPDATE_TEST,
        probe=ExecutableProbe(probe_id=f"probe-{record_id}", probe_code="assert normalise('ab') == 'Ab'"),
        evidence_ids=["ev-msg"], approved_at_commit=PARENT,
    )


def episode_input(episode_id: str, tests: list[FailingTest], items: list[EvidenceItem] | None, *,
                  memory: list[MemoryRecord] | None = None, message: str | None = "Capitalise normalise()",
                  availability: Availability = Availability.AVAILABLE,
                  groups: list[FailureGroup] | None = None) -> EpisodeInput:
    return EpisodeInput(
        episode_id=episode_id,
        repo=RepoRef(repo_id="synthlib", license="MIT"),
        environment=EnvironmentSpec(python_version="3.12.3", docker_image="env@sha256:dead", stability_runs=5),
        ci_event=CIEvent(
            event_id="ci-1", commit=CommitRef(sha=HEAD, stream_index=10),
            parent_commit=CommitRef(sha=PARENT, stream_index=9),
            commit_message=message, commit_message_availability=availability,
            changed_files=[FileChange(path="src/synthlib/core.py", change_kind="modified",
                                      unified_diff="@@ -1 +1 @@\n-    return s.lower()\n+    return s.capitalize()")],
            failing_tests=tests,
            failure_groups=groups if groups is not None else [
                FailureGroup(group_id="g-1", function_qualname=NORM, test_ids=[t.test_id for t in tests])],
        ),
        corpus=RetrievalCorpusSpec(corpus_id="synthlib-corpus", history_cutoff_commit=HEAD,
                                   history_cutoff_index=10, items=items),
        visible_memory=memory,
        budget=EpisodeBudget(max_tool_calls=40, max_llm_calls=20, max_tokens=200_000),
    )


def prov(method: str = "tests_with_source_only_change") -> LabelProvenance:
    return LabelProvenance(method=method, stability_runs=5, verified_by="synthetic")


def ref_repair() -> ReferenceRepair:
    return ReferenceRepair(unified_diff=TEST_DIFF, changed_files=["tests/test_synthlib.py"])


def mutant(mutant_id: str, function: str) -> MutantSpec:
    return MutantSpec(mutant_id=mutant_id, function_qualname=function, file_path="src/synthlib/core.py")


def label(test_id: str, direction: RepairDirection, *, function: str | None, group: str | None = "g-1",
          action: DecisionLabel | None = None, evidence: list[str] | None = None,
          mutant_id: str | None = None, memory_id: str | None = None,
          memory_applies: bool | None = None) -> PerTestGoldLabel:
    return PerTestGoldLabel(
        test_id=test_id, group_id=group, function_qualname=function, gold_direction=direction,
        expected_action=action or DecisionLabel(direction.value), gold_evidence_ids=evidence,
        injected_mutant_id=mutant_id, related_memory_record_id=memory_id, memory_applies=memory_applies)


# Opaque ids on purpose: the episode id is agent-visible and must not encode the scenario.
S1, S2, S3, S4, S5 = "ep-0001", "ep-0002", "ep-0003", "ep-0004", "ep-0005"


def make_s1() -> Episode:
    t = [failing_test(T_NORM, NORM)]
    return Episode(
        episode_id=S1, scenario=ScenarioType.S1_OUTDATED_TEST, stream_id="st", stream_position=10,
        agent_input=episode_input(S1, t, intent_evidence()),
        ground_truth=EpisodeGroundTruth(
            episode_id=S1, scenario=ScenarioType.S1_OUTDATED_TEST,
            per_test_labels=[label(T_NORM, RepairDirection.UPDATE_TEST, function=NORM,
                                   evidence=["ev-msg", "ev-changelog"])],
            reference_repairs=[ref_repair()], evidence_availability=Availability.AVAILABLE,
            future_commit_shas=[FUTURE], label_provenance=prov()))


def make_s2() -> Episode:
    t = [failing_test(T_SLUG, SLUG)]
    return Episode(
        episode_id=S2, scenario=ScenarioType.S2_INJECTED_REGRESSION, stream_id="st", stream_position=11,
        agent_input=episode_input(S2, t, intent_evidence(), memory=None,
                                  groups=[FailureGroup(group_id="g-1", function_qualname=SLUG, test_ids=[T_SLUG])]),
        ground_truth=EpisodeGroundTruth(
            episode_id=S2, scenario=ScenarioType.S2_INJECTED_REGRESSION,
            per_test_labels=[label(T_SLUG, RepairDirection.FIX_CODE, function=SLUG, evidence=[], mutant_id="mut-1")],
            injected_mutants=[mutant("mut-1", SLUG)],
            mutation_fixtures=[MutationFixture(function_qualname=SLUG,
                                               mutants=[mutant(f"mut-{i}", SLUG) for i in range(1, 5)],
                                               reference_killed_mutant_ids=["mut-1", "mut-2", "mut-3"])],
            evidence_availability=Availability.ABSENT, label_provenance=prov("tests_with_mutant_applied")))


def make_s3() -> Episode:
    t = [failing_test(T_NORM, NORM), failing_test(T_SLUG, SLUG)]
    groups = [FailureGroup(group_id="g-n", function_qualname=NORM, test_ids=[T_NORM]),
              FailureGroup(group_id="g-s", function_qualname=SLUG, test_ids=[T_SLUG])]
    return Episode(
        episode_id=S3, scenario=ScenarioType.S3_MIXED, difficulty=Difficulty.HARD, stream_id="st",
        stream_position=12, agent_input=episode_input(S3, t, intent_evidence(), groups=groups),
        ground_truth=EpisodeGroundTruth(
            episode_id=S3, scenario=ScenarioType.S3_MIXED,
            per_test_labels=[
                label(T_NORM, RepairDirection.UPDATE_TEST, function=NORM, group="g-n", evidence=["ev-msg"]),
                label(T_SLUG, RepairDirection.FIX_CODE, function=SLUG, group="g-s", evidence=[], mutant_id="mut-9")],
            reference_repairs=[ref_repair()], injected_mutants=[mutant("mut-9", SLUG)],
            evidence_availability=Availability.AVAILABLE, label_provenance=prov("manual_review")))


def make_s4() -> Episode:
    t = [failing_test("tests/test_synthlib.py::test_normalise_suffix", NORM)]
    rec = memory_record("mem-approved")
    return Episode(
        episode_id=S4, scenario=ScenarioType.S4_DECOY_REGRESSION, difficulty=Difficulty.HARD,
        stream_id="st", stream_position=15,
        agent_input=episode_input(S4, t, intent_evidence(), memory=[rec]),
        ground_truth=EpisodeGroundTruth(
            episode_id=S4, scenario=ScenarioType.S4_DECOY_REGRESSION,
            per_test_labels=[label(t[0].test_id, RepairDirection.FIX_CODE, function=NORM, evidence=[],
                                   mutant_id="mut-decoy", memory_id="mem-approved", memory_applies=False)],
            injected_mutants=[mutant("mut-decoy", NORM)], decoy_memory_record_ids=["mem-approved"],
            evidence_availability=Availability.ABSENT, label_provenance=prov("tests_with_mutant_applied")))


def make_s5() -> Episode:
    t = [failing_test(T_NORM, NORM)]
    return Episode(
        episode_id=S5, scenario=ScenarioType.S5_INTENT_STRIPPED, difficulty=Difficulty.MEDIUM,
        stream_id="st", stream_position=18,
        agent_input=episode_input(S5, t, [], memory=[], message=None, availability=Availability.WITHHELD),
        ground_truth=EpisodeGroundTruth(
            episode_id=S5, scenario=ScenarioType.S5_INTENT_STRIPPED,
            per_test_labels=[label(T_NORM, RepairDirection.UPDATE_TEST, function=NORM,
                                   action=DecisionLabel.ESCALATE, evidence=[])],
            expected_episode_outcome=DecisionLabel.ESCALATE, reference_repairs=[ref_repair()],
            evidence_availability=Availability.WITHHELD,
            stripped_evidence=[
                StrippedEvidence(source_type=EvidenceSourceType.COMMIT_MESSAGE, source_ref=HEAD,
                                 original_evidence_id="ev-msg"),
                StrippedEvidence(source_type=EvidenceSourceType.DOCUMENTATION, source_ref="docs/api.md")],
            label_provenance=prov()))


@pytest.fixture
def s1_episode() -> Episode: return make_s1()
@pytest.fixture
def s2_episode() -> Episode: return make_s2()
@pytest.fixture
def s3_episode() -> Episode: return make_s3()
@pytest.fixture
def s4_episode() -> Episode: return make_s4()
@pytest.fixture
def s5_episode() -> Episode: return make_s5()
