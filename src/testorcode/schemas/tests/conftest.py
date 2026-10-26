"""Synthetic fixtures for the schema tests.

Every commit hash, test id, diff and evidence snippet below is invented for
testing.  Nothing here describes a real repository, a real commit or a real
experimental result.
"""

from __future__ import annotations

import pytest

from testorcode.episode_schemas import (
    Availability,
    CIEvent,
    CommitRef,
    DecisionLabel,
    Difficulty,
    Episode,
    EpisodeBudget,
    EpisodeGroundTruth,
    EpisodeInput,
    EnvironmentSpec,
    EvidenceItem,
    EvidenceSourceType,
    ExecutableProbe,
    FailingTest,
    FailureGroup,
    FileChange,
    LabelProvenance,
    MemoryRecord,
    MutantSpec,
    MutationFixture,
    PerTestGoldLabel,
    ReferenceRepair,
    RepairDirection,
    RepoRef,
    RetrievalCorpusSpec,
    ScenarioType,
    StrippedEvidence,
    TestSuiteSnapshot,
)

HEAD = "a1b2c3d4e5f60718293a4b5c6d7e8f9012345678"
PARENT = "0f1e2d3c4b5a69788796a5b4c3d2e1f098765432"
FUTURE = "ffffffffffffffffffffffffffffffffffffffff"


def repo() -> RepoRef:
    return RepoRef(repo_id="synthlib", package_name="synthlib", license="MIT")


def environment() -> EnvironmentSpec:
    return EnvironmentSpec(
        python_version="3.12.3",
        docker_image="synthlib-env@sha256:deadbeef",
        dependency_lock_hash="sha256:1234",
        pytest_version="8.2.0",
        stability_runs=5,
        tool_versions={"git": "2.43.0"},
    )


def failing_test(test_id: str, function: str) -> FailingTest:
    return FailingTest(
        test_id=test_id,
        test_file="tests/test_synthlib.py",
        test_name=test_id.split("::")[-1],
        target_function=function,
        failure_message="assert 'A' == 'a'",
        traceback="Traceback (most recent call last): ...",
        assertion_diff="- 'a'\n+ 'A'",
        stability_runs=5,
        stable=True,
    )


def corpus(
    items: list[EvidenceItem] | None = None,
    *,
    cutoff_index: int = 10,
) -> RetrievalCorpusSpec:
    return RetrievalCorpusSpec(
        corpus_id="synthlib-corpus",
        corpus_version="v1",
        index_version="bm25-v1",
        history_cutoff_commit=HEAD,
        history_cutoff_index=cutoff_index,
        included_source_types=[
            EvidenceSourceType.COMMIT_MESSAGE,
            EvidenceSourceType.CHANGELOG,
            EvidenceSourceType.DOCUMENTATION,
            EvidenceSourceType.DOCSTRING,
        ],
        items=items,
    )


def intent_evidence() -> list[EvidenceItem]:
    return [
        EvidenceItem(
            evidence_id="ev-commit-msg-1",
            source_type=EvidenceSourceType.COMMIT_MESSAGE,
            source_ref=HEAD,
            commit_sha=HEAD,
            commit_index=10,
            text="Capitalise the first letter of normalise() output on purpose.",
        ),
        EvidenceItem(
            evidence_id="ev-changelog-1",
            source_type=EvidenceSourceType.CHANGELOG,
            source_ref="CHANGELOG.md",
            commit_index=10,
            section="Unreleased",
            line_start=3,
            line_end=4,
            text="normalise() now returns a capitalised string.",
        ),
    ]


def ci_event(
    tests: list[FailingTest],
    *,
    commit_message: str | None = "Capitalise normalise() output",
    availability: Availability = Availability.AVAILABLE,
    groups: list[FailureGroup] | None = None,
) -> CIEvent:
    return CIEvent(
        event_id="ci-1",
        commit=CommitRef(sha=HEAD, stream_index=10),
        parent_commit=CommitRef(sha=PARENT, stream_index=9),
        commit_message=commit_message,
        commit_message_availability=availability,
        changed_files=[
            FileChange(
                path="src/synthlib/core.py",
                change_kind="modified",
                unified_diff="@@ -1 +1 @@\n-    return s.lower()\n+    return s.capitalize()",
                lines_added=1,
                lines_removed=1,
            )
        ],
        failing_tests=tests,
        failure_groups=groups
        or [
            FailureGroup(
                group_id="g-normalise",
                function_qualname="synthlib.core.normalise",
                test_ids=[t.test_id for t in tests],
            )
        ],
        suite_snapshot=TestSuiteSnapshot(
            total=40,
            passed=40 - len(tests),
            failed=len(tests),
            failing_test_ids=[t.test_id for t in tests],
        ),
    )


def memory_record(
    record_id: str = "mem-1",
    function: str = "synthlib.core.normalise",
) -> MemoryRecord:
    return MemoryRecord(
        record_id=record_id,
        repo_id="synthlib",
        function_qualname=function,
        behaviour_signature="normalise() returns a capitalised string",
        direction=RepairDirection.UPDATE_TEST,
        probe=ExecutableProbe(
            probe_id=f"probe-{record_id}",
            probe_code="assert normalise('ab') == 'Ab'",
        ),
        evidence_ids=["ev-commit-msg-1"],
        approved_at_commit=PARENT,
        created_in_episode="ep-prev",
    )


def episode_input(
    *,
    tests: list[FailingTest],
    corpus_items: list[EvidenceItem] | None,
    memory: list[MemoryRecord] | None = None,
    commit_message: str | None = "Capitalise normalise() output",
    availability: Availability = Availability.AVAILABLE,
    groups: list[FailureGroup] | None = None,
    episode_id: str = "ep-001",
) -> EpisodeInput:
    return EpisodeInput(
        episode_id=episode_id,
        repo=repo(),
        environment=environment(),
        ci_event=ci_event(
            tests,
            commit_message=commit_message,
            availability=availability,
            groups=groups,
        ),
        corpus=corpus(corpus_items),
        visible_memory=memory,
        budget=EpisodeBudget(max_tool_calls=40, max_llm_calls=20, max_tokens=200_000),
    )


def provenance_label(method: str = "tests_with_source_only_change") -> LabelProvenance:
    return LabelProvenance(method=method, stability_runs=5, verified_by="synthetic")


# --------------------------------------------------------------------------- #
# Episode fixtures, one per scenario
# --------------------------------------------------------------------------- #


@pytest.fixture
def s1_episode() -> Episode:
    tests = [failing_test("tests/test_synthlib.py::test_normalise", "synthlib.core.normalise")]
    return Episode(
        episode_id="ep-s1",
        scenario=ScenarioType.S1_OUTDATED_TEST,
        difficulty=Difficulty.EASY,
        stream_id="synthlib-stream",
        stream_position=10,
        agent_input=episode_input(
            tests=tests, corpus_items=intent_evidence(), episode_id="ep-s1"
        ),
        ground_truth=EpisodeGroundTruth(
            episode_id="ep-s1",
            scenario=ScenarioType.S1_OUTDATED_TEST,
            per_test_labels=[
                PerTestGoldLabel(
                    test_id=tests[0].test_id,
                    group_id="g-normalise",
                    function_qualname="synthlib.core.normalise",
                    gold_direction=RepairDirection.UPDATE_TEST,
                    expected_action=DecisionLabel.UPDATE_TEST,
                    cause="outdated_test",
                    gold_evidence_ids=["ev-commit-msg-1", "ev-changelog-1"],
                )
            ],
            reference_repairs=[
                ReferenceRepair(
                    kind="developer_test_diff",
                    unified_diff="@@\n-    assert normalise('ab') == 'ab'\n+    assert normalise('ab') == 'Ab'",
                    changed_files=["tests/test_synthlib.py"],
                    source_commit=HEAD,
                )
            ],
            gold_evidence_ids=["ev-commit-msg-1", "ev-changelog-1"],
            evidence_availability=Availability.AVAILABLE,
            future_commit_shas=[FUTURE],
            label_provenance=provenance_label(),
        ),
    )


@pytest.fixture
def s2_episode() -> Episode:
    tests = [failing_test("tests/test_synthlib.py::test_slugify", "synthlib.core.slugify")]
    return Episode(
        episode_id="ep-s2",
        scenario=ScenarioType.S2_INJECTED_REGRESSION,
        difficulty=Difficulty.EASY,
        stream_id="synthlib-stream",
        stream_position=11,
        agent_input=episode_input(
            tests=tests,
            corpus_items=intent_evidence(),
            episode_id="ep-s2",
            groups=[
                FailureGroup(
                    group_id="g-slugify",
                    function_qualname="synthlib.core.slugify",
                    test_ids=[tests[0].test_id],
                )
            ],
        ),
        ground_truth=EpisodeGroundTruth(
            episode_id="ep-s2",
            scenario=ScenarioType.S2_INJECTED_REGRESSION,
            per_test_labels=[
                PerTestGoldLabel(
                    test_id=tests[0].test_id,
                    group_id="g-slugify",
                    function_qualname="synthlib.core.slugify",
                    gold_direction=RepairDirection.FIX_CODE,
                    expected_action=DecisionLabel.FIX_CODE,
                    cause="injected_regression",
                    gold_evidence_ids=[],
                    injected_mutant_id="mut-1",
                )
            ],
            injected_mutants=[
                MutantSpec(
                    mutant_id="mut-1",
                    function_qualname="synthlib.core.slugify",
                    file_path="src/synthlib/core.py",
                    operator="comparison_swap",
                    killed_by_reference_tests=[tests[0].test_id],
                )
            ],
            mutation_fixtures=[
                MutationFixture(
                    catalogue_id="cat-slugify",
                    function_qualname="synthlib.core.slugify",
                    mutants=[
                        MutantSpec(
                            mutant_id=f"mut-{i}",
                            function_qualname="synthlib.core.slugify",
                            file_path="src/synthlib/core.py",
                        )
                        for i in range(1, 5)
                    ],
                    reference_killed_mutant_ids=["mut-1", "mut-2", "mut-3"],
                    generated_by="mutmut (synthetic)",
                )
            ],
            gold_evidence_ids=[],
            evidence_availability=Availability.ABSENT,
            label_provenance=provenance_label("tests_with_mutant_applied"),
        ),
    )


@pytest.fixture
def s3_episode() -> Episode:
    outdated = failing_test(
        "tests/test_synthlib.py::test_normalise", "synthlib.core.normalise"
    )
    regressed = failing_test(
        "tests/test_synthlib.py::test_slugify", "synthlib.core.slugify"
    )
    groups = [
        FailureGroup(
            group_id="g-normalise",
            function_qualname="synthlib.core.normalise",
            test_ids=[outdated.test_id],
        ),
        FailureGroup(
            group_id="g-slugify",
            function_qualname="synthlib.core.slugify",
            test_ids=[regressed.test_id],
        ),
    ]
    return Episode(
        episode_id="ep-s3",
        scenario=ScenarioType.S3_MIXED,
        difficulty=Difficulty.HARD,
        stream_id="synthlib-stream",
        stream_position=12,
        agent_input=episode_input(
            tests=[outdated, regressed],
            corpus_items=intent_evidence(),
            groups=groups,
            episode_id="ep-s3",
        ),
        ground_truth=EpisodeGroundTruth(
            episode_id="ep-s3",
            scenario=ScenarioType.S3_MIXED,
            per_test_labels=[
                PerTestGoldLabel(
                    test_id=outdated.test_id,
                    group_id="g-normalise",
                    function_qualname="synthlib.core.normalise",
                    gold_direction=RepairDirection.UPDATE_TEST,
                    expected_action=DecisionLabel.UPDATE_TEST,
                    cause="outdated_test",
                    gold_evidence_ids=["ev-commit-msg-1"],
                ),
                PerTestGoldLabel(
                    test_id=regressed.test_id,
                    group_id="g-slugify",
                    function_qualname="synthlib.core.slugify",
                    gold_direction=RepairDirection.FIX_CODE,
                    expected_action=DecisionLabel.FIX_CODE,
                    cause="injected_regression",
                    gold_evidence_ids=[],
                    injected_mutant_id="mut-9",
                ),
            ],
            reference_repairs=[
                ReferenceRepair(
                    kind="developer_test_diff",
                    unified_diff="@@\n-    assert normalise('ab') == 'ab'\n+    assert normalise('ab') == 'Ab'",
                    changed_files=["tests/test_synthlib.py"],
                )
            ],
            injected_mutants=[
                MutantSpec(
                    mutant_id="mut-9",
                    function_qualname="synthlib.core.slugify",
                    file_path="src/synthlib/core.py",
                )
            ],
            gold_evidence_ids=["ev-commit-msg-1"],
            evidence_availability=Availability.AVAILABLE,
            label_provenance=provenance_label("manual_review"),
        ),
    )


@pytest.fixture
def s4_episode() -> Episode:
    tests = [failing_test("tests/test_synthlib.py::test_normalise_suffix", "synthlib.core.normalise")]
    record = memory_record("mem-approved-normalise")
    return Episode(
        episode_id="ep-s4",
        scenario=ScenarioType.S4_DECOY_REGRESSION,
        difficulty=Difficulty.HARD,
        stream_id="synthlib-stream",
        stream_position=15,
        agent_input=episode_input(
            tests=tests,
            corpus_items=intent_evidence(),
            memory=[record],
            episode_id="ep-s4",
        ),
        ground_truth=EpisodeGroundTruth(
            episode_id="ep-s4",
            scenario=ScenarioType.S4_DECOY_REGRESSION,
            per_test_labels=[
                PerTestGoldLabel(
                    test_id=tests[0].test_id,
                    group_id="g-normalise",
                    function_qualname="synthlib.core.normalise",
                    gold_direction=RepairDirection.FIX_CODE,
                    expected_action=DecisionLabel.FIX_CODE,
                    cause="decoy_regression",
                    gold_evidence_ids=[],
                    related_memory_record_id="mem-approved-normalise",
                    memory_applies=False,
                    injected_mutant_id="mut-decoy",
                )
            ],
            injected_mutants=[
                MutantSpec(
                    mutant_id="mut-decoy",
                    function_qualname="synthlib.core.normalise",
                    file_path="src/synthlib/core.py",
                    operator="boundary_shift",
                )
            ],
            decoy_memory_record_ids=["mem-approved-normalise"],
            gold_evidence_ids=[],
            evidence_availability=Availability.ABSENT,
            label_provenance=provenance_label("tests_with_mutant_applied"),
        ),
    )


@pytest.fixture
def s5_episode() -> Episode:
    tests = [failing_test("tests/test_synthlib.py::test_normalise", "synthlib.core.normalise")]
    return Episode(
        episode_id="ep-s5",
        scenario=ScenarioType.S5_INTENT_STRIPPED,
        difficulty=Difficulty.MEDIUM,
        stream_id="synthlib-stream",
        stream_position=18,
        agent_input=episode_input(
            tests=tests,
            corpus_items=[],  # empty corpus, not an unavailable one
            memory=[],  # memory enabled, nothing applies
            commit_message=None,
            availability=Availability.WITHHELD,
            episode_id="ep-s5",
        ),
        ground_truth=EpisodeGroundTruth(
            episode_id="ep-s5",
            scenario=ScenarioType.S5_INTENT_STRIPPED,
            per_test_labels=[
                PerTestGoldLabel(
                    test_id=tests[0].test_id,
                    group_id="g-normalise",
                    function_qualname="synthlib.core.normalise",
                    gold_direction=RepairDirection.UPDATE_TEST,
                    expected_action=DecisionLabel.ESCALATE,
                    cause="outdated_test",
                    gold_evidence_ids=[],
                )
            ],
            expected_episode_outcome=DecisionLabel.ESCALATE,
            reference_repairs=[
                ReferenceRepair(
                    kind="developer_test_diff",
                    unified_diff="@@\n-    assert normalise('ab') == 'ab'\n+    assert normalise('ab') == 'Ab'",
                    changed_files=["tests/test_synthlib.py"],
                )
            ],
            gold_evidence_ids=None,
            evidence_availability=Availability.WITHHELD,
            stripped_evidence=[
                StrippedEvidence(
                    source_type=EvidenceSourceType.COMMIT_MESSAGE,
                    source_ref=HEAD,
                    reason="intent_stripped_scenario",
                ),
                StrippedEvidence(
                    source_type=EvidenceSourceType.DOCUMENTATION,
                    source_ref="docs/api.md",
                    reason="intent_stripped_scenario",
                ),
            ],
            label_provenance=provenance_label(),
        ),
    )
