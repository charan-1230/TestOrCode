"""Episode data models for TestOrCode.

TestOrCode: Intent-Grounded Triage of CI Test Failures with Scoped Change Memory
and Test-Integrity Checking.

This module defines *only* the episode side of the stable interface:

  * shared enums and identifier types used across the project,
  * the agent-visible episode input (:class:`EpisodeInput`),
  * the evaluation-only ground truth (:class:`EpisodeGroundTruth`),
  * the complete dataset record (:class:`Episode`) that joins the two.

Design rule enforced here
-------------------------
Gold labels, reference repairs, injected mutants, scenario identity and the
tool-fault plan live in models that are *structurally* unreachable from
:class:`EpisodeInput`.  The orchestrator builds an agent payload with
``episode.agent_payload()``; there is no field on ``EpisodeInput`` capable of
holding evaluation-only data, so a leak cannot happen by forgetting to exclude
a field.  :func:`find_leaked_ground_truth` provides a belt-and-braces check.

What this module does *not* do
------------------------------
Pydantic validates data shapes.  It cannot enforce filesystem permissions,
patch scope, sandbox isolation or memory-write authority at run time.  Those
checks belong to the tools and the orchestrator; see ``RUNTIME_ENFORCEMENT``
at the bottom of this file for the authoritative list.

Targets Python 3.12 and Pydantic v2.  No dependency beyond ``pydantic``.
"""

from __future__ import annotations

import fnmatch
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "SCHEMA_VERSION",
    # identifier aliases
    "Identifier",
    "CommitSha",
    # enums
    "DecisionLabel",
    "RepairDirection",
    "HypothesisKind",
    "ScenarioType",
    "Difficulty",
    "Availability",
    "EvidenceSourceType",
    "PatchScope",
    "PathClass",
    "ToolName",
    "ToolFaultKind",
    "TestOutcome",
    # models
    "RepoRef",
    "CommitRef",
    "EnvironmentSpec",
    "EditPolicy",
    "EpisodeBudget",
    "FileChange",
    "FailingTest",
    "FailureGroup",
    "TestSuiteSnapshot",
    "CIEvent",
    "EvidenceItem",
    "RetrievalCorpusSpec",
    "ExecutableProbe",
    "MemoryRecord",
    "MutantSpec",
    "MutationFixture",
    "ToolFaultInjection",
    "ToolFaultPlan",
    "StrippedEvidence",
    "LabelProvenance",
    "ReferenceRepair",
    "PerTestGoldLabel",
    "EpisodeGroundTruth",
    "EpisodeInput",
    "Episode",
    # helpers / constants
    "AGENT_FORBIDDEN_KEYS",
    "find_leaked_ground_truth",
    "DEFAULT_ALLOWED_TOOLS",
    "RUNTIME_ENFORCEMENT",
]

SCHEMA_VERSION: Final[str] = "1.0.0"


# --------------------------------------------------------------------------- #
# Identifier aliases
# --------------------------------------------------------------------------- #

Identifier = Annotated[str, Field(min_length=1, max_length=256)]
"""A non-empty stable identifier (episode, evidence, test, record, ...)."""

CommitSha = Annotated[
    str, Field(min_length=7, max_length=40, pattern=r"^[0-9a-fA-F]{7,40}$")
]
"""A Git commit hash, abbreviated (>= 7 hex chars) or full."""


# --------------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------------- #


class DecisionLabel(StrEnum):
    """The three permitted outcomes of the TestOrCode decision policy."""

    UPDATE_TEST = "UPDATE_TEST"
    FIX_CODE = "FIX_CODE"
    ESCALATE = "ESCALATE"


class RepairDirection(StrEnum):
    """The underlying *side* that is wrong for a failing test.

    Distinct from :class:`DecisionLabel`: ``ESCALATE`` is an action the agent
    may take, never a statement about which side is wrong.  In S5 the direction
    is ``UPDATE_TEST`` by construction, but the expected *action* is
    ``ESCALATE`` because the intent evidence was stripped.
    """

    UPDATE_TEST = "UPDATE_TEST"
    FIX_CODE = "FIX_CODE"


class HypothesisKind(StrEnum):
    """Intent Analyst hypothesis space (proposal section 8)."""

    OUTDATED_TEST = "outdated_test"
    REGRESSION = "regression"
    UNKNOWN = "unknown"


class ScenarioType(StrEnum):
    """Episode construction types S1-S5 (proposal Table 3).

    Tool faults (``T``) are deliberately *not* a scenario: they are injected on
    top of any scenario and carry no gold decision label of their own.
    """

    S1_OUTDATED_TEST = "S1"
    S2_INJECTED_REGRESSION = "S2"
    S3_MIXED = "S3"
    S4_DECOY_REGRESSION = "S4"
    S5_INTENT_STRIPPED = "S5"


class Difficulty(StrEnum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class Availability(StrEnum):
    """Why a piece of information is or is not present.

    Needed to keep "we looked and there is none" apart from "we never
    annotated this" and from "we removed it on purpose" (S5).
    """

    AVAILABLE = "available"
    ABSENT = "absent"
    WITHHELD = "withheld"
    UNKNOWN = "unknown"


class EvidenceSourceType(StrEnum):
    """Where a retrievable item came from (proposal section 7, Retrieval)."""

    COMMIT_MESSAGE = "commit_message"
    CHANGELOG = "changelog"
    DOCUMENTATION = "documentation"
    DOCSTRING = "docstring"
    SOURCE_FILE = "source_file"
    TEST_FILE = "test_file"
    MEMORY_RECORD = "memory_record"
    TEST_RESULT = "test_result"
    OTHER = "other"


class PatchScope(StrEnum):
    """``apply_patch(scope=...)`` from the agent tool table."""

    SRC = "src"
    TEST = "test"


class PathClass(StrEnum):
    """Classification of a repository path under an :class:`EditPolicy`."""

    SRC = "src"
    TEST = "test"
    FORBIDDEN = "forbidden"
    OUTSIDE = "outside"


class ToolName(StrEnum):
    """Sandboxed tools available to the agents (proposal Table 2)."""

    RUN_TESTS = "run_tests"
    RUN_MUTANTS = "run_mutants"
    READ_FILE = "read_file"
    SEARCH_REPO = "search_repo"
    GIT_LOG = "git_log"
    GIT_SHOW = "git_show"
    GIT_BLAME = "git_blame"
    APPLY_PATCH = "apply_patch"
    MEMORY_READ = "memory.read"
    MEMORY_WRITE = "memory.write"
    ESCALATE = "escalate"


class ToolFaultKind(StrEnum):
    """Injected tool faults (proposal Table 3, row T)."""

    PATCH_DOES_NOT_APPLY = "patch_does_not_apply"
    INVALID_TEST_SELECTOR = "invalid_test_selector"
    TIMEOUT = "timeout"
    FILE_NOT_FOUND = "file_not_found"
    MALFORMED_OUTPUT = "malformed_output"


class TestOutcome(StrEnum):
    """Per-test outcome in a suite snapshot."""

    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"
    SKIPPED = "skipped"
    XFAILED = "xfailed"
    XPASSED = "xpassed"
    NOT_RUN = "not_run"


DEFAULT_ALLOWED_TOOLS: Final[tuple[ToolName, ...]] = (
    ToolName.RUN_TESTS,
    ToolName.READ_FILE,
    ToolName.SEARCH_REPO,
    ToolName.GIT_LOG,
    ToolName.GIT_SHOW,
    ToolName.GIT_BLAME,
    ToolName.APPLY_PATCH,
    ToolName.MEMORY_READ,
    ToolName.ESCALATE,
)
"""Tools an agent role may be given.  ``memory.write`` and ``run_mutants`` are
reserved for the Integrity Guard and are granted by the orchestrator, not by
the episode."""


# --------------------------------------------------------------------------- #
# Base models
# --------------------------------------------------------------------------- #


class _Frozen(BaseModel):
    """Immutable, strict base for every episode-side model.

    ``extra="forbid"`` means an unknown key (for example a stray ``gold_label``)
    is a validation error rather than a silently carried payload.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
        use_enum_values=False,
    )


# --------------------------------------------------------------------------- #
# Repository, environment and policy
# --------------------------------------------------------------------------- #


class RepoRef(_Frozen):
    """Identifies the library under test."""

    repo_id: Identifier = Field(description="Stable short id, e.g. 'toolz'.")
    repo_url: str | None = Field(
        default=None, description="Clone URL, for provenance only."
    )
    package_name: str | None = None
    license: str | None = None


class CommitRef(_Frozen):
    """A pinned point in the repository history."""

    sha: CommitSha
    committed_at: datetime | None = None
    stream_index: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Chronological position of this commit in the episode stream. "
            "Used to reject corpus items that post-date the current commit."
        ),
    )


class EnvironmentSpec(_Frozen):
    """Everything needed to re-create the sandbox for this episode."""

    python_version: str = Field(description="e.g. '3.12.3'.")
    docker_image: str | None = Field(
        default=None, description="Image reference, ideally digest-pinned."
    )
    dependency_lock_hash: str | None = Field(
        default=None, description="Hash of the pinned dependency set."
    )
    pytest_version: str | None = None
    network_enabled: Literal[False] = Field(
        default=False,
        description="Sandbox is always offline; LLM calls go through the orchestrator.",
    )
    test_timeout_seconds: int | None = Field(default=None, gt=0)
    stability_runs: int | None = Field(
        default=None, gt=0, description="Repeat count of the stability filter (5)."
    )
    tool_versions: dict[str, str] = Field(
        default_factory=dict,
        description="e.g. {'git': '2.43.0', 'mutmut': '2.5.0'}.",
    )


class EditPolicy(_Frozen):
    """Declarative statement of which paths may be edited.

    This is a *specification* consumed by ``apply_patch`` and the Integrity
    Guard.  The schema cannot stop a write; it only states the rule and offers
    :meth:`classify_path` so every component applies the same rule.
    """

    allowed_source_roots: tuple[str, ...] = ("src/",)
    allowed_test_roots: tuple[str, ...] = ("tests/", "test/")
    forbidden_globs: tuple[str, ...] = (
        "**/conftest.py",
        "conftest.py",
        ".github/**",
        "pyproject.toml",
        "setup.py",
        "setup.cfg",
        "tox.ini",
        "requirements*.txt",
        "*.lock",
        ".pre-commit-config.yaml",
        "Dockerfile*",
    )

    def classify_path(self, path: str) -> PathClass:
        """Classify a repository-relative path.

        ``FORBIDDEN`` wins over ``SRC``/``TEST`` so that ``tests/conftest.py``
        is never editable.
        """
        norm = path.replace("\\", "/")
        while norm.startswith("./"):
            norm = norm[2:]
        norm = norm.lstrip("/")
        for pattern in self.forbidden_globs:
            if fnmatch.fnmatch(norm, pattern) or fnmatch.fnmatch(
                norm, pattern.replace("**/", "*")
            ):
                return PathClass.FORBIDDEN
        if any(norm.startswith(root) for root in self.allowed_test_roots):
            return PathClass.TEST
        if any(norm.startswith(root) for root in self.allowed_source_roots):
            return PathClass.SRC
        return PathClass.OUTSIDE

    def is_editable(self, path: str, scope: PatchScope) -> bool:
        """True iff ``path`` may be edited by a patch declared with ``scope``."""
        cls = self.classify_path(path)
        if scope is PatchScope.SRC:
            return cls is PathClass.SRC
        return cls is PathClass.TEST


class EpisodeBudget(_Frozen):
    """Per-episode budget; identical across conditions by experimental design."""

    max_tool_calls: int | None = Field(default=None, gt=0)
    max_llm_calls: int | None = Field(default=None, gt=0)
    max_tokens: int | None = Field(default=None, gt=0)
    max_revisions: int = Field(
        default=1, ge=0, description="Guard-rejection revision rounds (proposal: 1)."
    )
    wall_clock_seconds: int | None = Field(default=None, gt=0)


# --------------------------------------------------------------------------- #
# CI event
# --------------------------------------------------------------------------- #


class FileChange(_Frozen):
    """One file touched by the commit that triggered the CI event."""

    path: str = Field(min_length=1)
    change_kind: Literal["added", "modified", "deleted", "renamed"]
    unified_diff: str | None = Field(
        default=None, description="None means the diff body was not materialised."
    )
    lines_added: int | None = Field(default=None, ge=0)
    lines_removed: int | None = Field(default=None, ge=0)


class FailingTest(_Frozen):
    """A single failing pytest node confirmed stable by the Triage agent."""

    test_id: Identifier = Field(description="pytest node id.")
    test_file: str = Field(min_length=1)
    test_name: str = Field(min_length=1)
    target_function: str | None = Field(
        default=None,
        description="Qualified name of the function under test, used for grouping.",
    )
    failure_type: Literal["assertion", "error"] = "assertion"
    failure_message: str | None = None
    traceback: str | None = None
    assertion_diff: str | None = Field(
        default=None, description="Expected/actual rendering from pytest."
    )
    stability_runs: int | None = Field(default=None, gt=0)
    stable: bool | None = Field(
        default=None,
        description="None means the stability filter was not run for this test.",
    )


class FailureGroup(_Frozen):
    """Failures grouped by changed function (Triage output)."""

    group_id: Identifier
    function_qualname: str | None = None
    test_ids: list[Identifier] = Field(min_length=1)


class TestSuiteSnapshot(_Frozen):
    """Counts and outcomes for one execution of the suite."""

    phase: Literal["episode_start", "parent_commit", "reference"] = "episode_start"
    collected: bool = True
    total: int | None = Field(default=None, ge=0)
    passed: int | None = Field(default=None, ge=0)
    failed: int | None = Field(default=None, ge=0)
    errors: int | None = Field(default=None, ge=0)
    skipped: int | None = Field(default=None, ge=0)
    failing_test_ids: list[Identifier] = Field(default_factory=list)
    outcomes: dict[str, TestOutcome] | None = Field(
        default=None, description="None means per-test outcomes were not recorded."
    )


class CIEvent(_Frozen):
    """The red CI run the agent is asked to triage.

    Contains only information a maintainer would see at that commit.  The
    parent commit is included because the episode is built by applying a
    source-only change to it; no *later* commit may appear here.
    """

    event_id: Identifier
    commit: CommitRef = Field(description="Current (head) commit of the episode.")
    parent_commit: CommitRef | None = None
    commit_message: str | None = Field(
        default=None,
        description="None together with availability != AVAILABLE for S5.",
    )
    commit_message_availability: Availability = Availability.AVAILABLE
    changed_files: list[FileChange] = Field(default_factory=list)
    failing_tests: list[FailingTest] = Field(min_length=1)
    failure_groups: list[FailureGroup] = Field(default_factory=list)
    suite_snapshot: TestSuiteSnapshot | None = None

    @model_validator(mode="after")
    def _check_consistency(self) -> CIEvent:
        ids = [t.test_id for t in self.failing_tests]
        if len(ids) != len(set(ids)):
            raise ValueError("failing_tests contains duplicate test_id values")
        known = set(ids)
        for group in self.failure_groups:
            unknown = [t for t in group.test_ids if t not in known]
            if unknown:
                raise ValueError(
                    f"failure_group {group.group_id!r} references unknown test ids: "
                    f"{sorted(unknown)}"
                )
        if self.commit_message_availability is Availability.AVAILABLE:
            if self.commit_message is None:
                raise ValueError(
                    "commit_message_availability is AVAILABLE but commit_message is None; "
                    "use ABSENT, WITHHELD or UNKNOWN to record a missing message"
                )
        elif self.commit_message is not None:
            raise ValueError(
                "commit_message must be None unless commit_message_availability "
                "is AVAILABLE"
            )
        if self.parent_commit is not None and self.parent_commit.sha == self.commit.sha:
            raise ValueError("parent_commit.sha must differ from commit.sha")
        return self


# --------------------------------------------------------------------------- #
# Retrieval corpus and evidence
# --------------------------------------------------------------------------- #


class EvidenceItem(_Frozen):
    """One retrievable unit in the per-library RAG corpus.

    Carries only the fields needed to cite it.  Relevance scores and ranks are
    produced at retrieval time and live in ``result_schemas.RetrievedEvidence``,
    never here: the corpus must not pre-judge which item is correct.
    """

    evidence_id: Identifier
    source_type: EvidenceSourceType
    source_ref: str = Field(
        min_length=1,
        description="Repository path, commit sha, doc section or memory record id.",
    )
    commit_sha: str | None = None
    commit_index: int | None = Field(
        default=None,
        ge=0,
        description="Chronological index; must not exceed the corpus cutoff.",
    )
    section: str | None = None
    line_start: int | None = Field(default=None, ge=1)
    line_end: int | None = Field(default=None, ge=1)
    text: str | None = Field(default=None, description="Snippet, if materialised.")
    created_at: datetime | None = None

    @model_validator(mode="after")
    def _check_span(self) -> EvidenceItem:
        if (
            self.line_start is not None
            and self.line_end is not None
            and self.line_end < self.line_start
        ):
            raise ValueError("line_end must be >= line_start")
        return self


class RetrievalCorpusSpec(_Frozen):
    """What the Intent Analyst is allowed to search for this episode.

    ``items is None`` means the corpus is materialised at run time from the
    pinned repository; ``items == []`` means the corpus is genuinely empty.
    The two are not the same and must not be collapsed.
    """

    corpus_id: Identifier
    corpus_version: str | None = None
    index_version: str | None = None
    history_cutoff_commit: CommitSha = Field(
        description="Inclusive cutoff; equals the episode's current commit."
    )
    history_cutoff_index: int | None = Field(default=None, ge=0)
    included_source_types: list[EvidenceSourceType] = Field(default_factory=list)
    items: list[EvidenceItem] | None = None
    excluded_note: str | None = Field(
        default=None,
        description="Human note on deliberate exclusions, e.g. stripped docs in S5.",
    )

    @model_validator(mode="after")
    def _check_cutoff(self) -> RetrievalCorpusSpec:
        if self.items is None:
            return self
        seen: set[str] = set()
        for item in self.items:
            if item.evidence_id in seen:
                raise ValueError(f"duplicate evidence_id {item.evidence_id!r}")
            seen.add(item.evidence_id)
            if (
                self.history_cutoff_index is not None
                and item.commit_index is not None
                and item.commit_index > self.history_cutoff_index
            ):
                raise ValueError(
                    f"evidence item {item.evidence_id!r} has commit_index "
                    f"{item.commit_index} beyond history_cutoff_index "
                    f"{self.history_cutoff_index}: future commits must never be indexed"
                )
        return self


# --------------------------------------------------------------------------- #
# Scoped memory
# --------------------------------------------------------------------------- #


class ExecutableProbe(_Frozen):
    """Executable check derived from the assertions of an approved test patch.

    A memory record counts as evidence only while its probe passes on the
    current code (proposal, Decision policy).
    """

    probe_id: Identifier
    probe_code: str = Field(
        min_length=1, description="Self-contained pytest-compatible snippet."
    )
    probe_selector: str | None = Field(
        default=None, description="Node id used to run the probe, if pre-installed."
    )
    derived_from: Literal["approved_test_patch", "manual"] = "approved_test_patch"
    expected_outcome: Literal["pass"] = "pass"


class MemoryRecord(_Frozen):
    """A maintainer-approved behaviour change, scoped and probe-backed.

    Shape follows the proposal: ``{function, behaviour signature, direction,
    executable probe, evidence IDs, commit}``.  ``scope`` exists so the coarse
    module-level memory ablation can be represented with the same model.
    """

    record_id: Identifier
    repo_id: Identifier
    function_qualname: str = Field(min_length=1)
    behaviour_signature: str = Field(
        min_length=1, description="Short description of the approved new behaviour."
    )
    direction: RepairDirection
    probe: ExecutableProbe
    evidence_ids: list[Identifier] = Field(default_factory=list)
    approved_at_commit: CommitSha
    created_in_episode: Identifier | None = None
    scope: Literal["function", "module"] = "function"
    supersedes: Identifier | None = None
    superseded_by: Identifier | None = None
    flagged_stale: bool = Field(
        default=False, description="Set when the probe keeps failing."
    )
    approved_by_maintainer: Literal[True] = Field(
        default=True,
        description=(
            "A stored record exists only after simulated maintainer approval. "
            "Unapproved write attempts are recorded in result_schemas, not here."
        ),
    )

    @model_validator(mode="after")
    def _check_self_reference(self) -> MemoryRecord:
        if self.supersedes == self.record_id or self.superseded_by == self.record_id:
            raise ValueError("a memory record cannot supersede itself")
        return self


# --------------------------------------------------------------------------- #
# Mutation fixtures (evaluation-only)
# --------------------------------------------------------------------------- #


class MutantSpec(_Frozen):
    """One mutant of a function.

    Used both for the regression injected into S2/S3/S4 and for the cached
    mutant catalogue the Integrity Guard runs through ``run_mutants``.  Mutant
    identity is evaluation-only: the agent observes the *effect* of an injected
    mutant on the source, never this record.
    """

    mutant_id: Identifier
    function_qualname: str = Field(min_length=1)
    file_path: str = Field(min_length=1)
    operator: str | None = Field(default=None, description="e.g. 'comparison_swap'.")
    unified_diff: str | None = None
    killed_by_reference_tests: list[Identifier] | None = Field(
        default=None,
        description="None means not yet computed; [] means no reference test kills it.",
    )


class MutationFixture(_Frozen):
    """Precomputed mutant catalogue for one function, for deterministic scoring."""

    catalogue_id: Identifier
    function_qualname: str = Field(min_length=1)
    mutants: list[MutantSpec] = Field(default_factory=list)
    reference_killed_mutant_ids: list[Identifier] | None = Field(
        default=None,
        description=(
            "Mutants killed by the developer's reference tests; the denominator "
            "of mutant-kill retention.  None means not computed."
        ),
    )
    generated_by: str | None = Field(default=None, description="e.g. 'mutmut 2.5.0'.")

    @model_validator(mode="after")
    def _check_reference_subset(self) -> MutationFixture:
        if self.reference_killed_mutant_ids is None:
            return self
        known = {m.mutant_id for m in self.mutants}
        unknown = [m for m in self.reference_killed_mutant_ids if m not in known]
        if unknown:
            raise ValueError(
                f"reference_killed_mutant_ids reference unknown mutants: {sorted(unknown)}"
            )
        return self


# --------------------------------------------------------------------------- #
# Tool faults (orchestrator-only)
# --------------------------------------------------------------------------- #


class ToolFaultInjection(_Frozen):
    """A single scripted tool fault.

    Lives in :attr:`Episode.fault_plan`, i.e. outside the agent input: the
    agent must not know a fault is coming.  Observed faults are recorded by
    ``result_schemas.ToolEvent``.
    """

    fault_id: Identifier
    tool: ToolName
    fault_kind: ToolFaultKind
    trigger: Literal["call_index", "first_call", "probabilistic"] = "call_index"
    call_index: int | None = Field(default=None, ge=0)
    probability: float | None = Field(default=None, ge=0.0, le=1.0)
    max_occurrences: int = Field(default=1, gt=0)
    message: str | None = None

    @model_validator(mode="after")
    def _check_trigger(self) -> ToolFaultInjection:
        if self.trigger == "call_index" and self.call_index is None:
            raise ValueError("trigger 'call_index' requires call_index")
        if self.trigger == "probabilistic" and self.probability is None:
            raise ValueError("trigger 'probabilistic' requires probability")
        return self


class ToolFaultPlan(_Frozen):
    """Deterministic fault schedule for scenario T (about 10% of tool calls)."""

    enabled: bool = False
    seed: int | None = Field(
        default=None, description="None means unseeded, not seed 0."
    )
    target_failure_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    injections: list[ToolFaultInjection] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_enabled(self) -> ToolFaultPlan:
        if not self.enabled and (self.injections or self.target_failure_rate):
            raise ValueError(
                "fault plan is disabled but defines injections or a target rate"
            )
        if self.enabled and not self.injections and self.target_failure_rate is None:
            raise ValueError(
                "enabled fault plan must define injections or a target_failure_rate"
            )
        return self


# --------------------------------------------------------------------------- #
# Ground truth (evaluation-only)
# --------------------------------------------------------------------------- #


class StrippedEvidence(_Frozen):
    """Evidence that was deliberately removed or verified not to exist.

    Records *absence* accurately for S5 instead of leaving a silent gap.
    """

    source_type: EvidenceSourceType
    source_ref: str | None = None
    reason: Literal[
        "intent_stripped_scenario",
        "not_present_in_history",
        "excluded_by_cutoff",
    ]
    original_evidence_id: Identifier | None = None
    note: str | None = None


class LabelProvenance(_Frozen):
    """How the per-test gold labels were established."""

    method: Literal[
        "tests_with_source_only_change",
        "tests_with_mutant_applied",
        "manual_review",
    ]
    stability_runs: int | None = Field(default=None, gt=0)
    verified_by: str | None = None
    verified_at: datetime | None = None
    note: str | None = None


class ReferenceRepair(_Frozen):
    """The developer's own diff, used as the repair reference.

    Never visible to the agent (proposal section 9, Environment).
    """

    kind: Literal["developer_test_diff", "developer_source_diff"]
    unified_diff: str = Field(min_length=1)
    changed_files: list[str] = Field(default_factory=list)
    source_commit: CommitSha | None = None


class PerTestGoldLabel(_Frozen):
    """Ground truth for one failing test (or one failure group).

    ``gold_direction`` is which side is actually wrong.  ``expected_action`` is
    what a correct agent should *do*, which differs only in S5, where the
    evidence needed to justify ``UPDATE_TEST`` has been stripped and the correct
    action is ``ESCALATE``.
    """

    test_id: Identifier
    group_id: Identifier | None = None
    function_qualname: str | None = None
    gold_direction: RepairDirection
    expected_action: DecisionLabel
    cause: Literal["outdated_test", "injected_regression", "decoy_regression"]
    gold_evidence_ids: list[Identifier] | None = Field(
        default=None,
        description="None = not annotated; [] = annotated and no such evidence exists.",
    )
    related_memory_record_id: Identifier | None = None
    memory_applies: bool | None = Field(
        default=None,
        description="S4 decoys set this False: a prior record must not justify an update.",
    )
    injected_mutant_id: Identifier | None = None
    note: str | None = None

    @model_validator(mode="after")
    def _check_action(self) -> PerTestGoldLabel:
        if (
            self.expected_action is not DecisionLabel.ESCALATE
            and self.expected_action.value != self.gold_direction.value
        ):
            raise ValueError(
                f"expected_action {self.expected_action} contradicts gold_direction "
                f"{self.gold_direction}; only ESCALATE may differ"
            )
        if self.cause == "injected_regression" and self.gold_direction is not (
            RepairDirection.FIX_CODE
        ):
            raise ValueError("cause 'injected_regression' requires FIX_CODE direction")
        if self.cause == "decoy_regression" and self.gold_direction is not (
            RepairDirection.FIX_CODE
        ):
            raise ValueError("cause 'decoy_regression' requires FIX_CODE direction")
        if self.cause == "outdated_test" and self.gold_direction is not (
            RepairDirection.UPDATE_TEST
        ):
            raise ValueError("cause 'outdated_test' requires UPDATE_TEST direction")
        return self


class EpisodeGroundTruth(_Frozen):
    """Everything the evaluation harness knows and the agent must never see."""

    episode_id: Identifier
    scenario: ScenarioType
    per_test_labels: list[PerTestGoldLabel] = Field(min_length=1)
    expected_episode_outcome: DecisionLabel | None = Field(
        default=None,
        description="Set to ESCALATE for S5; None when the gold is per-test.",
    )
    reference_repairs: list[ReferenceRepair] = Field(default_factory=list)
    injected_mutants: list[MutantSpec] = Field(default_factory=list)
    mutation_fixtures: list[MutationFixture] = Field(default_factory=list)
    gold_evidence_ids: list[Identifier] | None = Field(
        default=None,
        description="Episode-level union; None = not annotated, [] = none exists.",
    )
    evidence_availability: Availability = Availability.UNKNOWN
    stripped_evidence: list[StrippedEvidence] = Field(default_factory=list)
    decoy_memory_record_ids: list[Identifier] = Field(default_factory=list)
    future_commit_shas: list[CommitSha] = Field(
        default_factory=list,
        description="Known later commits; checked against the agent input for leakage.",
    )
    label_provenance: LabelProvenance
    note: str | None = None

    @model_validator(mode="after")
    def _check_scenario_rules(self) -> EpisodeGroundTruth:
        labels = self.per_test_labels
        test_ids = [lbl.test_id for lbl in labels]
        if len(test_ids) != len(set(test_ids)):
            raise ValueError("per_test_labels contains duplicate test_id values")

        directions = {lbl.gold_direction for lbl in labels}
        has_mutant = bool(self.injected_mutants)
        has_ref_test_diff = any(
            r.kind == "developer_test_diff" for r in self.reference_repairs
        )
        scenario = self.scenario

        if scenario is ScenarioType.S1_OUTDATED_TEST:
            if directions != {RepairDirection.UPDATE_TEST}:
                raise ValueError("S1 requires every gold_direction to be UPDATE_TEST")
            if has_mutant:
                raise ValueError("S1 must not inject a mutant")
            if not has_ref_test_diff:
                raise ValueError(
                    "S1 requires a developer_test_diff as the reference repair"
                )
        elif scenario is ScenarioType.S2_INJECTED_REGRESSION:
            if directions != {RepairDirection.FIX_CODE}:
                raise ValueError("S2 requires every gold_direction to be FIX_CODE")
            if not has_mutant:
                raise ValueError("S2 requires at least one injected mutant")
            if has_ref_test_diff:
                raise ValueError(
                    "S2 tests must stay untouched: no developer_test_diff reference"
                )
        elif scenario is ScenarioType.S3_MIXED:
            if directions != {RepairDirection.UPDATE_TEST, RepairDirection.FIX_CODE}:
                raise ValueError(
                    "S3 requires both UPDATE_TEST and FIX_CODE per-test labels"
                )
            if not has_mutant:
                raise ValueError("S3 requires at least one injected mutant")
            if not has_ref_test_diff:
                raise ValueError(
                    "S3 requires a developer_test_diff for its outdated tests"
                )
            outdated_fns = {
                lbl.function_qualname
                for lbl in labels
                if lbl.gold_direction is RepairDirection.UPDATE_TEST
            } - {None}
            regressed_fns = {
                lbl.function_qualname
                for lbl in labels
                if lbl.gold_direction is RepairDirection.FIX_CODE
            } - {None}
            if outdated_fns & regressed_fns:
                raise ValueError(
                    "S3 requires the injected regression to sit in a different "
                    f"function; overlap: {sorted(outdated_fns & regressed_fns)}"
                )
        elif scenario is ScenarioType.S4_DECOY_REGRESSION:
            if directions != {RepairDirection.FIX_CODE}:
                raise ValueError("S4 requires every gold_direction to be FIX_CODE")
            if not has_mutant:
                raise ValueError("S4 requires at least one injected mutant")
            if not self.decoy_memory_record_ids:
                raise ValueError(
                    "S4 requires the memory record(s) that must not justify an update"
                )
            if any(lbl.memory_applies for lbl in labels):
                raise ValueError(
                    "S4 labels must not mark the prior memory record as applicable"
                )
        elif scenario is ScenarioType.S5_INTENT_STRIPPED:
            if self.expected_episode_outcome is not DecisionLabel.ESCALATE:
                raise ValueError("S5 requires expected_episode_outcome == ESCALATE")
            if any(
                lbl.expected_action is not DecisionLabel.ESCALATE for lbl in labels
            ):
                raise ValueError("S5 requires expected_action ESCALATE on every label")
            if self.gold_evidence_ids:
                raise ValueError(
                    "S5 has no sufficient intent evidence: gold_evidence_ids must be "
                    "None or empty"
                )
            if self.evidence_availability not in (
                Availability.WITHHELD,
                Availability.ABSENT,
            ):
                raise ValueError(
                    "S5 requires evidence_availability WITHHELD or ABSENT"
                )
            if not self.stripped_evidence:
                raise ValueError(
                    "S5 must record what was removed in stripped_evidence"
                )

        if (
            self.expected_episode_outcome is DecisionLabel.ESCALATE
            and scenario is not ScenarioType.S5_INTENT_STRIPPED
        ):
            raise ValueError(
                "episode-level ESCALATE is only defined for S5 in the current design"
            )

        mutant_ids = {m.mutant_id for m in self.injected_mutants}
        unknown = {
            lbl.injected_mutant_id
            for lbl in labels
            if lbl.injected_mutant_id is not None
        } - mutant_ids
        if unknown:
            raise ValueError(
                f"labels reference unknown injected mutants: {sorted(unknown)}"
            )
        return self


# --------------------------------------------------------------------------- #
# Agent-facing input
# --------------------------------------------------------------------------- #


class EpisodeInput(_Frozen):
    """Everything the agent is allowed to see, and nothing else.

    There is deliberately no scenario, difficulty, gold-label, reference-repair,
    injected-mutant or fault-plan field on this model, and ``extra="forbid"``
    prevents one being attached later.  Serialising this object can therefore
    never leak ground truth.
    """

    schema_version: str = SCHEMA_VERSION
    episode_id: Identifier
    repo: RepoRef
    environment: EnvironmentSpec
    ci_event: CIEvent
    corpus: RetrievalCorpusSpec
    visible_memory: list[MemoryRecord] | None = Field(
        default=None,
        description=(
            "None = memory disabled for this condition (A2); "
            "[] = memory enabled but nothing applies yet."
        ),
    )
    edit_policy: EditPolicy = Field(default_factory=EditPolicy)
    budget: EpisodeBudget = Field(default_factory=EpisodeBudget)
    allowed_tools: list[ToolName] = Field(
        default_factory=lambda: list(DEFAULT_ALLOWED_TOOLS)
    )
    allowed_actions: tuple[DecisionLabel, ...] = (
        DecisionLabel.UPDATE_TEST,
        DecisionLabel.FIX_CODE,
        DecisionLabel.ESCALATE,
    )

    @model_validator(mode="after")
    def _check_cutoff_matches_event(self) -> EpisodeInput:
        if self.corpus.history_cutoff_commit.lower() not in (
            self.ci_event.commit.sha.lower(),
            self.ci_event.commit.sha.lower()[: len(self.corpus.history_cutoff_commit)],
        ) and not self.ci_event.commit.sha.lower().startswith(
            self.corpus.history_cutoff_commit.lower()
        ):
            raise ValueError(
                "corpus.history_cutoff_commit must be the episode's current commit; "
                f"got {self.corpus.history_cutoff_commit!r} vs "
                f"{self.ci_event.commit.sha!r}"
            )
        if self.visible_memory is not None:
            ids = [r.record_id for r in self.visible_memory]
            if len(ids) != len(set(ids)):
                raise ValueError("visible_memory contains duplicate record_id values")
        return self


# --------------------------------------------------------------------------- #
# Complete dataset record
# --------------------------------------------------------------------------- #


class Episode(_Frozen):
    """One JSONL record of ``datasets/pilot_episodes.jsonl``.

    An episode is a single CI event in a chronological stream: a pinned commit
    of one library whose suite is red, together with the per-failing-test ground
    truth needed to score an agent's handling of it.
    """

    schema_version: str = SCHEMA_VERSION
    episode_id: Identifier
    scenario: ScenarioType
    difficulty: Difficulty = Difficulty.EASY
    stream_id: Identifier = Field(
        description="Chronological stream (usually one per library)."
    )
    stream_position: int = Field(
        ge=0, description="Order within the stream; memory flows forward only."
    )
    created_at: datetime | None = None
    agent_input: EpisodeInput
    ground_truth: EpisodeGroundTruth
    fault_plan: ToolFaultPlan = Field(
        default_factory=ToolFaultPlan,
        description="Orchestrator-only; never part of the agent input.",
    )
    tags: list[str] = Field(default_factory=list)
    note: str | None = None

    @model_validator(mode="after")
    def _check_episode(self) -> Episode:
        if self.agent_input.episode_id != self.episode_id:
            raise ValueError("agent_input.episode_id does not match episode_id")
        if self.ground_truth.episode_id != self.episode_id:
            raise ValueError("ground_truth.episode_id does not match episode_id")
        if self.ground_truth.scenario is not self.scenario:
            raise ValueError("ground_truth.scenario does not match scenario")

        failing = {t.test_id for t in self.agent_input.ci_event.failing_tests}
        labelled = {lbl.test_id for lbl in self.ground_truth.per_test_labels}
        missing = labelled - failing
        if missing:
            raise ValueError(
                f"gold labels reference tests that are not failing in the CI event: "
                f"{sorted(missing)}"
            )
        unlabelled = failing - labelled
        if unlabelled:
            raise ValueError(
                f"failing tests without a gold label: {sorted(unlabelled)}"
            )

        # Future-commit leakage check.
        future = {sha.lower() for sha in self.ground_truth.future_commit_shas}
        if future:
            event = self.agent_input.ci_event
            visible = {event.commit.sha.lower()}
            if event.parent_commit is not None:
                visible.add(event.parent_commit.sha.lower())
            if visible & future:
                raise ValueError(
                    "ci_event references a commit listed as a future commit"
                )
            items = self.agent_input.corpus.items or []
            leaked = [
                item.evidence_id
                for item in items
                if item.commit_sha is not None and item.commit_sha.lower() in future
            ]
            if leaked:
                raise ValueError(
                    f"retrieval corpus indexes future commits via items: {sorted(leaked)}"
                )

        # Decoy memory records must actually be visible for S4 to be meaningful.
        if self.ground_truth.decoy_memory_record_ids:
            visible_records = {
                r.record_id for r in (self.agent_input.visible_memory or [])
            }
            missing_records = (
                set(self.ground_truth.decoy_memory_record_ids) - visible_records
            )
            if missing_records and self.agent_input.visible_memory is not None:
                raise ValueError(
                    "decoy memory records are not present in visible_memory: "
                    f"{sorted(missing_records)}"
                )
        return self

    # -- orchestrator helpers ------------------------------------------------ #

    def to_agent_input(self) -> EpisodeInput:
        """Return the agent-visible view.  Contains no evaluation-only data."""
        return self.agent_input

    def agent_payload(self) -> dict[str, Any]:
        """JSON-ready dict for prompting.  The only sanctioned way to build one."""
        return self.agent_input.model_dump(mode="json")

    def gold_actions(self) -> dict[str, DecisionLabel]:
        """Expected action per failing test, for the evaluation layer only."""
        return {
            lbl.test_id: lbl.expected_action
            for lbl in self.ground_truth.per_test_labels
        }


# --------------------------------------------------------------------------- #
# Leakage helper
# --------------------------------------------------------------------------- #

AGENT_FORBIDDEN_KEYS: Final[frozenset[str]] = frozenset(
    {
        "ground_truth",
        "scenario",
        "difficulty",
        "per_test_labels",
        "gold_direction",
        "gold_evidence_ids",
        "expected_action",
        "expected_episode_outcome",
        "reference_repairs",
        "reference_repair",
        "injected_mutants",
        "injected_mutant_id",
        "mutation_fixtures",
        "reference_killed_mutant_ids",
        "killed_by_reference_tests",
        "decoy_memory_record_ids",
        "future_commit_shas",
        "stripped_evidence",
        "label_provenance",
        "fault_plan",
        "memory_applies",
        "cause",
    }
)
"""Keys that must never appear anywhere in a payload sent to the model."""


def find_leaked_ground_truth(payload: Any, _path: str = "") -> list[str]:
    """Return dotted paths of forbidden keys found in ``payload``.

    A defence-in-depth check for the orchestrator and the test suite.  An empty
    list means the payload is safe with respect to :data:`AGENT_FORBIDDEN_KEYS`.
    """
    found: list[str] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            path = f"{_path}.{key}" if _path else str(key)
            if key in AGENT_FORBIDDEN_KEYS:
                found.append(path)
            found.extend(find_leaked_ground_truth(value, path))
    elif isinstance(payload, (list, tuple)):
        for index, value in enumerate(payload):
            found.extend(find_leaked_ground_truth(value, f"{_path}[{index}]"))
    return found


# --------------------------------------------------------------------------- #
# Runtime enforcement notes
# --------------------------------------------------------------------------- #

RUNTIME_ENFORCEMENT: Final[tuple[str, ...]] = (
    "Patch scope: apply_patch must reject writes outside EditPolicy-permitted "
    "paths. The schema only declares the policy.",
    "Forbidden files: CI configuration, dependency manifests and conftest.py are "
    "blocked by the patch tool and re-checked by the Integrity Guard.",
    "FIX_CODE must not touch test files: enforced by apply_patch(scope=src) and "
    "verified post hoc from the observed changed-file list.",
    "UPDATE_TEST requires cited positive evidence: enforced by the decision "
    "policy in the orchestrator and re-checked by the Integrity Guard.",
    "Memory writes: only the Integrity Guard holds memory.write, and only after "
    "simulated maintainer approval. The orchestrator grants the tool.",
    "Memory probes: a record may be used as evidence only when its probe passes "
    "on the current code. The memory tool must run the probe.",
    "Test weakening: assertion counting, skip/xfail detection, comparison "
    "loosening and exception swallowing are AST/diff checks in the Guard.",
    "Future-commit isolation: the retrieval indexer must cut history at the "
    "current commit; the Episode validator can only catch declared leaks.",
    "Sandbox isolation: Docker, no network, resource limits and per-episode "
    "worktrees are environment concerns, not schema concerns.",
    "Tool-fault injection: the fault plan is applied by the tool layer and must "
    "never be exposed to the agent.",
    "Budget enforcement: tool-call, LLM-call and token budgets are counted and "
    "cut off by the orchestrator.",
)
