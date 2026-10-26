"""Episode data models for TestOrCode.

Three layers, kept structurally apart:

* ``EpisodeInput``       - everything the agent may see (and nothing else),
* ``EpisodeGroundTruth`` - evaluation-only labels, reference repairs, mutants,
* ``Episode``            - one JSONL record joining both plus the tool-fault plan.

``EpisodeInput`` has no field capable of holding a gold label, scenario id,
reference repair, mutant identity or fault plan, and ``extra="forbid"`` stops
one being attached later.  Build the agent prompt only from
``Episode.agent_payload()``.

Pydantic validates shapes and cross-references.  It cannot enforce filesystem
permissions, patch scope, sandbox isolation or memory-write authority; see
``RUNTIME_ENFORCEMENT``.  Python 3.12, Pydantic v2, no other dependency.
"""

from __future__ import annotations

import fnmatch
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION: Final[str] = "2.0.0"

Identifier = Annotated[str, Field(min_length=1, max_length=256)]
CommitSha = Annotated[
    str, Field(min_length=7, max_length=40, pattern=r"^[0-9a-fA-F]{7,40}$")
]


def same_commit(a: str, b: str) -> bool:
    """True if two full or abbreviated SHAs can name the same commit."""
    a, b = a.lower(), b.lower()
    return a.startswith(b) or b.startswith(a)


# --------------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------------- #


class DecisionLabel(StrEnum):
    """The three permitted actions."""

    UPDATE_TEST = "UPDATE_TEST"
    FIX_CODE = "FIX_CODE"
    ESCALATE = "ESCALATE"


class RepairDirection(StrEnum):
    """Which side is actually wrong.  ESCALATE is an action, never a direction."""

    UPDATE_TEST = "UPDATE_TEST"
    FIX_CODE = "FIX_CODE"


class HypothesisKind(StrEnum):
    OUTDATED_TEST = "outdated_test"
    REGRESSION = "regression"
    UNKNOWN = "unknown"


class ScenarioType(StrEnum):
    """S1-S5.  Tool faults are orthogonal and are not a scenario."""

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
    """Keeps 'looked and none exists', 'removed on purpose' and 'never annotated' apart."""

    AVAILABLE = "available"
    ABSENT = "absent"
    WITHHELD = "withheld"
    UNKNOWN = "unknown"


class EvidenceSourceType(StrEnum):
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
    SRC = "src"
    TEST = "test"


class PathClass(StrEnum):
    SRC = "src"
    TEST = "test"
    FORBIDDEN = "forbidden"
    OUTSIDE = "outside"


class ToolName(StrEnum):
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
    """Scenario T faults (proposal Table 3)."""

    PATCH_DOES_NOT_APPLY = "patch_does_not_apply"
    INVALID_TEST_SELECTOR = "invalid_test_selector"
    TIMEOUT = "timeout"


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
"""Default agent tools.  ``memory.write`` and ``run_mutants`` belong to the
Integrity Guard and are granted by the orchestrator."""


class _Frozen(BaseModel):
    model_config = ConfigDict(
        extra="forbid", frozen=True, str_strip_whitespace=True, use_enum_values=False
    )


# --------------------------------------------------------------------------- #
# Repository, environment, policy
# --------------------------------------------------------------------------- #


class RepoRef(_Frozen):
    repo_id: Identifier
    repo_url: str | None = None
    package_name: str | None = None
    license: str | None = None


class CommitRef(_Frozen):
    sha: CommitSha
    committed_at: datetime | None = None
    stream_index: int | None = Field(
        default=None, ge=0, description="Chronological position in the stream."
    )


class EnvironmentSpec(_Frozen):
    """What is needed to rebuild the sandbox for this episode."""

    python_version: str
    docker_image: str | None = None
    dependency_lock_hash: str | None = None
    pytest_version: str | None = None
    network_enabled: Literal[False] = False
    test_timeout_seconds: int | None = Field(default=None, gt=0)
    stability_runs: int | None = Field(default=None, gt=0)


class EditPolicy(_Frozen):
    """Declares which paths may be edited.

    A specification, not enforcement: ``apply_patch`` and the Integrity Guard
    must apply it.  ``classify_path`` is a pure helper so both use one rule.
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
        """FORBIDDEN wins over SRC/TEST, so ``tests/conftest.py`` is never editable."""
        norm = path.replace("\\", "/")
        while norm.startswith("./"):
            norm = norm[2:]
        norm = norm.lstrip("/")
        for pattern in self.forbidden_globs:
            if fnmatch.fnmatch(norm, pattern) or fnmatch.fnmatch(
                norm, pattern.replace("**/", "*")
            ):
                return PathClass.FORBIDDEN
        if any(norm.startswith(r) for r in self.allowed_test_roots):
            return PathClass.TEST
        if any(norm.startswith(r) for r in self.allowed_source_roots):
            return PathClass.SRC
        return PathClass.OUTSIDE

    def is_editable(self, path: str, scope: PatchScope) -> bool:
        cls = self.classify_path(path)
        return cls is (PathClass.SRC if scope is PatchScope.SRC else PathClass.TEST)


class EpisodeBudget(_Frozen):
    """Identical across conditions by experimental design."""

    max_tool_calls: int | None = Field(default=None, gt=0)
    max_llm_calls: int | None = Field(default=None, gt=0)
    max_tokens: int | None = Field(default=None, gt=0)
    max_revisions: int = Field(default=1, ge=0)
    wall_clock_seconds: int | None = Field(default=None, gt=0)


# --------------------------------------------------------------------------- #
# CI event
# --------------------------------------------------------------------------- #


class FileChange(_Frozen):
    path: str = Field(min_length=1)
    change_kind: Literal["added", "modified", "deleted", "renamed"]
    unified_diff: str | None = None


class FailingTest(_Frozen):
    test_id: Identifier = Field(description="pytest node id.")
    test_file: str = Field(min_length=1)
    test_name: str = Field(min_length=1)
    target_function: str | None = None
    failure_message: str | None = None
    traceback: str | None = None
    assertion_diff: str | None = None
    stable: bool | None = Field(
        default=None, description="None = stability filter not run for this test."
    )


class FailureGroup(_Frozen):
    """Failures grouped by changed function (Triage output)."""

    group_id: Identifier
    function_qualname: str | None = None
    test_ids: list[Identifier] = Field(min_length=1)


class CIEvent(_Frozen):
    """The red CI run.  Contains only what a maintainer sees at that commit."""

    event_id: Identifier
    commit: CommitRef
    parent_commit: CommitRef | None = None
    commit_message: str | None = None
    commit_message_availability: Availability = Availability.AVAILABLE
    changed_files: list[FileChange] = Field(default_factory=list)
    failing_tests: list[FailingTest] = Field(min_length=1)
    failure_groups: list[FailureGroup] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check(self) -> CIEvent:
        ids = [t.test_id for t in self.failing_tests]
        if len(ids) != len(set(ids)):
            raise ValueError("failing_tests contains duplicate test_id values")
        known = set(ids)
        seen_groups: set[str] = set()
        for g in self.failure_groups:
            if g.group_id in seen_groups:
                raise ValueError(f"duplicate group_id {g.group_id!r}")
            seen_groups.add(g.group_id)
            unknown = sorted(set(g.test_ids) - known)
            if unknown:
                raise ValueError(
                    f"failure_group {g.group_id!r} references unknown test ids: {unknown}"
                )
        if self.commit_message_availability is Availability.AVAILABLE:
            if self.commit_message is None:
                raise ValueError(
                    "commit_message_availability is AVAILABLE but commit_message is "
                    "None; use ABSENT, WITHHELD or UNKNOWN for a missing message"
                )
        elif self.commit_message is not None:
            raise ValueError(
                "commit_message must be None unless commit_message_availability is "
                "AVAILABLE"
            )
        if self.parent_commit and same_commit(self.parent_commit.sha, self.commit.sha):
            raise ValueError("parent_commit must differ from commit")
        return self


# --------------------------------------------------------------------------- #
# Retrieval corpus and memory (agent-visible)
# --------------------------------------------------------------------------- #


class EvidenceItem(_Frozen):
    """One retrievable unit.  Scores and ranks are run-time and live in results."""

    evidence_id: Identifier
    source_type: EvidenceSourceType
    source_ref: str = Field(min_length=1, description="Path, sha, doc section or record id.")
    commit_sha: CommitSha | None = None
    commit_index: int | None = Field(default=None, ge=0)
    line_start: int | None = Field(default=None, ge=1)
    line_end: int | None = Field(default=None, ge=1)
    text: str | None = None

    @model_validator(mode="after")
    def _check(self) -> EvidenceItem:
        if (
            self.line_start is not None
            and self.line_end is not None
            and self.line_end < self.line_start
        ):
            raise ValueError("line_end must be >= line_start")
        return self


class RetrievalCorpusSpec(_Frozen):
    """``items=None`` = built at run time from the pinned repo; ``[]`` = genuinely empty."""

    corpus_id: Identifier
    corpus_version: str | None = None
    history_cutoff_commit: CommitSha
    history_cutoff_index: int | None = Field(default=None, ge=0)
    items: list[EvidenceItem] | None = None

    @model_validator(mode="after")
    def _check(self) -> RetrievalCorpusSpec:
        seen: set[str] = set()
        for item in self.items or []:
            if item.evidence_id in seen:
                raise ValueError(f"duplicate evidence_id {item.evidence_id!r}")
            seen.add(item.evidence_id)
            if (
                self.history_cutoff_index is not None
                and item.commit_index is not None
                and item.commit_index > self.history_cutoff_index
            ):
                raise ValueError(
                    f"evidence {item.evidence_id!r} has commit_index {item.commit_index} "
                    f"beyond history_cutoff_index {self.history_cutoff_index}: future "
                    "commits must never be indexed"
                )
        return self


class ExecutableProbe(_Frozen):
    """Check derived from an approved test patch; the record counts only while it passes."""

    probe_id: Identifier
    probe_code: str = Field(min_length=1)
    probe_selector: str | None = None


class MemoryRecord(_Frozen):
    """``{function, behaviour signature, direction, probe, evidence IDs, commit}``.

    A stored record exists only after simulated maintainer approval.  Rejected
    write attempts are logged in ``result_schemas.MemoryWriteAttempt``.
    """

    record_id: Identifier
    repo_id: Identifier
    function_qualname: str = Field(min_length=1)
    behaviour_signature: str = Field(min_length=1)
    direction: RepairDirection
    probe: ExecutableProbe
    evidence_ids: list[Identifier] = Field(default_factory=list)
    approved_at_commit: CommitSha
    scope: Literal["function", "module"] = "function"
    supersedes: Identifier | None = None
    flagged_stale: bool = False

    @model_validator(mode="after")
    def _check(self) -> MemoryRecord:
        if self.supersedes == self.record_id:
            raise ValueError("a memory record cannot supersede itself")
        return self


# --------------------------------------------------------------------------- #
# Evaluation-only: mutants, tool faults, ground truth
# --------------------------------------------------------------------------- #


class MutantSpec(_Frozen):
    """A mutant: the injected regression, or one entry of the cached catalogue."""

    mutant_id: Identifier
    function_qualname: str = Field(min_length=1)
    file_path: str = Field(min_length=1)
    operator: str | None = None
    unified_diff: str | None = None


class MutationFixture(_Frozen):
    """Precomputed mutants for one function; denominator of mutant-kill retention."""

    function_qualname: str = Field(min_length=1)
    mutants: list[MutantSpec] = Field(default_factory=list)
    reference_killed_mutant_ids: list[Identifier] | None = Field(
        default=None,
        description="Killed by the developer's reference tests; None = not computed.",
    )

    @model_validator(mode="after")
    def _check(self) -> MutationFixture:
        known = {m.mutant_id for m in self.mutants}
        if len(known) != len(self.mutants):
            raise ValueError("duplicate mutant_id in fixture")
        unknown = sorted(set(self.reference_killed_mutant_ids or []) - known)
        if unknown:
            raise ValueError(f"reference_killed_mutant_ids reference unknown mutants: {unknown}")
        return self


class ToolFaultInjection(_Frozen):
    tool: ToolName
    fault_kind: ToolFaultKind
    call_index: int = Field(ge=0, description="Zero-based index of the call to fail.")


class ToolFaultPlan(_Frozen):
    """Scenario T.  Orchestrator-only: the agent must not know a fault is coming."""

    enabled: bool = False
    seed: int | None = Field(default=None, description="None = unseeded, not 0.")
    failure_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    injections: list[ToolFaultInjection] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check(self) -> ToolFaultPlan:
        has_plan = bool(self.injections) or self.failure_rate is not None
        if not self.enabled and has_plan:
            raise ValueError("fault plan is disabled but defines injections or a rate")
        if self.enabled and not has_plan:
            raise ValueError("enabled fault plan must define injections or failure_rate")
        return self


class StrippedEvidence(_Frozen):
    """Evidence deliberately removed (S5), recorded instead of leaving a silent gap."""

    source_type: EvidenceSourceType
    source_ref: str | None = None
    original_evidence_id: Identifier | None = None
    reason: Literal["intent_stripped_scenario", "not_present_in_history"] = (
        "intent_stripped_scenario"
    )


class LabelProvenance(_Frozen):
    method: Literal[
        "tests_with_source_only_change", "tests_with_mutant_applied", "manual_review"
    ]
    stability_runs: int | None = Field(default=None, gt=0)
    verified_by: str | None = None


class ReferenceRepair(_Frozen):
    """The developer's test diff.  Never visible to the agent."""

    unified_diff: str = Field(min_length=1)
    changed_files: list[str] = Field(default_factory=list)
    source_commit: CommitSha | None = None


class PerTestGoldLabel(_Frozen):
    """Ground truth for one failing test.

    ``gold_direction`` is the side that is wrong; ``expected_action`` is what a
    correct agent should do.  They differ only when the action is ESCALATE (S5).
    """

    test_id: Identifier
    group_id: Identifier | None = None
    function_qualname: str | None = None
    gold_direction: RepairDirection
    expected_action: DecisionLabel
    gold_evidence_ids: list[Identifier] | None = Field(
        default=None, description="None = not annotated; [] = annotated, none exists."
    )
    related_memory_record_id: Identifier | None = None
    memory_applies: bool | None = Field(
        default=None, description="S4 requires False: the prior record must not justify an update."
    )
    injected_mutant_id: Identifier | None = None

    @model_validator(mode="after")
    def _check(self) -> PerTestGoldLabel:
        if (
            self.expected_action is not DecisionLabel.ESCALATE
            and self.expected_action.value != self.gold_direction.value
        ):
            raise ValueError(
                f"expected_action {self.expected_action.value} contradicts "
                f"gold_direction {self.gold_direction.value}; only ESCALATE may differ"
            )
        if self.injected_mutant_id and self.gold_direction is not RepairDirection.FIX_CODE:
            raise ValueError("a label linked to an injected mutant must be FIX_CODE")
        return self


def _unique_functions(labels: list[PerTestGoldLabel], direction: RepairDirection) -> set[str]:
    return {
        lbl.function_qualname
        for lbl in labels
        if lbl.gold_direction is direction and lbl.function_qualname
    }


class EpisodeGroundTruth(_Frozen):
    """Everything the harness knows and the agent must never see."""

    episode_id: Identifier
    scenario: ScenarioType
    per_test_labels: list[PerTestGoldLabel] = Field(min_length=1)
    expected_episode_outcome: DecisionLabel | None = Field(
        default=None, description="ESCALATE for S5; None otherwise."
    )
    reference_repairs: list[ReferenceRepair] = Field(default_factory=list)
    injected_mutants: list[MutantSpec] = Field(default_factory=list)
    mutation_fixtures: list[MutationFixture] = Field(default_factory=list)
    evidence_availability: Availability = Availability.UNKNOWN
    stripped_evidence: list[StrippedEvidence] = Field(default_factory=list)
    decoy_memory_record_ids: list[Identifier] = Field(default_factory=list)
    future_commit_shas: list[CommitSha] = Field(
        default_factory=list, description="Known later commits, for leakage checks."
    )
    label_provenance: LabelProvenance
    note: str | None = None

    def all_gold_evidence_ids(self) -> set[str]:
        return {e for lbl in self.per_test_labels for e in (lbl.gold_evidence_ids or [])}

    @model_validator(mode="after")
    def _check(self) -> EpisodeGroundTruth:
        labels = self.per_test_labels
        ids = [lbl.test_id for lbl in labels]
        if len(ids) != len(set(ids)):
            raise ValueError("per_test_labels contains duplicate test_id values")

        # mutant references: id exists, and function identity agrees when both known
        mutants = {m.mutant_id: m for m in self.injected_mutants}
        if len(mutants) != len(self.injected_mutants):
            raise ValueError("duplicate injected mutant_id")
        for lbl in labels:
            if lbl.injected_mutant_id is None:
                continue
            m = mutants.get(lbl.injected_mutant_id)
            if m is None:
                raise ValueError(
                    f"label {lbl.test_id!r} references unknown injected mutant "
                    f"{lbl.injected_mutant_id!r}"
                )
            if lbl.function_qualname and lbl.function_qualname != m.function_qualname:
                raise ValueError(
                    f"label {lbl.test_id!r} function {lbl.function_qualname!r} differs "
                    f"from its mutant's function {m.function_qualname!r}"
                )

        directions = {lbl.gold_direction for lbl in labels}
        sc = self.scenario
        S = ScenarioType

        if (self.expected_episode_outcome is DecisionLabel.ESCALATE) != (
            sc is S.S5_INTENT_STRIPPED
        ):
            raise ValueError(
                "expected_episode_outcome must be ESCALATE for S5 and None for S1-S4"
            )
        if self.decoy_memory_record_ids and sc is not S.S4_DECOY_REGRESSION:
            raise ValueError("decoy_memory_record_ids is only valid for S4")

        if sc is not S.S5_INTENT_STRIPPED:
            for lbl in labels:
                if lbl.expected_action.value != lbl.gold_direction.value:
                    raise ValueError(
                        f"{sc.value} label {lbl.test_id!r}: expected_action must equal "
                        "gold_direction (only S5 expects ESCALATE)"
                    )

        def need_mutants() -> None:
            if not self.injected_mutants:
                raise ValueError(f"{sc.value} requires at least one injected mutant")

        def need_mutant_links() -> None:
            for lbl in labels:
                if lbl.gold_direction is RepairDirection.FIX_CODE and not lbl.injected_mutant_id:
                    raise ValueError(
                        f"{sc.value} FIX_CODE label {lbl.test_id!r} must name its "
                        "injected_mutant_id"
                    )

        def need_gold_evidence(direction: RepairDirection) -> None:
            for lbl in labels:
                if lbl.gold_direction is direction and not lbl.gold_evidence_ids:
                    raise ValueError(
                        f"{sc.value} UPDATE_TEST label {lbl.test_id!r} needs non-empty "
                        "gold_evidence_ids"
                    )

        if sc is S.S1_OUTDATED_TEST:
            if directions != {RepairDirection.UPDATE_TEST}:
                raise ValueError("S1 requires every gold_direction to be UPDATE_TEST")
            if self.injected_mutants:
                raise ValueError("S1 must not inject a mutant")
            if not self.reference_repairs:
                raise ValueError("S1 requires the developer test diff as reference_repairs")
            need_gold_evidence(RepairDirection.UPDATE_TEST)

        elif sc is S.S2_INJECTED_REGRESSION:
            if directions != {RepairDirection.FIX_CODE}:
                raise ValueError("S2 requires every gold_direction to be FIX_CODE")
            need_mutants()
            need_mutant_links()
            if self.reference_repairs:
                raise ValueError("S2 tests must stay untouched: no reference test repair")

        elif sc is S.S3_MIXED:
            if directions != {RepairDirection.UPDATE_TEST, RepairDirection.FIX_CODE}:
                raise ValueError("S3 requires both UPDATE_TEST and FIX_CODE labels")
            if not self.reference_repairs:
                raise ValueError("S3 requires the developer test diff as reference_repairs")
            need_mutants()
            need_mutant_links()
            need_gold_evidence(RepairDirection.UPDATE_TEST)
            # Decision: missing function identity is REJECTED, not left unverified.
            # S3 is defined by two different functions and the builder always knows them.
            missing = [lbl.test_id for lbl in labels if not lbl.function_qualname]
            if missing:
                raise ValueError(
                    f"S3 requires function_qualname on every label to verify that the "
                    f"outdated and regressed functions differ; missing for {missing}"
                )
            overlap = _unique_functions(labels, RepairDirection.UPDATE_TEST) & _unique_functions(
                labels, RepairDirection.FIX_CODE
            )
            if overlap:
                raise ValueError(
                    "S3 requires the regression to be in a different function from the "
                    f"outdated test; overlap: {sorted(overlap)}"
                )

        elif sc is S.S4_DECOY_REGRESSION:
            if directions != {RepairDirection.FIX_CODE}:
                raise ValueError("S4 requires every gold_direction to be FIX_CODE")
            need_mutants()
            need_mutant_links()
            if not self.decoy_memory_record_ids:
                raise ValueError("S4 requires decoy_memory_record_ids")
            for lbl in labels:
                if not lbl.function_qualname:
                    raise ValueError(
                        f"S4 label {lbl.test_id!r} requires a non-empty function_qualname"
                    )
                if lbl.memory_applies is not False:
                    raise ValueError(
                        f"S4 label {lbl.test_id!r} must set memory_applies=False "
                        "(None does not mark the decoy as inapplicable)"
                    )
                if lbl.related_memory_record_id not in self.decoy_memory_record_ids:
                    raise ValueError(
                        f"S4 label {lbl.test_id!r} must reference one of "
                        "decoy_memory_record_ids as related_memory_record_id"
                    )

        elif sc is S.S5_INTENT_STRIPPED:
            for lbl in labels:
                if not lbl.function_qualname:
                    raise ValueError(
                        f"S5 label {lbl.test_id!r} requires a non-empty function_qualname "
                        "to verify memory-coverage rules"
                    )
                if lbl.gold_direction is not RepairDirection.UPDATE_TEST:
                    raise ValueError(
                        f"S5 label {lbl.test_id!r}: gold_direction must be UPDATE_TEST"
                    )
                if lbl.expected_action is not DecisionLabel.ESCALATE:
                    raise ValueError(
                        f"S5 label {lbl.test_id!r}: expected_action must be ESCALATE"
                    )
                if lbl.gold_evidence_ids:
                    raise ValueError(
                        f"S5 label {lbl.test_id!r}: gold_evidence_ids must be None or []"
                    )
            if self.evidence_availability not in (Availability.WITHHELD, Availability.ABSENT):
                raise ValueError("S5 requires evidence_availability WITHHELD or ABSENT")
            if not self.stripped_evidence:
                raise ValueError("S5 must record what was removed in stripped_evidence")
        return self


# --------------------------------------------------------------------------- #
# Agent-facing input and complete record
# --------------------------------------------------------------------------- #


class EpisodeInput(_Frozen):
    """Everything the agent may see.  Contains no evaluation-only field by construction.

    Use opaque ``episode_id`` values: the id is visible to the agent and must
    not encode the scenario.
    """

    episode_id: Identifier
    repo: RepoRef
    environment: EnvironmentSpec
    ci_event: CIEvent
    corpus: RetrievalCorpusSpec
    visible_memory: list[MemoryRecord] | None = Field(
        default=None,
        description="None = memory disabled (A2); [] = enabled, nothing recorded yet.",
    )
    edit_policy: EditPolicy = Field(default_factory=EditPolicy)
    budget: EpisodeBudget = Field(default_factory=EpisodeBudget)
    allowed_tools: list[ToolName] = Field(default_factory=lambda: list(DEFAULT_ALLOWED_TOOLS))

    @model_validator(mode="after")
    def _check(self) -> EpisodeInput:
        if not same_commit(self.corpus.history_cutoff_commit, self.ci_event.commit.sha):
            raise ValueError(
                "corpus.history_cutoff_commit must be the episode's current commit"
            )
        ids = [r.record_id for r in self.visible_memory or []]
        if len(ids) != len(set(ids)):
            raise ValueError("visible_memory contains duplicate record_id values")
        return self


class Episode(_Frozen):
    """One JSONL record of ``datasets/pilot_episodes.jsonl``."""

    schema_version: str = SCHEMA_VERSION
    episode_id: Identifier
    scenario: ScenarioType
    difficulty: Difficulty = Difficulty.EASY
    stream_id: Identifier
    stream_position: int = Field(ge=0)
    agent_input: EpisodeInput
    ground_truth: EpisodeGroundTruth
    fault_plan: ToolFaultPlan = Field(default_factory=ToolFaultPlan)
    note: str | None = None

    @model_validator(mode="after")
    def _check(self) -> Episode:
        ai, gt = self.agent_input, self.ground_truth
        if ai.episode_id != self.episode_id or gt.episode_id != self.episode_id:
            raise ValueError("episode_id mismatch between record, agent_input and ground_truth")
        if gt.scenario is not self.scenario:
            raise ValueError("ground_truth.scenario does not match scenario")

        event = ai.ci_event
        failing = {t.test_id for t in event.failing_tests}
        labelled = {lbl.test_id for lbl in gt.per_test_labels}
        if labelled - failing:
            raise ValueError(
                f"gold labels reference tests not failing in the CI event: "
                f"{sorted(labelled - failing)}"
            )
        if failing - labelled:
            raise ValueError(f"failing tests without a gold label: {sorted(failing - labelled)}")

        # label.group_id must agree with the failure groups when groups are given
        groups = {g.group_id: g for g in event.failure_groups}
        if groups:
            for lbl in gt.per_test_labels:
                if lbl.group_id is None:
                    continue
                g = groups.get(lbl.group_id)
                if g is None or lbl.test_id not in g.test_ids:
                    raise ValueError(
                        f"label {lbl.test_id!r} names group {lbl.group_id!r} which does "
                        "not contain it"
                    )
            if self.scenario is ScenarioType.S3_MIXED:
                by_test = {lbl.test_id: lbl.gold_direction for lbl in gt.per_test_labels}
                for g in groups.values():
                    if len({by_test[t] for t in g.test_ids if t in by_test}) > 1:
                        raise ValueError(
                            f"S3 failure group {g.group_id!r} mixes UPDATE_TEST and "
                            "FIX_CODE tests; groups are formed per changed function"
                        )

        # future-commit leakage (abbreviation-safe)
        future = gt.future_commit_shas
        if future:
            visible = [event.commit.sha] + ([event.parent_commit.sha] if event.parent_commit else [])
            visible += [r.approved_at_commit for r in ai.visible_memory or []]
            visible += [i.commit_sha for i in ai.corpus.items or [] if i.commit_sha]
            if any(same_commit(v, f) for v in visible for f in future):
                raise ValueError("agent input references a commit listed as a future commit")

        # evidence-id references: gold ids must exist in a materialised corpus
        if ai.corpus.items is not None:
            known = {i.evidence_id for i in ai.corpus.items}
            known |= {r.record_id for r in ai.visible_memory or []}
            bad = sorted(gt.all_gold_evidence_ids() - known)
            if bad:
                raise ValueError(f"gold_evidence_ids not found in the corpus or memory: {bad}")

        visible_ids = {r.record_id for r in ai.visible_memory or []}
        records = {r.record_id: r for r in ai.visible_memory or []}

        if self.scenario is ScenarioType.S4_DECOY_REGRESSION:
            if ai.visible_memory is None:
                raise ValueError(
                    "S4 requires memory to be enabled: visible_memory must be a list "
                    "containing the decoy record(s), not None"
                )
            missing = sorted(set(gt.decoy_memory_record_ids) - visible_ids)
            if missing:
                raise ValueError(f"decoy memory records not in visible_memory: {missing}")
            for lbl in gt.per_test_labels:
                if not lbl.function_qualname:
                    raise ValueError(
                        f"S4 label {lbl.test_id!r} requires a non-empty function_qualname "
                        "to validate against decoy memory"
                    )
                rec = records[lbl.related_memory_record_id]  # presence checked above
                if lbl.function_qualname != rec.function_qualname:
                    raise ValueError(
                        f"S4 label {lbl.test_id!r} function {lbl.function_qualname!r} differs "
                        f"from decoy record function {rec.function_qualname!r}"
                    )

        if self.scenario is ScenarioType.S5_INTENT_STRIPPED:
            corpus_items = ai.corpus.items or []
            corpus_ids = {i.evidence_id for i in corpus_items}
            for s in gt.stripped_evidence:
                if s.original_evidence_id and s.original_evidence_id in corpus_ids:
                    raise ValueError(
                        f"S5 corpus still exposes stripped evidence {s.original_evidence_id!r}"
                    )
                # A replacement 'uninformative' commit message may legitimately share its
                # source_ref, so only non-commit-message sources are matched by reference.
                if s.source_ref and s.source_type is not EvidenceSourceType.COMMIT_MESSAGE:
                    if any(
                        i.source_type is s.source_type and i.source_ref == s.source_ref
                        for i in corpus_items
                    ):
                        raise ValueError(
                            f"S5 corpus still exposes stripped {s.source_type.value} "
                            f"{s.source_ref!r}"
                        )
            missing = [lbl.test_id for lbl in gt.per_test_labels if not lbl.function_qualname]
            if missing:
                raise ValueError(
                    f"S5 requires non-empty function_qualname on every label to verify "
                    f"memory coverage; missing for {missing}"
                )
            label_fns = {lbl.function_qualname for lbl in gt.per_test_labels}
            covered = sorted(
                r.record_id for r in ai.visible_memory or [] if r.function_qualname in label_fns
            )
            if covered:
                raise ValueError(f"S5 requires no memory coverage; covering records: {covered}")
        return self

    def agent_payload(self) -> dict[str, Any]:
        """JSON-ready agent view.

        The only sanctioned method for constructing prompt payloads sent to agents.
        Guarantees ground-truth isolation by serializing strictly the agent-facing
        ``agent_input`` (an ``EpisodeInput`` model) which structurally contains no
        evaluation-only fields (gold labels, reference repairs, hidden mutants,
        decoy memory identities) or tool-fault plans.
        """
        return self.agent_input.model_dump(mode="json")

    def gold_actions(self) -> dict[str, DecisionLabel]:
        """Expected action per failing test.  Evaluation layer only."""
        return {lbl.test_id: lbl.expected_action for lbl in self.ground_truth.per_test_labels}


# --------------------------------------------------------------------------- #
# Leakage helper and runtime-enforcement notes
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
        "injected_mutants",
        "injected_mutant_id",
        "mutation_fixtures",
        "reference_killed_mutant_ids",
        "decoy_memory_record_ids",
        "future_commit_shas",
        "stripped_evidence",
        "label_provenance",
        "fault_plan",
        "memory_applies",
    }
)


def find_leaked_ground_truth(payload: Any, _path: str = "") -> list[str]:
    """Dotted paths of forbidden keys found anywhere in ``payload`` (empty = clean).

    A defense-in-depth and diagnostic audit utility for tests and orchestrators,
    NOT a security boundary. True isolation is enforced structurally by using
    ``Episode.agent_payload()`` to construct agent inputs.
    """
    found: list[str] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            path = f"{_path}.{key}" if _path else str(key)
            if key in AGENT_FORBIDDEN_KEYS:
                found.append(path)
            found.extend(find_leaked_ground_truth(value, path))
    elif isinstance(payload, (list, tuple)):
        for i, value in enumerate(payload):
            found.extend(find_leaked_ground_truth(value, f"{_path}[{i}]"))
    return found


RUNTIME_ENFORCEMENT: Final[tuple[str, ...]] = (
    "apply_patch must reject writes outside EditPolicy (CI config, dependency files, conftest.py).",
    "FIX_CODE must not touch tests: apply_patch(scope=src), re-checked by the Integrity Guard.",
    "UPDATE_TEST needs cited positive evidence: decision policy plus Guard re-check.",
    "Only the Integrity Guard holds memory.write, and only after simulated maintainer approval.",
    "A memory record is evidence only if its probe passes on current code: the memory tool runs it.",
    "Weakening checks (assertions, skip/xfail, loosened comparisons, swallowed exceptions) are Guard AST/diff checks.",
    "The indexer must cut history at the current commit; the validator catches only declared leaks.",
    "Sandbox isolation, budgets and fault injection are orchestrator/tool concerns, not schema concerns.",
    "Episode and repo ids are visible to the agent: use opaque ids that do not encode the scenario.",
)
