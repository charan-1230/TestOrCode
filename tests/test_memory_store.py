"""``JsonlMemoryStore``: approval-gated writes, probe-gated reads, persistence, superseding and invalidation.

Probes really execute (bubblewrap sandbox) against a package-layout fixture repository.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from testorcode.errors import MemoryAuthorizationError, ToolError
from testorcode.memory import JsonlMemoryStore, record_from_repair
from testorcode.retrieval import memory_evidence
from testorcode.schemas.contracts import (
    EpisodeContext,
    FailureGroupDetail,
    GuardResult,
    Hypothesis,
    IntentAnalysis,
    MemoryApproval,
    PatchProposal,
    PytestExecution,
)
from testorcode.schemas.episode_schemas import (
    DecisionLabel,
    EditPolicy,
    ExecutableProbe,
    FailingTest,
    FailureGroup,
    HypothesisKind,
    MemoryRecord,
    PatchScope,
    RepairDirection,
    RepoRef,
)
from testorcode.schemas.result_schemas import (
    GuardDecision,
    IntegrityReport,
    ProbeStatus,
    TestRunStatus,
)
from testorcode.tools import sandbox
from testorcode.tools.probes import probe_from_tests

needs_bwrap = pytest.mark.skipif(not sandbox.bwrap_available(), reason="bubblewrap unavailable: probes not executed")

POLICY = EditPolicy(allowed_source_roots=("pkg/",), allowed_test_roots=("tests/",))
FUNCTION = "pkg.core.shout"
TESTS = "from pkg.core import shout\n\n\ndef test_shout():\n    assert shout('a') == 'A!'\n"


def git(root: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    files = {"pkg/__init__.py": "", "pkg/core.py": "def shout(s):\n    return s.upper() + '!'\n",
             "tests/test_core.py": TESTS}
    for rel, text in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)
    git(root, "init", "-q")
    git(root, "add", ".")
    git(root, "-c", "user.name=t", "-c", "user.email=t@x", "commit", "-qm", "c1")
    return root


def context(root: Path, repo_id: str = "r", episode: str = "ep-1") -> EpisodeContext:
    return EpisodeContext(episode_id=episode, repo=RepoRef(repo_id=repo_id), worktree_path=root,
                          commit_sha=git(root, "rev-parse", "HEAD"), edit_policy=POLICY)


def record(root: Path, record_id: str = "mem-1", code: str | None = None, **update) -> MemoryRecord:
    return MemoryRecord(
        record_id=record_id, repo_id="r", function_qualname=FUNCTION, behaviour_signature="shout adds '!'",
        direction=RepairDirection.UPDATE_TEST,
        probe=ExecutableProbe(probe_id=f"probe-{record_id}", probe_code=code or probe_from_tests(TESTS, ["test_shout"])),
        evidence_ids=["ev-1"], approved_at_commit=git(root, "rev-parse", "HEAD"), **update,
    )


def approval(rec: MemoryRecord, **update) -> MemoryApproval:
    return MemoryApproval(**{"record_id": rec.record_id, "guard_decision": GuardDecision.APPROVE,
                             "maintainer_approved": True, "approved_at_commit": rec.approved_at_commit, **update})


def store(tmp_path: Path, **kw) -> JsonlMemoryStore:
    return JsonlMemoryStore(tmp_path / "mem" / "memory.jsonl", isolation="bwrap", timeout_seconds=30, **kw)


# --------------------------------------------------------------------------- #
# Writes
# --------------------------------------------------------------------------- #


def test_mismatched_or_missing_approvals_are_refused_and_nothing_is_written(tmp_path: Path, repo: Path) -> None:
    s, rec = store(tmp_path), record(repo)
    with pytest.raises(MemoryAuthorizationError, match="not 'mem-1'"):
        s.write(rec, approval(rec, record_id="mem-other"))
    with pytest.raises(MemoryAuthorizationError, match="different approval commits"):
        s.write(rec, approval(rec, approved_at_commit="abcdef1"))
    with pytest.raises(MemoryAuthorizationError):  # a forged approval that bypassed the contract's validator
        s.write(rec, MemoryApproval.model_construct(record_id="mem-1", guard_decision=GuardDecision.REJECT,
                                                    maintainer_approved=True, approved_at_commit=rec.approved_at_commit))
    with pytest.raises(MemoryAuthorizationError):
        s.write(rec, {"record_id": "mem-1"})  # type: ignore[arg-type]
    with pytest.raises(ValueError):  # the contract itself refuses an approval without maintainer approval
        approval(rec, maintainer_approved=False)
    assert not s.path.exists() and s.records() == []


def test_a_record_with_an_unusable_probe_is_never_stored(tmp_path: Path, repo: Path) -> None:
    s, rec = store(tmp_path), record(repo, code="x = 1\n")
    with pytest.raises(ToolError, match="defines no test"):
        s.write(rec, approval(rec))
    assert s.records() == []


@needs_bwrap
def test_approved_record_persists_and_is_usable_while_its_probe_passes(tmp_path: Path, repo: Path) -> None:
    rec = record(repo)
    store(tmp_path).write(rec, approval(rec))
    reopened = store(tmp_path)  # a new process would see the same file
    result = reopened.read(context(repo), FUNCTION)
    [check] = result.checks
    assert check.probe_status is ProbeStatus.PASSED and check.blocked_reason is None
    assert [r.record_id for r in result.usable] == ["mem-1"]
    assert [e.evidence_id for e in memory_evidence(result)] == ["mem-1"]
    assert reopened.read(context(repo, repo_id="other"), FUNCTION).checks == ()  # scoped by repository
    assert reopened.read(context(repo), "pkg.core.other").checks == ()  # and by function


def test_rewriting_the_same_record_is_idempotent_but_a_different_one_is_refused(tmp_path: Path, repo: Path) -> None:
    s, rec = store(tmp_path), record(repo)
    s.write(rec, approval(rec))
    s.write(rec, approval(rec))
    assert len(s.records()) == 1
    changed = rec.model_copy(update={"behaviour_signature": "something else"})
    with pytest.raises(MemoryAuthorizationError, match="already stored"):
        s.write(changed, approval(changed))


@needs_bwrap
def test_a_newer_approved_record_supersedes_the_older_one(tmp_path: Path, repo: Path) -> None:
    s = store(tmp_path)
    old, new = record(repo, "mem-old"), record(repo, "mem-new")
    s.write(old, approval(old))
    s.write(new, approval(new))
    stored = {r.record_id: r for r in store(tmp_path).records()}
    assert stored["mem-new"].supersedes == "mem-old"
    checks = {c.record.record_id: c for c in s.read(context(repo), FUNCTION).checks}
    assert checks["mem-old"].blocked_reason == "superseded" and checks["mem-old"].probe_status is ProbeStatus.NOT_RUN
    assert checks["mem-new"].valid_as_evidence


# --------------------------------------------------------------------------- #
# Invalidation
# --------------------------------------------------------------------------- #


@needs_bwrap
def test_failing_probe_blocks_the_record_and_repeated_failure_flags_it_stale(tmp_path: Path, repo: Path) -> None:
    s, rec = store(tmp_path), record(repo)
    s.write(rec, approval(rec))
    (repo / "pkg/core.py").write_text("def shout(s):\n    return s.upper()\n")  # the approved behaviour is gone

    first = s.read(context(repo), FUNCTION).checks[0]
    assert first.probe_status is ProbeStatus.FAILED and first.blocked_reason == "probe_failed"
    assert not first.record.flagged_stale and not first.valid_as_evidence
    second = s.read(context(repo), FUNCTION).checks[0]
    assert second.probe_status is ProbeStatus.FAILED and second.record.flagged_stale  # "keeps failing"
    assert store(tmp_path).records()[0].flagged_stale  # persisted

    (repo / "pkg/core.py").write_text("def shout(s):\n    return s.upper() + '!'\n")
    third = s.read(context(repo), FUNCTION).checks[0]  # stale stays stale: not re-run, never evidence
    assert third.probe_status is ProbeStatus.NOT_RUN and third.blocked_reason == "probe_failed"
    assert not third.valid_as_evidence


@needs_bwrap
def test_a_passing_probe_resets_the_failure_count(tmp_path: Path, repo: Path) -> None:
    s, rec = store(tmp_path), record(repo)
    s.write(rec, approval(rec))
    core = repo / "pkg/core.py"
    good = core.read_text()
    core.write_text("def shout(s):\n    return s\n")
    assert s.read(context(repo), FUNCTION).checks[0].probe_status is ProbeStatus.FAILED
    core.write_text(good)
    assert s.read(context(repo), FUNCTION).checks[0].valid_as_evidence
    core.write_text("def shout(s):\n    return s\n")
    assert not s.read(context(repo), FUNCTION).checks[0].record.flagged_stale  # one failure again, not two


@needs_bwrap
def test_probe_errors_block_the_record_without_counting_as_failures(tmp_path: Path, repo: Path) -> None:
    s, rec = store(tmp_path), record(repo, code="def test_x():\n    assert missing_name() == 1\n")
    s.write(rec, approval(rec))
    for _ in range(3):
        check = s.read(context(repo), FUNCTION).checks[0]
        assert check.probe_status is ProbeStatus.ERROR and check.blocked_reason == "probe_not_run"
    assert not store(tmp_path).records()[0].flagged_stale


@needs_bwrap
def test_a_record_from_the_future_is_out_of_scope(tmp_path: Path, repo: Path) -> None:
    first = git(repo, "rev-parse", "HEAD")
    (repo / "README").write_text("x\n")
    git(repo, "add", ".")
    git(repo, "-c", "user.name=t", "-c", "user.email=t@x", "commit", "-qm", "c2")
    s, rec = store(tmp_path), record(repo)  # approved at the newer commit
    s.write(rec, approval(rec))
    past = context(repo).model_copy(update={"commit_sha": first})  # an episode at the older commit
    check = s.read(past, FUNCTION).checks[0]
    assert check.blocked_reason == "out_of_scope" and check.probe_status is ProbeStatus.NOT_RUN
    unknown = record(repo, "mem-x").model_copy(update={"approved_at_commit": "abcdef1234567"})
    s.write(unknown, approval(unknown))
    checks = {c.record.record_id: c for c in s.read(context(repo), FUNCTION).checks}
    assert checks["mem-x"].blocked_reason == "out_of_scope"


@needs_bwrap
def test_existence_in_the_file_is_not_trust(tmp_path: Path, repo: Path) -> None:
    """A record injected into the JSONL file without approval is still only as good as its probe."""
    s = store(tmp_path)
    s.path.parent.mkdir(parents=True)
    bad = record(repo, "mem-injected", code="from pkg.core import shout\n\n\ndef test_x():\n    assert shout('a') == 'a'\n")
    s.path.write_text("not json at all\n" + json.dumps({"record": bad.model_dump(mode="json")}) + "\n"
                      + json.dumps({"record": {"record_id": "half"}}) + "\n")
    [check] = s.read(context(repo), FUNCTION).checks  # the malformed lines are ignored, the injected record probed
    assert check.probe_status is ProbeStatus.FAILED and not check.valid_as_evidence
    ok = record(repo, "mem-ok")
    s.write(ok, approval(ok))
    lines = s.path.read_text().splitlines()
    assert lines[0] == "not json at all" and len(lines) == 4  # malformed lines kept verbatim, never used


@needs_bwrap
def test_seeded_dataset_probe_gets_support_code_from_the_pristine_tests(tmp_path: Path, repo: Path) -> None:
    bare = record(repo, "mem-seed", code="def test_shout():\n    assert shout('a') == 'A!'\n")
    bare = bare.model_copy(update={"probe": bare.probe.model_copy(update={"probe_selector": "tests/test_core.py::test_shout"})})
    s = store(tmp_path)
    assert s.seed([bare], worktree_path=repo) == ["mem-seed"]
    assert s.seed([bare], worktree_path=repo) == []  # idempotent
    (repo / "tests/test_core.py").write_text("")  # an agent later empties the test file: the stored probe is unaffected
    assert "from pkg.core import shout" in s.records()[0].probe.probe_code
    assert s.read(context(repo), FUNCTION).checks[0].valid_as_evidence


# --------------------------------------------------------------------------- #
# Records from approved repairs
# --------------------------------------------------------------------------- #


def _guard(decision: GuardDecision) -> GuardResult:
    run = PytestExecution(selectors=("tests",), command=(), status=TestRunStatus.PASSED, passed=1, failed=0)
    return GuardResult(decision=decision, reasons=() if decision is GuardDecision.APPROVE else ("x",),
                       post_repair_run=run, integrity=IntegrityReport(guard_ran=True, guard_decision=decision))


def _group(function: str | None = FUNCTION) -> FailureGroupDetail:
    test = FailingTest(test_id="tests/test_core.py::test_shout", test_file="tests/test_core.py", test_name="test_shout")
    return FailureGroupDetail(group=FailureGroup(group_id="g1", function_qualname=function, test_ids=[test.test_id]),
                              failing_tests=(test,))


ANALYSIS = IntentAnalysis(group_id="g1", hypothesis=Hypothesis(kind=HypothesisKind.OUTDATED_TEST, explanation="e"),
                          expected_behaviour="shout() appends '!'", cited_evidence_ids=("ev-1",), confidence=0.9,
                          recommended_decision=DecisionLabel.UPDATE_TEST)
PROPOSAL = PatchProposal(group_id="g1", decision=RepairDirection.UPDATE_TEST, scope=PatchScope.TEST,
                         target_files=("tests/test_core.py",), unified_diff="--- a/x\n+++ b/x\n", rationale="r",
                         evidence_ids=("ev-1", "ev-2"))


@needs_bwrap
def test_record_from_an_approved_repair_carries_a_runnable_probe(tmp_path: Path, repo: Path) -> None:
    ctx = context(repo)
    rec = record_from_repair(ctx, _group(), ANALYSIS, PROPOSAL, _guard(GuardDecision.APPROVE))
    assert rec.record_id == "ep-1-g1" and rec.function_qualname == FUNCTION and rec.repo_id == "r"
    assert rec.direction is RepairDirection.UPDATE_TEST and rec.approved_at_commit == ctx.commit_sha
    assert rec.evidence_ids == ["ev-1", "ev-2"] and "appends '!'" in rec.behaviour_signature
    assert rec.probe.probe_selector == "tests/test_core.py::test_shout" and "from pkg.core import shout" in rec.probe.probe_code
    s = store(tmp_path)
    s.write(rec, approval(rec))
    assert s.read(ctx, FUNCTION).checks[0].valid_as_evidence


def test_records_come_only_from_approved_and_scoped_repairs(repo: Path) -> None:
    with pytest.raises(MemoryAuthorizationError):
        record_from_repair(context(repo), _group(), ANALYSIS, PROPOSAL, _guard(GuardDecision.ESCALATE))
    with pytest.raises(ToolError, match="no mapped function"):
        record_from_repair(context(repo), _group(None), ANALYSIS, PROPOSAL, _guard(GuardDecision.APPROVE))


# --------------------------------------------------------------------------- #
# Superseding rules and store isolation
# --------------------------------------------------------------------------- #


def _entries(s: JsonlMemoryStore) -> dict:
    with s._locked():
        return {e.record.record_id: e for e in s._load()[0]}


def test_a_fix_code_record_never_supersedes_an_approved_behaviour_change(tmp_path: Path, repo: Path) -> None:
    s = store(tmp_path)
    change = record(repo, "mem-change")  # UPDATE_TEST: an approved behaviour change
    fix = record(repo, "mem-fix").model_copy(update={"direction": RepairDirection.FIX_CODE})
    s.write(change, approval(change))
    s.write(fix, approval(fix))
    entries = _entries(s)
    assert entries["mem-fix"].record.supersedes is None
    assert all(e.superseded_by is None for e in entries.values())  # both stay active


def test_a_newer_behaviour_change_supersedes_older_fix_and_change_records(tmp_path: Path, repo: Path) -> None:
    s = store(tmp_path)
    first, fix = record(repo, "mem-1"), record(repo, "mem-fix").model_copy(update={"direction": RepairDirection.FIX_CODE})
    second = record(repo, "mem-2")
    for r in (first, fix, second):
        s.write(r, approval(r))
    entries = _entries(s)
    assert entries["mem-1"].superseded_by == entries["mem-fix"].superseded_by == "mem-2"
    assert entries["mem-2"].superseded_by is None and entries["mem-2"].record.supersedes == "mem-fix"


def test_seeded_memory_never_supersedes_approved_records(tmp_path: Path, repo: Path) -> None:
    s = store(tmp_path)
    approved = record(repo, "mem-approved")
    s.write(approved, approval(approved))
    assert s.seed([record(repo, "mem-seeded")], worktree_path=repo) == ["mem-seeded"]
    entries = _entries(s)
    assert entries["mem-approved"].superseded_by is None and entries["mem-seeded"].record.supersedes is None
    assert entries["mem-seeded"].origin == "seeded"


def test_a_store_is_bound_to_one_condition_seed_and_stream(tmp_path: Path, repo: Path) -> None:
    from testorcode.errors import ConfigError
    from testorcode.memory import memory_namespace, memory_path

    path = memory_path(tmp_path / "mem", "FULL", 42, "dev")
    assert path == tmp_path / "mem" / "FULL" / "seed-42" / "dev.jsonl"
    full = JsonlMemoryStore(path, namespace=memory_namespace("FULL", 42, "dev"))
    rec = record(repo)
    full.write(rec, approval(rec))
    JsonlMemoryStore(path, namespace=memory_namespace("FULL", 42, "dev"))  # the same run may reopen it
    for other in (memory_namespace("FULL", 43, "dev"), memory_namespace("A3", 42, "dev"),
                  memory_namespace("FULL", 42, "test")):
        with pytest.raises(ConfigError, match="needs its own store"):
            JsonlMemoryStore(path, namespace=other)
    legacy = tmp_path / "legacy.jsonl"
    JsonlMemoryStore(legacy).write(rec, approval(rec))
    with pytest.raises(ConfigError, match="without a namespace"):
        JsonlMemoryStore(legacy, namespace=memory_namespace("FULL", 42, "dev"))


def test_a_replayed_stream_in_the_same_store_is_refused_explicitly(tmp_path: Path, repo: Path) -> None:
    s, rec = store(tmp_path), record(repo)
    s.write(rec, approval(rec))
    replay = rec.model_copy(update={"behaviour_signature": "a different approved repair in a replay"})
    with pytest.raises(MemoryAuthorizationError, match="must use its own store"):
        s.write(replay, approval(replay))
