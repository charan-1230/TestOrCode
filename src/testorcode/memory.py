"""Scoped change memory.

Read: ``MemoryStore.read`` returns every record for a function with the status of
its executable probe on the *current* code.  Only ``MemoryReadResult.usable``
records may be cited as evidence; stale or failing records are returned but marked.

Write: ``MemoryStore.write`` requires a ``MemoryApproval``, which only the
Integrity Guard's workflow builds after a Guard APPROVE and maintainer approval.
Without one it raises ``MemoryAuthorizationError``.  This is a convention
enforced by the store, not a security boundary: Python cannot stop other code
from constructing an approval, so the orchestrator must not hand the write
path to any agent except the Integrity Guard.

``JsonlMemoryStore`` persists records as JSON lines.  A stored record is never trusted because it exists: every read
re-runs its probe (``tools.probes.run_executable_probe``, sandboxed, against the worktree's current source) and only a
passing probe makes it usable.  Records are scoped to ``(repo_id, function_qualname)``; their behaviour signature and
direction travel with them so the Intent Analyst can judge relevance (a passing probe proves the *approved* behaviour
still holds, not that a new failure is the same change: S4 decoys keep passing probes by design).
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from pydantic import ValidationError

from testorcode.errors import ConfigError, MemoryAuthorizationError, ToolError
from testorcode.logging_config import get_logger
from testorcode.schemas.contracts import (
    EpisodeContext,
    FailureGroupDetail,
    GuardResult,
    IntentAnalysis,
    MemoryApproval,
    MemoryReadResult,
    MemoryRecordCheck,
    PatchProposal,
)
from testorcode.schemas.episode_schemas import (
    ExecutableProbe,
    MemoryRecord,
    PathClass,
    RepairDirection,
    same_commit,
)
from testorcode.schemas.result_schemas import GuardDecision, ProbeStatus
from testorcode.tools.git_tools import run_git
from testorcode.tools.probes import (
    probe_from_tests,
    probe_problem,
    qualname_of_test,
    run_executable_probe,
    support_code,
)
from testorcode.tools.sandbox import Isolation

try:  # advisory lock for concurrent writers (POSIX); without it writes are still atomic
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX
    fcntl = None  # type: ignore[assignment]

logger = get_logger(__name__)

STALE_AFTER_FAILURES = 2  # consecutive probe failures before a record is flagged stale ("keeps failing")


class MemoryStore(Protocol):
    def read(self, context: EpisodeContext, function_qualname: str) -> MemoryReadResult:
        """Records for ``function_qualname`` with probes run against the current code."""
        ...

    def write(self, record: MemoryRecord, approval: MemoryApproval) -> None:
        """Persist ``record``.

        Raises:
            MemoryAuthorizationError: ``approval`` is for another record or invalid.
        """
        ...


@dataclass
class _Entry:
    record: MemoryRecord
    probe_failures: int = 0
    superseded_by: str | None = None
    origin: Literal["approved", "seeded"] = "approved"

    def to_json(self) -> str:
        return json.dumps({
            "record": self.record.model_dump(mode="json"), "probe_failures": self.probe_failures,
            "superseded_by": self.superseded_by, "origin": self.origin,
        }, sort_keys=True)


class JsonlMemoryStore:
    """JSONL-backed scoped memory with probe-gated reads and approval-gated writes.

    ``python_executable``/``timeout_seconds``/``isolation``/``docker_image`` configure probe runs (the target
    repository's interpreter, as for its tests).  ``check_history`` refuses records approved at a commit that is not an
    ancestor of (or equal to) the episode's commit, so memory never carries information from the future.

    **Isolation contract.**  One store holds one chronological stream of one experimental run: memory carries over
    between the episodes of that stream and nowhere else.  ``namespace`` (e.g. ``memory_namespace(condition, seed,
    stream)``) is recorded next to the file on first use; opening the file with a different namespace raises
    ``ConfigError``, so conditions or seeds can never share records by accident.  ``memory_path`` gives each
    namespace its own file.  A replayed stream needs a fresh store (record ids are ``<episode>-<group>``).
    """

    def __init__(
        self,
        path: str | Path,
        *,
        python_executable: str | Path | None = None,
        timeout_seconds: int = 120,
        isolation: Isolation | str = Isolation.AUTO,
        docker_image: str | None = None,
        stale_after_failures: int = STALE_AFTER_FAILURES,
        check_history: bool = True,
        namespace: str | None = None,
    ) -> None:
        self.path = Path(path)
        self.namespace = namespace
        if namespace is not None:
            self._claim_namespace(namespace)
        self.python_executable = python_executable
        self.timeout_seconds = timeout_seconds
        self.isolation = isolation
        self.docker_image = docker_image
        self.stale_after_failures = max(1, stale_after_failures)
        self.check_history = check_history

    # -- read ---------------------------------------------------------------- #

    def read(self, context: EpisodeContext, function_qualname: str) -> MemoryReadResult:
        """Every record for ``(context.repo, function_qualname)``, each with its probe run on the current code.

        Superseded, stale-flagged and out-of-history records are returned blocked without running their probe.  A
        probe failure counts towards stale flagging; an execution error (``ERROR``) does not.
        """
        checks: list[MemoryRecordCheck] = []
        with self._locked():  # snapshot only: probes run without holding the lock
            entries, _ = self._load()
        outcomes: dict[str, ProbeStatus] = {}
        for entry in entries:
            record = entry.record
            if record.repo_id != context.repo.repo_id or record.function_qualname != function_qualname:
                continue
            if entry.superseded_by is not None:
                checks.append(MemoryRecordCheck(record=record, blocked_reason="superseded"))
                continue
            if record.flagged_stale:
                checks.append(MemoryRecordCheck(record=record, blocked_reason="probe_failed"))
                continue
            if self.check_history and not _at_or_before(context, record.approved_at_commit):
                checks.append(MemoryRecordCheck(record=record, blocked_reason="out_of_scope"))
                continue
            status = run_executable_probe(
                context.worktree_path, record.probe, policy=context.edit_policy,
                timeout_seconds=self.timeout_seconds, python_executable=self.python_executable,
                isolation=self.isolation, docker_image=self.docker_image,
            ).status
            outcomes[record.record_id] = status
            checks.append(MemoryRecordCheck(
                record=record, probe_status=status,
                blocked_reason=None if status is ProbeStatus.PASSED
                else "probe_failed" if status is ProbeStatus.FAILED else "probe_not_run",
            ))
        if outcomes:
            stale = self._record_probe_outcomes(outcomes)
            checks = [c.model_copy(update={"record": c.record.model_copy(update={"flagged_stale": True})})
                      if c.record.record_id in stale else c for c in checks]
        logger.info("memory.read %s: %d records, %d usable", function_qualname, len(checks),
                    sum(c.valid_as_evidence for c in checks))
        return MemoryReadResult(checks=tuple(checks))

    def _record_probe_outcomes(self, outcomes: dict[str, ProbeStatus]) -> set[str]:
        """Merge probe results into the current file (it may have changed meanwhile); return the ids now stale.

        PASSED resets a record's consecutive-failure count, FAILED increments it (``stale_after_failures`` consecutive
        failures flag the record stale), ERROR leaves it unchanged (an execution problem is not a behaviour failure).
        """
        stale: set[str] = set()
        with self._locked():
            entries, invalid = self._load()
            dirty = False
            for entry in entries:
                status = outcomes.get(entry.record.record_id)
                if status is ProbeStatus.PASSED and entry.probe_failures:
                    entry.probe_failures, dirty = 0, True
                elif status is ProbeStatus.FAILED:
                    entry.probe_failures, dirty = entry.probe_failures + 1, True
                    if entry.probe_failures >= self.stale_after_failures and not entry.record.flagged_stale:
                        entry.record = entry.record.model_copy(update={"flagged_stale": True})
                        logger.info("memory: record %s flagged stale after %d probe failures",
                                    entry.record.record_id, entry.probe_failures)
                if entry.record.flagged_stale and status is not None:
                    stale.add(entry.record.record_id)
            if dirty:
                self._save(entries, invalid)
        return stale

    def records(self) -> list[MemoryRecord]:
        """Every stored record (for inspection and tests; never evidence by itself)."""
        with self._locked():
            return [e.record for e in self._load()[0]]

    # -- write --------------------------------------------------------------- #

    def write(self, record: MemoryRecord, approval: MemoryApproval) -> None:
        """Persist an approved record.  An approved UPDATE_TEST record (a behaviour change) supersedes the function's
        older records; a FIX_CODE record supersedes nothing.

        Raises:
            MemoryAuthorizationError: no valid approval, or the approval is for another record or commit, or a
                different record with this id is already stored.
            ToolError: the record's probe is not runnable code (an unusable probe is never stored).
        """
        if not isinstance(record, MemoryRecord):
            raise MemoryAuthorizationError("memory write needs a MemoryRecord")
        if not isinstance(approval, MemoryApproval):
            raise MemoryAuthorizationError("memory write needs a MemoryApproval")
        if approval.guard_decision is not GuardDecision.APPROVE or approval.maintainer_approved is not True:
            raise MemoryAuthorizationError("memory write needs a Guard APPROVE and maintainer approval")
        if approval.record_id != record.record_id:
            raise MemoryAuthorizationError(
                f"approval is for record {approval.record_id!r}, not {record.record_id!r}"
            )
        if not same_commit(approval.approved_at_commit, record.approved_at_commit):
            raise MemoryAuthorizationError("approval and record name different approval commits")
        problem = probe_problem(record.probe.probe_code)
        if problem:
            raise ToolError(f"memory record {record.record_id} rejected: {problem}")
        self._add(record, origin="approved")

    def seed(self, records: Iterable[MemoryRecord], *, worktree_path: Path) -> list[str]:
        """Harness only: load prior approved memory an episode declares (``EpisodeInput.visible_memory``).

        Call it **before any agent acts**: a dataset probe that is a bare test function is completed with the support
        code (imports, helpers) of the test module its ``probe_selector`` names, read from the pristine CI state.
        From then on the stored probe never depends on the worktree's tests.  Already stored ids are skipped.
        Returns the ids added.
        """
        added = []
        stored = {r.record_id for r in self.records()}
        for record in records:
            if record.record_id in stored:
                continue
            probe = record.probe
            selector_file = (probe.probe_selector or "").split("::", 1)[0]
            module = Path(worktree_path) / selector_file if selector_file else None
            if module is not None and module.is_file() and not module.is_symlink():
                code = support_code(module.read_text(encoding="utf-8", errors="replace")) + "\n" + probe.probe_code
                probe = probe.model_copy(update={"probe_code": code})
            self._add(record.model_copy(update={"probe": probe}), origin="seeded")
            added.append(record.record_id)
        return added

    def _add(self, record: MemoryRecord, *, origin: Literal["approved", "seeded"]) -> None:
        with self._locked():
            entries, invalid = self._load()
            for entry in entries:
                if entry.record.record_id == record.record_id:
                    if entry.record == record:
                        return  # idempotent re-write of the same approved record
                    raise MemoryAuthorizationError(
                        f"a different record {record.record_id!r} is already stored: a replayed stream, seed or "
                        "condition must use its own store (memory_path)"
                    )
            # Only a newer *approved behaviour change* replaces earlier records of the function (proposal section 7).
            # A FIX_CODE approval restores behaviour and replaces nothing; seeded (declared prior) memory never
            # replaces records this store approved.
            replaces = origin == "approved" and record.direction is RepairDirection.UPDATE_TEST
            active = [e for e in entries if replaces and e.superseded_by is None and e.record.repo_id == record.repo_id
                      and e.record.function_qualname == record.function_qualname]
            if active and record.supersedes is None:
                record = record.model_copy(update={"supersedes": active[-1].record.record_id})
            for entry in active:
                entry.superseded_by = record.record_id
            entries.append(_Entry(record=record, origin=origin))
            self._save(entries, invalid)
        logger.info("memory: stored %s record %s (%d superseded)", origin, record.record_id, len(active))

    # -- persistence --------------------------------------------------------- #

    def _claim_namespace(self, namespace: str) -> None:
        marker = self.path.with_name(self.path.name + ".namespace")
        with self._locked():
            if marker.is_file():
                existing = marker.read_text(encoding="utf-8").strip()
                if existing != namespace:
                    raise ConfigError(
                        f"memory store {self.path} belongs to {existing!r}, not {namespace!r}: "
                        "each condition, seed and stream needs its own store"
                    )
                return
            if self.path.is_file() and self.path.stat().st_size:
                raise ConfigError(f"memory store {self.path} already holds records without a namespace")
            marker.write_text(namespace + "\n", encoding="utf-8")

    def _load(self) -> tuple[list[_Entry], list[str]]:
        """Valid entries plus the raw lines that are not (kept verbatim, never used)."""
        entries: list[_Entry] = []
        invalid: list[str] = []
        if not self.path.is_file():
            return entries, invalid
        for number, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                entry = _Entry(
                    record=MemoryRecord.model_validate(data["record"]),
                    probe_failures=int(data.get("probe_failures", 0)),
                    superseded_by=data.get("superseded_by"),
                    origin=data.get("origin", "approved"),
                )
            except (ValueError, KeyError, TypeError, ValidationError) as exc:
                logger.warning("memory: ignoring malformed line %d of %s (%s)", number, self.path, type(exc).__name__)
                invalid.append(line)
                continue
            entries.append(entry)
        return entries, invalid

    def _save(self, entries: list[_Entry], invalid: list[str]) -> None:
        """Atomic replace: write a temporary file in the same directory, fsync, rename."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lines = [*invalid, *(e.to_json() for e in entries)]
        fd, tmp = tempfile.mkstemp(prefix=f".{self.path.name}.", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write("".join(f"{line}\n" for line in lines))
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, self.path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise

    @contextlib.contextmanager
    def _locked(self) -> Iterator[None]:
        if fcntl is None:
            yield
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path.with_name(self.path.name + ".lock"), "a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)


def memory_namespace(condition: str, seed: int, stream: str) -> str:
    """The isolation key of one store: one experimental condition, one seed, one chronological stream."""
    return f"{condition}/seed={seed}/{stream}"


def memory_path(base_dir: str | Path, condition: str, seed: int, stream: str) -> Path:
    """``<base>/<condition>/seed-<seed>/<stream>.jsonl``: a separate store per condition, seed and stream."""
    return Path(base_dir) / str(condition) / f"seed-{seed}" / f"{stream}.jsonl"


def _at_or_before(context: EpisodeContext, commit: str) -> bool:
    """``commit`` is an ancestor of (or equal to) the episode's commit in its worktree's history."""
    if same_commit(commit, context.commit_sha):
        return True
    try:
        result = run_git(context.worktree_path, ["merge-base", "--is-ancestor", commit, context.commit_sha])
    except ToolError:
        return False
    return result.returncode == 0


# --------------------------------------------------------------------------- #
# Records from approved repairs (the Integrity Guard's ``build_memory_record``)
# --------------------------------------------------------------------------- #


def record_from_repair(
    context: EpisodeContext,
    group: FailureGroupDetail,
    analysis: IntentAnalysis,
    proposal: PatchProposal,
    result: GuardResult,
) -> MemoryRecord:
    """The record for an approved repair: function, behaviour signature, direction, executable probe, evidence, commit.

    The probe is derived from the group's tests as they stand **after** the approved repair (the worktree holds it):
    for UPDATE_TEST the updated tests (the proposal's "assertions in the approved test patch"), for FIX_CODE the
    untouched tests that verified the fix.  It is made self-contained (``probes.probe_from_tests``), so later reads do
    not depend on the worktree's tests.

    Raises:
        MemoryAuthorizationError: ``result`` is not an APPROVE.
        ToolError: the group has no mapped function, or no probe can be derived from its tests.
    """
    if result.decision is not GuardDecision.APPROVE:
        raise MemoryAuthorizationError(f"memory records come only from approved repairs, not {result.decision.value}")
    function = group.group.function_qualname
    if not function:
        raise ToolError("cannot scope a memory record: the failure group has no mapped function")
    by_file: dict[str, list[str]] = {}
    for test_id in group.group.test_ids:
        by_file.setdefault(test_id.split("::", 1)[0], []).append(qualname_of_test(test_id))
    parts = []
    for path, names in sorted(by_file.items()):
        target = context.worktree_path / path
        if context.edit_policy.classify_path(path) is not PathClass.TEST or not target.is_file() or target.is_symlink():
            raise ToolError(f"cannot derive a probe: {path} is not a readable test file")
        try:
            parts.append(probe_from_tests(target.read_text(encoding="utf-8", errors="replace"), names))
        except (SyntaxError, ValueError) as exc:
            raise ToolError(f"cannot derive a probe from {path}: {exc}") from None
    record_id = f"{context.episode_id}-{group.group.group_id}"
    direction: RepairDirection = proposal.decision
    behaviour = analysis.expected_behaviour or analysis.hypothesis.explanation
    return MemoryRecord(
        record_id=record_id,
        repo_id=context.repo.repo_id,
        function_qualname=function,
        behaviour_signature=f"{function}: {behaviour}"[:500] + f" [{direction.value}]",
        direction=direction,
        probe=ExecutableProbe(probe_id=f"probe-{record_id}", probe_code="\n".join(parts),
                              probe_selector=group.group.test_ids[0]),
        evidence_ids=list(dict.fromkeys([*analysis.cited_evidence_ids, *proposal.evidence_ids])),
        approved_at_commit=context.commit_sha,
    )
