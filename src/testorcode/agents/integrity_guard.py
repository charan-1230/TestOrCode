"""Integrity Guard: independently verify a repair; the only path to memory writes.

Day-1 version: deterministic, no model call.  The worktree already holds the
applied patch.  The Guard

1. re-runs every test file of the group in the sandbox (the failing tests plus
   their neighbours, to catch collateral breakage);
2. re-checks the touched files against the ``EditPolicy`` and protected paths;
3. counts ``assert`` statements before/after in touched test files and looks
   for new skip/xfail markers (weakening);
4. for FIX_CODE, confirms no test file was touched; for UPDATE_TEST, that
   evidence was cited;
5. asks ``tools.verification.trusted_outcome`` for the verdict.

A failing re-run or any finding is REJECT.  A passing re-run is *not*
approval: without a harness-supplied ``TrustedProbe`` the verdict is ESCALATE
(human verification required).  Mutation testing is not run yet.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

from testorcode.agents._common import read_worktree_file
from testorcode.errors import MemoryAuthorizationError
from testorcode.llm_client import LLMClient
from testorcode.logging_config import get_logger
from testorcode.schemas.contracts import (
    EpisodeContext,
    FailureGroupDetail,
    GuardResult,
    IntentAnalysis,
    MemoryApproval,
    PatchProposal,
)
from testorcode.schemas.episode_schemas import MemoryRecord, PathClass, RepairDirection
from testorcode.schemas.result_schemas import (
    GuardDecision,
    IntegrityReport,
    ProhibitedEditCheck,
    TestRunStatus,
    TestWeakeningFindings,
)
from testorcode.tools.git_tools import run_git
from testorcode.tools.patch_tools import is_protected_path
from testorcode.tools.pytest_runner import run_tests
from testorcode.tools.verification import TrustedProbe, trusted_outcome, verify_behaviour

logger = get_logger(__name__)

_SKIP_MARK = re.compile(r"^\+.*(pytest\.mark\.(skip|skipif|xfail)|pytest\.(skip|xfail)\()")


def verify_repair(
    context: EpisodeContext,
    group: FailureGroupDetail,
    analysis: IntentAnalysis,
    proposal: PatchProposal,
    *,
    llm: LLMClient | None = None,
    python_executable: str | Path | None = None,
    timeout_seconds: int = 120,
    probe: TrustedProbe | None = None,
) -> GuardResult:
    """Check the applied repair: tests, weakening, scope, evidence, mutants.

    Passing tests alone never yields APPROVE.  The worktree already contains
    the applied patch.  ``python_executable``/``timeout_seconds`` select the
    target repository's interpreter; ``probe`` is a harness-supplied trusted
    behaviour check (None = none available, so at best ESCALATE).

    Raises:
        ToolError, ToolTimeoutError.
    """
    worktree = context.worktree_path
    test_files = sorted({t.test_file for t in group.failing_tests})
    run = run_tests(worktree, test_files, timeout_seconds=timeout_seconds, python_executable=python_executable)

    policy = context.edit_policy
    touched = list(proposal.target_files)
    out_of_scope = [p for p in touched if not policy.is_editable(p, proposal.scope) or is_protected_path(p)]
    touched_tests = [p for p in touched if policy.classify_path(p) is PathClass.TEST]
    edits = ProhibitedEditCheck(
        checks_run=True, files_modified=touched, out_of_scope_files=out_of_scope, test_files_modified=touched_tests
    )

    before = after = 0
    for path in touched_tests:
        before += _count_asserts(_head_text(worktree, path))
        after += _count_asserts(read_worktree_file(worktree, path))
    removed = max(0, before - after)
    skips = sum(1 for line in proposal.unified_diff.splitlines() if _SKIP_MARK.match(line))
    weakening = TestWeakeningFindings(
        checks_run=True,
        assertions_before=before,
        assertions_after=after,
        removed_assertions=removed,
        added_skip_markers=skips,
        weakening_detected=bool(removed or skips),
    )

    is_fix = proposal.decision is RepairDirection.FIX_CODE
    tests_untouched = (not _tests_dirty(worktree, policy.allowed_test_roots)) if is_fix else None
    evidence_met = bool(analysis.cited_evidence_ids) if not is_fix else None

    problems: list[str] = []
    if run.status is not TestRunStatus.PASSED:
        problems.append(
            f"post-repair run of {', '.join(test_files)} is {run.status.value} "
            f"(failed={run.failed}, errors={run.errors})"
        )
    if out_of_scope:
        problems.append("patch touches files outside the edit policy: " + ", ".join(out_of_scope))
    if weakening.weakening_detected:
        problems.append(f"test weakening: {removed} assertion(s) removed, {skips} skip/xfail marker(s) added")
    if tests_untouched is False:
        problems.append("FIX_CODE repair modified test files")
    if evidence_met is False:
        problems.append("UPDATE_TEST repair cites no evidence")

    if problems:
        decision, reasons = GuardDecision.REJECT, problems
    else:
        verification = None
        if probe is not None:
            verification = verify_behaviour(
                worktree, probe, timeout_seconds=timeout_seconds, policy=policy,
                python_executable=python_executable,
            )
        outcome = trusted_outcome(run, verification)
        decision = outcome.recommended_decision
        reasons = [] if decision is GuardDecision.APPROVE else [
            "post-repair tests pass and no integrity finding was raised, but "
            + "; ".join(outcome.reasons)
            + " - human verification required"
        ]
    logger.info("guard: group %s -> %s", group.group.group_id, decision)
    return GuardResult(
        decision=decision,
        reasons=tuple(reasons),
        post_repair_run=run,
        integrity=IntegrityReport(
            guard_ran=True,
            guard_decision=decision,
            weakening=weakening,
            prohibited_edits=edits,
            tests_untouched_for_fix_code=tests_untouched,
            evidence_requirement_met=evidence_met,
            rejection_reasons=list(reasons) if decision is GuardDecision.REJECT else [],
        ),
    )


def authorize_memory_write(
    result: GuardResult, record: MemoryRecord, *, maintainer_approved: bool
) -> MemoryApproval:
    """Build the approval ``MemoryStore.write`` requires.

    Raises:
        MemoryAuthorizationError: ``result`` is not APPROVE or the maintainer did not approve.
    """
    if result.decision is not GuardDecision.APPROVE:
        raise MemoryAuthorizationError(f"memory write needs a Guard APPROVE, got {result.decision.value}")
    if not maintainer_approved:
        raise MemoryAuthorizationError("memory write needs maintainer approval")
    return MemoryApproval(
        record_id=record.record_id,
        guard_decision=result.decision,
        maintainer_approved=True,
        approved_at_commit=record.approved_at_commit,
    )


def _count_asserts(text: str | None) -> int:
    if not text:
        return 0
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return 0
    return sum(isinstance(node, ast.Assert) for node in ast.walk(tree))


def _head_text(worktree: Path, path: str) -> str | None:
    """The file at the worktree's HEAD.  The CI state changes source files only, so for a
    test file HEAD is the pre-repair content."""
    result = run_git(worktree, ["show", f"HEAD:{path}"])
    return result.stdout if result.returncode == 0 else None


def _tests_dirty(worktree: Path, test_roots: tuple[str, ...]) -> bool:
    result = run_git(worktree, ["status", "--porcelain", "--", *test_roots])
    return result.returncode != 0 or bool(result.stdout.strip())
