"""Integrity Guard: independently verify a repair; the only path to memory writes.

The worktree already holds the applied patch.  The Guard

1. re-runs the group's failing tests, then the whole test suite in the sandbox
   (``verify_group``): the group's tests must pass and no test outside other
   groups' still-unresolved failures may fail;
2. re-checks the touched files against the ``EditPolicy`` and protected paths;
3. judges only this patch's own change (the state just before it, rebuilt by reversing
   its diff, versus now; earlier groups' approved repairs are never attributed to it)
   with the deterministic weakening analysis (``tools.weakening``: removed assertions,
   skip/xfail, loosened comparisons, swallowed exceptions, deleted tests), and rejects
   a test patch that changes any test function other than the group's failing tests;
4. for FIX_CODE, confirms no test file was touched; for UPDATE_TEST, that
   evidence was cited;
5. if ``llm`` is given and nothing was found, asks the model for a semantic
   review (``prompts.GUARD_SYSTEM``): does a changed test still check what its
   name and setup say, is the repair legitimate?  The review can only lower the
   verdict to ESCALATE, never approve;
6. for UPDATE_TEST, measures the updated tests' strength on cached mutants of the
   group's function (``tools.mutation.mutation_check``; baseline: the same tests
   before this repair, never the reference repair);
7. asks ``tools.verification.repair_outcome`` for the verdict.

A failing re-run or any finding is REJECT.  A passing re-run is *not* approval
of a test edit: UPDATE_TEST needs measured mutant-kill strength (measured weakening
is REJECT, unmeasured strength ESCALATE).  FIX_CODE is verified by the untouched
original tests plus the whole-suite regression run; a harness ``TrustedProbe``,
when supplied, must pass as well.
"""

from __future__ import annotations

import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

from testorcode.agents.prompts import GUARD_SYSTEM, GuardReview, build_guard_user
from testorcode.errors import MemoryAuthorizationError, ToolError
from testorcode.llm_client import LLMClient
from testorcode.logging_config import get_logger
from testorcode.memory import record_from_repair
from testorcode.schemas.contracts import (
    EpisodeContext,
    FailureGroupDetail,
    GuardResult,
    IntentAnalysis,
    MemoryApproval,
    PatchProposal,
    PytestExecution,
)
from testorcode.schemas.episode_schemas import MemoryRecord, MutantSpec, PathClass, RepairDirection
from testorcode.schemas.result_schemas import (
    AgentRole,
    GuardDecision,
    IntegrityReport,
    ProhibitedEditCheck,
    TestRunStatus,
)
from testorcode.tools.git_tools import run_git
from testorcode.tools.mutation import MutationCheck, mutation_check
from testorcode.tools.patch_tools import is_protected_path
from testorcode.tools.pytest_runner import run_tests
from testorcode.tools.verification import TrustedProbe, repair_outcome, verify_behaviour
from testorcode.tools.weakening import analyse_test_changes

logger = get_logger(__name__)

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
    known_failures: frozenset[str] = frozenset(),
    cached_mutants: Mapping[str, Sequence[MutantSpec]] | None = None,
) -> GuardResult:
    """Check the applied repair: tests, weakening, scope, evidence, mutants.

    Passing tests alone never approves a test edit.  The worktree already contains
    the applied patch.  ``python_executable``/``timeout_seconds`` select the
    target repository's interpreter; ``probe`` is a harness-supplied trusted
    behaviour check (optional; when given it must pass).
    ``known_failures`` are other groups' still-unresolved CI failures, which may
    keep failing (see ``verify_group``).  ``cached_mutants`` (function -> mutants,
    ``tools.mutation.load_cached_mutants``) replaces mutant generation for the
    functions it lists; the mutants are executed now either way.

    Raises:
        ToolError, ToolTimeoutError.
    """
    worktree = context.worktree_path
    run, regression, test_problems = verify_group(
        context, group, known_failures, timeout_seconds=timeout_seconds, python_executable=python_executable
    )

    policy = context.edit_policy
    # Judge only what THIS patch changed.  The worktree may already hold earlier groups' approved repairs, so the
    # state to compare with is the one just before this patch: the current files with this diff reversed (not HEAD).
    touched = _diff_paths(proposal.unified_diff) or list(proposal.target_files)
    pre_patch, post_patch = _patch_states(worktree, proposal.unified_diff, touched)
    out_of_scope = [p for p in touched if not policy.is_editable(p, proposal.scope) or is_protected_path(p)]
    touched_tests = [p for p in touched if policy.classify_path(p) is PathClass.TEST]
    edits = ProhibitedEditCheck(
        checks_run=True, files_modified=touched, out_of_scope_files=out_of_scope, test_files_modified=touched_tests
    )

    found = analyse_test_changes({p: pre_patch.get(p) for p in touched_tests}, {p: post_patch.get(p) for p in touched_tests})
    weakening = found.findings
    allowed = {f"{t.test_file}::{_test_qualname(t.test_id)}" for t in group.failing_tests}
    unrelated = sorted(found.changed_tests - allowed)

    is_fix = proposal.decision is RepairDirection.FIX_CODE
    tests_untouched = (not touched_tests) if is_fix else None  # this patch's own files, not earlier approvals
    evidence_met = bool(analysis.cited_evidence_ids) if not is_fix else None

    problems: list[str] = list(test_problems)
    if out_of_scope:
        problems.append("patch touches files outside the edit policy: " + ", ".join(out_of_scope))
    problems += found.problems
    if tests_untouched is False:
        problems.append("FIX_CODE repair modified test files")
    if evidence_met is False:
        problems.append("UPDATE_TEST repair cites no evidence")
    if unrelated:
        problems.append("patch changes test functions outside this failure group: " + ", ".join(unrelated))

    mutation = MutationCheck()
    review = None
    if not problems and llm is not None:
        review = llm.complete_structured(
            role=AgentRole.INTEGRITY_GUARD,
            system=GUARD_SYSTEM,
            user=build_guard_user(context, group, analysis, proposal),
            response_model=GuardReview,
        ).parsed
    if problems:
        decision, reasons = GuardDecision.REJECT, problems
    elif review is not None and (review.verdict == "suspicious" or not review.test_meaning_preserved):
        # The model review can only lower the verdict: a concern sends the repair to a human.
        decision = GuardDecision.ESCALATE
        reasons = ["semantic review flagged the repair: " + ("; ".join(review.reasons) or "test meaning not preserved")]
    else:
        verification = None
        if probe is not None:
            verification = verify_behaviour(
                worktree, probe, timeout_seconds=timeout_seconds, policy=policy,
                python_executable=python_executable,
            )
        cached = (cached_mutants or {}).get(group.group.function_qualname or "")
        mutation = mutation_check(  # UPDATE_TEST: strength of the edited tests; FIX_CODE: oracle sensitivity
            context, group, proposal, pre_patch, timeout_seconds=timeout_seconds,
            python_executable=python_executable, mutants=list(cached) if cached else None,
        )
        outcome = repair_outcome(
            proposal.decision, run, probe_verification=verification, mutation=mutation.result,
            mutation_problems=mutation.problems, mutation_gaps=mutation.gaps,
        )
        decision = outcome.recommended_decision
        reviewed = " and the semantic review found it consistent" if review is not None else ""
        if decision is GuardDecision.APPROVE:
            reasons = []
        elif decision is GuardDecision.REJECT:
            reasons = list(outcome.reasons)
        else:
            reasons = [
                f"post-repair tests pass, no integrity finding was raised{reviewed}, but "
                + "; ".join(outcome.reasons)
                + " - human verification required"
            ]
    logger.info("guard: group %s -> %s", group.group.group_id, decision)
    return GuardResult(
        decision=decision,
        reasons=tuple(reasons),
        post_repair_run=run,
        regression_run=regression,
        integrity=IntegrityReport(
            guard_ran=True,
            guard_decision=decision,
            weakening=weakening,
            prohibited_edits=edits,
            mutation=mutation.result,
            tests_untouched_for_fix_code=tests_untouched,
            evidence_requirement_met=evidence_met,
            rejection_reasons=list(reasons) if decision is GuardDecision.REJECT else [],
        ),
    )


def verify_group(
    context: EpisodeContext,
    group: FailureGroupDetail,
    known_failures: frozenset[str],
    *,
    timeout_seconds: int,
    python_executable: str | Path | None = None,
) -> tuple[PytestExecution, PytestExecution | None, list[str]]:
    """Test-level verification of one group's applied repair: ``(targeted run, regression run, problems)``.

    1. **Targeted:** the group's own failing tests must now pass.
    2. **Regression:** the whole test suite (the policy's test roots) is run; every failure must be one of this
       group's tests (already reported) or a ``known_failures`` test, i.e. another group's CI failure that is
       still unresolved.  Anything else is a test that passed before the repair and fails now.

    Other groups' known failures may share a test file with this group without blocking a correct repair, while
    a repair that breaks a previously passing test, or a group that was already accepted (the caller leaves its
    tests out of ``known_failures``), is caught.  Shared by the Integrity Guard and the B1 baseline.  The
    regression run is skipped when the targeted run already failed.
    """
    worktree = context.worktree_path
    targets = list(group.group.test_ids)
    targeted = run_tests(worktree, targets, timeout_seconds=timeout_seconds, python_executable=python_executable)
    if targeted.status is not TestRunStatus.PASSED:
        still = sorted(set(targeted.failing_test_ids) & set(targets))
        return targeted, None, [
            f"post-repair run of the group's tests is {targeted.status.value}"
            + (f"; still failing: {', '.join(still)}" if still else "")
            + (f" ({targeted.error_message[:200]})" if targeted.error_message else "")
        ]
    regression = run_tests(
        worktree, suite_selectors(context), timeout_seconds=timeout_seconds, python_executable=python_executable
    )
    if regression.status not in (TestRunStatus.PASSED, TestRunStatus.FAILED):
        return targeted, regression, [f"regression run of the test suite did not execute: {regression.status.value}"]
    new = sorted(set(regression.failing_test_ids) - set(targets) - set(known_failures))
    if new:
        return targeted, regression, [f"previously passing tests now fail: {', '.join(new)}"]
    return targeted, regression, []


def suite_selectors(context: EpisodeContext) -> list[str]:
    """The policy's test roots that exist in the worktree (the scope of a whole-suite run)."""
    roots = [r.rstrip("/") for r in context.edit_policy.allowed_test_roots if (context.worktree_path / r).is_dir()]
    return roots or [r.rstrip("/") for r in context.edit_policy.allowed_test_roots[:1]]


def build_memory_record(
    context: EpisodeContext,
    group: FailureGroupDetail,
    analysis: IntentAnalysis,
    proposal: PatchProposal,
    result: GuardResult,
) -> MemoryRecord:
    """The record the Guard writes after an approved repair (function, behaviour signature,
    direction, executable probe derived from the approved test patch, evidence ids, commit).

    The probe is derived from the group's tests as they stand after the approved repair
    (``memory.record_from_repair``, Human 2's memory work).

    Raises:
        MemoryAuthorizationError: ``result`` is not APPROVE.
        ToolError: no function to scope the record to, or no probe can be derived.
    """
    return record_from_repair(context, group, analysis, proposal, result)


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


def _diff_paths(diff: str) -> list[str]:
    """Files a unified diff changes (``+++ b/<path>``; ``--- a/<path>`` for deletions)."""
    paths: list[str] = []
    for line in diff.splitlines():
        if line.startswith(("+++ b/", "--- a/")):
            path = line[6:].split("\t", 1)[0]
            if path not in paths:
                paths.append(path)
    return paths


def _patch_states(worktree: Path, diff: str, paths: list[str]) -> tuple[dict[str, str | None], dict[str, str | None]]:
    """Text of ``paths`` just before and just after this patch (the worktree already holds it applied).

    The diff is reversed in a scratch copy of the touched files, so earlier groups' accepted repairs are part of both
    states and never attributed to this patch.  ``--recount`` tolerates a stored diff whose trailing blank context was
    trimmed (``PatchProposal`` strips surrounding whitespace).

    Raises:
        ToolError: the diff does not reverse cleanly on the worktree.
    """
    diff = diff if diff.endswith("\n") else diff + "\n"
    after = {p: _read_text(worktree / p) for p in paths}
    with tempfile.TemporaryDirectory(prefix="testorcode-guard-") as tmp:
        work = Path(tmp)
        for path, text in after.items():
            if text is not None:
                (work / path).parent.mkdir(parents=True, exist_ok=True)
                (work / path).write_text(text)
        run_git(work, ["init", "-q"])
        result = run_git(work, ["apply", "--recount", "--whitespace=nowarn", "-R", "-"], input_text=diff)
        if result.returncode != 0:
            raise ToolError(f"patch could not be reversed in a scratch copy: {result.stderr.strip()[:200]}")
        before = {p: _read_text(work / p) for p in paths}
    return before, after


def _read_text(path: Path) -> str | None:
    return path.read_text(encoding="utf-8", errors="replace") if path.is_file() else None


def _test_qualname(test_id: str) -> str:
    """``tests/x.py::TestC::test_m[p]`` -> ``TestC.test_m``."""
    return ".".join(test_id.split("::")[1:]).split("[", 1)[0]
