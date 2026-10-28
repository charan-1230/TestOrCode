"""Triage: run tests, confirm stable failures, group them by affected function.

The code runs the CI's failing tests in the sandbox ``context.stability_runs``
times (``environment.stability_runs``; 5 for the pilot).  The first run is the
baseline; a test is kept only if it failed in every run (``stable=True`` when
more than one run was made), otherwise it is reported as unstable.  Each
reproduced test is then mapped to the changed function it exercises, in order:

1. ``FailingTest.target_function`` when the episode names one;
2. otherwise the deepest traceback frame that falls inside a function the CI
   diff changed (rule-based, no model call);
3. otherwise one model call (``prompts.TRIAGE_SYSTEM``) for all tests still
   unmapped, choosing among the changed functions or "unknown".

Tests that stay unknown get one group each (``mapped_by="unmapped"``).  Groups keep first-seen order.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

from testorcode.agents._common import read_worktree_file, split_diff_by_file
from testorcode.agents.prompts import TRIAGE_SYSTEM, TriageGrouping, build_triage_user
from testorcode.config import Settings
from testorcode.errors import ToolError, ToolTimeoutError
from testorcode.llm_client import LLMClient
from testorcode.logging_config import get_logger
from testorcode.schemas.contracts import (
    CommitInfo,
    EpisodeContext,
    FailureGroupDetail,
    PytestExecution,
    TriageResult,
)
from testorcode.schemas.episode_schemas import (
    Availability,
    CIEvent,
    FailingTest,
    FailureGroup,
    PathClass,
)
from testorcode.schemas.result_schemas import AgentRole, TestRunStatus
from testorcode.tools.pytest_runner import run_tests

logger = get_logger(__name__)

_FRAME = re.compile(r"([\w./\-]+\.py):(\d+)")
_HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@")


def run_triage(
    context: EpisodeContext, ci_event: CIEvent, *, llm: LLMClient, settings: Settings
) -> TriageResult:
    """Run the failing tests, drop unstable ones, group the rest.

    Raises:
        ToolError, ToolTimeoutError: tests cannot be run.
        LLMError: provider failure.
    """
    baseline, reproduced, unstable = stable_failures(context, ci_event, settings)

    source_changes = [
        c for c in ci_event.changed_files
        if c.unified_diff and context.edit_policy.classify_path(c.path) is PathClass.SRC
    ]
    diff = "\n".join(c.unified_diff or "" for c in source_changes) or None
    spans = changed_functions(context.worktree_path, diff or "")
    mapping: dict[str, str | None] = {}
    source: dict[str, str] = {}
    for test in reproduced:
        if test.target_function:
            mapping[test.test_id], source[test.test_id] = test.target_function, "target_function"
        else:
            mapping[test.test_id], source[test.test_id] = _from_traceback(test, spans), "traceback"
    unmapped = [t for t in reproduced if mapping[t.test_id] is None]
    if unmapped and spans:
        mapping.update(_ask_model(context, unmapped, spans, diff, llm))
        source.update({t.test_id: "model" for t in unmapped})
    logger.info(
        "triage: %d test(s) mapped by rules, %d by the model, %d unknown",
        len(reproduced) - len(unmapped), sum(1 for t in unmapped if mapping[t.test_id]),
        sum(1 for t in reproduced if mapping[t.test_id] is None),
    )

    commits: tuple[CommitInfo, ...] = ()
    if ci_event.commit_message_availability is Availability.AVAILABLE and ci_event.commit_message:
        commits = (
            CommitInfo(
                sha=ci_event.commit.sha,
                message=ci_event.commit_message,
                committed_at=ci_event.commit.committed_at,
                changed_files=tuple(c.path for c in ci_event.changed_files),
            ),
        )

    # A test whose function could not be identified gets a group of its own: sharing a test file is no
    # reason to give several tests one decision (a file can hold an outdated test and a regression).
    by_key: dict[str, list[FailingTest]] = {}
    for test in reproduced:
        by_key.setdefault(mapping[test.test_id] or f"unmapped:{test.test_id}", []).append(test)
    groups = []
    for index, (key, tests) in enumerate(by_key.items(), 1):
        qualname = mapping[tests[0].test_id]
        test_files = sorted({t.test_file for t in tests})
        sources = {source[t.test_id] for t in tests}
        mapped_by = (
            "unmapped" if qualname is None
            else next(s for s in ("model", "traceback", "target_function") if s in sources)
        )
        groups.append(
            FailureGroupDetail(
                group=FailureGroup(group_id=f"g{index}", function_qualname=qualname, test_ids=[t.test_id for t in tests]),
                failing_tests=tuple(_with_rerun_failure(t, baseline.failures) for t in tests),
                affected_files=(*(c.path for c in source_changes), *test_files),
                context_commits=commits,
                diff_context=diff,
                mapped_by=mapped_by,
            )
        )
        logger.info("triage: group g%d (%s, %s) has %d test(s)", index, key, mapped_by, len(tests))
    return TriageResult(groups=tuple(groups), unstable_test_ids=unstable, baseline_run=baseline)


def stable_failures(
    context: EpisodeContext, ci_event: CIEvent, settings: Settings
) -> tuple[PytestExecution, list[FailingTest], tuple[str, ...]]:
    """(baseline run, failures that failed in every one of ``context.stability_runs`` runs, unstable test ids).

    Code only, no model: shared by Triage and the B1 baseline so both start from the same failures.
    Only PASSED and FAILED are test outcomes.  A run that ends in an error, a collection error or a
    timeout says nothing about the tests, so it is raised as an execution failure instead of being read
    as "the test passed this time" (which would mislabel a real failure as unstable).

    Raises:
        ToolTimeoutError: a stability run timed out.
        ToolError: a stability run could not execute the tests (error or collection error).
    """
    selectors = [t.test_id for t in ci_event.failing_tests]
    baseline = _checked(_run(context, selectors, settings), 1, context.stability_runs)
    failed_every_run = set(baseline.failing_test_ids)
    for index in range(2, context.stability_runs + 1):
        if not failed_every_run:
            break
        rerun = _checked(_run(context, sorted(failed_every_run), settings), index, context.stability_runs)
        failed_every_run &= set(rerun.failing_test_ids)
    stable_flag = True if context.stability_runs > 1 else None  # None = no stability filter was run
    reproduced = [t.model_copy(update={"stable": stable_flag}) for t in ci_event.failing_tests
                  if t.test_id in failed_every_run]
    unstable = tuple(t.test_id for t in ci_event.failing_tests if t.test_id not in failed_every_run)
    logger.info(
        "baseline %s, %d of %d failures failed in all %d run(s)",
        baseline.status, len(reproduced), len(selectors), context.stability_runs,
    )
    return baseline, reproduced, unstable


def _checked(run: PytestExecution, index: int, total: int) -> PytestExecution:
    """``run`` if it is a test outcome (PASSED/FAILED); otherwise the execution failure is raised."""
    if run.status in (TestRunStatus.PASSED, TestRunStatus.FAILED):
        return run
    message = f"stability run {index}/{total} did not execute the tests: {run.status.value}"
    if run.error_message:
        message += f" ({run.error_message[:300]})"
    if run.status is TestRunStatus.TIMEOUT:
        raise ToolTimeoutError(message)
    raise ToolError(message)


def _run(context: EpisodeContext, selectors: list[str], settings: Settings) -> PytestExecution:
    return run_tests(
        context.worktree_path,
        selectors,
        timeout_seconds=settings.test_timeout_seconds,
        python_executable=settings.test_python,
    )


# --------------------------------------------------------------------------- #
# Changed functions and traceback mapping
# --------------------------------------------------------------------------- #

Span = tuple[str, int, int, str]  # path, first line, last line, qualname


def changed_functions(worktree: Path, diff: str) -> list[Span]:
    """Innermost functions/methods of the current source that contain a changed line of ``diff``."""
    spans: list[Span] = []
    for path, file_diff in split_diff_by_file(diff).items():
        text = read_worktree_file(worktree, path)
        if text is None:
            continue
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        functions = list(_functions(tree, path.removesuffix(".py").replace("/", ".")))
        for line in _changed_lines(file_diff):
            inside = [f for f in functions if f[0] <= line <= f[1]]
            if inside:
                first, last, name = max(inside, key=lambda f: f[0])  # innermost = latest start
                span = (path, first, last, name)
                if span not in spans:
                    spans.append(span)
    return spans


def _functions(tree: ast.Module, prefix: str):
    stack = [(node, prefix) for node in tree.body]
    while stack:
        node, scope = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            name = f"{scope}.{node.name}"
            if not isinstance(node, ast.ClassDef):
                yield node.lineno, node.end_lineno or node.lineno, name
            stack.extend((child, name) for child in node.body)


def _changed_lines(file_diff: str) -> list[int]:
    """New-file line numbers of added lines, and of the line after each deletion."""
    lines: list[int] = []
    new_line = 0
    in_hunk = False
    for raw in file_diff.splitlines():
        match = _HUNK.match(raw)
        if match:
            new_line, in_hunk = int(match.group(1)), True
            continue
        if not in_hunk or raw.startswith(("+++", "---", "\\")):
            continue
        if raw.startswith("+"):
            lines.append(new_line)
            new_line += 1
        elif raw.startswith("-"):
            lines.append(new_line)
        else:
            new_line += 1
    return lines


def _from_traceback(test: FailingTest, spans: list[Span]) -> str | None:
    """Changed function of the deepest traceback frame that lies inside one."""
    found = None
    for path, line in _FRAME.findall(test.traceback or ""):
        for span_path, first, last, name in spans:
            if path.endswith(span_path) and first <= int(line) <= last:
                found = name
    return found


def _ask_model(
    context: EpisodeContext, tests: list[FailingTest], spans: list[Span], diff: str | None, llm: LLMClient
) -> dict[str, str | None]:
    names = [name for _, _, _, name in spans]
    reply = llm.complete_structured(
        role=AgentRole.TRIAGE,
        system=TRIAGE_SYSTEM,
        user=build_triage_user(context, tests, names, diff),
        response_model=TriageGrouping,
    ).parsed
    wanted = {t.test_id for t in tests}
    out: dict[str, str | None] = {}
    for item in reply.assignments:
        if item.test_id in wanted:
            out[item.test_id] = item.function if item.function in names else None
    return out


def _with_rerun_failure(recorded: FailingTest, rerun: tuple[FailingTest, ...]) -> FailingTest:
    """The CI record, with the message/traceback from our own re-run when the CI one is missing."""
    fresh = next((f for f in rerun if f.test_id == recorded.test_id), None)
    if fresh is None:
        return recorded
    return recorded.model_copy(
        update={
            "failure_message": recorded.failure_message or fresh.failure_message,
            "traceback": recorded.traceback or fresh.traceback,
            "assertion_diff": recorded.assertion_diff or fresh.assertion_diff,
        }
    )
