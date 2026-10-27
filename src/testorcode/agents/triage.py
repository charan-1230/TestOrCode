"""Triage: run tests, confirm stable failures, group them by affected function.

Day-1 version: deterministic, no model call.  It re-runs the CI's failing tests
once in the sandbox (the baseline run), keeps the tests that fail again and
groups them by ``target_function`` (falling back to the test file when the
episode does not name one).  Tests that pass on the re-run are reported as
unstable.  ``llm`` is accepted for the shared signature and not used.
"""

from __future__ import annotations

from testorcode.config import Settings
from testorcode.llm_client import LLMClient
from testorcode.logging_config import get_logger
from testorcode.schemas.contracts import (
    CommitInfo,
    EpisodeContext,
    FailureGroupDetail,
    TriageResult,
)
from testorcode.schemas.episode_schemas import (
    Availability,
    CIEvent,
    FailingTest,
    FailureGroup,
    PathClass,
)
from testorcode.tools.pytest_runner import run_tests

logger = get_logger(__name__)


def run_triage(
    context: EpisodeContext, ci_event: CIEvent, *, llm: LLMClient, settings: Settings
) -> TriageResult:
    """Run the failing tests, drop unstable ones, group the rest.

    Raises:
        ToolError, ToolTimeoutError: tests cannot be run.
        LLMError: provider failure.
    """
    selectors = [t.test_id for t in ci_event.failing_tests]
    baseline = run_tests(
        context.worktree_path,
        selectors,
        timeout_seconds=settings.test_timeout_seconds,
        python_executable=settings.test_python,
    )
    failed_now = set(baseline.failing_test_ids)
    reproduced = [t for t in ci_event.failing_tests if t.test_id in failed_now]
    unstable = tuple(t.test_id for t in ci_event.failing_tests if t.test_id not in failed_now)
    logger.info(
        "triage: baseline %s, %d of %d failures reproduced", baseline.status, len(reproduced), len(selectors)
    )

    by_key: dict[str, list[FailingTest]] = {}
    for test in reproduced:
        by_key.setdefault(test.target_function or test.test_file, []).append(test)

    source_changes = [
        c for c in ci_event.changed_files
        if c.unified_diff and context.edit_policy.classify_path(c.path) is PathClass.SRC
    ]
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

    groups = []
    for index, (key, tests) in enumerate(by_key.items(), 1):
        qualname = tests[0].target_function
        test_files = sorted({t.test_file for t in tests})
        groups.append(
            FailureGroupDetail(
                group=FailureGroup(
                    group_id=f"g{index}", function_qualname=qualname, test_ids=[t.test_id for t in tests]
                ),
                failing_tests=tuple(_with_rerun_failure(t, baseline.failures) for t in tests),
                affected_files=(*(c.path for c in source_changes), *test_files),
                context_commits=commits,
                diff_context="\n".join(c.unified_diff or "" for c in source_changes) or None,
            )
        )
        logger.info("triage: group g%d (%s) has %d test(s)", index, key, len(tests))
    return TriageResult(groups=tuple(groups), unstable_test_ids=unstable, baseline_run=baseline)


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
