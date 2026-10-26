"""Shared exception hierarchy.

Every exception that crosses a module boundary derives from ``TestOrCodeError``
and maps onto the run-record vocabulary in ``testorcode.schemas.result_schemas``
so the orchestrator can turn it into an ``ErrorRecord`` without guessing.

Convention: *escalation is a decision, not an error*.  An agent that cannot
decide returns an ``ESCALATE`` result.  It raises only for execution failures
(configuration, provider, tool, budget, authorization).
"""

from __future__ import annotations

from typing import ClassVar

from testorcode.schemas.result_schemas import ErrorCategory, ErrorRecord, RunStatus


class TestOrCodeError(Exception):
    """Base class for all package errors."""

    __test__ = False

    category: ClassVar[ErrorCategory] = ErrorCategory.ORCHESTRATOR_ERROR
    run_status: ClassVar[RunStatus] = RunStatus.ORCHESTRATOR_ERROR
    counted_against_agent: ClassVar[bool] = True

    def to_error_record(
        self, error_id: str, *, operation: str | None = None, fatal: bool = False
    ) -> ErrorRecord:
        """Convert to the run-record form (``fatal`` is decided by the caller)."""
        return ErrorRecord(
            error_id=error_id,
            category=self.category,
            message=str(self) or type(self).__name__,
            operation=operation,
            fatal=fatal,
            counted_against_agent=self.counted_against_agent,
        )


class ConfigError(TestOrCodeError):
    """Settings are invalid, or a setting needed for the requested operation is missing."""

    category = ErrorCategory.VALIDATION_ERROR


# --- LLM client ------------------------------------------------------------ #


class LLMError(TestOrCodeError):
    """Provider call failed."""

    category = ErrorCategory.API_ERROR
    run_status = RunStatus.API_FAILURE


class LLMRateLimitError(LLMError):
    """Logged, never scored against the agent."""

    category = ErrorCategory.RATE_LIMIT
    counted_against_agent = False


class LLMProviderOutageError(LLMError):
    """Logged, never scored against the agent."""

    category = ErrorCategory.PROVIDER_OUTAGE
    counted_against_agent = False


class LLMStructuredOutputError(LLMError):
    """The reply could not be parsed into the requested Pydantic model."""

    category = ErrorCategory.VALIDATION_ERROR


# --- Tools ----------------------------------------------------------------- #


class ToolError(TestOrCodeError):
    """A repository, git, pytest or patch operation failed."""

    category = ErrorCategory.TOOL_ERROR
    run_status = RunStatus.TOOL_FAILURE


class ToolTimeoutError(ToolError):
    category = ErrorCategory.TIMEOUT


class PatchError(ToolError):
    """The patch is malformed or does not apply (not the same as a Guard rejection)."""

    category = ErrorCategory.PATCH_ERROR


class ScopeViolationError(PatchError):
    """The patch touches paths the ``EditPolicy`` forbids."""


class HistoryCutoffError(ToolError):
    """A git or retrieval request reached past the episode's history cutoff commit."""


# --- Policy ---------------------------------------------------------------- #


class BudgetExceededError(TestOrCodeError):
    """A tool-call, LLM-call, token, revision or wall-clock budget was exhausted."""

    run_status = RunStatus.BUDGET_EXHAUSTED


class MemoryAuthorizationError(TestOrCodeError):
    """A memory write was attempted without a valid ``MemoryApproval``."""
