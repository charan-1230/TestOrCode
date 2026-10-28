"""Tool-call trajectory and tool budget for one episode.

The orchestrator turns a ``ToolRecorder`` on for the duration of an episode
(``recording``) and says which role is acting (``recorder.role``) and which
budget the current calls belong to (``recorder.budget``):

* ``agent``: calls an agent chose or that serve its own investigation (searches,
  file reads, its ``run_tests``, memory reads); limited by ``max_calls``, the
  episode's ``budget.max_tool_calls``;
* ``operational``: mandatory harness work that every condition pays whatever the
  agent does (Triage stability runs, patch validation/application, Guard or B1
  verification runs); limited by the finite safety cap ``max_operational``;
* ``exempt``: cleanup and the one final suite run (reverts must never be blocked).

Every tool wraps its work in ``tool_call(...)``, which

* refuses the call with ``BudgetExceededError`` once the current budget is used up;
* appends one ``ToolEvent`` (tool, role, small arguments, status, duration); the
  arguments carry ``budget`` so every real call is logged with its category.

Outside an episode no recorder is active and ``tool_call`` does nothing, so
tools stay usable on their own.  Arguments must be small and content-free:
test ids, counts, scopes; never diffs, file contents or prompts.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from testorcode.errors import BudgetExceededError, ScopeViolationError, ToolTimeoutError
from testorcode.schemas.episode_schemas import ToolName
from testorcode.schemas.result_schemas import AgentRole, ToolCallStatus, ToolEvent

_MAX_MESSAGE = 300


BUDGETS = ("agent", "operational", "exempt")


class ToolRecorder:
    """Ordered ``ToolEvent`` list for one episode, with an agent budget and an operational safety cap."""

    def __init__(self, max_calls: int | None = None, max_operational: int | None = None) -> None:
        self.max_calls = max_calls
        self.max_operational = max_operational
        self.role: AgentRole = AgentRole.ORCHESTRATOR
        self.budget: str = "agent"
        self.events: list[ToolEvent] = []
        self.used: dict[str, int] = dict.fromkeys(BUDGETS, 0)


class ToolCall:
    """Handle for one call: lets the tool mark a returned (not raised) failure."""

    def __init__(self) -> None:
        self.status = ToolCallStatus.OK
        self.error_message: str | None = None

    def fail(self, status: ToolCallStatus, message: str) -> None:
        self.status, self.error_message = status, message[:_MAX_MESSAGE]


_current: ContextVar[ToolRecorder | None] = ContextVar("testorcode_tool_recorder", default=None)


@contextmanager
def recording(recorder: ToolRecorder) -> Iterator[ToolRecorder]:
    token = _current.set(recorder)
    try:
        yield recorder
    finally:
        _current.reset(token)


def active_recorder() -> ToolRecorder | None:
    return _current.get()


@contextmanager
def tool_call(tool: ToolName, **arguments: Any) -> Iterator[ToolCall]:
    """Record one call of ``tool``.  Exceptions are recorded and re-raised.

    Raises:
        BudgetExceededError: the episode's tool-call budget is used up (nothing is run).
    """
    recorder = _current.get()
    call = ToolCall()
    if recorder is None:
        yield call
        return
    budget = recorder.budget
    if budget == "agent" and recorder.max_calls is not None and recorder.used[budget] >= recorder.max_calls:
        raise BudgetExceededError(f"tool-call budget of {recorder.max_calls} exhausted")
    if budget == "operational" and recorder.max_operational is not None and recorder.used[budget] >= recorder.max_operational:
        raise BudgetExceededError(f"operational tool-call safety cap of {recorder.max_operational} exhausted")
    started = time.monotonic()
    try:
        yield call
    except ToolTimeoutError as exc:
        call.fail(ToolCallStatus.TIMEOUT, str(exc) or type(exc).__name__)
        raise
    except ScopeViolationError as exc:
        call.fail(ToolCallStatus.NOT_PERMITTED, str(exc) or type(exc).__name__)
        raise
    except Exception as exc:
        call.fail(ToolCallStatus.ERROR, f"{type(exc).__name__}: {exc}")
        raise
    finally:
        recorder.used[budget] += 1
        recorder.events.append(
            ToolEvent(
                event_id=f"tool-{len(recorder.events) + 1}",
                step_index=len(recorder.events),
                tool=tool,
                called_by=recorder.role,
                arguments={**arguments, "budget": budget},
                status=call.status,
                error_message=call.error_message,
                duration_ms=round((time.monotonic() - started) * 1000, 1),
            )
        )
