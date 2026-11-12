"""Scenario T: injected tool faults (proposal Table 3, "about 10% of tool calls fail").

``FaultInjector`` decides, call by call, whether an agent's own tool call fails.  It is consulted at the two places where
an agent's action is executed: ``AgentTools.call`` (the model-callable tools of the Repairer and the B1 agent) and the
orchestrator's application of a proposed patch.  Harness work (Triage stability runs, the Guard's and the harness's
verification runs, reverts, the final suite, the Analyst's automatic searches) is never faulted, so a fault cannot
corrupt what is measured.

A plan (``ToolFaultPlan``, harness-only: it lives in the full ``Episode``, never in the agent view) either lists
injections (``tool``, ``fault_kind``, ``call_index`` = zero-based index among the agent's calls of that tool) or gives a
``failure_rate``.  Rate decisions are deterministic per (plan seed, tool, call index), so the same call fails in every
condition that reaches it.  The injected call is recorded as a ``ToolEvent`` with ``INJECTED_FAULT`` and counted against
the budget like a real call (``trace.record_injected_fault``); the real tool does not run.

The model is told what a real failure would say (``agent_message``): the plan must not leak through the error text.
"""

from __future__ import annotations

import random
from collections import Counter
from typing import Final

from testorcode.schemas.episode_schemas import ToolFaultKind, ToolFaultPlan, ToolName

# What the model (or the Repairer's revision prompt) sees.  Worded like the real tool's failure.
_AGENT_MESSAGES: Final[dict[ToolFaultKind, str]] = {
    ToolFaultKind.PATCH_DOES_NOT_APPLY: "git apply failed to apply the patch: error: patch failed: hunk does not apply",
    ToolFaultKind.INVALID_TEST_SELECTOR: "ERROR: not found: the given test selector matched no collected test",
    ToolFaultKind.TIMEOUT: "the tool call timed out before it finished",
}


def agent_message(kind: ToolFaultKind, tool: ToolName) -> str:
    """The error text shown for an injected fault (never mentions the injection)."""
    if tool is ToolName.RUN_TESTS and kind is ToolFaultKind.TIMEOUT:
        return "status=timeout passed=0 failed=0 errors=0\nerror: test run exceeded the time limit"
    if tool is ToolName.RUN_TESTS and kind is ToolFaultKind.INVALID_TEST_SELECTOR:
        return "status=error passed=0 failed=0 errors=0\nerror: " + _AGENT_MESSAGES[kind]
    if tool is not ToolName.APPLY_PATCH and kind is ToolFaultKind.PATCH_DOES_NOT_APPLY:
        return f"ERROR: {tool.value} failed: internal tool error"
    return f"ERROR: {_AGENT_MESSAGES[kind]}"


def _default_kind(tool: ToolName, rng: random.Random) -> ToolFaultKind:
    """The fault a rate-based plan injects into ``tool``: the realistic failure for that tool."""
    if tool is ToolName.APPLY_PATCH:
        return ToolFaultKind.PATCH_DOES_NOT_APPLY
    if tool is ToolName.RUN_TESTS:
        return ToolFaultKind.INVALID_TEST_SELECTOR if rng.random() < 0.5 else ToolFaultKind.TIMEOUT
    return ToolFaultKind.TIMEOUT


class FaultInjector:
    """Per-episode fault decisions; ``fired`` lists ``(tool, call_index, kind)`` of every injected fault, in order."""

    def __init__(self, plan: ToolFaultPlan) -> None:
        self.plan = plan
        self._calls: Counter[ToolName] = Counter()
        self.fired: list[tuple[ToolName, int, ToolFaultKind]] = []

    @classmethod
    def for_plan(cls, plan: ToolFaultPlan | None) -> FaultInjector | None:
        """An injector for an enabled plan; None when there is nothing to inject."""
        return cls(plan) if plan is not None and plan.enabled else None

    def fault_for(self, tool: ToolName) -> ToolFaultKind | None:
        """Count one agent call of ``tool`` and return the fault to inject into it, if any."""
        index = self._calls[tool]
        self._calls[tool] += 1
        kind = next((i.fault_kind for i in self.plan.injections if i.tool is tool and i.call_index == index), None)
        if kind is None and self.plan.failure_rate:
            rng = random.Random(f"{self.plan.seed}:{tool.value}:{index}")
            if rng.random() < self.plan.failure_rate:
                kind = _default_kind(tool, rng)
        if kind is not None:
            self.fired.append((tool, index, kind))
        return kind
