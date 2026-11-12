"""Provider and model check (proposal section 13.2): which model and structured-output mode give reliable tool steps?

The agents never use a provider's native tool-calling API: every tool call and every decision is one JSON reply
validated against a Pydantic model (``agents.tool_loop``).  So "tool-call reliability" here is: does the reply parse,
is the chosen action valid, and are the arguments right?  ``run_check`` asks each configured structured-output mode
(``json_schema``, ``json_object``, ``prompt``) a fixed set of synthetic probes built from the **real system prompts**:

* ``tool_step`` - the B1 agent must call ``read_file`` on a named file (a valid tool step with the right argument);
* ``finish``    - the B1 agent must finish with UPDATE_TEST citing the one id that states the intent;
* ``intent``    - the Intent Analyst must return an ``IntentAnalysis`` recommending UPDATE_TEST with that id;
* ``guard``     - the Guard's semantic review must parse as ``GuardReview``.

Every probe's input is synthetic (no pilot episode text).  The result records the model id the provider *returned*
(the dated id to freeze), the parse rate, the correct-action rate, latency and tokens per mode, and recommends the first
mode whose replies all parsed.  It spends ``len(modes) x 4 x repeats`` model calls.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlparse

from pydantic import BaseModel

from testorcode.agents.prompts import (
    B1_SYSTEM,
    GUARD_SYSTEM,
    INTENT_SYSTEM,
    TOOL_INSTRUCTIONS,
    B1Step,
    GuardReview,
)
from testorcode.config import Settings
from testorcode.errors import LLMError, LLMStructuredOutputError
from testorcode.llm_client import LLMClient
from testorcode.schemas.contracts import IntentAnalysis
from testorcode.schemas.result_schemas import AgentRole

CHECK_VERSION = "provider-check-v1"
MODES: tuple[str, ...] = ("json_schema", "json_object", "prompt")
INTENT_ID = "msg:probe00000001"
_FAILING = """\
FAILING TESTS:
- tests/test_units.py::test_to_meters
  message: AssertionError: assert to_meters(1, 'ft') == 0.3
  test source:
def test_to_meters():
    assert to_meters(1, 'ft') == 0.3"""
_DIFF = """\
CI SOURCE DIFF:
--- a/units/convert.py
+++ b/units/convert.py
@@ -4 +4 @@ def to_meters(value, unit):
-        return round(value * 0.3048, 1)
+        return round(value * 0.3048, 4)"""
_EVIDENCE = f"[{INTENT_ID}] (commit_message) Use four decimals in to_meters: one decimal lost precision (#12)"


@dataclass(frozen=True)
class Probe:
    name: str
    role: AgentRole
    system: str
    user: str
    response_model: type[BaseModel]
    correct: Callable[[Any], bool]


def _finish_ok(step: B1Step) -> bool:
    if step.action != "finish" or len(step.decisions) != 1:
        return False
    item = step.decisions[0]
    return item.decision == "UPDATE_TEST" and INTENT_ID in item.cited_evidence_ids and bool(item.edits)


PROBES: tuple[Probe, ...] = (
    Probe(
        "tool_step", AgentRole.SINGLE_AGENT, B1_SYSTEM.format(tool_instructions=TOOL_INSTRUCTIONS),
        "REPOSITORY: probe    CI COMMIT: 000000000001\n\n" + _FAILING + "\n\n" + _DIFF
        + "\n\nBefore deciding, read the first 40 lines of units/convert.py with read_file.  This step only.",
        B1Step, lambda s: s.action == "read_file" and s.path.removeprefix("./") == "units/convert.py",
    ),
    Probe(
        "finish", AgentRole.SINGLE_AGENT, B1_SYSTEM.format(tool_instructions=TOOL_INSTRUCTIONS),
        "REPOSITORY: probe    CI COMMIT: 000000000001\n\n" + _FAILING + "\n\nCI COMMIT MESSAGE:\nUse four decimals in "
        "to_meters\n\n" + _DIFF + "\n\nSEARCH RESULTS for the failing tests (cite only these ids):\n" + _EVIDENCE
        + "\n\nYou have all you need: finish now.",
        B1Step, _finish_ok,
    ),
    Probe(
        "intent", AgentRole.INTENT_ANALYST, INTENT_SYSTEM,
        "REPOSITORY: probe    CI COMMIT: 000000000001\nGROUP ID: g1    FUNCTION: units.convert.to_meters\n\n"
        + _FAILING + "\n\nCI COMMIT MESSAGE:\nUse four decimals in to_meters\n\n" + _DIFF
        + "\n\nEVIDENCE (cite only these ids):\n" + _EVIDENCE,
        IntentAnalysis,
        lambda a: a.group_id == "g1" and a.recommended_decision.value == "UPDATE_TEST" and INTENT_ID in a.cited_evidence_ids,
    ),
    Probe(
        "guard", AgentRole.INTEGRITY_GUARD, GUARD_SYSTEM,
        "GROUP ID: g1    REPAIR: UPDATE_TEST (scope test)\n" + _EVIDENCE + "\n\nTEST DIFF:\n-    assert to_meters(1, "
        "'ft') == 0.3\n+    assert to_meters(1, 'ft') == 0.3048",
        GuardReview, lambda r: r.verdict == "consistent" and r.test_meaning_preserved,
    ),
)


def run_check(
    settings: Settings,
    client_for: Callable[[Settings], LLMClient],
    *,
    modes: Sequence[str] = MODES,
    repeats: int = 2,
    probes: Sequence[Probe] = PROBES,
    clock: Callable[[], float] = time.monotonic,
) -> dict:
    """Run every probe ``repeats`` times per mode; never raises for a provider or parse failure (it is a result)."""
    per_mode: dict[str, dict] = {}
    returned_ids: set[str] = set()
    for mode in modes:
        llm = client_for(settings.model_copy(update={"llm_structured_mode": mode}))
        rows: dict[str, dict] = {}
        for probe in probes:
            row = {"calls": 0, "parsed": 0, "correct": 0, "errors": [], "seconds": 0.0, "tokens": 0}
            for _ in range(repeats):
                row["calls"] += 1
                start = clock()
                try:
                    reply = llm.complete_structured(role=probe.role, system=probe.system, user=probe.user,
                                                    response_model=probe.response_model)
                except LLMStructuredOutputError:
                    row["errors"].append("unparsable reply")
                except LLMError as exc:
                    row["errors"].append(f"{type(exc).__name__}")
                else:
                    row["parsed"] += 1
                    row["correct"] += bool(probe.correct(reply.parsed))
                    row["tokens"] += reply.usage.total_tokens or 0
                    returned_ids.add(reply.model_id)
                row["seconds"] += clock() - start
            rows[probe.name] = row
        calls = sum(r["calls"] for r in rows.values())
        per_mode[mode] = {
            "probes": rows,
            "parse_rate": sum(r["parsed"] for r in rows.values()) / calls if calls else None,
            "correct_rate": sum(r["correct"] for r in rows.values()) / calls if calls else None,
        }
    recommended = next((m for m in modes if per_mode[m]["parse_rate"] == 1.0), None)
    host = urlparse(settings.llm_base_url).netloc if settings.llm_base_url else None
    return {
        "check_version": CHECK_VERSION,
        "checked_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "provider": settings.llm_provider,
        "endpoint_host": host,  # host only: never a key or a full URL with parameters
        "model_requested": settings.llm_model,
        "model_ids_returned": sorted(returned_ids),
        "repeats": repeats,
        "modes": per_mode,
        "recommended_structured_mode": recommended,
    }
