"""A bounded tool loop on top of ``LLMClient.complete_structured``.

Each model reply is one validated JSON *step* (a Pydantic model with an
``action`` field plus tool arguments, see ``prompts.ToolArgs``).  If ``action``
names a tool, the tool runs and its result is appended to the conversation that
is sent with the next step; any other action is final and is returned.  This
needs no provider-specific function calling: it works with any endpoint that
returns JSON (Gemini's ``json_schema`` mode today).

The loop is bounded by ``max_steps`` model calls.  The last step is told that it
must be final; if it still asks for a tool, ``run_tool_loop`` returns ``None``
and the caller escalates (running out of steps is a decision outcome, not an
execution failure).  LLM and tool budgets are enforced underneath
(``_MeteredLLM``, ``tools.trace``).
"""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel

from testorcode.agents._common import clip
from testorcode.agents.tools import TOOL_NAMES, AgentTools
from testorcode.llm_client import LLMClient
from testorcode.logging_config import get_logger
from testorcode.schemas.result_schemas import AgentRole

logger = get_logger(__name__)


class _Step(Protocol):
    action: str
    query: str
    path: str
    start_line: int
    end_line: int
    selectors: list[str]


def run_tool_loop[S: BaseModel](
    llm: LLMClient,
    *,
    role: AgentRole,
    system: str,
    user: str,
    step_model: type[S],
    tools: AgentTools,
    max_steps: int,
) -> S | None:
    """Return the first final step, or ``None`` if ``max_steps`` model calls end without one."""
    transcript: list[str] = []
    for index in range(1, max_steps + 1):
        last = index == max_steps
        prompt = user
        if transcript:
            prompt += "\n\nTOOL CALLS SO FAR (oldest first):\n" + "\n\n".join(transcript)
        prompt += f"\n\nSTEP {index} of {max_steps}." + (
            " This is your LAST step: you must give a final action now, not a tool call." if last else ""
        )
        step: _Step = llm.complete_structured(role=role, system=system, user=prompt, response_model=step_model).parsed
        if step.action not in TOOL_NAMES:
            return step  # type: ignore[return-value]
        if last:
            logger.warning("role=%s asked for a tool on its last step; giving up", role)
            return None
        result = tools.call(
            step.action, query=step.query, path=step.path, start_line=step.start_line,
            end_line=step.end_line, selectors=step.selectors,
        )
        transcript.append(f"[{index}] {_describe(step)}\n{clip(result, 4000)}")
        logger.info("role=%s step %d: %s", role, index, step.action)
    return None  # pragma: no cover - the last step returns above


def _describe(step: _Step) -> str:
    if step.action == "search_repo":
        return f"search_repo(query={step.query!r})"
    if step.action == "read_file":
        return f"read_file(path={step.path!r}, start_line={step.start_line}, end_line={step.end_line})"
    return f"run_tests(selectors={step.selectors!r})"
