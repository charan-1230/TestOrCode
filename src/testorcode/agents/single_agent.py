"""B1 baseline: one tool-using agent with standard RAG and no memory (proposal section 10).

One prompt (``prompts.B1_SYSTEM``) does what the four roles split up in the
multi-agent system: it decides per failing test and writes the repair.  It gets
the failing output, the CI diff and commit message, and **standard RAG**: the
top-k ``search_repo`` results for the failing test's own text
(``retrieval.naive_query``), not intent-oriented queries.  It may then call the
same tools as the Repairer (``search_repo``, ``read_file``, ``run_tests``) and
finishes with ``B1Decision`` items.  The orchestrator (``SingleAgentRun``)
validates, applies and re-tests them; there is no independent Integrity Guard.
"""

from __future__ import annotations

from testorcode.agents.prompts import (
    B1_SYSTEM,
    TOOL_INSTRUCTIONS,
    B1Step,
    build_b1_user,
)
from testorcode.agents.tool_loop import run_tool_loop
from testorcode.agents.tools import AgentTools
from testorcode.llm_client import LLMClient
from testorcode.retrieval import naive_query
from testorcode.schemas.contracts import EpisodeContext
from testorcode.schemas.episode_schemas import CIEvent, FailingTest
from testorcode.schemas.result_schemas import AgentRole

# Model calls per failing test the agent is asked about.  The multi-agent pipeline spends roughly
# Intent 1 + Repairer up to 4 + Guard review 1 per group; B1 decides all groups in one loop, so its cap scales
# with the tests in front of it.  The shared episode budget (``_MeteredLLM``) is the binding limit for both.
TOOL_STEPS_PER_TEST = 6


def run_single_agent(
    context: EpisodeContext,
    ci_event: CIEvent,
    tests: list[FailingTest],
    *,
    llm: LLMClient,
    tools: AgentTools,
    rejection_reasons: tuple[str, ...] = (),
) -> B1Step | None:
    """The agent's final ``finish`` step for ``tests``, or ``None`` if it ran out of steps.

    Raises:
        LLMError, ToolError, BudgetExceededError.
    """
    query = naive_query(ci_event.model_copy(update={"failing_tests": tests}))
    search_results = tools.search_repo(query)
    diff = "\n".join(c.unified_diff or "" for c in ci_event.changed_files if c.unified_diff) or None
    user = build_b1_user(context, tests, ci_event.commit_message, diff, search_results, rejection_reasons)
    return run_tool_loop(
        llm, role=AgentRole.SINGLE_AGENT, system=B1_SYSTEM.format(tool_instructions=TOOL_INSTRUCTIONS),
        user=user, step_model=B1Step, tools=tools, max_steps=TOOL_STEPS_PER_TEST * max(1, len(tests)),
    )
