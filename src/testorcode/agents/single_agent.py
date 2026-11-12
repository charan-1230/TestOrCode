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

from testorcode.agents._common import clip
from testorcode.agents.prompts import (
    A5_SYSTEM,
    B1_SYSTEM,
    B2_SYSTEM,
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
from testorcode.schemas.result_schemas import AgentRole, RetrievedEvidence

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
    strong_prompt: bool = False,
    intent_evidence: list[RetrievedEvidence] | None = None,
    memory_lines: tuple[str, ...] = (),
    probe_checked: bool = True,
) -> B1Step | None:
    """The agent's final ``finish`` step for ``tests``, or ``None`` if it ran out of steps.

    ``strong_prompt`` (B2): the same agent with ``prompts.B2_SYSTEM``, i.e. B1's prompt plus the integrity
    instructions; nothing else changes.  ``intent_evidence`` (A5): evidence the harness retrieved with intent queries
    replaces the standard-RAG search, ``memory_lines`` are the usable memory records, and the prompt is
    ``prompts.A5_SYSTEM``.

    Raises:
        LLMError, ToolError, BudgetExceededError.
    """
    if intent_evidence is None:  # B1/B2: standard RAG on the failing tests' own text
        query = naive_query(ci_event.model_copy(update={"failing_tests": tests}))
        search_results = tools.search_repo(query)
        system = B2_SYSTEM if strong_prompt else B1_SYSTEM
    else:
        search_results = "\n".join(
            f"[{e.evidence_id}] ({e.source_type}) {clip(e.snippet, 500)}" for e in intent_evidence
        )
        system = A5_SYSTEM
    diff = "\n".join(c.unified_diff or "" for c in ci_event.changed_files if c.unified_diff) or None
    user = build_b1_user(
        context, tests, ci_event.commit_message, diff, search_results, rejection_reasons,
        intent_evidence=intent_evidence is not None, memory_lines=memory_lines, probe_checked=probe_checked,
    )
    return run_tool_loop(
        llm, role=AgentRole.SINGLE_AGENT,
        system=system.format(tool_instructions=TOOL_INSTRUCTIONS),
        user=user, step_model=B1Step, tools=tools, max_steps=TOOL_STEPS_PER_TEST * max(1, len(tests)),
    )
