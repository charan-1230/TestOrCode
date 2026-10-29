"""Intent Analyst: retrieve change-intent evidence and form a hypothesis per group.

One model call per group (prompt: ``prompts.INTENT_SYSTEM``).  The prompt carries the failing tests (message,
traceback, test source), the CI source diff, the commit message when visible,
and the BM25-retrieved evidence with its ids.  The reply must validate as
``IntentAnalysis``; its contract already forbids, e.g., UPDATE_TEST without a
cited id, and the orchestrator checks every cited id was really retrieved.
"""

from __future__ import annotations

import re
from collections import Counter

from testorcode.agents._common import clip
from testorcode.agents.prompts import INTENT_SYSTEM, build_intent_user
from testorcode.llm_client import LLMClient
from testorcode.logging_config import get_logger
from testorcode.memory import MemoryStore
from testorcode.retrieval import Retriever, search_many
from testorcode.schemas.contracts import EpisodeContext, FailureGroupDetail, IntentAnalysis
from testorcode.schemas.result_schemas import AgentRole

logger = get_logger(__name__)

TOP_K = 8
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")
_STOP = frozenset({"self", "return", "def", "class", "import", "from", "none", "true", "false", "elif", "else", "for", "with", "not", "and"})


def analyse_intent(
    context: EpisodeContext,
    group: FailureGroupDetail,
    *,
    retriever: Retriever | None,
    memory: MemoryStore | None,
    llm: LLMClient,
) -> IntentAnalysis:
    """Return a hypothesis and recommended decision for one failure group.

    Every cited id must come from ``retriever`` or ``memory`` (the orchestrator
    verifies with ``unknown_evidence_ids``).  Insufficient or conflicting
    evidence yields ``ESCALATE``, not a guess.  ``memory=None`` = disabled;
    ``retriever=None`` = no intent retrieval (A1): no search is made and the
    prompt says retrieval is disabled, so only memory records are citable.

    Raises:
        LLMError, ToolError.
    """
    retrieval = retriever is not None
    evidence = search_many(retriever, group_queries(group), top_k=TOP_K) if retrieval else []
    memory_lines: list[str] = []
    if memory is not None and group.group.function_qualname:
        for record in memory.read(context, group.group.function_qualname).usable:
            memory_lines.append(
                f"[{record.record_id}] (approved memory, probe passed) {record.function_qualname}: "
                f"{clip(record.behaviour_signature, 400)} -> {record.direction.value}"
            )
    user = build_intent_user(context, group, evidence, memory_lines, retrieval=retrieval)
    logger.info("intent: group %s, %d evidence items, %d memory items", group.group.group_id, len(evidence), len(memory_lines))
    response = llm.complete_structured(
        role=AgentRole.INTENT_ANALYST, system=INTENT_SYSTEM, user=user, response_model=IntentAnalysis
    )
    return response.parsed


def group_queries(group: FailureGroupDetail) -> list[str]:
    """Intent queries from what Triage handed over (mirrors ``retrieval.intent_queries``)."""
    diff = group.diff_context or ""
    names: list[str] = []
    if group.group.function_qualname:
        names += [p for p in group.group.function_qualname.split(".") if p != "boltons"]
    names += re.findall(r"^@@[^@]*@@\s*(?:def|class)\s+([A-Za-z_]\w*)", diff, flags=re.MULTILINE)
    names += re.findall(r"^[+-]\s*(?:async\s+)?(?:def|class)\s+([A-Za-z_]\w*)", diff, flags=re.MULTILINE)
    names += [p.rsplit("/", 1)[-1].removesuffix(".py") for p in group.affected_files if not p.startswith("tests")]
    changed = [ln[1:] for ln in diff.splitlines() if ln[:1] in "+-" and ln[:3] not in ("+++", "---")]
    idents = Counter(w.lower() for ln in changed for w in _IDENT.findall(ln) if w.lower() not in _STOP)
    queries = [" ".join(dict.fromkeys(names)), " ".join(w for w, _ in idents.most_common(12))]
    queries += [c.message.splitlines()[0] for c in group.context_commits if c.message]
    return [q for q in queries if q.strip()] or [" ".join(t.test_name for t in group.failing_tests)]

