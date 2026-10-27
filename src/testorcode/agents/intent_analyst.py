"""Intent Analyst: retrieve change-intent evidence and form a hypothesis per group.

One model call per group.  The prompt carries the failing tests (message,
traceback, test source), the CI source diff, the commit message when visible,
and the BM25-retrieved evidence with its ids.  The reply must validate as
``IntentAnalysis``; its contract already forbids, e.g., UPDATE_TEST without a
cited id, and the orchestrator checks every cited id was really retrieved.
"""

from __future__ import annotations

import re
from collections import Counter

from testorcode.agents._common import clip, function_source, read_worktree_file
from testorcode.llm_client import LLMClient
from testorcode.logging_config import get_logger
from testorcode.memory import MemoryStore
from testorcode.retrieval import Retriever, search_many
from testorcode.schemas.contracts import EpisodeContext, FailureGroupDetail, IntentAnalysis
from testorcode.schemas.result_schemas import AgentRole, RetrievedEvidence

logger = get_logger(__name__)

PROMPT_VERSION = "intent-v1"
TOP_K = 8
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")
_STOP = frozenset({"self", "return", "def", "class", "import", "from", "none", "true", "false", "elif", "else", "for", "with", "not", "and"})

SYSTEM_PROMPT = """\
You are the Intent Analyst of a CI triage system.  A commit made existing tests fail.  Decide, for ONE group of
failing tests, which side is wrong:

- outdated_test: the code change was INTENDED (a deliberate behaviour change, bug fix or API change) and the test
  still encodes the old behaviour.  Recommend UPDATE_TEST.
- regression: the code change broke behaviour that should have been kept.  Recommend FIX_CODE.
- unknown: you cannot tell.  Recommend ESCALATE.

Rules:
1. Ground the decision in evidence.  Cite evidence ONLY by the ids listed under EVIDENCE (e.g. "msg:...",
   "doc:...", "changelog:...", "docstring:...").  Never invent ids.  The CI commit message and the diff are context,
   not citable evidence, unless they also appear in the EVIDENCE list.
2. UPDATE_TEST requires at least one cited id that shows the new behaviour was intended.
3. Put ids that argue against your hypothesis in contradicting_evidence_ids.
4. If evidence is missing, weak or conflicting, recommend ESCALATE (hypothesis "unknown", or keep your
   hypothesis and still escalate).  A wrong confident answer is worse than an escalation.
5. Allowed pairs: outdated_test -> UPDATE_TEST or ESCALATE; regression -> FIX_CODE or ESCALATE;
   unknown -> ESCALATE.
6. expected_behaviour: one or two sentences on what the code should do.
7. confidence: a number from 0 to 1.
8. group_id must be exactly the given group id.

Reply with one JSON object matching the IntentAnalysis schema and nothing else.
"""


def analyse_intent(
    context: EpisodeContext,
    group: FailureGroupDetail,
    *,
    retriever: Retriever,
    memory: MemoryStore | None,
    llm: LLMClient,
) -> IntentAnalysis:
    """Return a hypothesis and recommended decision for one failure group.

    Every cited id must come from ``retriever`` or ``memory`` (the orchestrator
    verifies with ``unknown_evidence_ids``).  Insufficient or conflicting
    evidence yields ``ESCALATE``, not a guess.  ``memory=None`` = disabled.

    Raises:
        LLMError, ToolError.
    """
    evidence = search_many(retriever, group_queries(group), top_k=TOP_K)
    memory_lines: list[str] = []
    if memory is not None and group.group.function_qualname:
        for record in memory.read(context, group.group.function_qualname).usable:
            memory_lines.append(
                f"[{record.record_id}] (approved memory, probe passed) {record.function_qualname}: "
                f"{clip(record.behaviour_signature, 400)} -> {record.direction.value}"
            )
    user = build_user_prompt(context, group, evidence, memory_lines)
    logger.info("intent: group %s, %d evidence items, %d memory items", group.group.group_id, len(evidence), len(memory_lines))
    response = llm.complete_structured(
        role=AgentRole.INTENT_ANALYST, system=SYSTEM_PROMPT, user=user, response_model=IntentAnalysis
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


def build_user_prompt(
    context: EpisodeContext,
    group: FailureGroupDetail,
    evidence: list[RetrievedEvidence],
    memory_lines: list[str],
) -> str:
    parts = [
        f"REPOSITORY: {context.repo.repo_id}    CI COMMIT: {context.commit_sha[:12]}",
        f"GROUP ID: {group.group.group_id}    FUNCTION: {group.group.function_qualname or '(not named)'}",
        "",
        "FAILING TESTS:",
    ]
    for test in group.failing_tests:
        parts += [
            f"- {test.test_id}",
            "  message: " + clip(test.failure_message, 800),
            "  traceback:\n" + clip(test.traceback, 1500),
        ]
        source = read_worktree_file(context.worktree_path, test.test_file)
        body = function_source(source, test.test_name.split("[", 1)[0]) if source else None
        if body:
            parts.append("  test source:\n" + clip(body, 2500))
    commits = [c for c in group.context_commits if c.message]
    parts += ["", "CI COMMIT MESSAGE:", clip(commits[0].message, 1500) if commits else "(not available)"]
    parts += ["", "CI SOURCE DIFF (the change under suspicion):", clip(group.diff_context, 6000) or "(none)"]
    parts += ["", "EVIDENCE (cite only these ids):"]
    for item in evidence:
        parts.append(f"[{item.evidence_id}] ({item.source_type}) {clip(item.snippet, 700)}")
    if not evidence:
        parts.append("(nothing retrieved)")
    if memory_lines:
        parts += ["", "MEMORY (citable by record id):", *memory_lines]
    return "\n".join(parts)
