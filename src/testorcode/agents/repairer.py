"""Repairer: propose a scoped test or source patch (never applies it).

The model does not write a diff (LLM-written hunks rarely apply).  It returns
exact ``old`` -> ``new`` snippet replacements; this module checks each ``old``
occurs exactly once in an editable file of the right scope and builds the
unified diff with ``difflib``.  If the edits cannot be placed, the model is
told why once; a second failure becomes an ``EscalationRequest``.  The
orchestrator re-validates the diff with ``patch_tools`` either way.  Prompt: ``prompts.REPAIR_SYSTEM``.
"""

from __future__ import annotations

import difflib
import re

from testorcode.agents._common import clip, read_worktree_file
from testorcode.agents.prompts import (
    REPAIR_SCOPE_RULES,
    REPAIR_SYSTEM,
    TOOL_INSTRUCTIONS,
    RepairReply,
    RepairStep,
    SnippetEdit,
    build_repair_user,
)
from testorcode.agents.tool_loop import run_tool_loop
from testorcode.agents.tools import AgentTools
from testorcode.llm_client import LLMClient
from testorcode.logging_config import get_logger
from testorcode.schemas.contracts import (
    EpisodeContext,
    EscalationRequest,
    FailureGroupDetail,
    GuardResult,
    IntentAnalysis,
    PatchProposal,
    RepairerOutput,
)
from testorcode.schemas.episode_schemas import DecisionLabel, PatchScope, RepairDirection
from testorcode.schemas.result_schemas import AgentRole
from testorcode.tools.patch_tools import scope_for_decision

logger = get_logger(__name__)

TOOL_STEPS = 4  # model calls per attempt when the Repairer may use tools

def propose_repair(
    context: EpisodeContext,
    group: FailureGroupDetail,
    analysis: IntentAnalysis,
    *,
    llm: LLMClient,
    previous_rejection: GuardResult | None = None,
    rejection_reasons: tuple[str, ...] = (),
    tools: AgentTools | None = None,
) -> RepairerOutput:
    """Return a ``PatchProposal``, or an ``EscalationRequest`` if it cannot repair safely.

    On the single permitted revision, ``previous_rejection`` carries a Guard
    rejection and ``rejection_reasons`` a rejection from patch validation,
    application or the direction check.  The proposal must stay inside
    ``context.edit_policy``; the orchestrator re-validates it.

    With ``tools`` the model may call ``search_repo``/``read_file``/``run_tests``
    first (at most ``TOOL_STEPS`` model calls per attempt); running out of steps
    is an escalation.  Without ``tools`` it answers in one call.

    Raises:
        LLMError, ToolError.
    """
    group_id = group.group.group_id
    if analysis.recommended_decision is DecisionLabel.ESCALATE:
        return EscalationRequest(group_id=group_id, trigger="insufficient_evidence", summary="analysis escalated")
    scope = scope_for_decision(analysis.recommended_decision)
    files = editable_files(context, group, scope)
    if not files:
        return EscalationRequest(
            group_id=group_id, trigger="insufficient_evidence",
            summary=f"no editable {scope.value} file is in scope for this group",
        )
    system = REPAIR_SYSTEM.format(scope_rule=REPAIR_SCOPE_RULES[scope])
    if tools is not None:
        system = f"{system}\n{TOOL_INSTRUCTIONS}\nFinal actions: \"patch\" or \"escalate\".\n"
    user = build_repair_user(context, group, analysis, files, previous_rejection, rejection_reasons)
    revision = 1 if previous_rejection is not None or rejection_reasons else 0

    problem = ""
    for attempt in range(2):
        prompt = user if not problem else f"{user}\n\nYOUR PREVIOUS EDITS COULD NOT BE APPLIED: {problem}\nTry again."
        if tools is None:
            reply = llm.complete_structured(
                role=AgentRole.REPAIRER, system=system, user=prompt, response_model=RepairReply
            ).parsed
        else:
            reply = run_tool_loop(
                llm, role=AgentRole.REPAIRER, system=system, user=prompt, step_model=RepairStep,
                tools=tools, max_steps=TOOL_STEPS,
            )
            if reply is None:
                return EscalationRequest(
                    group_id=group_id, trigger="insufficient_evidence",
                    summary=f"repairer used all {TOOL_STEPS} steps without proposing a repair",
                )
        if reply.action == "escalate":
            return EscalationRequest(
                group_id=group_id, trigger="insufficient_evidence", summary=clip(reply.rationale, 1000)
            )
        diff, problem = build_diff(context, reply.edits, set(files))
        if diff:
            evidence = list(dict.fromkeys([*reply.evidence_ids, *analysis.cited_evidence_ids]))
            direction = RepairDirection(analysis.recommended_decision.value)
            return PatchProposal(
                group_id=group_id,
                decision=direction,
                scope=scope,
                target_files=tuple(sorted({e.path for e in reply.edits})),
                unified_diff=diff,
                rationale=clip(reply.rationale, 2000),
                evidence_ids=tuple(evidence) if direction is RepairDirection.UPDATE_TEST else tuple(reply.evidence_ids),
                revision_index=revision,
            )
        logger.warning("repairer: edits not applicable (attempt %d/2): %s", attempt + 1, problem)
    return EscalationRequest(
        group_id=group_id, trigger="insufficient_evidence",
        summary=clip(f"repairer could not produce an applicable edit: {problem}", 1000),
    )


def build_diff(context: EpisodeContext, edits: list[SnippetEdit], allowed: set[str]) -> tuple[str, str]:
    """``(unified_diff, "")`` or ``("", problem)``.  Edits to one file are applied in order."""
    if not edits:
        return "", "no edits were given"
    originals: dict[str, str] = {}
    updated: dict[str, str] = {}
    for edit in edits:
        path = edit.path.removeprefix("./").removeprefix("a/").removeprefix("b/")
        if path not in allowed:
            return "", f"{path} is not one of the editable files listed"
        if path not in updated:
            text = read_worktree_file(context.worktree_path, path)
            if text is None:
                return "", f"{path} does not exist"
            originals[path] = updated[path] = text
        count = updated[path].count(edit.old)
        if count != 1:
            return "", f"the 'old' text for {path} occurs {count} times (must be exactly once)"
        updated[path] = updated[path].replace(edit.old, edit.new, 1)
    chunks = []
    for path in sorted(updated):
        if updated[path] == originals[path]:
            continue
        chunks.append(
            "".join(
                difflib.unified_diff(
                    originals[path].splitlines(keepends=True),
                    updated[path].splitlines(keepends=True),
                    fromfile=f"a/{path}",
                    tofile=f"b/{path}",
                )
            )
        )
    if not chunks:
        return "", "the edits change nothing"
    return _trim_trailing_blank_context("".join(chunks)), ""


_HUNK = re.compile(r"^@@ -(\d+),(\d+) \+(\d+),(\d+) @@(.*)$")


def _trim_trailing_blank_context(diff: str) -> str:
    """Drop whitespace-only context lines at the very end of the diff.

    ``PatchProposal`` strips edge whitespace, which would delete such a line
    (a blank source line shown as ``" "``) and leave the last hunk one line
    short.  Removing trailing context and shrinking the hunk header keeps the
    patch equivalent.
    """
    lines = diff.rstrip("\n").split("\n")
    header_at = max((i for i, ln in enumerate(lines) if ln.startswith("@@")), default=None)
    if header_at is None:
        return diff
    match = _HUNK.match(lines[header_at])
    if match is None:
        return diff
    old_start, old_len, new_start, new_len, tail = match.groups()
    old_len, new_len = int(old_len), int(new_len)
    while len(lines) > header_at + 1 and lines[-1][:1] in (" ", "") and not lines[-1].strip():
        lines.pop()
        old_len, new_len = old_len - 1, new_len - 1
    lines[header_at] = f"@@ -{old_start},{old_len} +{new_start},{new_len} @@{tail}"
    return "\n".join(lines) + "\n"


def editable_files(context: EpisodeContext, group: FailureGroupDetail, scope: PatchScope) -> list[str]:
    if scope is PatchScope.TEST:
        candidates = [t.test_file for t in group.failing_tests]
    else:
        candidates = list(group.affected_files)
    return [p for p in dict.fromkeys(candidates) if context.edit_policy.is_editable(p, scope)]

