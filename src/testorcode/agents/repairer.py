"""Repairer: propose a scoped test or source patch (never applies it).

The model does not write a diff (LLM-written hunks rarely apply).  It returns
exact ``old`` -> ``new`` snippet replacements; this module checks each ``old``
occurs exactly once in an editable file of the right scope and builds the
unified diff with ``difflib``.  If the edits cannot be placed, the model is
told why once; a second failure becomes an ``EscalationRequest``.  The
orchestrator re-validates the diff with ``patch_tools`` either way.
"""

from __future__ import annotations

import difflib
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from testorcode.agents._common import (
    changed_windows,
    clip,
    function_source,
    read_worktree_file,
    split_diff_by_file,
)
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

PROMPT_VERSION = "repair-v1"


class SnippetEdit(BaseModel):
    """Replace ``old`` (copied exactly from the file, occurring once) with ``new``."""

    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=1, description="Repository-relative path, e.g. tests/test_x.py")
    old: str = Field(min_length=1, description="Exact existing text, including indentation; must occur once.")
    new: str = Field(description="Replacement text.")


class RepairReply(BaseModel):
    """What the model returns; turned into ``PatchProposal`` or ``EscalationRequest`` here."""

    model_config = ConfigDict(extra="forbid")

    # Every field is required: with optional arrays, Gemini's structured output tended to omit ``edits``.
    action: Literal["patch", "escalate"]
    edits: list[SnippetEdit] = Field(description="At least one edit for action 'patch'; [] for 'escalate'.")
    rationale: str = Field(min_length=1)
    evidence_ids: list[str] = Field(description="Ids the analysis cited; [] if none.")


_SCOPE_RULES = {
    PatchScope.TEST: (
        "Edit ONLY test files listed under FILES.  The test encodes outdated behaviour.  You may ONLY change the "
        "expected literal value in an existing `assert <expr> == <literal>` or `assert <expr> != <literal>` "
        "statement.  Do not add, delete or rename tests or assertions, do not change inputs, imports, fixtures or "
        "the tested expression, do not add skips.  Any other change is rejected automatically; if the fix needs "
        "one, choose action \"escalate\"."
    ),
    PatchScope.SRC: (
        "Edit ONLY source files listed under FILES.  The code regressed: restore the intended behaviour with the "
        "smallest change.  Never edit tests, conftest.py or configuration, and never add pytest imports, skips "
        "or process exits."
    ),
}

SYSTEM_PROMPT = """\
You are the Repairer of a CI triage system.  The Intent Analyst has decided which side is wrong for one group of
failing tests.  Produce the smallest repair on that side only, as exact snippet replacements.

Each edit: "path" (repository-relative), "old" (text copied EXACTLY from the file shown, including indentation,
long enough to occur only once, without the line-number prefix), "new" (replacement).

{scope_rule}

For action "patch", "edits" MUST contain at least one edit.
If you cannot repair safely within these rules, reply with action "escalate", "edits": [] and explain why in rationale.
evidence_ids: copy the ids the analysis cited (required when updating tests; [] if none).
Reply with one JSON object matching the RepairReply schema and nothing else.
"""


def propose_repair(
    context: EpisodeContext,
    group: FailureGroupDetail,
    analysis: IntentAnalysis,
    *,
    llm: LLMClient,
    previous_rejection: GuardResult | None = None,
) -> RepairerOutput:
    """Return a ``PatchProposal``, or an ``EscalationRequest`` if it cannot repair safely.

    ``previous_rejection`` is set on the single permitted revision.  The proposal
    must stay inside ``context.edit_policy``; the orchestrator re-validates it.

    Raises:
        LLMError, ToolError.
    """
    group_id = group.group.group_id
    if analysis.recommended_decision is DecisionLabel.ESCALATE:
        return EscalationRequest(group_id=group_id, trigger="insufficient_evidence", summary="analysis escalated")
    scope = scope_for_decision(analysis.recommended_decision)
    files = _editable_files(context, group, scope)
    if not files:
        return EscalationRequest(
            group_id=group_id, trigger="insufficient_evidence",
            summary=f"no editable {scope.value} file is in scope for this group",
        )
    system = SYSTEM_PROMPT.format(scope_rule=_SCOPE_RULES[scope])
    user = _user_prompt(context, group, analysis, files, previous_rejection)
    revision = 1 if previous_rejection is not None else 0

    problem = ""
    for attempt in range(2):
        prompt = user if not problem else f"{user}\n\nYOUR PREVIOUS EDITS COULD NOT BE APPLIED: {problem}\nTry again."
        reply = llm.complete_structured(
            role=AgentRole.REPAIRER, system=system, user=prompt, response_model=RepairReply
        ).parsed
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


def _editable_files(context: EpisodeContext, group: FailureGroupDetail, scope: PatchScope) -> list[str]:
    if scope is PatchScope.TEST:
        candidates = [t.test_file for t in group.failing_tests]
    else:
        candidates = list(group.affected_files)
    return [p for p in dict.fromkeys(candidates) if context.edit_policy.is_editable(p, scope)]


def _user_prompt(
    context: EpisodeContext,
    group: FailureGroupDetail,
    analysis: IntentAnalysis,
    files: list[str],
    previous_rejection: GuardResult | None,
) -> str:
    parts = [
        f"GROUP ID: {group.group.group_id}",
        (f"DECISION: {analysis.recommended_decision.value}  (hypothesis {analysis.hypothesis.kind.value}, "
        f"confidence {analysis.confidence:.2f})"),
        f"ANALYSIS: {clip(analysis.hypothesis.explanation, 1500)}",
        f"EXPECTED BEHAVIOUR: {clip(analysis.expected_behaviour, 600) or '(not stated)'}",
        f"CITED EVIDENCE IDS: {', '.join(analysis.cited_evidence_ids) or '(none)'}",
        "",
        "FAILING TESTS:",
    ]
    for test in group.failing_tests:
        parts += [f"- {test.test_id}", "  message: " + clip(test.failure_message, 800)]
    diffs = split_diff_by_file(group.diff_context or "")
    parts += ["", "FILES (line numbers are for orientation only; do not copy them):"]
    for path in files:
        text = read_worktree_file(context.worktree_path, path) or ""
        if path in diffs:
            shown = changed_windows(text, diffs[path])
        else:
            names = [t.test_name.split("[", 1)[0] for t in group.failing_tests if t.test_file == path]
            bodies = [b for n in dict.fromkeys(names) if (b := function_source(text, n))]
            shown = "\n\n".join(bodies) or clip(text, 12000)
        parts += [f"--- {path} ---", shown]
    if group.diff_context:
        parts += ["", "CI SOURCE DIFF (what the commit changed):", clip(group.diff_context, 5000)]
    if previous_rejection is not None:
        parts += [
            "",
            "YOUR PREVIOUS REPAIR WAS REJECTED:",
            *[f"- {clip(r, 600)}" for r in previous_rejection.reasons],
        ]
    return "\n".join(parts)
