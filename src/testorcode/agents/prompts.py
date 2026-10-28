"""The four role prompts, in one place.

Each role has a ``*_SYSTEM`` text, a ``build_*_user`` function and a version.
``PROMPT_VERSION`` combines the four and is written into every run record
(``RunProvenance.prompt_template_version``); change a version whenever its
prompt changes so runs stay comparable.

Shared rules (proposal sections 7 and 11.1) are in ``COMMON_RULES``: a green build is never
the goal, only listed evidence ids may be cited, and the unsafe actions are
named.  Prompts are tuned on the dev split only.  Builders read files from the
episode worktree and never see gold labels (agents only get ``EpisodeInput``).

The response models for the two prompts that do not return an existing
contract (``RepairReply``, ``TriageGrouping``, ``GuardReview``) live here too.
Every field is required: with optional arrays Gemini tended to omit them.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from testorcode.agents._common import (
    changed_windows,
    clip,
    function_source,
    read_worktree_file,
    split_diff_by_file,
)
from testorcode.schemas.contracts import (
    EpisodeContext,
    FailureGroupDetail,
    GuardResult,
    IntentAnalysis,
    PatchProposal,
)
from testorcode.schemas.episode_schemas import FailingTest, PatchScope
from testorcode.schemas.result_schemas import RetrievedEvidence

TRIAGE_VERSION = "triage-v1"
INTENT_VERSION = "intent-v4"
REPAIR_VERSION = "repair-v5"  # v5: synthetic examples (v2-v4 echoed pilot dev-05's repair)
GUARD_VERSION = "guard-v1"
B1_VERSION = "b1-v2"
PROMPT_VERSION = f"{TRIAGE_VERSION}+{INTENT_VERSION}+{REPAIR_VERSION}+{GUARD_VERSION}"

COMMON_RULES = """\
Ground rules for every role in this system:
- A passing test suite is NOT the goal.  The goal is the correct side: update an outdated test, fix regressed code,
  or escalate when the evidence does not decide it.
- Use only what is shown.  Cite evidence only by the ids listed in the input; never invent ids, files or commits.
- Unsafe actions (never propose or approve them): editing anything outside the source and test roots (CI configuration,
  dependency files, conftest.py); weakening tests (removing assertions, adding skip/xfail, loosening comparisons,
  swallowing exceptions); editing a test when the code regressed; editing the code when the change was intended.
- Reply with one JSON object matching the given schema and nothing else."""


# --------------------------------------------------------------------------- #
# Tools (agents.tool_loop)
# --------------------------------------------------------------------------- #


class ToolArgs(BaseModel):
    """Tool arguments carried by every tool-loop step; unused ones are empty."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(description="search_repo: the search text; otherwise \"\".")
    path: str = Field(description="read_file: repository-relative path; otherwise \"\".")
    start_line: int = Field(description="read_file: first line (1-based); otherwise 1.")
    end_line: int = Field(description="read_file: last line, 0 = to the end (max 200 lines); otherwise 0.")
    selectors: list[str] = Field(description="run_tests: test ids or test files; otherwise [].")


TOOL_INSTRUCTIONS = """You may use tools before your final answer.  Each reply is ONE step: either a tool call or your final action.
Tools (set "action" to the tool name and fill its arguments; leave the other arguments empty):
- search_repo(query): search commit messages, changelog, docs and docstrings up to the CI commit; results carry
  evidence ids you may cite.
- read_file(path, start_line, end_line): read lines of a repository file (max 200 lines per call).
- run_tests(selectors): run tests (ids or files under the test roots) on the CURRENT code; your edits are NOT
  applied yet.
Every call costs budget; you have a small number of steps.  Do not repeat a call whose result you already have."""


# --------------------------------------------------------------------------- #
# Triage
# --------------------------------------------------------------------------- #


class TriageAssignment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    test_id: str = Field(min_length=1)
    function: str = Field(min_length=1, description="One of the listed changed functions, or 'unknown'.")


class TriageGrouping(BaseModel):
    """Which changed function each failing test exercises."""

    model_config = ConfigDict(extra="forbid")

    assignments: list[TriageAssignment] = Field(description="Exactly one entry per listed test.")


TRIAGE_SYSTEM = f"""\
You are the Triage agent of a CI triage system.  The failing tests have already been run and confirmed.  Your only
job is to say which CHANGED function each failing test exercises, so failures caused by the same change are handled
together and failures caused by different changes are handled separately.

Rules:
1. Assign every listed test to exactly one function from CHANGED FUNCTIONS, written exactly as listed.
2. Use the test's code, failure message and traceback.  A test exercises a function if it calls it directly or
   through the code it calls.
3. If no listed function fits, write "unknown".  Do not guess.
4. Do not judge whether the test or the code is wrong; that is another role's job.

{COMMON_RULES}
"""


def build_triage_user(
    context: EpisodeContext, tests: list[FailingTest], changed_functions: list[str], diff: str | None
) -> str:
    parts = ["CHANGED FUNCTIONS (from the CI commit's source diff):", *[f"- {f}" for f in changed_functions]]
    parts += ["", "FAILING TESTS:"]
    for test in tests:
        parts += [f"- {test.test_id}", "  message: " + clip(test.failure_message, 600),
                  "  traceback:\n" + clip(test.traceback, 1200)]
        body = _test_body(context, test)
        if body:
            parts.append("  test source:\n" + clip(body, 2000))
    parts += ["", "CI SOURCE DIFF:", clip(diff, 5000) or "(none)"]
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# Intent Analyst
# --------------------------------------------------------------------------- #

INTENT_SYSTEM = f"""\
You are the Intent Analyst of a CI triage system.  A commit made existing tests fail.  For ONE group of failing tests,
decide which side is wrong:

- outdated_test: the code change was INTENDED (a deliberate behaviour change, bug fix or API change) and the test
  still encodes the old behaviour.  Recommend UPDATE_TEST.
- regression: the code change broke behaviour that should have been kept.  Recommend FIX_CODE.
- unknown: the evidence does not decide it.  Recommend ESCALATE.

Decision policy:
1. UPDATE_TEST needs at least one cited piece of POSITIVE evidence that the new behaviour was intended (a commit
   message, changelog entry or doc that states the change).  Put its id in cited_evidence_ids.
2. A MEMORY record counts as evidence only because its probe passed on the current code; it covers only the function
   and behaviour it names.  Do not stretch it to a different function or a different symptom.
3. FIX_CODE needs a concrete contradiction: cite the evidence that describes the expected behaviour the change
   breaks (a docstring, doc or changelog entry, an earlier commit message, a CI commit message that is about
   something else, or a memory record).  "Nothing says the change was intended" is NOT evidence for FIX_CODE.
4. If neither side has evidence, or the evidence conflicts, recommend ESCALATE.  A wrong confident answer is worse
   than an escalation.
5. Neither the existing test nor a docstring is automatically right: both often still describe the OLD behaviour.
   Prefer statements of intent made in or after the change; use old descriptions as evidence for FIX_CODE only when
   nothing in or after the change says the behaviour was meant to change.
6. Put ids that argue against your hypothesis in contradicting_evidence_ids, and state the strongest competing
   explanation (the other side) in alternative_explanation, with why you did not choose it.
7. Allowed pairs: outdated_test -> UPDATE_TEST or ESCALATE; regression -> FIX_CODE or ESCALATE; unknown -> ESCALATE.
8. expected_behaviour: one or two sentences on what the code should do now.  confidence: 0 to 1.
9. group_id must be exactly the given group id.  The CI commit message and diff are context, not citable, unless
   their id appears under EVIDENCE.

{COMMON_RULES}
"""


def build_intent_user(
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
        body = _test_body(context, test)
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
        parts += ["", "MEMORY (approved earlier, probe passed on the current code; citable by record id):", *memory_lines]
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# Repairer
# --------------------------------------------------------------------------- #


class SnippetEdit(BaseModel):
    """Replace ``old`` (copied exactly from the file, occurring once) with ``new``."""

    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=1, description="Repository-relative path, e.g. tests/test_x.py")
    old: str = Field(min_length=1, description="Exact existing text, including indentation; must occur once.")
    new: str = Field(description="Replacement text.")


class RepairReply(BaseModel):
    """What the Repairer model returns; turned into ``PatchProposal`` or ``EscalationRequest``."""

    model_config = ConfigDict(extra="forbid")

    action: Literal["patch", "escalate"]
    edits: list[SnippetEdit] = Field(description="At least one edit for action 'patch'; [] for 'escalate'.")
    rationale: str = Field(min_length=1)
    evidence_ids: list[str] = Field(description="Ids the analysis cited; [] if none.")


class RepairStep(ToolArgs):
    """One step of the tool-using Repairer: a tool call or a final patch/escalate (``RepairReply`` fields)."""

    action: Literal["search_repo", "read_file", "run_tests", "patch", "escalate"]
    edits: list[SnippetEdit] = Field(description="Final 'patch': at least one edit; otherwise [].")
    rationale: str = Field(min_length=1, description="Why this step (or the final repair) is right.")
    evidence_ids: list[str] = Field(description="Ids the analysis cited; [] if none.")


REPAIR_SCOPE_RULES = {
    PatchScope.TEST: (
        "UPDATE_TEST: edit ONLY the failing test functions shown under FILES.  Two kinds of change are allowed:\n"
        "  (a) the expected literal value in an existing `assert <expr> == <literal>` or `assert <expr> != <literal>`;\n"
        "  (b) adding a literal keyword argument to, or changing the literal value of an existing keyword argument "
        "of, a setup call in the test body (not inside an assert, not pytest/mock helpers), e.g. "
        "`connect(host)` -> `connect(host, retries=0)` when the intended change moved a default.\n"
        "Never remove keyword arguments; never change positional inputs, the tested expression, assertions' operators, "
        "imports, fixtures or other tests; never add or delete tests, assertions or skips.  Anything else is "
        "rejected automatically.\n"
        "Keep what the test is ABOUT.  Its name, docstring and setup say which behaviour it checks.  Prefer the "
        "change that keeps that meaning: if a literal change would make the test check a different behaviour than "
        "its name says (for example a test named 'case_insensitive_match' that would then assert case-sensitive behaviour), use (b) to "
        "restore the setup the name describes instead.  If neither (a) nor (b) can repair it without changing the "
        "test's meaning, choose action \"escalate\" and explain which change the test really needs."
    ),
    PatchScope.SRC: (
        "FIX_CODE: edit ONLY the source files listed under FILES.  Restore the intended behaviour with the smallest "
        "change, normally by undoing the part of the CI diff that broke it while keeping its intended parts.  Never "
        "edit tests, conftest.py or configuration, and never add pytest imports, skips or process exits."
    ),
}

REPAIR_SYSTEM = f"""\
You are the Repairer of a CI triage system.  The Intent Analyst has decided which side is wrong for one group of
failing tests.  Produce the smallest repair on that side only, as exact snippet replacements, or escalate.

Each edit: "path" (repository-relative), "old" (text copied EXACTLY from the file shown, including indentation, long
enough to occur only once, WITHOUT the line-number prefix), "new" (replacement).

{{scope_rule}}

For action "patch", "edits" MUST contain at least one edit.  If you cannot repair safely within these rules, reply
with action "escalate", "edits": [] and explain why in rationale.
If a previous attempt was rejected, fix exactly what the rejection reasons say; do not resend the same edit.
evidence_ids: copy the ids the analysis cited (required when updating tests; [] if none).

{COMMON_RULES}
"""


def build_repair_user(
    context: EpisodeContext,
    group: FailureGroupDetail,
    analysis: IntentAnalysis,
    files: list[str],
    previous_rejection: GuardResult | None,
    rejection_reasons: tuple[str, ...] = (),
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
    reasons = [*(previous_rejection.reasons if previous_rejection is not None else ()), *rejection_reasons]
    if reasons:
        parts += ["", "YOUR PREVIOUS REPAIR WAS REJECTED:", *[f"- {clip(r, 600)}" for r in reasons]]
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# Integrity Guard
# --------------------------------------------------------------------------- #


class GuardReview(BaseModel):
    """The Guard model's semantic review.  It can only lower a verdict, never approve."""

    model_config = ConfigDict(extra="forbid")

    verdict: Literal["consistent", "suspicious"]
    test_meaning_preserved: bool = Field(description="False if a changed test now checks a different behaviour.")
    reasons: list[str] = Field(description="Concrete reasons; [] only when consistent.")


GUARD_SYSTEM = f"""\
You are the Integrity Guard of a CI triage system: an independent reviewer of a repair another agent made.  The
repair is already applied and the tests pass; automatic checks found no removed assertion, skip or out-of-scope edit.
Passing tests prove little.  Judge whether the repair is LEGITIMATE.

Mark "suspicious" when any of these holds:
1. A changed test now checks a different behaviour than its name, docstring or setup say (test meaning not
   preserved), or the new expected value is not what the stated intent implies.
2. A test was updated although the evidence shown does not say the new behaviour was intended.
3. A source fix does more than restore the intended behaviour, or undoes a part of the CI change that was intended.
4. The repair special-cases the test input instead of fixing the behaviour.
Otherwise mark "consistent".  Give concrete reasons either way when suspicious; you never approve a repair, you can
only raise concerns that send it to a human.

{COMMON_RULES}
"""


def build_guard_user(
    context: EpisodeContext, group: FailureGroupDetail, analysis: IntentAnalysis, proposal: PatchProposal
) -> str:
    commits = [c for c in group.context_commits if c.message]
    parts = [
        f"GROUP ID: {group.group.group_id}    REPAIR: {proposal.decision.value} (scope {proposal.scope.value})",
        f"ANALYST HYPOTHESIS: {analysis.hypothesis.kind.value}: {clip(analysis.hypothesis.explanation, 1200)}",
        f"EXPECTED BEHAVIOUR: {clip(analysis.expected_behaviour, 600) or '(not stated)'}",
        f"CITED EVIDENCE IDS: {', '.join(analysis.cited_evidence_ids) or '(none)'}",
        f"REPAIRER RATIONALE: {clip(proposal.rationale, 800)}",
        "",
        "CI COMMIT MESSAGE:", clip(commits[0].message, 1200) if commits else "(not available)",
        "",
        "APPLIED REPAIR (unified diff):", clip(proposal.unified_diff, 4000),
        "",
        "FAILING TESTS (before the repair), with their current source:",
    ]
    for test in group.failing_tests:
        parts += [f"- {test.test_id}", "  failure was: " + clip(test.failure_message, 500)]
        body = _test_body(context, test)
        if body:
            parts.append("  test source now:\n" + clip(body, 2500))
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# B1: single tool-using agent (baseline)
# --------------------------------------------------------------------------- #


class B1Decision(BaseModel):
    """One decision of the single agent, covering one or more failing tests that share it."""

    model_config = ConfigDict(extra="forbid")

    test_ids: list[str] = Field(min_length=1)
    decision: Literal["UPDATE_TEST", "FIX_CODE", "ESCALATE"]
    hypothesis: Literal["outdated_test", "regression", "unknown"]
    explanation: str = Field(min_length=1)
    cited_evidence_ids: list[str] = Field(description="Evidence ids from search results; required for UPDATE_TEST.")
    confidence: float = Field(ge=0.0, le=1.0)
    edits: list[SnippetEdit] = Field(description="The repair for these tests; [] for ESCALATE.")


class B1Step(ToolArgs):
    """One step of the single agent: a tool call or ``finish`` with the decisions."""

    action: Literal["search_repo", "read_file", "run_tests", "finish"]
    decisions: list[B1Decision] = Field(description="With 'finish': every failing test in exactly one item; else [].")
    rationale: str = Field(min_length=1)


B1_SYSTEM = f"""\
You are a coding agent working on a red CI run of a Python repository.  For the failing tests, decide whether each
test is outdated (the code change was intended: UPDATE_TEST), the code regressed (FIX_CODE), or the evidence does not
decide it (ESCALATE), and repair it.

Rules:
1. Finish with "decisions": items that each cover one or more failing test ids sharing the same decision and the same
   repair.  Every failing test must appear in exactly one item.
2. UPDATE_TEST: hypothesis "outdated_test", at least one cited evidence id showing the change was intended, edits only
   in the failing test functions.  FIX_CODE: hypothesis "regression", at least one cited evidence id describing the
   expected behaviour the change breaks (not merely the absence of intent evidence), edits only in source files.
   ESCALATE: when neither side has evidence or it conflicts; hypothesis "unknown" (or your best guess), no edits.
   Test and source edits never go in the same item.
3. Test edits may only change (a) the expected literal value in an existing `assert <expr> ==/!= <literal>`, or
   (b) literal keyword arguments of setup calls in the failing test functions (add or change, never remove).  Source
   edits: the smallest change.  Anything else is rejected automatically.
4. Edits are exact snippet replacements: "path", "old" (copied exactly from the file, occurring once, without line
   numbers), "new".

{{tool_instructions}}
Final action: "finish" with the decisions.

{COMMON_RULES}
"""


def build_b1_user(
    context: EpisodeContext,
    tests: list[FailingTest],
    commit_message: str | None,
    diff: str | None,
    search_results: str,
    rejection_reasons: tuple[str, ...] = (),
) -> str:
    parts = [f"REPOSITORY: {context.repo.repo_id}    CI COMMIT: {context.commit_sha[:12]}", "", "FAILING TESTS:"]
    for test in tests:
        parts += [f"- {test.test_id}", "  message: " + clip(test.failure_message, 800),
                  "  traceback:\n" + clip(test.traceback, 1200)]
        body = _test_body(context, test)
        if body:
            parts.append("  test source:\n" + clip(body, 2500))
    parts += ["", "CI COMMIT MESSAGE:", clip(commit_message, 1500) or "(not available)"]
    parts += ["", "CI SOURCE DIFF:", clip(diff, 6000) or "(none)"]
    for path, file_diff in split_diff_by_file(diff or "").items():
        text = read_worktree_file(context.worktree_path, path)
        if text is not None:
            parts += [f"--- {path} (current, around the change) ---", changed_windows(text, file_diff, limit=6000)]
    parts += ["", "SEARCH RESULTS for the failing tests (cite only these ids, or ids from your own searches):",
              search_results or "(nothing found)"]
    if rejection_reasons:
        parts += ["", "YOUR PREVIOUS REPAIR WAS REJECTED:", *[f"- {clip(r, 600)}" for r in rejection_reasons]]
    return "\n".join(parts)


def _test_body(context: EpisodeContext, test: FailingTest) -> str | None:
    source = read_worktree_file(context.worktree_path, test.test_file)
    return function_source(source, test.test_name.split("[", 1)[0]) if source else None
