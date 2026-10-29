"""The four role prompts: shared rules, decision policy, rendering, and no gold-label leakage."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from testorcode.agents import prompts
from testorcode.schemas.contracts import (
    EpisodeContext,
    FailureGroupDetail,
    GuardResult,
    Hypothesis,
    IntentAnalysis,
    PatchProposal,
)
from testorcode.schemas.episode_schemas import (
    DecisionLabel,
    EvidenceSourceType,
    FailureGroup,
    HypothesisKind,
    PatchScope,
    RepairDirection,
)
from testorcode.schemas.result_schemas import GuardDecision, IntegrityReport, RetrievedEvidence
from testorcode.schemas.tests.conftest import HEAD, make_s1

EPISODE = make_s1()
S1 = EPISODE.agent_input
TESTS = "def test_normalise():\n    assert normalise(' a ') == 'a'\n"
SYSTEMS = {
    "triage": prompts.TRIAGE_SYSTEM,
    "intent": prompts.INTENT_SYSTEM,
    "repair-test": prompts.REPAIR_SYSTEM.format(scope_rule=prompts.REPAIR_SCOPE_RULES[PatchScope.TEST]),
    "repair-src": prompts.REPAIR_SYSTEM.format(scope_rule=prompts.REPAIR_SCOPE_RULES[PatchScope.SRC]),
    "guard": prompts.GUARD_SYSTEM,
    "b1": prompts.B1_SYSTEM.format(tool_instructions=prompts.TOOL_INSTRUCTIONS),
}


@pytest.fixture
def context(tmp_path: Path) -> EpisodeContext:
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests/test_synthlib.py").write_text(TESTS)
    subprocess.run(["git", "-C", str(tmp_path), "init", "-q"], check=True)
    return EpisodeContext.from_episode_input(S1, tmp_path)


def group() -> FailureGroupDetail:
    tests = tuple(S1.ci_event.failing_tests)
    return FailureGroupDetail(
        group=FailureGroup(group_id="g1", function_qualname="synthlib.core.normalise", test_ids=[t.test_id for t in tests]),
        failing_tests=tests,
        diff_context="--- a/src/synthlib/core.py\n+++ b/src/synthlib/core.py\n@@ -1 +1 @@\n-x\n+y\n",
    )


def analysis() -> IntentAnalysis:
    return IntentAnalysis(
        group_id="g1", hypothesis=Hypothesis(kind=HypothesisKind.OUTDATED_TEST, explanation="intended"),
        cited_evidence_ids=("ev-msg",), confidence=0.8, recommended_decision=DecisionLabel.UPDATE_TEST,
    )


def test_version_names_all_four_roles() -> None:
    assert prompts.PROMPT_VERSION.count("+") == 3
    for role in ("triage", "intent", "repair", "guard"):
        assert role in prompts.PROMPT_VERSION


@pytest.mark.parametrize("name", list(SYSTEMS))
def test_every_role_carries_the_shared_rules(name: str) -> None:
    text = SYSTEMS[name]
    assert prompts.COMMON_RULES in text
    assert "NOT the goal" in text and "never invent ids" in text
    assert "skip/xfail" in text and "conftest.py" in text  # proposal 11.1 unsafe actions
    assert "{scope_rule}" not in text and "{{" not in text  # template fully filled


def test_intent_prompt_states_the_decision_policy() -> None:
    text = prompts.INTENT_SYSTEM
    assert "POSITIVE evidence" in text  # UPDATE_TEST needs positive evidence of intent
    assert "probe passed" in text  # memory counts only with a passing probe
    assert "FIX_CODE needs a concrete contradiction" in text
    assert "is NOT evidence for FIX_CODE" in text
    assert "or when nothing" not in text  # the old clause that allowed FIX_CODE without evidence is gone
    assert "recommend ESCALATE" in text
    assert "Neither the existing test nor a docstring is automatically right" in text


def test_repair_test_scope_keeps_test_meaning() -> None:
    rule = prompts.REPAIR_SCOPE_RULES[PatchScope.TEST]
    assert "Keep what the test is ABOUT" in rule and "escalate" in rule


def test_intent_user_lists_evidence_and_source_but_no_gold(context: EpisodeContext) -> None:
    evidence = [RetrievedEvidence(evidence_id="ev-msg", source_type=EvidenceSourceType.COMMIT_MESSAGE,
                                  source_ref=HEAD, snippet="normalise now upper-cases")]
    text = prompts.build_intent_user(context, group(), evidence, ["[mem-1] approved memory"])
    assert "[ev-msg]" in text and "def test_normalise" in text and "[mem-1]" in text
    # ev-changelog is a gold evidence id of this episode that was NOT retrieved: it must not appear.
    assert "ev-changelog" in EPISODE.ground_truth.per_test_labels[0].gold_evidence_ids
    assert "ev-changelog" not in text
    assert "gold" not in text.lower() and "expected_action" not in text and "reference_repair" not in text


def test_repair_user_includes_both_kinds_of_rejection(context: EpisodeContext) -> None:
    report = IntegrityReport(guard_ran=True, guard_decision=GuardDecision.REJECT, rejection_reasons=["weakened"])
    rejected = GuardResult(decision=GuardDecision.REJECT, reasons=("weakened",), integrity=report)
    text = prompts.build_repair_user(context, group(), analysis(), ["tests/test_synthlib.py"], rejected,
                                     ("hunk is truncated",))
    assert "YOUR PREVIOUS REPAIR WAS REJECTED" in text
    assert "- weakened" in text and "- hunk is truncated" in text
    assert "def test_normalise" in text


def test_triage_user_lists_changed_functions(context: EpisodeContext) -> None:
    text = prompts.build_triage_user(context, list(S1.ci_event.failing_tests), ["src.synthlib.core.normalise"], None)
    assert "- src.synthlib.core.normalise" in text and S1.ci_event.failing_tests[0].test_id in text


def test_guard_user_shows_the_patch_and_current_test(context: EpisodeContext) -> None:
    patch = PatchProposal(
        group_id="g1", decision=RepairDirection.UPDATE_TEST, scope=PatchScope.TEST,
        target_files=("tests/test_synthlib.py",), unified_diff="--- a/t\n+++ b/t\n@@ -1 +1 @@\n-a\n+b",
        rationale="intended", evidence_ids=("ev-msg",),
    )
    text = prompts.build_guard_user(context, group(), analysis(), patch)
    assert "APPLIED REPAIR" in text and "+b" in text and "def test_normalise" in text


def test_response_models_require_every_field() -> None:
    for model in (prompts.RepairReply, prompts.RepairStep, prompts.TriageGrouping, prompts.GuardReview,
                  prompts.B1Step, prompts.B1Decision):
        schema = model.model_json_schema()
        assert set(schema["required"]) == set(schema["properties"]), model.__name__



def test_b1_user_shows_standard_rag_results_but_no_gold(context: EpisodeContext) -> None:
    text = prompts.build_b1_user(
        context, list(S1.ci_event.failing_tests), S1.ci_event.commit_message, group().diff_context,
        "[ev-msg] (commit_message) normalise now upper-cases", ("hunk is truncated",),
    )
    assert "[ev-msg]" in text and "def test_normalise" in text and "- hunk is truncated" in text
    assert "ev-changelog" not in text and "gold" not in text.lower() and "expected_action" not in text


def test_b1_and_tool_prompts_name_the_tools_and_final_action() -> None:
    b1 = SYSTEMS["b1"]
    for tool in ("search_repo(query)", "read_file(path", "run_tests(selectors)"):
        assert tool in b1 and tool in prompts.TOOL_INSTRUCTIONS
    assert 'Final action: "finish"' in b1 and "exactly one item" in b1
    assert prompts.B1_VERSION == "b1-v2"
    assert "not merely the absence of intent evidence" in b1


def test_tool_instructions_omit_search_only_when_it_is_not_allowed() -> None:
    from testorcode.schemas.episode_schemas import ToolName

    assert prompts.tool_instructions(list(ToolName)) == prompts.TOOL_INSTRUCTIONS  # FULL/A2/B1: byte-identical
    no_search = prompts.tool_instructions([ToolName.READ_FILE, ToolName.RUN_TESTS])  # A1
    assert "search_repo" not in no_search and "read_file(" in no_search and "run_tests(" in no_search
    a1 = prompts.tool_instructions([ToolName.READ_FILE, ToolName.RUN_TESTS], code_only=True)
    assert "read lines of a source or test file" in a1 and "changelog files are not readable" in a1
    assert "read lines of a repository file" not in a1
