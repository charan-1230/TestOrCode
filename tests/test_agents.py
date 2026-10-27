"""The four agents, offline: ``FakeLLMClient`` for the model, a temporary git repository for files and
patches, and a scripted ``run_tests`` (pytest execution itself is covered by ``test_pytest_runner``)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from testorcode.agents import integrity_guard, intent_analyst, repairer, triage
from testorcode.agents._common import split_diff_by_file
from testorcode.config import Settings
from testorcode.errors import MemoryAuthorizationError
from testorcode.llm_client import FakeLLMClient, create_llm_client
from testorcode.retrieval import build_retriever
from testorcode.schemas.contracts import (
    EpisodeContext,
    EscalationRequest,
    FailureGroupDetail,
    GuardResult,
    Hypothesis,
    IntentAnalysis,
    PatchProposal,
    PytestExecution,
)
from testorcode.schemas.episode_schemas import (
    Availability,
    DecisionLabel,
    FailingTest,
    FailureGroup,
    HypothesisKind,
    PatchScope,
    RepairDirection,
)
from testorcode.schemas.result_schemas import (
    AgentRole,
    GuardDecision,
    IntegrityReport,
    TestRunStatus,
)
from testorcode.schemas.tests.conftest import make_s1, memory_record
from testorcode.tools.patch_tools import apply_patch, revert_patch, validate_patch

S1 = make_s1().agent_input
TEST_ID = S1.ci_event.failing_tests[0].test_id  # tests/test_synthlib.py::test_normalise
SETTINGS = Settings.from_env({"LLM_MODEL": "fake-model-1"})

CORE = "def normalise(value):\n    return value.strip().upper()\n\n\ndef other():\n    return 1\n"
# The assertion is followed by a blank line and then more code, so the generated hunk ends on blank context.
TESTS = (
    "from synthlib.core import normalise\n\n\n"
    "def test_normalise():\n    assert normalise(' a ') == 'a'\n\n    return\n\n\n"
    "def test_other():\n    assert 1 == 1\n"
)


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    for rel, text in {"src/synthlib/core.py": CORE, "tests/test_synthlib.py": TESTS}.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)
    git(root, "init", "-q")
    git(root, "add", ".")
    git(root, "-c", "user.name=t", "-c", "user.email=t@example.invalid", "commit", "-qm", "init")
    return root


def execution(status: TestRunStatus, failing: tuple[str, ...] = (), selectors=("tests",)) -> PytestExecution:
    failures = tuple(
        FailingTest(test_id=t, test_file=t.split("::")[0], test_name=t.split("::")[1], failure_message="assert 'A' == 'a'")
        for t in failing
    )
    return PytestExecution(
        selectors=tuple(selectors), command=("python", "-m", "pytest"), status=status,
        exit_code=0 if status is TestRunStatus.PASSED else 1,
        passed=0 if failing else 2, failed=len(failing), errors=0, skipped=0, failures=failures,
    )


def group_detail(repo_diff: str | None = None) -> FailureGroupDetail:
    tests = tuple(S1.ci_event.failing_tests)
    return FailureGroupDetail(
        group=FailureGroup(group_id="g1", function_qualname="synthlib.core.normalise", test_ids=[TEST_ID]),
        failing_tests=tests,
        affected_files=("src/synthlib/core.py", "tests/test_synthlib.py"),
        diff_context=repo_diff,
    )


def analysis(decision: DecisionLabel, kind: HypothesisKind, cited=("ev-msg",)) -> IntentAnalysis:
    return IntentAnalysis(
        group_id="g1", hypothesis=Hypothesis(kind=kind, explanation="why"), cited_evidence_ids=cited,
        confidence=0.9, recommended_decision=decision,
    )


def core_diff() -> str:
    return (
        "--- a/src/synthlib/core.py\n+++ b/src/synthlib/core.py\n@@ -1,2 +1,2 @@\n"
        " def normalise(value):\n-    return value.strip()\n+    return value.strip().upper()\n"
    )


# --------------------------------------------------------------------------- #
# Triage
# --------------------------------------------------------------------------- #


def test_triage_groups_reproduced_failures_and_marks_the_rest_unstable(monkeypatch, repo: Path) -> None:
    extra = S1.ci_event.failing_tests[0].model_copy(
        update={"test_id": "tests/test_synthlib.py::test_other", "test_name": "test_other", "target_function": None}
    )
    event = S1.ci_event.model_copy(update={"failing_tests": [*S1.ci_event.failing_tests, extra]})
    calls = []

    def fake_run(worktree, selectors, *, timeout_seconds, python_executable=None, **_):
        calls.append((list(selectors), python_executable))
        return execution(TestRunStatus.FAILED, (TEST_ID,), selectors)

    monkeypatch.setattr(triage, "run_tests", fake_run)
    settings = Settings.from_env({"LLM_MODEL": "m", "TEST_PYTHON": "/venv/bin/python"})
    result = triage.run_triage(EpisodeContext.from_episode_input(S1, repo), event, llm=FakeLLMClient(), settings=settings)

    assert calls == [([TEST_ID, extra.test_id], Path("/venv/bin/python"))]
    [g] = result.groups
    assert g.group.test_ids == [TEST_ID] and g.group.function_qualname == "synthlib.core.normalise"
    assert result.unstable_test_ids == (extra.test_id,)
    assert g.diff_context and "capitalize" in g.diff_context  # the source diff
    assert g.affected_files == ("src/synthlib/core.py", "tests/test_synthlib.py")
    assert g.context_commits and g.context_commits[0].message == S1.ci_event.commit_message


def test_triage_hides_a_withheld_commit_message(monkeypatch, repo: Path) -> None:
    event = S1.ci_event.model_copy(update={"commit_message": None, "commit_message_availability": Availability.WITHHELD})
    monkeypatch.setattr(triage, "run_tests", lambda *a, **k: execution(TestRunStatus.FAILED, (TEST_ID,)))
    result = triage.run_triage(EpisodeContext.from_episode_input(S1, repo), event, llm=FakeLLMClient(), settings=SETTINGS)
    assert result.groups[0].context_commits == ()


# --------------------------------------------------------------------------- #
# Intent Analyst
# --------------------------------------------------------------------------- #


def test_intent_analyst_sends_retrieved_ids_and_returns_the_validated_reply(repo: Path) -> None:
    reply = analysis(DecisionLabel.UPDATE_TEST, HypothesisKind.OUTDATED_TEST)
    fake = FakeLLMClient({AgentRole.INTENT_ANALYST: [reply.model_dump(mode="json")]})
    out = intent_analyst.analyse_intent(
        EpisodeContext.from_episode_input(S1, repo), group_detail(core_diff()),
        retriever=build_retriever(S1.corpus), memory=None, llm=fake,
    )
    assert out == reply
    [call] = fake.calls
    assert call["response_model"] is IntentAnalysis and call["role"] is AgentRole.INTENT_ANALYST
    assert "[ev-msg]" in call["user"] and "def test_normalise" in call["user"]
    assert "gold" not in call["user"].lower()


def test_intent_analyst_reply_that_breaks_the_contract_is_rejected(repo: Path) -> None:
    bad = analysis(DecisionLabel.UPDATE_TEST, HypothesisKind.OUTDATED_TEST).model_dump(mode="json")
    bad["cited_evidence_ids"] = []  # UPDATE_TEST without evidence
    fake = FakeLLMClient({AgentRole.INTENT_ANALYST: [bad]})
    with pytest.raises(Exception, match="UPDATE_TEST requires"):
        intent_analyst.analyse_intent(
            EpisodeContext.from_episode_input(S1, repo), group_detail(), retriever=build_retriever(S1.corpus),
            memory=None, llm=fake,
        )


def test_intent_queries_use_function_diff_and_commit_message() -> None:
    queries = intent_analyst.group_queries(group_detail(core_diff()))
    assert any("normalise" in q for q in queries) and any("upper" in q for q in queries)


# --------------------------------------------------------------------------- #
# Repairer
# --------------------------------------------------------------------------- #


def test_test_repair_diff_survives_contract_stripping_and_applies(repo: Path) -> None:
    edit = {"path": "tests/test_synthlib.py", "old": "== 'a'", "new": "== 'A'"}
    fake = FakeLLMClient({AgentRole.REPAIRER: [
        {"action": "patch", "edits": [edit], "rationale": "intended upper-casing", "evidence_ids": ["ev-msg"]}
    ]})
    context = EpisodeContext.from_episode_input(S1, repo)
    out = repairer.propose_repair(
        context, group_detail(), analysis(DecisionLabel.UPDATE_TEST, HypothesisKind.OUTDATED_TEST), llm=fake
    )
    assert isinstance(out, PatchProposal)
    assert out.scope is PatchScope.TEST and out.target_files == ("tests/test_synthlib.py",)
    assert out.evidence_ids == ("ev-msg",)
    # The proposal's diff is whitespace-stripped by the contract; it must still validate, apply and revert.
    assert validate_patch(repo, out.unified_diff, policy=context.edit_policy, scope=out.scope).ok
    apply_patch(repo, out.unified_diff, policy=context.edit_policy, scope=out.scope)
    assert "== 'A'" in (repo / "tests/test_synthlib.py").read_text()
    revert_patch(repo, out.unified_diff, policy=context.edit_policy)
    assert (repo / "tests/test_synthlib.py").read_text() == TESTS


def test_source_repair_edits_only_files_of_the_ci_diff(repo: Path) -> None:
    edit = {"path": "src/synthlib/core.py", "old": "value.strip().upper()", "new": "value.strip()"}
    fake = FakeLLMClient({AgentRole.REPAIRER: [{"action": "patch", "edits": [edit], "rationale": "revert regression", "evidence_ids": []}]})
    out = repairer.propose_repair(
        EpisodeContext.from_episode_input(S1, repo), group_detail(core_diff()),
        analysis(DecisionLabel.FIX_CODE, HypothesisKind.REGRESSION, cited=()), llm=fake,
    )
    assert isinstance(out, PatchProposal) and out.decision is RepairDirection.FIX_CODE and out.scope is PatchScope.SRC
    assert "src/synthlib/core.py" in fake.calls[0]["user"]


def test_unplaceable_edits_are_retried_once_then_escalated(repo: Path) -> None:
    wrong = {"action": "patch", "edits": [{"path": "tests/test_synthlib.py", "old": "not in file", "new": "x"}],
             "rationale": "r", "evidence_ids": ["ev-msg"]}
    outside = {**wrong, "edits": [{"path": "src/synthlib/core.py", "old": "upper", "new": "lower"}]}
    fake = FakeLLMClient({AgentRole.REPAIRER: [wrong, outside]})
    out = repairer.propose_repair(
        EpisodeContext.from_episode_input(S1, repo), group_detail(),
        analysis(DecisionLabel.UPDATE_TEST, HypothesisKind.OUTDATED_TEST), llm=fake,
    )
    assert isinstance(out, EscalationRequest) and "not one of the editable files" in out.summary
    assert "occurs 0 times" in fake.calls[1]["user"]


def test_model_escalation_becomes_an_escalation_request(repo: Path) -> None:
    fake = FakeLLMClient({AgentRole.REPAIRER: [{"action": "escalate", "edits": [], "rationale": "needs a new assertion", "evidence_ids": []}]})
    out = repairer.propose_repair(
        EpisodeContext.from_episode_input(S1, repo), group_detail(),
        analysis(DecisionLabel.UPDATE_TEST, HypothesisKind.OUTDATED_TEST), llm=fake,
    )
    assert out == EscalationRequest(group_id="g1", trigger="insufficient_evidence", summary="needs a new assertion")


def test_trailing_blank_context_is_trimmed_with_header_counts() -> None:
    diff = "--- a/f.py\n+++ b/f.py\n@@ -1,4 +1,4 @@\n a\n-b\n+B\n \n \n"
    assert repairer._trim_trailing_blank_context(diff) == "--- a/f.py\n+++ b/f.py\n@@ -1,2 +1,2 @@\n a\n-b\n+B\n"


def test_split_diff_by_file() -> None:
    two = core_diff() + "--- a/src/synthlib/other.py\n+++ b/src/synthlib/other.py\n@@ -1 +1 @@\n-x\n+y\n"
    assert list(split_diff_by_file(two)) == ["src/synthlib/core.py", "src/synthlib/other.py"]


# --------------------------------------------------------------------------- #
# Integrity Guard
# --------------------------------------------------------------------------- #


def applied_test_patch(repo: Path, new_tests: str) -> PatchProposal:
    (repo / "tests/test_synthlib.py").write_text(new_tests)
    return PatchProposal(
        group_id="g1", decision=RepairDirection.UPDATE_TEST, scope=PatchScope.TEST,
        target_files=("tests/test_synthlib.py",), unified_diff=git(repo, "diff"), rationale="r", evidence_ids=("ev-msg",),
    )


def verify(monkeypatch, repo: Path, proposal: PatchProposal, status: TestRunStatus) -> GuardResult:
    failing = () if status is TestRunStatus.PASSED else (TEST_ID,)
    monkeypatch.setattr(integrity_guard, "run_tests", lambda *a, **k: execution(status, failing))
    return integrity_guard.verify_repair(
        EpisodeContext.from_episode_input(S1, repo), group_detail(),
        analysis(DecisionLabel.UPDATE_TEST, HypothesisKind.OUTDATED_TEST), proposal,
    )


def test_passing_rerun_without_probe_escalates_never_approves(monkeypatch, repo: Path) -> None:
    result = verify(monkeypatch, repo, applied_test_patch(repo, TESTS.replace("== 'a'", "== 'A'")), TestRunStatus.PASSED)
    assert result.decision is GuardDecision.ESCALATE
    assert "human verification required" in result.reasons[0]
    report = result.integrity
    assert report.weakening.weakening_detected is False and report.weakening.assertions_before == 2
    assert report.prohibited_edits.out_of_scope_files == [] and report.evidence_requirement_met is True


def test_failing_rerun_is_rejected(monkeypatch, repo: Path) -> None:
    result = verify(monkeypatch, repo, applied_test_patch(repo, TESTS.replace("== 'a'", "== 'b'")), TestRunStatus.FAILED)
    assert result.decision is GuardDecision.REJECT and "post-repair run" in result.reasons[0]
    assert result.post_repair_run is not None and result.post_repair_run.status is TestRunStatus.FAILED


def test_removed_assertion_is_weakening(monkeypatch, repo: Path) -> None:
    weakened = TESTS.replace("    assert normalise(' a ') == 'a'\n", "    pass\n")
    result = verify(monkeypatch, repo, applied_test_patch(repo, weakened), TestRunStatus.PASSED)
    assert result.decision is GuardDecision.REJECT
    assert result.integrity.weakening.removed_assertions == 1


def test_memory_write_needs_approve_and_maintainer() -> None:
    report = IntegrityReport(guard_ran=True, guard_decision=GuardDecision.ESCALATE)
    escalated = GuardResult(decision=GuardDecision.ESCALATE, reasons=("x",), integrity=report)
    with pytest.raises(MemoryAuthorizationError):
        integrity_guard.authorize_memory_write(escalated, memory_record(), maintainer_approved=True)


# --------------------------------------------------------------------------- #
# Settings used by the live path
# --------------------------------------------------------------------------- #


def test_live_settings_reach_the_client() -> None:
    settings = Settings.from_env({
        "LLM_MODEL": "m", "LLM_API_KEY": "k", "LLM_STRUCTURED_MODE": "json_object",
        "LLM_MAX_RETRIES": "6", "LLM_TIMEOUT_SECONDS": "90",
    })
    client = create_llm_client(settings)
    assert (client._mode, client._max_retries, client._timeout) == ("json_object", 6, 90.0)
    with pytest.raises(Exception, match="LLM_STRUCTURED_MODE"):
        Settings.from_env({"LLM_STRUCTURED_MODE": "xml"})
