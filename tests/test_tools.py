"""Model-callable tools (``agents.tools``) and the bounded tool loop (``agents.tool_loop``)."""

from __future__ import annotations

from pathlib import Path

import pytest

from testorcode.agents import tools as agent_tools
from testorcode.agents.prompts import RepairStep
from testorcode.agents.tool_loop import run_tool_loop
from testorcode.agents.tools import AgentTools
from testorcode.llm_client import FakeLLMClient
from testorcode.schemas.contracts import EpisodeContext, PytestExecution
from testorcode.schemas.episode_schemas import EvidenceSourceType, FailingTest, ToolName
from testorcode.schemas.result_schemas import (
    AgentRole,
    RetrievedEvidence,
    TestRunStatus,
    ToolCallStatus,
)
from testorcode.schemas.tests.conftest import HEAD, make_s1
from testorcode.tools.trace import ToolRecorder, recording

S1 = make_s1().agent_input
NO_TOOL = {"query": "", "path": "", "start_line": 1, "end_line": 0, "selectors": []}


class Searcher:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def search(self, query: str, *, top_k: int) -> list[RetrievedEvidence]:
        self.queries.append(query)
        return [RetrievedEvidence(evidence_id="ev-msg", source_type=EvidenceSourceType.COMMIT_MESSAGE,
                                  source_ref=HEAD, snippet="normalise now capitalises")]


@pytest.fixture
def worktree(tmp_path: Path) -> Path:
    (tmp_path / "src/synthlib").mkdir(parents=True)
    (tmp_path / "src/synthlib/core.py").write_text("".join(f"line {i}\n" for i in range(1, 301)))
    (tmp_path / ".env").write_text("LLM_API_KEY=secret\n")
    return tmp_path


def make_tools(worktree: Path, allowed=None) -> tuple[AgentTools, Searcher]:
    context = EpisodeContext.from_episode_input(S1, worktree)
    if allowed is not None:
        context = context.model_copy(update={"allowed_tools": tuple(allowed)})
    searcher = Searcher()
    return AgentTools(context, retriever=searcher, timeout_seconds=30), searcher


def test_read_file_returns_numbered_bounded_lines(worktree: Path) -> None:
    tools, _ = make_tools(worktree)
    recorder = ToolRecorder()
    with recording(recorder):
        text = tools.read_file("src/synthlib/core.py", 5, 0)
    assert "    5| line 5" in text and "  204| line 204" in text and "line 205" not in text
    assert "(96 more lines)" in text
    [event] = recorder.events
    assert event.tool is ToolName.READ_FILE and event.status is ToolCallStatus.OK


@pytest.mark.parametrize("path", ["../secret", "/etc/passwd", ".git/config", ".env", "src/../.env", ""])
def test_read_file_refuses_unsafe_paths(worktree: Path, path: str) -> None:
    tools, _ = make_tools(worktree)
    recorder = ToolRecorder()
    with recording(recorder):
        text = tools.read_file(path)
    assert text.startswith("INVALID") and "secret" not in text.lower().replace("../secret", "")
    assert recorder.events[0].status is ToolCallStatus.INVALID_ARGUMENTS


def test_tool_not_in_allowed_tools_is_refused_and_recorded(worktree: Path) -> None:
    tools, searcher = make_tools(worktree, allowed=[ToolName.RUN_TESTS])
    recorder = ToolRecorder()
    with recording(recorder):
        assert tools.search_repo("normalise").startswith("REFUSED")
        assert tools.read_file("src/synthlib/core.py").startswith("REFUSED")
    assert searcher.queries == []
    assert [(e.tool, e.status) for e in recorder.events] == [
        (ToolName.SEARCH_REPO, ToolCallStatus.NOT_PERMITTED), (ToolName.READ_FILE, ToolCallStatus.NOT_PERMITTED),
    ]


def test_search_repo_lists_citable_ids(worktree: Path) -> None:
    tools, searcher = make_tools(worktree)
    assert "[ev-msg]" in tools.search_repo("why capitalise")
    assert searcher.queries == ["why capitalise"]


@pytest.mark.parametrize("selectors", [[], ["src/synthlib/core.py"], ["tests/../src/x.py"], ["/abs/tests/x.py"]])
def test_run_tests_only_accepts_test_selectors(worktree: Path, monkeypatch, selectors) -> None:
    monkeypatch.setattr(agent_tools, "run_tests", lambda *a, **k: pytest.fail("must not run"))
    tools, _ = make_tools(worktree)
    recorder = ToolRecorder()
    with recording(recorder):
        assert tools.run_tests(selectors).startswith("INVALID")
    assert recorder.events[0].status is ToolCallStatus.INVALID_ARGUMENTS


def test_run_tests_summarises_failures(worktree: Path, monkeypatch) -> None:
    failure = FailingTest(test_id="tests/test_synthlib.py::test_normalise", test_file="tests/test_synthlib.py",
                          test_name="test_normalise", failure_message="assert 'A' == 'a'")
    result = PytestExecution(selectors=("tests",), command=("pytest",), status=TestRunStatus.FAILED, exit_code=1,
                             passed=1, failed=1, errors=0, failures=(failure,))
    seen = []
    monkeypatch.setattr(agent_tools, "run_tests", lambda root, sel, **k: seen.append(sel) or result)
    tools, _ = make_tools(worktree)
    text = tools.run_tests(["tests/test_synthlib.py"])
    assert seen == [["tests/test_synthlib.py"]]
    assert "status=failed passed=1 failed=1" in text and "assert 'A' == 'a'" in text


def final(**kw) -> dict:
    return {"action": "patch", "edits": [], "rationale": "done", "evidence_ids": [], **NO_TOOL, **kw}


def test_loop_runs_tools_then_returns_the_final_step(worktree: Path) -> None:
    tools, _ = make_tools(worktree)
    llm = FakeLLMClient({AgentRole.REPAIRER: [
        final(action="read_file", path="src/synthlib/core.py", start_line=1, end_line=2), final(),
    ]})
    step = run_tool_loop(llm, role=AgentRole.REPAIRER, system="s", user="u", step_model=RepairStep,
                         tools=tools, max_steps=3)
    assert step.action == "patch"
    assert "read_file(path='src/synthlib/core.py'" in llm.calls[1]["user"] and "line 2" in llm.calls[1]["user"]
    assert "STEP 2 of 3" in llm.calls[1]["user"]


def test_loop_gives_up_when_the_last_step_is_still_a_tool_call(worktree: Path) -> None:
    tools, searcher = make_tools(worktree)
    llm = FakeLLMClient({AgentRole.REPAIRER: [final(action="search_repo", query="x")] * 2})
    step = run_tool_loop(llm, role=AgentRole.REPAIRER, system="s", user="u", step_model=RepairStep,
                         tools=tools, max_steps=2)
    assert step is None and searcher.queries == ["x"]  # the last step's tool call is not executed
    assert "This is your LAST step" in llm.calls[1]["user"]


def test_read_file_refuses_symlinks_to_protected_or_outside_files(worktree: Path, tmp_path_factory) -> None:
    outside = tmp_path_factory.mktemp("outside") / "secret.txt"
    outside.write_text("outside secret\n")
    (worktree / "src/notes.txt").symlink_to(worktree / ".env")  # innocent name, points at .env inside the worktree
    (worktree / "src/escape.txt").symlink_to(outside)  # points outside the worktree
    tools, _ = make_tools(worktree)
    recorder = ToolRecorder()
    with recording(recorder):
        via_name = tools.read_file("src/notes.txt")
        escape = tools.read_file("src/escape.txt")
    assert via_name.startswith("INVALID") and "LLM_API_KEY" not in via_name
    assert escape.startswith("INVALID") and "outside secret" not in escape
    assert [e.status for e in recorder.events] == [ToolCallStatus.INVALID_ARGUMENTS] * 2
