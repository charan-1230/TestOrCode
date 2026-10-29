"""Integrity Guard across several groups of one episode (A2): an earlier approved repair must not be blamed on a later
group, and a later patch must not be allowed to alter an earlier group's tests.

Real orchestrator, real agents, real git, real pytest (``trusted-host``: a trusted fixture repository), real patch
tools and probe runs; only the model replies are scripted.  Approval needs a trusted probe (production policy, kept),
so this test harness supplies one probe per function, like a dataset would.
"""

from __future__ import annotations

import subprocess
from functools import partial
from pathlib import Path

import pytest

from testorcode import orchestrator
from testorcode.agents import integrity_guard, triage
from testorcode.agents import tools as agent_tools
from testorcode.config import Settings
from testorcode.llm_client import FakeLLMClient
from testorcode.orchestrator import Agents, run_episode
from testorcode.schemas.episode_schemas import (
    DecisionLabel,
    EvidenceItem,
    EvidenceSourceType,
    FailingTest,
)
from testorcode.schemas.result_schemas import (
    AgentRole,
    GuardDecision,
    PatchApplicationStatus,
    TestRunStatus,
)
from testorcode.schemas.tests.conftest import make_s1
from testorcode.tools.mutation import mutation_check
from testorcode.tools.pytest_runner import run_tests
from testorcode.tools.verification import ProbeCase, TrustedProbe, verify_behaviour

CORE = (
    "def normalise(s):\n    return s.capitalize()\n\n\n"  # regression: should lower-case (FIX_CODE)
    "def twice(x):\n    return x * 3\n\n\n"  # intended change: now triples (UPDATE_TEST)
    "def label():\n    return 'v2'\n\n\n"  # intended change: renamed (UPDATE_TEST)
    "def shout(s):\n    return s.upper() + '!'\n"  # unchanged, passing
)
TESTS = {
    "normalise": "def test_normalise():\n    assert normalise('a') == 'a'\n",
    "twice": "def test_twice():\n    assert twice(2) == 4\n",
    "label": "def test_label():\n    assert label() == 'v1'\n",
    "shout": "def test_shout():\n    assert shout('a') == 'A!'\n",
}
IMPORTS = "from synthlib.core import label, normalise, shout, twice\n\n\n"
FUNCTION = {name: f"synthlib.core.{name}" for name in TESTS}
FAILURE = {"normalise": "assert 'A' == 'a'", "twice": "assert 6 == 4", "label": "assert 'v2' == 'v1'"}
EVIDENCE = {  # FIX_CODE cites the expected behaviour; UPDATE_TEST cites intent
    "normalise": ("docstring:normalise", EvidenceSourceType.DOCSTRING, "normalise(s): return s stripped and lower-cased."),
    "twice": ("changelog:twice", EvidenceSourceType.CHANGELOG, "twice() now triples its argument on purpose."),
    "label": ("changelog:label", EvidenceSourceType.CHANGELOG, "label() now returns v2 on purpose."),
}
PROBES = {
    "normalise": TrustedProbe(probe_id="p-n", module="synthlib.core", function="normalise",
                              cases=(ProbeCase(args=("AbC",), expected="abc"),), provenance="dataset"),
    "twice": TrustedProbe(probe_id="p-t", module="synthlib.core", function="twice",
                          cases=(ProbeCase(args=(2,), expected=6),), provenance="dataset"),
    "label": TrustedProbe(probe_id="p-l", module="synthlib.core", function="label",
                          cases=(ProbeCase(expected="v2"),), provenance="dataset"),
}
NO_TOOL = {"query": "", "path": "", "start_line": 1, "end_line": 0, "selectors": []}
SETTINGS = Settings.from_env({"LLM_MODEL": "fake-model-1", "EXPERIMENT_CONDITION": "A2", "TEST_TIMEOUT_SECONDS": "60"})
CONSISTENT = {"verdict": "consistent", "test_meaning_preserved": True, "reasons": []}


def build(tmp_path: Path, monkeypatch, order: list[str], files: dict[str, str]):
    """A repository whose CI failures are ``order`` (in that order); ``files`` maps each test name to its file."""
    root = tmp_path / "repo"
    layout: dict[str, list[str]] = {}
    for name in (*order, "shout"):  # the CI state fails exactly the listed tests
        layout.setdefault(files.get(name, "tests/test_core.py"), []).append(TESTS[name])
    contents = {"pyproject.toml": '[tool.pytest.ini_options]\npythonpath = ["src"]\n', "src/synthlib/__init__.py": "",
                "src/synthlib/core.py": CORE}
    contents |= {path: IMPORTS + "\n\n".join(bodies) for path, bodies in layout.items()}
    for rel, text in contents.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)
    for args in (["init", "-q"], ["add", "."], ["-c", "user.name=t", "-c", "user.email=t@x", "commit", "-qm", "ci"]):
        subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)
    trusted = partial(run_tests, isolation="trusted-host")
    for module in (triage, integrity_guard, agent_tools, orchestrator):
        monkeypatch.setattr(module, "run_tests", trusted)
    monkeypatch.setattr(integrity_guard, "verify_behaviour", partial(verify_behaviour, isolation="trusted-host"))
    monkeypatch.setattr(integrity_guard, "mutation_check", partial(mutation_check, isolation="trusted-host"))

    base = make_s1().agent_input
    ids = {name: f"{files.get(name, 'tests/test_core.py')}::test_{name}" for name in order}
    failing = [FailingTest(test_id=ids[n], test_file=ids[n].split("::")[0], test_name=f"test_{n}",
                           target_function=FUNCTION[n], failure_message=FAILURE[n]) for n in order]
    items = [*base.corpus.items, *(EvidenceItem(evidence_id=e, source_type=t, source_ref="docs", text=x)
                                   for e, t, x in EVIDENCE.values())]
    episode = base.model_copy(update={
        "ci_event": base.ci_event.model_copy(update={"failing_tests": failing}),
        "corpus": base.corpus.model_copy(update={"items": items}),
    })
    return root, episode, ids


def guard(context, group, analysis, proposal, **kw):
    name = group.group.function_qualname.rsplit(".", 1)[-1]
    return integrity_guard.verify_repair(context, group, analysis, proposal, probe=PROBES[name], **kw)


def intent(group_id: str, name: str) -> dict:
    fix = name == "normalise"
    return {"group_id": group_id, "hypothesis": {"kind": "regression" if fix else "outdated_test", "explanation": "s"},
            "cited_evidence_ids": [EVIDENCE[name][0]], "confidence": 0.9,
            "recommended_decision": "FIX_CODE" if fix else "UPDATE_TEST"}


def patch(*edits, cited=()) -> dict:
    return {"action": "patch", "edits": [{"path": p, "old": o, "new": n} for p, o, n in edits],
            "rationale": "s", "evidence_ids": list(cited), **NO_TOOL}


def repair_for(name: str, test_file: str) -> dict:
    if name == "normalise":
        return patch(("src/synthlib/core.py", "return s.capitalize()", "return s.lower()"))
    if name == "twice":
        return patch((test_file, "twice(2) == 4", "twice(2) == 6"), cited=[EVIDENCE["twice"][0]])
    return patch((test_file, "label() == 'v1'", "label() == 'v2'"), cited=[EVIDENCE["label"][0]])


def run(root, episode, order, repairs, files) -> tuple:
    llm = FakeLLMClient({
        AgentRole.INTENT_ANALYST: [intent(f"g{i}", name) for i, name in enumerate(order, 1)],
        AgentRole.REPAIRER: repairs,
        AgentRole.INTEGRITY_GUARD: [CONSISTENT] * 4,
    })
    return run_episode(episode, worktree_path=root, settings=SETTINGS, llm=llm, agents=Agents(integrity_guard=guard))


def verdicts(record) -> dict[str, str]:
    return {g.group_id: g.guard_decision for g in record.failure_groups}


SAME = {}
APART = {"twice": "tests/test_twice.py", "label": "tests/test_label.py"}


@pytest.mark.parametrize(
    ("order", "files"),
    [
        (["twice", "normalise"], SAME),  # 1. UPDATE_TEST approved first, then a correct FIX_CODE
        (["normalise", "twice"], SAME),  # 2. FIX_CODE approved first, then UPDATE_TEST
        (["twice", "label"], SAME),  # 3. two UPDATE_TEST groups in the same test file
        (["twice", "normalise"], APART),  # 4. groups in different files
        (["normalise", "twice"], APART),
    ],
    ids=["update-then-fix", "fix-then-update", "two-updates-same-file", "update-then-fix-apart", "fix-then-update-apart"],
)
def test_every_correct_group_is_approved_regardless_of_order(tmp_path, monkeypatch, order, files) -> None:
    root, episode, ids = build(tmp_path, monkeypatch, order, files)
    repairs = [repair_for(name, ids[name].split("::")[0]) for name in order]
    record = run(root, episode, order, repairs, files)

    assert verdicts(record) == {"g1": GuardDecision.APPROVE, "g2": GuardDecision.APPROVE}, record.failure_groups
    assert {p.test_id: p.predicted_action for p in record.predictions} == {
        ids[n]: DecisionLabel.FIX_CODE if n == "normalise" else DecisionLabel.UPDATE_TEST for n in order}
    assert [p.application_status for p in record.patches] == [PatchApplicationStatus.APPLIED] * 2
    assert record.final_test_status is TestRunStatus.PASSED
    assert all(g.disposition == "accepted" for g in record.failure_groups)


def test_later_patch_may_not_edit_an_earlier_approved_groups_test(tmp_path, monkeypatch) -> None:
    order = ["twice", "label"]
    root, episode, _ = build(tmp_path, monkeypatch, order, SAME)
    sneaky = patch(("tests/test_core.py", "label() == 'v1'", "label() == 'v2'"),
                   ("tests/test_core.py", "twice(2) == 6", "twice(2) == 7"), cited=[EVIDENCE["label"][0]])
    repairs = [repair_for("twice", "tests/test_core.py"), sneaky, sneaky]  # the revision repeats it
    record = run(root, episode, order, repairs, SAME)

    groups = {g.group_id: g for g in record.failure_groups}
    assert groups["g1"].guard_decision is GuardDecision.APPROVE and groups["g1"].disposition == "accepted"
    assert groups["g2"].guard_decision is GuardDecision.REJECT
    assert any("outside this failure group: tests/test_core.py::test_twice" in r for r in groups["g2"].guard_reasons)
    text = (root / "tests/test_core.py").read_text()
    assert "twice(2) == 6" in text  # 6. the earlier approved repair survived both rollbacks
    assert "label() == 'v1'" in text and "twice(2) == 7" not in text  # the rejected patch is fully undone
    assert record.final_test_status is TestRunStatus.FAILED  # label is still red: honest end state


def test_rejected_later_fix_rolls_back_only_itself(tmp_path, monkeypatch) -> None:
    order = ["twice", "normalise"]
    root, episode, _ = build(tmp_path, monkeypatch, order, SAME)
    wrong = patch(("src/synthlib/core.py", "return s.capitalize()", "return s.title()"))
    record = run(root, episode, order, [repair_for("twice", "tests/test_core.py"), wrong, wrong], SAME)

    groups = {g.group_id: g for g in record.failure_groups}
    assert groups["g1"].disposition == "accepted"
    assert groups["g2"].guard_decision is GuardDecision.REJECT and groups["g2"].disposition == "reverted"
    assert "twice(2) == 6" in (root / "tests/test_core.py").read_text()
    assert "return s.capitalize()" in (root / "src/synthlib/core.py").read_text()


# --------------------------------------------------------------------------- #
# H2: the configured target interpreter and timeout reach every test run, including the default Guard's
# --------------------------------------------------------------------------- #


def _recording_runs(monkeypatch, seen: list[tuple[str, object, object]]) -> None:
    """Wrap every module's ``run_tests`` (already the trusted-host runner) to note who ran with what."""
    for module in (triage, integrity_guard, agent_tools, orchestrator):
        inner = module.run_tests

        def recorded(worktree, selectors, *, _inner=inner, _name=module.__name__, **kw):
            seen.append((_name, kw.get("python_executable"), kw.get("timeout_seconds")))
            return _inner(worktree, selectors, **kw)
        monkeypatch.setattr(module, "run_tests", recorded)
    inner_check = integrity_guard.mutation_check

    def recorded_check(*args, _inner=inner_check, **kw):  # cached-mutant runs must use the same interpreter/timeout
        seen.append(("testorcode.tools.mutation", kw.get("python_executable"), kw.get("timeout_seconds")))
        return _inner(*args, **kw)
    monkeypatch.setattr(integrity_guard, "mutation_check", recorded_check)


@pytest.mark.parametrize("injected", [False, True], ids=["default-guard", "injected-guard"])
def test_configured_interpreter_and_timeout_reach_every_run(tmp_path, monkeypatch, injected) -> None:
    import sys
    order = ["twice", "normalise"]
    root, episode, _ = build(tmp_path, monkeypatch, order, SAME)
    python = tmp_path / "target-python"  # a non-default interpreter path (a wrapper around this interpreter)
    python.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
    python.chmod(0o755)
    settings = Settings.from_env({"LLM_MODEL": "fake-model-1", "EXPERIMENT_CONDITION": "A2",
                                  "TEST_TIMEOUT_SECONDS": "37", "TEST_PYTHON": str(python)})
    seen: list[tuple[str, object, object]] = []
    _recording_runs(monkeypatch, seen)
    llm = FakeLLMClient({
        AgentRole.INTENT_ANALYST: [intent(f"g{i}", name) for i, name in enumerate(order, 1)],
        AgentRole.REPAIRER: [repair_for(name, "tests/test_core.py") for name in order] * 2,
        AgentRole.INTEGRITY_GUARD: [CONSISTENT] * 4,
    })
    agents = Agents(integrity_guard=guard) if injected else Agents()  # the default Guard has no probe
    record = run_episode(episode, worktree_path=root, settings=settings, llm=llm, agents=agents)

    modules = {name for name, _, _ in seen}
    assert {"testorcode.agents.triage", "testorcode.agents.integrity_guard", "testorcode.orchestrator",
            "testorcode.tools.mutation"} <= modules
    assert all(Path(p) == python and t == 37 for _, p, t in seen), seen
    # g1 (twice, UPDATE_TEST) is approved on measured mutant-kill strength with or without a harness probe.
    assert verdicts(record)["g1"] is GuardDecision.APPROVE


# --------------------------------------------------------------------------- #
# M1: fixed verification overhead does not consume the agents' tool budget
# --------------------------------------------------------------------------- #


def test_two_group_episode_overhead_does_not_exhaust_the_agent_budget(tmp_path, monkeypatch) -> None:
    from testorcode.schemas.episode_schemas import EpisodeBudget
    order = ["twice", "normalise"]
    root, episode, _ = build(tmp_path, monkeypatch, order, SAME)
    # The agents need exactly 8 calls (per group: the Analyst's 3 searches and the Repairer's read_file).  The harness
    # needs 13 more (5 stability runs; per group validate, apply, targeted run, regression run) plus the final suite.
    # Under the former single counter a budget of 8 ran out at the first Repairer read (5 + 3 = 8 calls).
    tight = episode.model_copy(update={"budget": EpisodeBudget(max_tool_calls=8, max_llm_calls=20)})
    read = {"action": "read_file", "edits": [], "rationale": "s", "evidence_ids": [], **NO_TOOL,
            "path": "src/synthlib/core.py"}
    repairs = [read, repair_for("twice", "tests/test_core.py"), read, repair_for("normalise", "tests/test_core.py")]
    record = run(root, episode.model_copy(update={"budget": tight.budget}), order, repairs, SAME)

    assert verdicts(record) == {"g1": GuardDecision.APPROVE, "g2": GuardDecision.APPROVE}, record.errors
    budgets = [e.arguments["budget"] for e in record.tool_events]
    # Operational: the 13 harness calls above plus one run_mutants call per group (twice: test strength; normalise:
    # FIX_CODE oracle sensitivity).
    assert (budgets.count("agent"), budgets.count("operational"), budgets.count("exempt")) == (8, 15, 1)
    assert {e.called_by for e in record.tool_events if e.arguments["budget"] == "agent"} <= {
        AgentRole.INTENT_ANALYST, AgentRole.REPAIRER}
    assert record.tool_events[-1].arguments["budget"] == "exempt"  # the final suite run
    assert record.final_test_status is TestRunStatus.PASSED


def test_agent_budget_exhaustion_stops_the_agent_not_the_cleanup(tmp_path, monkeypatch) -> None:
    from testorcode.schemas.episode_schemas import EpisodeBudget
    from testorcode.schemas.result_schemas import RunStatus
    order = ["twice", "normalise"]
    root, episode, _ = build(tmp_path, monkeypatch, order, SAME)
    tight = episode.model_copy(update={"budget": EpisodeBudget(max_tool_calls=2)})
    read = {"action": "read_file", "edits": [], "rationale": "s", "evidence_ids": [], **NO_TOOL,
            "path": "src/synthlib/core.py"}
    record = run(root, tight, order, [read] * 6, SAME)
    assert record.outcome.status is RunStatus.BUDGET_EXHAUSTED
    assert [e.arguments["budget"] for e in record.tool_events].count("agent") == 2
    assert subprocess.run(["git", "-C", str(root), "status", "--porcelain"], capture_output=True, text=True,
                          check=True).stdout == ""


def test_operational_safety_cap_is_finite(tmp_path, monkeypatch) -> None:
    from testorcode.schemas.result_schemas import RunStatus
    order = ["twice", "normalise"]
    root, episode, _ = build(tmp_path, monkeypatch, order, SAME)
    monkeypatch.setattr(orchestrator, "OPERATIONAL_CAP_FACTOR", 0)  # cap 0 x budget -> no harness call allowed
    monkeypatch.setattr(orchestrator, "OPERATIONAL_CAP_DEFAULT", 0)
    record = run(root, episode, order, [], SAME)
    assert record.outcome.status is RunStatus.BUDGET_EXHAUSTED
    assert "operational tool-call safety cap" in record.errors[0].message

