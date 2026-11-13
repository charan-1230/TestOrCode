"""Final-phase features: extended mutants and unmeasurable strength, sandbox defaults (Docker), several repositories per
experiment, git tools in prompts and in the A1 view."""

from __future__ import annotations

from pathlib import Path

import pytest

from testorcode.agents import prompts
from testorcode.conditions import episode_for_condition
from testorcode.config import Settings
from testorcode.dataset import agent_view_path, episode_path, load_agent_inputs, load_episodes
from testorcode.experiment import ExperimentConfig, RepoEnvironment, default_run_one
from testorcode.schemas.episode_schemas import ToolName
from testorcode.schemas.result_schemas import ExperimentCondition as C
from testorcode.tools import pytest_runner, sandbox
from testorcode.tools.mutation import generate_mutants
from testorcode.tools.sandbox import Isolation, apply_defaults, sandbox_defaults

# --------------------------------------------------------------------------- #
# Mutants
# --------------------------------------------------------------------------- #

SMALL = '''class Cached:
    def __call__(self, *args):
        key = (self.func,) + args if self.scoped else args
        return self.cache[key]
'''
LARGE = '''def f(a, b):
    if a > 1 and b < 2:
        return a + b
    return a - 3 * b == 4
'''


def test_small_functions_are_topped_up_with_extended_operators() -> None:
    mutants = generate_mutants(SMALL, "pkg/m.py", "Cached.__call__", module="pkg.m")
    operators = [m.operator for m in mutants]
    assert any("negate IfExp condition" in op for op in operators)
    assert "(not self.scoped)" in "".join(m.unified_diff for m in mutants)


def test_functions_with_enough_base_candidates_keep_their_mutants() -> None:
    mutants = generate_mutants(LARGE, "pkg/m.py", "f", module="pkg.m")
    assert len(mutants) == 6 and not any("negate" in m.operator for m in mutants)


def test_unmeasurable_strength_is_a_gap_unless_allowed(monkeypatch, tmp_path: Path) -> None:
    from testorcode.schemas.contracts import FailureGroupDetail, PatchProposal
    from testorcode.schemas.episode_schemas import FailureGroup, RepairDirection
    from testorcode.schemas.tests.conftest import make_s1
    from testorcode.tools import mutation

    ep = make_s1().agent_input
    test = ep.ci_event.failing_tests[0]
    group = FailureGroupDetail(group=FailureGroup(group_id="g", function_qualname="pkg.m.f", test_ids=[test.test_id]),
                               failing_tests=(test,))
    proposal = PatchProposal(group_id="g", decision=RepairDirection.UPDATE_TEST, scope="test",
                             unified_diff="--- a/x\n+++ b/x\n", target_files=(test.test_file,), rationale="r",
                             evidence_ids=("ev",))
    spec = generate_mutants(LARGE, "pkg/m.py", "f", module="pkg.m")[:2]
    survived = mutation.MutationRun(status=mutation.TestRunStatus.PASSED, outcomes=tuple(
        mutation.MutantOutcome(m.mutant_id, "survived") for m in spec))
    monkeypatch.setattr(mutation, "run_mutants", lambda *a, **k: survived)
    context = mutation.EpisodeContext.from_episode_input(ep, tmp_path)
    strict = mutation.mutation_check(context, group, proposal, {}, timeout_seconds=5, mutants=spec)
    allowed = mutation.mutation_check(context, group, proposal, {}, timeout_seconds=5, mutants=spec,
                                      allow_unmeasurable=True)
    assert strict.unmeasurable and strict.gaps == strict.unmeasurable
    assert allowed.unmeasurable and not allowed.gaps


# --------------------------------------------------------------------------- #
# Sandbox defaults (Docker)
# --------------------------------------------------------------------------- #


def test_sandbox_defaults_change_only_auto_requests() -> None:
    assert apply_defaults("auto", None) == (Isolation.AUTO, None)
    with sandbox_defaults("docker", "img:1"):
        assert apply_defaults("auto", None) == (Isolation.DOCKER, "img:1")
        assert apply_defaults("auto", "own:2") == (Isolation.DOCKER, "own:2")
        assert apply_defaults("bwrap", None) == (Isolation.BWRAP, None)
    assert apply_defaults("auto", None) == (Isolation.AUTO, None)


def test_docker_runs_use_the_images_python_not_a_host_path(monkeypatch, tmp_path: Path) -> None:
    captured = {}
    monkeypatch.setattr(sandbox, "docker_available", lambda: True)

    def fake_wrap(mode, inner, **kwargs):
        captured.update(mode=mode, inner=inner, image=kwargs["docker_image"])
        raise RuntimeError("stop before executing")

    monkeypatch.setattr(pytest_runner, "wrap_command", fake_wrap)
    (tmp_path / "tests").mkdir()
    with sandbox_defaults("docker", "testorcode/pilot:1"), pytest.raises(RuntimeError, match="stop"):
        pytest_runner.run_pytest(tmp_path, ["tests"], timeout_seconds=5, python_executable="/home/x/venv/bin/python")
    assert captured["mode"] is Isolation.DOCKER and captured["image"] == "testorcode/pilot:1"
    assert captured["inner"][0] == "python"


@pytest.mark.skipif(not sandbox.docker_available(), reason="needs a reachable docker daemon")
def test_live_docker_sandbox_runs_a_test(tmp_path: Path) -> None:  # pragma: no cover - not runnable here
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests/test_ok.py").write_text("def test_ok():\n    assert 1 + 1 == 2\n")
    image = "testorcode/pilot-py312:pytest-8.4.2"  # docker/pilot.Dockerfile
    run = pytest_runner.run_pytest(tmp_path, ["tests"], timeout_seconds=120, isolation="docker", docker_image=image)
    assert run.execution.passed == 1


# --------------------------------------------------------------------------- #
# Several repositories in one experiment
# --------------------------------------------------------------------------- #


def test_each_episode_runs_against_its_own_repository_and_interpreter(monkeypatch, tmp_path: Path) -> None:
    from testorcode import orchestrator

    episode = load_agent_inputs(agent_view_path("dev"))[4]
    gold = next(e for e in load_episodes(episode_path("dev")) if e.episode_id == episode.episode_id)
    seen = {}
    monkeypatch.setattr(orchestrator, "run_episode", lambda ep, **kw: seen.update(kw) or "record")
    config = ExperimentConfig(
        conditions=(C.A2_NO_MEMORY,), out=tmp_path / "out", worktrees=tmp_path / "wt",
        repos={episode.repo.repo_id: RepoEnvironment(path=tmp_path / "clone", python=tmp_path / "py",
                                                     docker_image="img:2")},
    )
    materialised = []
    run_one = default_run_one(config, settings=Settings.from_env({"LLM_MODEL": "m"}), llm_factory=lambda s: object(),
                              materializer=lambda repo, dest, ep: materialised.append(repo))
    assert run_one(episode, gold, C.A2_NO_MEMORY, 0) == "record"
    assert materialised == [tmp_path / "clone"]
    assert seen["settings"].test_python == tmp_path / "py" and seen["settings"].sandbox_docker_image == "img:2"
    assert seen["fault_plan"] == gold.fault_plan  # scenario T on by default
    no_faults = config.model_copy(update={"tool_faults": False})
    default_run_one(no_faults, settings=Settings.from_env({"LLM_MODEL": "m"}), llm_factory=lambda s: object(),
                    materializer=lambda *a: None)(episode, gold, C.A2_NO_MEMORY, 0)
    assert seen["fault_plan"] is None


def test_dataset_name_selects_the_files() -> None:
    assert agent_view_path("dev", dataset="final").name == "final_dev_agent_view.jsonl"
    assert episode_path("test", dataset="final").name == "final_test.jsonl"
    with pytest.raises(ValueError):
        agent_view_path("dev", dataset="../x")


# --------------------------------------------------------------------------- #
# Git tools in the prompts and the A1 view
# --------------------------------------------------------------------------- #


def test_a1_loses_every_tool_that_returns_change_intent_text() -> None:
    episode = load_agent_inputs(agent_view_path("dev"))[0]
    a1 = episode_for_condition(episode, C.A1_NO_INTENT_RETRIEVAL)
    assert not {ToolName.SEARCH_REPO, ToolName.GIT_LOG, ToolName.GIT_SHOW, ToolName.GIT_BLAME} & set(a1.allowed_tools)
    full = episode_for_condition(episode, C.FULL)
    assert {ToolName.GIT_LOG, ToolName.GIT_SHOW, ToolName.GIT_BLAME} <= set(full.allowed_tools)


def test_tool_instructions_list_git_tools_only_when_all_are_allowed() -> None:
    every = prompts.tool_instructions(list(ToolName))
    assert "git_show(sha)" in every and every == prompts.TOOL_INSTRUCTIONS
    assert "git_" not in prompts.tool_instructions([ToolName.READ_FILE, ToolName.RUN_TESTS])
    assert "git_log" in prompts.RepairStep.model_json_schema()["properties"]["action"]["enum"]
    assert prompts.RepairStep.model_validate({"action": "read_file", "query": "", "path": "a", "start_line": 1,
                                             "end_line": 0, "selectors": [], "edits": [], "rationale": "r",
                                             "evidence_ids": []}).sha == ""


# --------------------------------------------------------------------------- #
# Guard retention threshold (GUARD_MIN_RETENTION): killed-before vs killed-after
# --------------------------------------------------------------------------- #


def _retention_check(monkeypatch, tmp_path, *, total: int, lost: int, min_retention: float | None):
    from testorcode.schemas.contracts import FailureGroupDetail, PatchProposal
    from testorcode.schemas.episode_schemas import FailureGroup, RepairDirection
    from testorcode.schemas.tests.conftest import make_s1
    from testorcode.tools import mutation

    ep = make_s1().agent_input
    test = ep.ci_event.failing_tests[0]
    group = FailureGroupDetail(group=FailureGroup(group_id="g", function_qualname="pkg.m.f", test_ids=[test.test_id]),
                               failing_tests=(test,))
    proposal = PatchProposal(group_id="g", decision=RepairDirection.UPDATE_TEST, scope="test",
                             unified_diff="--- a/x\n+++ b/x\n", target_files=(test.test_file,), rationale="r",
                             evidence_ids=("ev",))
    base = generate_mutants(LARGE, "pkg/m.py", "f", module="pkg.m")
    spec = [base[i % len(base)].model_copy(update={"mutant_id": f"m{i}"}) for i in range(total)]  # run_mutants is faked
    ids = [m.mutant_id for m in spec]

    def fake(*args, **kwargs):
        before = "overrides" in kwargs  # the Guard runs the pre-repair tests with overrides
        killed = set(ids) if before else set(ids[lost:])
        return mutation.MutationRun(status=mutation.TestRunStatus.PASSED, outcomes=tuple(
            mutation.MutantOutcome(i, "killed" if i in killed else "survived",
                                   killed_by=(test.test_id,) if i in killed else ()) for i in ids))

    monkeypatch.setattr(mutation, "run_mutants", fake)
    context = mutation.EpisodeContext.from_episode_input(ep, tmp_path)
    extra = {} if min_retention is None else {"min_retention": min_retention}
    return mutation.mutation_check(context, group, proposal, {"tests/x.py": "old"}, timeout_seconds=5, mutants=spec, **extra)


def test_retention_threshold_accepts_a_small_loss_only_when_relaxed(monkeypatch, tmp_path) -> None:
    strict = _retention_check(monkeypatch, tmp_path, total=10, lost=1, min_retention=None)  # default 1.0: pilot rule
    relaxed = _retention_check(monkeypatch, tmp_path, total=10, lost=1, min_retention=0.9)
    assert strict.problems and strict.retention == pytest.approx(0.9)
    assert not relaxed.problems and relaxed.retention == pytest.approx(0.9)  # 0.90 is acceptable at the threshold


def test_retention_below_the_threshold_is_rejected(monkeypatch, tmp_path) -> None:
    check = _retention_check(monkeypatch, tmp_path, total=5, lost=1, min_retention=0.9)  # 4/5 = 0.80
    assert check.retention == pytest.approx(0.8) and check.problems and "retention 0.80 < 0.90" in check.problems[0]


def test_retention_is_not_measured_when_nothing_was_killed_before(monkeypatch, tmp_path) -> None:
    from testorcode.tools import mutation

    check = _retention_check(monkeypatch, tmp_path, total=5, lost=5, min_retention=0.9)  # after kills nothing, before kills all
    assert check.retention == 0.0 and check.problems  # all lost: rejected
    assert isinstance(mutation.MutationCheck().retention, type(None))  # default: not measured, never invented


def test_min_retention_setting_defaults_to_the_pilot_rule() -> None:
    assert Settings.from_env({}).guard_min_retention == 1.0
    assert Settings.from_env({"GUARD_MIN_RETENTION": "0.9"}).guard_min_retention == 0.9


# --------------------------------------------------------------------------- #
# Direction metric: hypothesis primary, action secondary (both always reported)
# --------------------------------------------------------------------------- #


def test_summary_reports_both_direction_rules_escalation_rate_and_repair_coverage() -> None:
    from testorcode.dataset import episode_path, load_episodes
    from testorcode.evaluation import ScoringRules, summarize
    from testorcode.experiment import read_records

    runs = Path(__file__).resolve().parents[1] / "results" / "final" / "dev" / "runs.jsonl"
    if not runs.is_file():
        pytest.skip("dev pass records not present")
    records, episodes = read_records(runs), load_episodes(episode_path("dev"))
    primary = {s.condition: s.metrics for s in summarize(records, episodes, rules=ScoringRules(direction_from="hypothesis"))}
    flipped = {s.condition: s.metrics for s in summarize(records, episodes, rules=ScoringRules(direction_from="action"))}
    for condition, m in primary.items():
        assert m["direction_macro_f1"] == m["direction_macro_f1_hypothesis"]  # primary follows the rule
        assert flipped[condition]["direction_macro_f1"] == m["direction_macro_f1_action"]
        # both are always present, whatever the rule: choosing one never hides the other
        assert m["direction_macro_f1_action"] == flipped[condition]["direction_macro_f1_action"]
        assert m["direction_macro_f1_hypothesis"] == flipped[condition]["direction_macro_f1_hypothesis"]
        assert 0.0 <= m["escalation_rate"] <= 1.0 and 0.0 <= m["repair_coverage"] <= 1.0
        assert m["escalation_rate"] + m["repair_coverage"] <= 1.0 + 1e-9  # an accepted repair is not an escalation
