"""Frozen final configuration, runner-level tool-fault rate and the model override."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from testorcode.config import Settings
from testorcode.dataset import episode_path, load_episodes
from testorcode.errors import ConfigError
from testorcode.experiment import ROOT, ExperimentConfig, fault_plan_for, run_experiment
from testorcode.freeze import FrozenConfig, load_frozen, settings_snapshot
from testorcode.schemas.result_schemas import ExperimentCondition as C
from testorcode.tools.faults import FaultInjector

EPISODES = {e.episode_id: e for e in load_episodes(episode_path("dev"))}
TRIVIAL = (C.TRIVIAL_ALWAYS_UPDATE_TEST,)


def frozen(**settings: str) -> FrozenConfig:
    return FrozenConfig(name="t", settings=settings)


def test_frozen_settings_override_typed_and_empty_means_unset() -> None:
    base = Settings.from_env({"LLM_MODEL": "m-dev", "MUTANT_CACHE_PATH": "a.json", "RETRIEVAL_MODE": "bm25"})
    out = frozen(LLM_MODEL="m-final", RETRIEVAL_MODE="hybrid", MUTANT_CACHE_PATH="",
                 GUARD_UNMEASURABLE_STRENGTH="allow").apply(base)
    assert out.llm_model == "m-final" and out.retrieval_mode == "hybrid" and out.mutant_cache_path is None
    assert out.guard_unmeasurable_strength == "allow"


@pytest.mark.parametrize("var", ["LLM_API_KEY", "LLM_API_KEY_2", "VOYAGE_API_KEY", "GITHUB_TOKEN", "NOT_A_SETTING"])
def test_secrets_and_unknown_variables_are_refused(var: str) -> None:
    with pytest.raises(ValidationError):
        frozen(**{var: "x"})


def test_an_invalid_value_is_a_config_error() -> None:
    with pytest.raises(ConfigError, match="RETRIEVAL_MODE"):
        frozen(RETRIEVAL_MODE="magic").apply(Settings())


def test_load_frozen_returns_the_file_hash(tmp_path: Path) -> None:
    path = tmp_path / "f.json"
    path.write_text(json.dumps({"name": "final-v1", "experiment": {"tool_fault_rate": 0.1}}))
    config, sha = load_frozen(path)
    assert config.experiment.tool_fault_rate == 0.1 and len(sha) == 64
    path.write_text("{")
    with pytest.raises(ConfigError):
        load_frozen(path)


def test_snapshot_has_no_secrets() -> None:
    snap = settings_snapshot(Settings.from_env({"LLM_API_KEY": "sk-zzzzzzzzzzzzzzzz", "LLM_MODEL": "m"}))
    assert snap["llm_model"] == "m" and "sk-zzzz" not in json.dumps(snap) and "llm_api_key" not in snap


def test_a_fault_rate_keeps_the_episodes_injections_and_is_the_same_for_every_condition() -> None:
    dev04 = EPISODES["dev-04"]  # its own plan: the second run_tests call gets an invalid selector
    plan = fault_plan_for(dev04, 0.5)
    assert plan.failure_rate == 0.5 and plan.injections == dev04.fault_plan.injections
    assert fault_plan_for(dev04, None) == dev04.fault_plan
    plain = fault_plan_for(EPISODES["dev-02"], 0.3)
    assert plain.enabled and plain.seed == fault_plan_for(EPISODES["dev-02"], 0.3).seed
    a, b = FaultInjector(plain), FaultInjector(plain)
    from testorcode.schemas.episode_schemas import ToolName

    assert [a.fault_for(ToolName.READ_FILE) for _ in range(40)] == [b.fault_for(ToolName.READ_FILE) for _ in range(40)]
    assert 0 < len(a.fired) < 40


def test_a_rate_without_tool_faults_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="contradict"):
        ExperimentConfig(conditions=TRIVIAL, out=tmp_path, tool_faults=False, tool_fault_rate=0.1)


def test_config_json_records_the_frozen_config_and_the_effective_settings(tmp_path: Path) -> None:
    env = tmp_path / "second.env"
    env.write_text("LLM_MODEL=from-env-file\nLLM_API_KEY=sk-should-not-leak-0123456789\n")
    config = ExperimentConfig(
        conditions=TRIVIAL, out=tmp_path / "out", episode_ids=("dev-02",), env_file=env, model="override-model",
        frozen=FrozenConfig(name="final-v1", settings={"RETRIEVAL_MODE": "hybrid"}), frozen_sha256="ab" * 32,
        tool_fault_rate=0.1,
    )
    run_experiment(config)
    text = (tmp_path / "out" / "config.json").read_text()
    written = json.loads(text)
    assert written["frozen"]["name"] == "final-v1" and written["frozen"]["sha256"] == "ab" * 32
    assert written["settings"]["llm_model"] == "override-model"  # env file, then frozen, then --model
    assert written["settings"]["retrieval_mode"] == "hybrid" and written["tool_fault_rate"] == 0.1
    assert "sk-should-not-leak" not in text


def test_cli_refuses_a_command_line_that_contradicts_the_frozen_config(tmp_path: Path) -> None:
    path = tmp_path / "f.json"
    path.write_text(json.dumps({"name": "x", "experiment": {"tool_fault_rate": 0.1}}))
    done = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "run_experiment.py"), "--conditions", "TRIVIAL_UPDATE_TEST",
         "--freeze", str(path), "--tool-fault-rate", "0.2", "--out", str(tmp_path / "o")],
        capture_output=True, text=True, check=False,
    )
    assert done.returncode == 2 and "contradicts the frozen config" in done.stderr


def test_the_committed_draft_is_valid_and_refused_for_the_final_runs(tmp_path: Path) -> None:
    frozen_draft, _ = load_frozen(ROOT / "configs" / "final.draft.json")
    assert frozen_draft.name.startswith("DRAFT") and frozen_draft.settings["TEST_PATCH_RULE"] == "literal"
    done = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "run_experiment.py"), "--conditions", "TRIVIAL_UPDATE_TEST",
         "--split", "test", "--final", "--freeze", str(ROOT / "configs" / "final.draft.json"),
         "--out", str(tmp_path / "o")],
        capture_output=True, text=True, check=False,
    )
    assert done.returncode == 2 and "not the draft" in done.stderr and not (tmp_path / "o").exists()
