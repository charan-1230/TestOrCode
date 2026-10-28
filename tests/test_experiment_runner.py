"""Experiment runner: scored JSONL output, resumable, frozen test split refused, nothing secret written."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from testorcode.baselines import trivial_record
from testorcode.errors import ConfigError
from testorcode.experiment import (
    ExperimentConfig,
    read_records,
    run_experiment,
)
from testorcode.schemas.result_schemas import ExperimentCondition as C

ROOT = Path(__file__).resolve().parents[1]
TRIVIAL = (C.TRIVIAL_ALWAYS_UPDATE_TEST, C.TRIVIAL_ALWAYS_FIX_CODE)


def config(tmp_path: Path, **kw) -> ExperimentConfig:
    base = {"conditions": TRIVIAL, "episode_ids": ("dev-02", "dev-05"), "seeds": (0, 1), "out": tmp_path / "out"}
    return ExperimentConfig(**{**base, **kw})


def test_trivial_conditions_run_end_to_end_and_every_record_is_scored(tmp_path: Path) -> None:
    cfg = config(tmp_path)
    result = run_experiment(cfg)
    assert (result.ran, result.skipped_done, result.skipped_unavailable) == (8, 0, [])  # 2 episodes x 2 conditions x 2 seeds
    records = read_records(cfg.out / "runs.jsonl")
    assert len(records) == 8
    for r in records:
        assert r.evaluation is not None and r.evaluation.decisions  # harness scoring is attached
        assert r.provenance.split == "dev" and r.provenance.dataset_version.startswith("pilot@")
        assert r.provenance.model_id is None and r.provenance.seed in (0, 1)


def test_a_second_call_resumes_instead_of_repeating(tmp_path: Path) -> None:
    cfg = config(tmp_path)
    run_experiment(cfg)
    again = run_experiment(cfg)
    assert (again.ran, again.skipped_done) == (0, 8)
    more = run_experiment(config(tmp_path, seeds=(0, 1, 2)))
    assert (more.ran, more.skipped_done) == (4, 8)  # only the new seed ran
    assert len(read_records(cfg.out / "runs.jsonl")) == 12


def test_config_records_what_ran_and_nothing_secret(tmp_path: Path) -> None:
    cfg = config(tmp_path)
    run_experiment(cfg)
    saved = json.loads((cfg.out / "config.json").read_text())
    assert saved["conditions"] == ["TRIVIAL_UPDATE_TEST", "TRIVIAL_FIX_CODE"] and saved["split"] == "dev"
    assert set(saved["dataset_files_sha256"]) >= {"pilot_dev.jsonl", "pilot_dev_agent_view.jsonl"}
    text = (cfg.out / "config.json").read_text().lower()
    assert not any(word in text for word in ("api_key", "secret", "token", "password"))


def test_the_frozen_test_split_is_refused_without_final(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="frozen"):
        config(tmp_path, split="test", episode_ids=())
    assert config(tmp_path, split="test", episode_ids=(), final=True).split == "test"


@pytest.mark.parametrize(
    "bad",
    [{"conditions": ()}, {"seeds": ()}, {"seeds": (0, 0)}, {"conditions": (C.B1, C.B1)}, {"split": "train"}],
)
def test_invalid_configurations_are_refused(tmp_path: Path, bad: dict) -> None:
    with pytest.raises(ValueError):
        config(tmp_path, **bad)


def test_an_unknown_episode_is_an_error_not_an_empty_run(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown dev episodes"):
        run_experiment(config(tmp_path, episode_ids=("dev-99",)))
    assert not (tmp_path / "out" / "runs.jsonl").exists()


def test_a_run_that_cannot_start_is_reported_and_writes_no_fake_record(tmp_path: Path) -> None:
    def no_model(episode, gold, condition, seed):
        raise ConfigError("no model configured")

    cfg = config(tmp_path, conditions=(C.A2_NO_MEMORY,), episode_ids=("dev-05",), seeds=(0,))
    result = run_experiment(cfg, run_one=no_model)
    assert result.ran == 0 and len(result.skipped_unavailable) == 1 and "no model configured" in result.skipped_unavailable[0]
    assert read_records(cfg.out / "runs.jsonl") == []


def test_non_trivial_conditions_go_through_the_injected_runner_with_the_agent_view_only(tmp_path: Path) -> None:
    seen = []

    def fake(episode, gold, condition, seed):
        seen.append((type(episode).__name__, condition, seed))
        return trivial_record(episode, C.TRIVIAL_ALWAYS_UPDATE_TEST, seed=seed).model_copy(
            update={"provenance": trivial_record(episode, C.TRIVIAL_ALWAYS_UPDATE_TEST, seed=seed).provenance.model_copy(
                update={"condition": condition, "model_id": "test-model-1"})}
        )  # fmt: skip

    cfg = config(tmp_path, conditions=(C.B1,), episode_ids=("dev-05",), seeds=(7,))
    assert run_experiment(cfg, run_one=fake).ran == 1
    assert seen == [("EpisodeInput", C.B1, 7)]  # the agent's input is the gold-free view
    [record] = read_records(cfg.out / "runs.jsonl")
    assert record.condition is C.B1 and record.evaluation is not None


def test_summarize_script_prints_a_table_and_writes_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    cfg = config(tmp_path)
    run_experiment(cfg)
    spec = importlib.util.spec_from_file_location("summarize_results", ROOT / "scripts" / "summarize_results.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(sys, "argv", ["summarize_results.py", str(cfg.out / "runs.jsonl")])
    assert module.main() == 0
    out = capsys.readouterr().out
    assert "TRIVIAL_FIX_CODE" in out and "TRIVIAL_UPDATE_TEST" in out and "Not computed" in out
    summary = json.loads((cfg.out / "summary.json").read_text())
    assert {s["condition"] for s in summary} == {"TRIVIAL_UPDATE_TEST", "TRIVIAL_FIX_CODE"}
    assert (cfg.out / "summary.md").read_text().startswith("| condition")
    assert module.cell(None) == "-" and module.cell(0.0) == "0" and module.cell(0.5) == "0.500"  # None is never shown as 0


def test_run_experiment_script_refuses_the_test_split(tmp_path: Path) -> None:
    import subprocess

    done = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "run_experiment.py"), "--conditions", "TRIVIAL_UPDATE_TEST",
         "--split", "test", "--out", str(tmp_path / "o")],
        capture_output=True, text=True, check=False,
    )  # fmt: skip
    assert done.returncode == 2 and "frozen" in done.stderr
    assert not (tmp_path / "o").exists()
