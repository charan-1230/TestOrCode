from __future__ import annotations

from pathlib import Path

import pytest

from testorcode.config import ENV_VARS, Settings
from testorcode.errors import ConfigError
from testorcode.schemas.result_schemas import ExperimentCondition


def test_defaults_need_no_environment_or_key() -> None:
    s = Settings.from_env({})
    assert s.llm_api_key is None and s.llm_model is None
    assert s.experiment_condition is ExperimentCondition.B1
    assert s.max_repair_revisions == 1 and s.retrieval_top_k > 0


def test_blank_values_are_unset() -> None:
    s = Settings.from_env({"LLM_MODEL": "", "LLM_API_KEY": "  ", "RANDOM_SEED": ""})
    assert s.llm_model is None and s.llm_api_key is None and s.random_seed == 42


def test_values_are_parsed() -> None:
    s = Settings.from_env(
        {
            "LLM_MODEL": "m-2026-01-01",
            "EXPERIMENT_CONDITION": "A2",
            "RANDOM_SEED": "0",
            "RESULTS_PATH": "out/",
            "TEST_TIMEOUT_SECONDS": "30",
            "LOG_LEVEL": "debug",
        }
    )
    assert s.llm_model == "m-2026-01-01"
    assert s.experiment_condition is ExperimentCondition.A2_NO_MEMORY
    assert s.random_seed == 0 and s.results_path == Path("out/")
    assert s.test_timeout_seconds == 30 and s.log_level == "DEBUG"


@pytest.mark.parametrize(
    ("var", "value"),
    [
        ("TEST_TIMEOUT_SECONDS", "0"),
        ("RETRIEVAL_TOP_K", "-1"),
        ("MAX_REPAIR_REVISIONS", "2"),
        ("RANDOM_SEED", "abc"),
        ("EXPERIMENT_CONDITION", "nope"),
        ("LOG_LEVEL", "LOUD"),
    ],
)
def test_invalid_values_raise_config_error(var: str, value: str) -> None:
    with pytest.raises(ConfigError, match=var):
        Settings.from_env({var: value})


def test_secrets_are_masked_and_not_echoed_in_errors() -> None:
    s = Settings.from_env({"LLM_API_KEY": "sk-supersecretvalue123"})
    assert "supersecret" not in repr(s) and "supersecret" not in s.model_dump_json()
    assert s.secret_values() == ["sk-supersecretvalue123"]
    with pytest.raises(ConfigError) as exc:
        Settings.from_env({"LLM_API_KEY": "sk-supersecretvalue123", "RANDOM_SEED": "x"})
    assert "supersecret" not in str(exc.value)


def test_require_llm() -> None:
    with pytest.raises(ConfigError, match="LLM_MODEL"):
        Settings.from_env({}).require_llm()
    with pytest.raises(ConfigError, match="LLM_API_KEY"):
        Settings.from_env({"LLM_MODEL": "m"}).require_llm()
    Settings.from_env({"LLM_MODEL": "m", "LLM_API_KEY": "k"}).require_llm()
    Settings.from_env({"LLM_MODEL": "m", "LLM_BASE_URL": "http://localhost:11434/v1"}).require_llm()


def test_env_file_is_read_without_touching_os_environ(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("LLM_MODEL", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("LLM_MODEL=from-file\nRANDOM_SEED=7\n")
    s = Settings.from_env({"RANDOM_SEED": "9"}, env_file=env_file)
    assert s.llm_model == "from-file" and s.random_seed == 9  # environment wins
    import os

    assert "LLM_MODEL" not in os.environ


def test_env_example_lists_every_variable() -> None:
    text = (Path(__file__).parent.parent / ".env.example").read_text()
    missing = [v for v in ENV_VARS.values() if f"{v}=" not in text]
    assert not missing, f".env.example lacks {missing}"
