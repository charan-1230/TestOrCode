"""Every shared module imports without side effects; deferred code fails loudly."""

from __future__ import annotations

import importlib
import logging
from pathlib import Path

import pytest

from testorcode.config import Settings
from testorcode.errors import ConfigError
from testorcode.llm_client import OpenAICompatibleClient, create_llm_client
from testorcode.schemas.contracts import EpisodeContext
from testorcode.schemas.episode_schemas import EditPolicy, PatchScope, RepoRef

MODULES = [
    "testorcode",
    "testorcode.config",
    "testorcode.errors",
    "testorcode.logging_config",
    "testorcode.llm_client",
    "testorcode.orchestrator",
    "testorcode.retrieval",
    "testorcode.memory",
    "testorcode.evaluation",
    "testorcode.agents",
    "testorcode.agents.triage",
    "testorcode.agents.intent_analyst",
    "testorcode.agents.repairer",
    "testorcode.agents.integrity_guard",
    "testorcode.tools",
    "testorcode.tools.pytest_runner",
    "testorcode.tools.git_tools",
    "testorcode.tools.patch_tools",
    "testorcode.schemas.episode_schemas",
    "testorcode.schemas.result_schemas",
    "testorcode.schemas.contracts",
]

CONTEXT = EpisodeContext(
    episode_id="ep-1", repo=RepoRef(repo_id="r"), worktree_path=Path("/tmp/wt"), commit_sha="a1b2c3d4"
)


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name: str) -> None:
    importlib.import_module(name)


def test_import_does_not_configure_logging() -> None:
    importlib.import_module("testorcode.logging_config")
    assert not logging.getLogger("testorcode").handlers or all(
        not getattr(h, "_testorcode_handler", False) for h in logging.getLogger("testorcode").handlers
    )


def test_deferred_functions_raise_not_implemented() -> None:
    from testorcode.tools import git_tools, patch_tools, pytest_runner

    with pytest.raises(NotImplementedError):
        pytest_runner.run_tests(Path("."), ["tests"], timeout_seconds=1)
    with pytest.raises(NotImplementedError):
        git_tools.git_log(Path("."), cutoff_commit="a1b2c3d4")
    with pytest.raises(NotImplementedError):
        patch_tools.validate_patch(Path("."), "", policy=EditPolicy(), scope=PatchScope.SRC)


def test_llm_client_needs_credentials_only_when_created() -> None:
    settings = Settings.from_env({})
    with pytest.raises(Exception, match="LLM_MODEL"):
        create_llm_client(settings)
    client = OpenAICompatibleClient(settings)
    with pytest.raises(ConfigError, match="LLM_MODEL"):
        client.complete_structured(role=None, system="", user="", response_model=Settings)  # type: ignore[arg-type]
