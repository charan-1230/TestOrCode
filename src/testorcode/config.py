"""Application settings, read from environment variables.

No settings code existed, ``.env.example`` defines the variable names, and every
module needs one shared model/provider/budget configuration, so this is the
single place that turns the environment into validated values.

* Nothing is read at import time.  Call ``Settings.from_env()`` explicitly.
* ``.env`` is read only when an ``env_file`` is passed, and it never mutates
  ``os.environ``.  Real environment variables take precedence over the file.
* Empty strings (``LLM_MODEL=``) count as unset.
* No API key is needed to build ``Settings``; call ``require_llm()`` right
  before a live API call.
* Secrets are ``SecretStr``: they are masked in ``repr``, logs and dumps.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Final, Literal

from dotenv import dotenv_values
from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError

from testorcode.errors import ConfigError
from testorcode.schemas.result_schemas import ExperimentCondition

LogLevelName = str

# field name -> environment variable
ENV_VARS: Final[dict[str, str]] = {
    "llm_provider": "LLM_PROVIDER",
    "llm_api_key": "LLM_API_KEY",
    "llm_base_url": "LLM_BASE_URL",
    "llm_model": "LLM_MODEL",
    "llm_structured_mode": "LLM_STRUCTURED_MODE",
    "llm_max_retries": "LLM_MAX_RETRIES",
    "llm_timeout_seconds": "LLM_TIMEOUT_SECONDS",
    "voyage_api_key": "VOYAGE_API_KEY",
    "voyage_embedding_model": "VOYAGE_EMBEDDING_MODEL",
    "voyage_rerank_model": "VOYAGE_RERANK_MODEL",
    "github_token": "GITHUB_TOKEN",
    "experiment_condition": "EXPERIMENT_CONDITION",
    "random_seed": "RANDOM_SEED",
    "dataset_path": "DATASET_PATH",
    "results_path": "RESULTS_PATH",
    "test_python": "TEST_PYTHON",
    "test_timeout_seconds": "TEST_TIMEOUT_SECONDS",
    "retrieval_top_k": "RETRIEVAL_TOP_K",
    "max_repair_revisions": "MAX_REPAIR_REVISIONS",
    "retrieval_mode": "RETRIEVAL_MODE",
    "embedding_backend": "EMBEDDING_BACKEND",
    "local_embedding_model": "LOCAL_EMBEDDING_MODEL",
    "rerank_backend": "RERANK_BACKEND",
    "local_rerank_model": "LOCAL_RERANK_MODEL",
    "embedding_cache_dir": "EMBEDDING_CACHE_DIR",
    "mutant_cache_path": "MUTANT_CACHE_PATH",
    "guard_unmeasurable_strength": "GUARD_UNMEASURABLE_STRENGTH",
    "test_patch_rule": "TEST_PATCH_RULE",
    "sandbox_isolation": "SANDBOX_ISOLATION",
    "sandbox_docker_image": "SANDBOX_DOCKER_IMAGE",
    "log_level": "LOG_LEVEL",
    "log_file": "LOG_FILE",
}

# Extra API keys used one after another when a key's quota runs out (LLM_API_KEY_1 .. LLM_API_KEY_9, in order).
KEY_POOL_VARS: Final[tuple[str, ...]] = tuple(f"LLM_API_KEY_{i}" for i in range(1, 10))

_VALID_LOG_LEVELS: Final[tuple[str, ...]] = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")


class Settings(BaseModel):
    """One configuration shared by every agent and the orchestrator."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    # LLM: a single provider/model for all agents.
    llm_provider: str | None = None
    llm_api_key: SecretStr | None = None
    llm_api_key_pool: tuple[SecretStr, ...] = Field(
        default=(), description="LLM_API_KEY_1..9: keys tried in order when the current one's quota is used up."
    )
    llm_base_url: str | None = None
    llm_model: str | None = None
    llm_structured_mode: Literal["json_schema", "json_object", "prompt"] = Field(
        default="json_schema", description="What the provider is asked for; replies are always validated client-side."
    )
    llm_max_retries: int = Field(default=2, ge=0, le=10, description="Retries for rate limits, timeouts and 5xx.")
    llm_timeout_seconds: float = Field(default=60.0, gt=0, description="Per-request timeout.")

    # Optional integrations.
    voyage_api_key: SecretStr | None = None
    voyage_embedding_model: str = "voyage-code-3"
    voyage_rerank_model: str = "rerank-2.5"
    github_token: SecretStr | None = None

    # Experiment.
    # Default A2: the only multi-agent condition that needs no MemoryStore.  run_episode also runs FULL, A1 and A3
    # (with a store), B1 and the trivial baselines; the non-proposal conditions fail explicitly.
    experiment_condition: ExperimentCondition = ExperimentCondition.A2_NO_MEMORY
    random_seed: int = 42
    dataset_path: Path = Path("datasets/pilot_dev.jsonl")
    results_path: Path = Path("results/")

    # Target repository: interpreter that runs its tests (None = the current one).
    # Keep it an unresolved venv path: resolving the symlink loses the venv.
    test_python: Path | None = None

    # Limits.
    test_timeout_seconds: int = Field(default=120, gt=0)
    retrieval_top_k: int = Field(default=8, gt=0)
    max_repair_revisions: int = Field(
        default=1, ge=0, le=1, description="A rejected repair gets at most one revision."
    )

    # Retrieval (``retrieval_hybrid``).  bm25 = keyword only (the pilot runs); hybrid = BM25 + dense embeddings fused
    # with RRF, optionally reranked.  The same index serves every condition that retrieves.
    retrieval_mode: Literal["bm25", "hybrid"] = "bm25"
    embedding_backend: Literal["voyage", "local"] = Field(
        default="voyage", description="voyage = voyage-code-3 over HTTPS (needs VOYAGE_API_KEY); local = sentence-transformers."
    )
    local_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    rerank_backend: Literal["none", "voyage", "local"] = "none"
    local_rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    embedding_cache_dir: Path | None = Field(default=None, description="On-disk embedding cache; None = memory only.")

    # Integrity Guard (``tools.mutation``).
    mutant_cache_path: Path | None = Field(
        default=None, description="Cached-mutant catalogue given to the Guard; None = the Guard generates mutants."
    )
    guard_unmeasurable_strength: Literal["escalate", "allow"] = Field(
        default="escalate",
        description=(
            "UPDATE_TEST repairs whose strength cannot be measured (neither the updated nor the pre-repair tests kill "
            "any executable mutant): escalate (the pilot behaviour) or allow on the Guard's other checks."
        ),
    )

    # Test-side patch content rule (``tools.patch_content``): literal = only expected literal values may change (the
    # pilot); expected_value = the expected side of an equality assertion may become any side-effect-free expression.
    test_patch_rule: Literal["literal", "expected_value"] = "literal"

    # Sandbox for every test run of an episode (``tools.sandbox``).  The episode's own docker_image wins when set.
    sandbox_isolation: Literal["auto", "docker", "bwrap"] = "auto"
    sandbox_docker_image: str | None = None

    # Logging.
    log_level: LogLevelName = "INFO"
    log_file: Path | None = None

    @classmethod
    def from_env(
        cls,
        environ: Mapping[str, str] | None = None,
        *,
        env_file: Path | str | None = None,
    ) -> Settings:
        """Build settings from ``environ`` (default ``os.environ``).

        Args:
            environ: Variable source; pass a dict in tests.
            env_file: Optional dotenv file for local development.  Values in
                ``environ`` win over values in the file.

        Raises:
            ConfigError: A value is invalid.  The message names the variable
                and never echoes the offending value (it may be a secret).
        """
        source: dict[str, str] = {}
        if env_file is not None:
            source.update({k: v for k, v in dotenv_values(env_file).items() if v is not None})
        source.update(os.environ if environ is None else environ)

        raw: dict[str, str] = {}
        for field, var in ENV_VARS.items():
            value = source.get(var)
            if value is not None and value.strip() != "":
                raw[field] = value.strip()
        pool = [source[var].strip() for var in KEY_POOL_VARS if (source.get(var) or "").strip()]
        if pool:
            raw["llm_api_key_pool"] = pool
        try:
            settings = cls.model_validate(raw)
        except ValidationError as exc:
            problems = []
            for err in exc.errors(include_input=False, include_url=False):
                field = str(err["loc"][0]) if err["loc"] else "?"
                field = "LLM_API_KEY_n" if field == "llm_api_key_pool" else field
                problems.append(f"{ENV_VARS.get(field, field)}: {err['msg']}")
            raise ConfigError("invalid configuration: " + "; ".join(problems)) from None
        level = settings.log_level.upper()
        if level not in _VALID_LOG_LEVELS:
            raise ConfigError(f"invalid configuration: LOG_LEVEL: must be one of {_VALID_LOG_LEVELS}")
        return settings.model_copy(update={"log_level": level})

    def require_llm(self) -> None:
        """Check the settings needed for a live model call.

        Raises:
            ConfigError: ``LLM_MODEL`` is unset, or no API key (``LLM_API_KEY_n`` or
                ``LLM_API_KEY``) is set while no ``LLM_BASE_URL`` points at a keyless endpoint.
        """
        missing = []
        if not self.llm_model:
            missing.append("LLM_MODEL")
        if not self.llm_api_keys() and not self.llm_base_url:
            missing.append("LLM_API_KEY (or LLM_BASE_URL for a keyless endpoint)")
        if missing:
            raise ConfigError("live LLM use requires: " + ", ".join(missing))

    def llm_api_keys(self) -> list[SecretStr]:
        """Keys in the order they are used: ``LLM_API_KEY_1..9``, then ``LLM_API_KEY``; duplicates dropped."""
        keys, seen = [], set()
        for key in (*self.llm_api_key_pool, *((self.llm_api_key,) if self.llm_api_key is not None else ())):
            if key.get_secret_value() not in seen:
                seen.add(key.get_secret_value())
                keys.append(key)
        return keys

    def secret_values(self) -> list[str]:
        """Plain secret values, for log redaction only."""
        secrets = (self.llm_api_key, *self.llm_api_key_pool, self.voyage_api_key, self.github_token)
        return [s.get_secret_value() for s in secrets if s is not None]
