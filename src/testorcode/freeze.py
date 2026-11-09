"""The frozen configuration of a final experiment: one committed JSON file, applied on top of ``.env``.

The proposal fixes the model, the retrieval set-up, the Guard policy and the thresholds on the dev split *before* the
test split is touched.  ``FrozenConfig`` is that decision written down::

    {
      "name": "final-v1",
      "settings": {"LLM_MODEL": "qwen/qwen3-coder-480b-a35b-instruct", "RETRIEVAL_MODE": "bm25",
                   "GUARD_UNMEASURABLE_STRENGTH": "escalate", "MUTANT_CACHE_PATH": ""},
      "experiment": {"tool_faults": true, "tool_fault_rate": 0.1},
      "scoring": {"direction_from": "hypothesis", "escalate_after_correct_repair": "wrong"},
      "hypothesis": {"min_direction_f1_gain": 0.15, "b2_gap": 0.05},
      "notes": "frozen on dev, 2026-11-10"
    }

* ``settings``: environment-variable names of ``config.ENV_VARS`` and the value to use, overriding both ``.env`` and the
  process environment; ``""`` means unset.  Secrets (API keys, tokens) are refused: they never belong in a committed file.
* ``experiment``: runner options (``ExperimentConfig`` fields), used when the command line does not give them.
* ``scoring``: the scoring rules (``evaluation.ScoringRules``), read by the report and statistics scripts.
* ``hypothesis``: the pre-registered thresholds (``hypothesis.HypothesisThresholds``), read by
  ``scripts/compare_conditions.py``.

``scripts/run_experiment.py --freeze FILE`` applies it and copies it (with its SHA-256) and the effective non-secret
settings into ``config.json``, so every final run says exactly what it used.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from testorcode.config import ENV_VARS, Settings
from testorcode.errors import ConfigError
from testorcode.evaluation import ScoringRules
from testorcode.hypothesis import HypothesisThresholds

SECRET_VARS = frozenset({"LLM_API_KEY", "VOYAGE_API_KEY", "GITHUB_TOKEN"})
_FIELD_FOR_VAR = {var: field for field, var in ENV_VARS.items()}
_SECRET_FIELDS = ("llm_api_key", "llm_api_key_pool", "voyage_api_key", "github_token")


class ExperimentOptions(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    tool_faults: bool | None = None
    tool_fault_rate: float | None = Field(default=None, ge=0.0, le=1.0)


class FrozenConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    settings: dict[str, str] = Field(default_factory=dict)
    experiment: ExperimentOptions = Field(default_factory=ExperimentOptions)
    scoring: ScoringRules = Field(default_factory=ScoringRules)
    hypothesis: HypothesisThresholds = Field(default_factory=HypothesisThresholds)
    notes: str | None = None

    @field_validator("settings")
    @classmethod
    def _known_and_not_secret(cls, value: dict[str, str]) -> dict[str, str]:
        for var in value:
            if var in SECRET_VARS or var.startswith("LLM_API_KEY_"):
                raise ValueError(f"{var} is a secret and must stay in .env, never in a frozen config")
            if var not in _FIELD_FOR_VAR:
                raise ValueError(f"{var} is not a setting (see config.ENV_VARS)")
        return value

    def apply(self, settings: Settings) -> Settings:
        """``settings`` with every frozen setting replacing its value (typed and validated like ``.env`` values)."""
        if not self.settings:
            return settings
        typed = Settings.from_env(self.settings)  # validates the values; "" = unset (the field's default)
        update = {_FIELD_FOR_VAR[var]: getattr(typed, _FIELD_FOR_VAR[var]) for var in self.settings}
        return settings.model_copy(update=update)


def load_frozen(path: str | Path) -> tuple[FrozenConfig, str]:
    """The frozen config and the SHA-256 of the file's bytes.

    Raises:
        ConfigError: unreadable, not JSON, or invalid.
    """
    try:
        data = Path(path).read_bytes()
        return FrozenConfig.model_validate(json.loads(data)), hashlib.sha256(data).hexdigest()
    except (OSError, ValueError) as exc:
        raise ConfigError(f"invalid frozen config {path}: {exc}") from None


def load_rules(path: str | Path | None) -> tuple[ScoringRules, HypothesisThresholds, FrozenConfig | None]:
    """The scoring rules and thresholds of a frozen config, or the defaults (the pilot's scoring) without one."""
    if path is None:
        return ScoringRules(), HypothesisThresholds(), None
    frozen, _ = load_frozen(path)
    return frozen.scoring, frozen.hypothesis, frozen


def settings_snapshot(settings: Settings) -> dict[str, Any]:
    """Every non-secret setting as JSON values (what ``config.json`` records)."""
    return settings.model_dump(mode="json", exclude=set(_SECRET_FIELDS))
