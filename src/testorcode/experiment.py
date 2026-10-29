"""Experiment runner: episodes x conditions x seeds -> scored run records in one JSONL file.

Harness side only: it holds the gold labels (for the simulated maintainer and for scoring) and never passes them
to an agent.  Trivial baselines need no repository and no API key; every other condition goes through
``orchestrator.run_episode`` on a safe, disposable worktree.  The frozen test split needs ``final=True``.

Output directory: ``runs.jsonl`` (one scored ``ExperimentRunRecord`` per line, appended as runs finish, so an
interrupted experiment resumes where it stopped), ``config.json`` (what was run; never a secret) and, for
conditions with memory, ``memory/<condition>/seed-<seed>/<split>.jsonl`` (one store per condition, seed and stream).
"""

from __future__ import annotations

import importlib.util
import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Final

from pydantic import BaseModel, ConfigDict, Field, model_validator

from testorcode.baselines import is_trivial, trivial_record
from testorcode.conditions import profile_for
from testorcode.config import Settings
from testorcode.dataset import agent_view_path, episode_path, load_agent_inputs, load_episodes
from testorcode.errors import TestOrCodeError
from testorcode.evaluation import evaluate_run
from testorcode.llm_client import LLMClient
from testorcode.memory import JsonlMemoryStore, memory_namespace, memory_path
from testorcode.schemas.episode_schemas import DecisionLabel, Episode, EpisodeInput
from testorcode.schemas.result_schemas import (
    ExperimentCondition,
    ExperimentRunRecord,
    TestRunStatus,
)

log = logging.getLogger(__name__)

ROOT: Final[Path] = Path(__file__).resolve().parents[2]
RUNS_FILE: Final[str] = "runs.jsonl"
CONFIG_FILE: Final[str] = "config.json"

RunOne = Callable[[EpisodeInput, Episode, ExperimentCondition, int], ExperimentRunRecord]


class ExperimentConfig(BaseModel):
    """What to run.  Frozen; contains no credentials (those come from ``.env`` when a model is called)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    split: str = "dev"
    episode_ids: tuple[str, ...] = Field(default=(), description="Empty = every episode of the split.")
    conditions: tuple[ExperimentCondition, ...]
    seeds: tuple[int, ...] = (0,)
    out: Path
    repo: Path = ROOT.parent / "boltons"
    worktrees: Path = ROOT.parent / "episode-worktrees"
    final: bool = Field(default=False, description="Required to touch the frozen test split.")

    @model_validator(mode="after")
    def _check(self) -> ExperimentConfig:
        if self.split not in ("dev", "test"):
            raise ValueError(f"split must be 'dev' or 'test', not {self.split!r}")
        if self.split == "test" and not self.final:
            raise ValueError("the test split is frozen; pass final=True only for the frozen evaluation")
        if not self.conditions:
            raise ValueError("at least one condition is required")
        if not self.seeds:
            raise ValueError("at least one seed is required")
        if len(set(self.conditions)) != len(self.conditions) or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("conditions and seeds must not repeat")
        return self


# --------------------------------------------------------------------------- #
# Harness helpers (shared with scripts/run_episode.py)
# --------------------------------------------------------------------------- #


def simulated_maintainer(gold: dict[str, DecisionLabel]):
    """Proposal section 9: approve when the direction matches the gold label and the tests pass."""

    def approve(group, decision, result) -> bool:
        passed = result.post_repair_run is not None and result.post_repair_run.status is TestRunStatus.PASSED
        return passed and all(gold.get(t) == decision for t in group.group.test_ids)

    return approve


def dataset_version(split: str) -> str:
    manifest = json.loads((ROOT / "datasets" / "pilot_manifest.json").read_text())
    return "pilot@" + manifest["files_sha256"][f"pilot_{split}_agent_view.jsonl"][:12]


def _materializer():
    spec = importlib.util.spec_from_file_location(
        "materialize_episode", ROOT / "scripts" / "pilot" / "materialize_episode.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.materialize


def _code_revision() -> str | None:
    from testorcode.orchestrator import _code_revision as revision

    return revision()


# --------------------------------------------------------------------------- #
# Running
# --------------------------------------------------------------------------- #


def load_split(config: ExperimentConfig) -> list[tuple[EpisodeInput, Episode]]:
    """The agent view and the gold episode for each selected episode (ids checked to agree)."""
    inputs = {e.episode_id: e for e in load_agent_inputs(agent_view_path(config.split))}
    gold = {e.episode_id: e for e in load_episodes(episode_path(config.split))}
    if inputs.keys() != gold.keys():
        raise ValueError("agent-view and gold files do not list the same episodes")
    wanted = list(config.episode_ids) or list(inputs)
    missing = [e for e in wanted if e not in inputs]
    if missing:
        raise ValueError(f"unknown {config.split} episodes: {missing}")
    # Chronological stream order, whatever order the ids were given in: memory carries over in that order.
    wanted = sorted(dict.fromkeys(wanted), key=lambda e: (gold[e].stream_id, gold[e].stream_position))
    return [(inputs[e], gold[e]) for e in wanted]


def read_records(path: Path) -> list[ExperimentRunRecord]:
    if not path.exists():
        return []
    return [ExperimentRunRecord.model_validate_json(line) for line in path.read_text().splitlines() if line.strip()]


def _key(record: ExperimentRunRecord) -> tuple[str, str, int | None]:
    return (record.episode_id, record.condition.value, record.provenance.seed)


def memory_dir(config: ExperimentConfig) -> Path:
    """Where the experiment keeps its memory stores: ``<out>/memory/<condition>/seed-<seed>/<split>.jsonl``."""
    return config.out / "memory"


def default_run_one(
    config: ExperimentConfig,
    *,
    settings: Settings | None = None,
    llm_factory: Callable[[Settings], LLMClient] | None = None,
    materializer: Callable[[Path, Path, dict], object] | None = None,
) -> RunOne:
    """Trivial baselines directly; every other condition through the orchestrator on a safe worktree.

    Conditions whose profile uses memory (FULL) get one ``JsonlMemoryStore`` per (condition, seed) for the split: the
    store isolation contract of ``memory`` (``memory_path``/``memory_namespace``), so memory carries over between the
    episodes of one stream (run in stream order by ``run_experiment``) and never between conditions, seeds or splits.
    Each episode's declared prior memory (``visible_memory``) is seeded into it before any agent acts.  Every other
    condition runs with ``memory=None``.  ``settings``/``llm_factory``/``materializer`` default to ``.env``, the hosted
    client and the pilot materialiser (tests inject fixtures).
    """
    stores: dict[tuple[str, int], JsonlMemoryStore] = {}

    def store_for(condition: ExperimentCondition, seed: int, run_settings: Settings, episode: EpisodeInput):
        key = (condition.value, seed)
        if key not in stores:
            stores[key] = JsonlMemoryStore(
                memory_path(memory_dir(config), condition.value, seed, config.split),
                python_executable=run_settings.test_python, timeout_seconds=run_settings.test_timeout_seconds,
                docker_image=episode.environment.docker_image,
                namespace=memory_namespace(condition.value, seed, config.split),
            )
        return stores[key]

    def run_one(episode: EpisodeInput, gold: Episode, condition: ExperimentCondition, seed: int) -> ExperimentRunRecord:
        if is_trivial(condition):
            return trivial_record(episode, condition, seed=seed, code_revision=_code_revision())
        from testorcode.llm_client import create_llm_client
        from testorcode.orchestrator import run_episode

        base = settings if settings is not None else Settings.from_env(env_file=ROOT / ".env")
        run_settings = base.model_copy(update={"experiment_condition": condition, "random_seed": seed})
        llm = (llm_factory or create_llm_client)(run_settings)
        worktree = (config.worktrees / f"{episode.episode_id}-{condition.value}-{seed}").absolute()
        (materializer or _materializer())(config.repo, worktree, episode.model_dump(mode="json"))
        memory = None
        if profile_for(condition).memory:
            memory = store_for(condition, seed, run_settings, episode)
            memory.seed(episode.visible_memory or [], worktree_path=worktree)  # harness side, before any agent acts
        return run_episode(
            episode, worktree_path=worktree, settings=run_settings, llm=llm, memory=memory,
            maintainer=simulated_maintainer(gold.gold_actions()),
        )  # fmt: skip

    return run_one


def scored(record: ExperimentRunRecord, gold: Episode, split: str) -> ExperimentRunRecord:
    """The record with provenance split/dataset version and the harness evaluation filled in (re-validated)."""
    provenance = record.provenance.model_copy(update={"split": split, "dataset_version": dataset_version(split)})
    draft = record.model_copy(update={"provenance": provenance})
    evaluation = evaluate_run(draft, gold)
    return ExperimentRunRecord.model_validate(draft.model_copy(update={"evaluation": evaluation}).model_dump())


class ExperimentResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ran: int = 0
    skipped_done: int = 0
    skipped_unavailable: list[str] = Field(default_factory=list)


def run_experiment(config: ExperimentConfig, *, run_one: RunOne | None = None) -> ExperimentResult:
    """Run every missing (episode, condition, seed) and append its scored record to ``runs.jsonl``.

    A run that cannot start (for example no model or API key configured) is skipped and reported in
    ``skipped_unavailable``: without a model id no valid record exists, and a fake one would be worse than none.
    """
    pairs = load_split(config)
    run_one = run_one or default_run_one(config)
    config.out.mkdir(parents=True, exist_ok=True)
    runs_path = config.out / RUNS_FILE
    done = {_key(r) for r in read_records(runs_path)}
    _write_config(config, [e.episode_id for e, _ in pairs])
    result = ExperimentResult()
    for episode, gold in pairs:
        for condition in config.conditions:
            for seed in config.seeds:
                if (episode.episode_id, condition.value, seed) in done:
                    result.skipped_done += 1
                    continue
                try:
                    record = run_one(episode, gold, condition, seed)
                except TestOrCodeError as exc:
                    result.skipped_unavailable.append(f"{episode.episode_id}/{condition.value}/seed {seed}: {exc}")
                    log.warning("skipped %s %s: %s", episode.episode_id, condition.value, exc)
                    continue
                final = scored(record, gold, config.split)
                with runs_path.open("a", encoding="utf-8") as fh:
                    fh.write(final.to_jsonl() + "\n")
                result.ran += 1
    return result


def _write_config(config: ExperimentConfig, episode_ids: list[str]) -> None:
    manifest = json.loads((ROOT / "datasets" / "pilot_manifest.json").read_text())
    payload = {
        "split": config.split,
        "episodes": episode_ids,
        "conditions": [c.value for c in config.conditions],
        "seeds": list(config.seeds),
        "dataset_files_sha256": manifest["files_sha256"],
        "code_revision": _code_revision(),
        "final": config.final,
    }
    (config.out / CONFIG_FILE).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

