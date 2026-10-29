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
import os
import tempfile
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
    RunStatus,
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
    retry_failed: bool = Field(
        default=False,
        description="Re-run runs that ended in api_failure (replacing that line). Off by default: a failed run is done.",
    )

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
    retried: int = 0  # of ``ran``: runs that replaced an api_failure record (``retry_failed``)
    skipped_done: int = 0
    skipped_unavailable: list[str] = Field(default_factory=list)


def _is_api_failure(record: ExperimentRunRecord) -> bool:
    return record.outcome.status is RunStatus.API_FAILURE


def _wrote_memory(record: ExperimentRunRecord) -> bool:
    return any(w.written for w in record.memory_writes)


def unsafe_retries(
    pairs: list[tuple[EpisodeInput, Episode]], config: ExperimentConfig, records: list[ExperimentRunRecord]
) -> list[str]:
    """Why each requested retry of a failed run would break the memory stream, for the conditions that have memory.

    A retry runs an *earlier* episode after later ones, so it could read memory those later episodes wrote or
    seeded, and a failed attempt that already wrote memory could be replayed into a different record.  Memory-free
    conditions are always safe to retry.
    """
    by_key = {_key(r): r for r in records}
    problems = []
    for index, (episode, _) in enumerate(pairs):
        for condition in config.conditions:
            if not profile_for(condition).memory:
                continue
            for seed in config.seeds:
                failed = by_key.get((episode.episode_id, condition.value, seed))
                if failed is None or not _is_api_failure(failed):
                    continue
                label = f"{episode.episode_id}/{condition.value}/seed {seed}"
                if _wrote_memory(failed):
                    problems.append(f"{label}: the failed attempt already wrote memory, a replay could differ")
                for later, _ in pairs[index + 1:]:
                    other = by_key.get((later.episode_id, condition.value, seed))
                    if other is not None and (_wrote_memory(other) or later.visible_memory):
                        problems.append(f"{label}: later episode {later.episode_id} already ran and wrote or seeded memory")
                        break
    return problems


def _replace_run_lines(runs_path: Path, replaced: ExperimentRunRecord, new: ExperimentRunRecord) -> None:
    """Swap the failed record's line for the new one in a single atomic file replacement (never two lines per run)."""
    kept = [r for r in read_records(runs_path) if r.provenance.run_id != replaced.provenance.run_id]
    text = "".join(r.to_jsonl() + "\n" for r in [*kept, new])
    fd, tmp = tempfile.mkstemp(dir=runs_path.parent, prefix=f".{runs_path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(tmp, runs_path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def run_experiment(config: ExperimentConfig, *, run_one: RunOne | None = None) -> ExperimentResult:
    """Run every missing (episode, condition, seed) and append its scored record to ``runs.jsonl``.

    A run that cannot start (for example no model or API key configured) is skipped and reported in
    ``skipped_unavailable``: without a model id no valid record exists, and a fake one would be worse than none.

    By default any existing record, an ``api_failure`` one included, counts as done.  With ``config.retry_failed`` an
    ``api_failure`` run is re-run and its line is replaced.  For a condition with memory that is refused (``ValueError``,
    nothing run) when it would break the stream's chronology: re-run the whole stream into a fresh ``out`` instead.
    """
    pairs = load_split(config)
    runs_path = config.out / RUNS_FILE
    existing = read_records(runs_path)
    failed_by_key = {_key(r): r for r in existing if _is_api_failure(r)} if config.retry_failed else {}
    if failed_by_key:
        problems = unsafe_retries(pairs, config, existing)
        if problems:
            raise ValueError(
                "refusing to retry: " + "; ".join(problems) + ". Memory carries over between episodes in stream "
                "order, so re-run the whole stream for this condition and seed into a fresh --out."
            )
    run_one = run_one or default_run_one(config)
    config.out.mkdir(parents=True, exist_ok=True)
    done = {_key(r) for r in existing}
    _write_config(config, [e.episode_id for e, _ in pairs])
    result = ExperimentResult()
    for episode, gold in pairs:
        for condition in config.conditions:
            for seed in config.seeds:
                key = (episode.episode_id, condition.value, seed)
                retrying = key in failed_by_key
                if key in done and not retrying:
                    result.skipped_done += 1
                    continue
                try:
                    record = run_one(episode, gold, condition, seed)
                except TestOrCodeError as exc:
                    result.skipped_unavailable.append(f"{episode.episode_id}/{condition.value}/seed {seed}: {exc}")
                    log.warning("skipped %s %s: %s", episode.episode_id, condition.value, exc)
                    continue
                final = scored(record, gold, config.split)
                if retrying:
                    _replace_run_lines(runs_path, failed_by_key[key], final)
                    result.retried += 1
                else:
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

