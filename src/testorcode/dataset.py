"""Load pilot episode files (``datasets/pilot_{dev,test}[_agent_view].jsonl``).

Two kinds of file, kept apart on purpose:

* ``*_agent_view.jsonl``: what a system under test may see (``EpisodeInput``).  Use ``load_agent_inputs``.
* ``pilot_dev.jsonl`` / ``pilot_test.jsonl``: full records with gold labels and reference repairs
  (``Episode``).  Use ``load_episodes`` only in the evaluation harness; never put these in a prompt.

This is a convention plus two guards, not a security boundary: ``load_agent_inputs`` refuses a file that
contains forbidden ground-truth keys, and ``load_episodes`` refuses an agent-view file.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from testorcode.schemas.episode_schemas import Episode, EpisodeInput, find_leaked_ground_truth

DATASETS_DIR = Path(__file__).resolve().parents[2] / "datasets"


def _records(path: str | Path) -> Iterator[tuple[int, dict]]:
    with open(path, encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                yield number, json.loads(line)


def load_episodes(path: str | Path) -> list[Episode]:
    """Full records including gold labels.  Evaluation harness only.

    Raises:
        ValueError: the file is an agent view (no ``ground_truth``) or a record is invalid.
    """
    episodes = []
    for number, record in _records(path):
        if "ground_truth" not in record:
            msg = f"{path}:{number}: no ground_truth; this looks like an agent-view file"
            raise ValueError(msg)
        episodes.append(Episode.model_validate(record))
    return episodes


def load_agent_inputs(path: str | Path) -> list[EpisodeInput]:
    """Agent-visible inputs, in stream order.

    Raises:
        ValueError: a record contains ground-truth keys (a full episode file was passed) or is invalid.
    """
    inputs = []
    for number, record in _records(path):
        leaked = find_leaked_ground_truth(record)
        if leaked:
            msg = f"{path}:{number}: ground-truth keys present ({', '.join(leaked[:3])}); pass an *_agent_view.jsonl file"
            raise ValueError(msg)
        inputs.append(EpisodeInput.model_validate(record))
    return inputs


def agent_view_path(split: str, datasets_dir: str | Path = DATASETS_DIR) -> Path:
    """Path of the agent-view file for ``split`` ('dev' or 'test')."""
    if split not in ("dev", "test"):
        msg = f"split must be 'dev' or 'test', not {split!r}"
        raise ValueError(msg)
    return Path(datasets_dir) / f"pilot_{split}_agent_view.jsonl"


def episode_path(split: str, datasets_dir: str | Path = DATASETS_DIR) -> Path:
    """Path of the full (gold-label) file for ``split``.  Evaluation harness only."""
    if split not in ("dev", "test"):
        msg = f"split must be 'dev' or 'test', not {split!r}"
        raise ValueError(msg)
    return Path(datasets_dir) / f"pilot_{split}.jsonl"
