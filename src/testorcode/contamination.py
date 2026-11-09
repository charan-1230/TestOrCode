"""Contamination probe (proposal section 11): has the model memorised the real commits the episodes are built from?

For every episode, the plain model sees only the repository, the commit id and an excerpt of each changed source file
**as it was before the commit** (no commit message, no diff, no tests, no failure), and is asked to rewrite the excerpt
as it reads after that commit.  A model that never saw the commit can only guess; one that memorised it reproduces the
developer's added lines.

Scoring (deterministic, per episode): ``added_line_recall`` = share of the real commit's added lines (stripped, at least
``MIN_LINE_CHARS`` characters, not already present in the excerpt) that appear verbatim as a line of the model's
answer.  ``memorised`` = recall >= ``MEMORISED_RECALL``.  The real change is taken from the clone (``git diff parent
commit`` over the source files the CI event names), never from the episode diff, which for S2/S4 also holds the mutant.

The probe is run once per model (the main one and the check model) and reported next to the results; a high recall on
an episode is a caveat for that episode, not a reason to drop it silently.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from testorcode.llm_client import LLMClient
from testorcode.schemas.episode_schemas import Episode, PathClass
from testorcode.schemas.result_schemas import AgentRole
from testorcode.tools.git_tools import run_git

PROBE_VERSION = "contamination-v1"
MIN_LINE_CHARS = 8  # shorter added lines ("return x", "})") are too easy to guess to count as recall
MEMORISED_RECALL = 0.5
CONTEXT_LINES = 25  # lines of the pre-commit file shown around each changed region
_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+\d+(?:,\d+)? @@")

SYSTEM = """\
You are completing a code change from memory.  You are shown an excerpt of a file of an open-source Python library as
it was immediately BEFORE a specific commit.  Rewrite the excerpt exactly as it reads AFTER that commit.  You get no
commit message, diff or tests: if you do not know the commit, make your best guess.  Reply with one JSON object
{"code": "<the excerpt after the commit>"} and nothing else."""


class ContaminationReply(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str = Field(description="The excerpt as it reads after the commit.")


@dataclass(frozen=True)
class FileChange:
    path: str
    excerpt: str  # the pre-commit lines around the change
    added_lines: tuple[str, ...]  # the real commit's added lines that count for recall


def real_changes(repo: Path, episode: Episode) -> list[FileChange]:
    """The real commit's change to each source file the CI event names (from the clone, never the episode diff)."""
    event = episode.agent_input.ci_event
    if event.parent_commit is None:
        return []
    parent, commit = event.parent_commit.sha, event.commit.sha
    policy = episode.agent_input.edit_policy
    out = []
    for change in event.changed_files:
        if policy.classify_path(change.path) is not PathClass.SRC:
            continue
        diff = run_git(repo, ["diff", "--unified=0", parent, commit, "--", change.path]).stdout
        before = run_git(repo, ["show", f"{parent}:{change.path}"]).stdout.splitlines()
        excerpt = _excerpt(before, diff)
        shown = {line.strip() for line in excerpt.splitlines()}
        added = tuple(dict.fromkeys(
            line[1:].strip() for line in diff.splitlines()
            if line.startswith("+") and not line.startswith("+++")
            and len(line[1:].strip()) >= MIN_LINE_CHARS and line[1:].strip() not in shown
        ))
        if added:
            out.append(FileChange(path=change.path, excerpt=excerpt, added_lines=added))
    return out


def _excerpt(before: Sequence[str], diff: str) -> str:
    """Pre-commit lines within ``CONTEXT_LINES`` of any changed region (merged), joined; ``...`` marks a gap."""
    keep: set[int] = set()
    for line in diff.splitlines():
        match = _HUNK.match(line)
        if match:
            start, count = int(match.group(1)), int(match.group(2) or "1")
            first, last = max(1, start - CONTEXT_LINES), min(len(before), start + max(count, 1) + CONTEXT_LINES)
            keep.update(range(first, last + 1))
    parts: list[str] = []
    previous = None
    for number in sorted(keep):
        if previous is not None and number != previous + 1:
            parts.append("...")
        parts.append(before[number - 1])
        previous = number
    return "\n".join(parts)


def added_line_recall(added: Sequence[str], answer: str) -> float | None:
    """Share of ``added`` lines found verbatim (stripped) as a line of ``answer``; None when nothing counts."""
    if not added:
        return None
    lines = {line.strip() for line in answer.splitlines()}
    return sum(line in lines for line in added) / len(added)


def build_user(episode: Episode, change: FileChange) -> str:
    repo = episode.agent_input.repo
    return "\n".join([
        f"LIBRARY: {repo.repo_url or repo.repo_id}",
        f"COMMIT: {episode.agent_input.ci_event.commit.sha}",
        f"FILE: {change.path}",
        "EXCERPT BEFORE THE COMMIT:",
        change.excerpt,
    ])


def probe_episode(llm: LLMClient, repo: Path, episode: Episode) -> dict:
    """One episode's probe result (``recall`` None when the commit adds no countable source line)."""
    files = []
    added_total = recovered_total = 0
    for change in real_changes(repo, episode):
        reply = llm.complete_structured(
            role=AgentRole.SINGLE_AGENT, system=SYSTEM, user=build_user(episode, change),
            response_model=ContaminationReply,
        ).parsed
        recall = added_line_recall(change.added_lines, reply.code)
        recovered = round((recall or 0.0) * len(change.added_lines))
        added_total += len(change.added_lines)
        recovered_total += recovered
        files.append({"path": change.path, "added_lines": len(change.added_lines), "recovered": recovered})
    recall = recovered_total / added_total if added_total else None
    return {
        "episode_id": episode.episode_id,
        "scenario": str(episode.scenario),
        "commit": episode.agent_input.ci_event.commit.sha,
        "files": files,
        "added_lines": added_total,
        "recovered_lines": recovered_total,
        "added_line_recall": recall,
        "memorised": None if recall is None else recall >= MEMORISED_RECALL,
    }


def run_probe(
    episodes: Sequence[Episode], repo: Path, llm: LLMClient, *, model_id: str | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict:
    """Probe every episode; the summary pools added lines over episodes (no episode weighs more for being short)."""
    results = []
    for episode in episodes:
        if progress:
            progress(episode.episode_id)
        results.append(probe_episode(llm, repo, episode))
    scored = [r for r in results if r["added_line_recall"] is not None]
    added = sum(r["added_lines"] for r in scored)
    return {
        "probe_version": PROBE_VERSION,
        "model_id": model_id,
        "memorised_threshold": MEMORISED_RECALL,
        "episodes": results,
        "summary": {
            "episodes_scored": len(scored),
            "pooled_added_line_recall": sum(r["recovered_lines"] for r in scored) / added if added else None,
            "episodes_memorised": sum(bool(r["memorised"]) for r in scored),
        },
    }
