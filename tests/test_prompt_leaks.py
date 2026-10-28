"""No pilot reference repair may reach a prompt.

repair-v2..v4 carried an example taken from pilot dev-05's developer fix (``cached(lru)`` -> ``cached(lru,
scoped=False)``) and a hint naming its test ('unscoped').  These checks keep every static prompt and every agent-view
episode free of the reference repairs' changed lines, the pilot failing-test names and those two tokens.
"""

from __future__ import annotations

import json

import pytest

from testorcode.agents import prompts
from testorcode.dataset import agent_view_path, episode_path, load_episodes
from testorcode.schemas.episode_schemas import PatchScope

WINDOW = 20  # shared substrings of this length or more count as a leak
EPISODES = [e for split in ("dev", "test") for e in load_episodes(episode_path(split))]
STATIC = {
    name: value for name, value in vars(prompts).items()
    if name.isupper() and isinstance(value, str) and not name.endswith("_VERSION")
} | {f"REPAIR_SCOPE_RULES[{scope}]": text for scope, text in prompts.REPAIR_SCOPE_RULES.items()}
STATIC["REPAIR_SYSTEM(test)"] = prompts.REPAIR_SYSTEM.format(scope_rule=prompts.REPAIR_SCOPE_RULES[PatchScope.TEST])


def changed_lines(diff: str, sign: str) -> list[str]:
    return [line[1:].strip() for line in diff.splitlines()
            if line.startswith(sign) and not line.startswith(sign * 3) and len(line[1:].strip()) >= WINDOW]


def _test_hunks(diff: str) -> str:
    """Only the hunks of test files (a reference commit may also carry source changes the CI diff already shows)."""
    keep, out = False, []
    for line in diff.splitlines():
        if line.startswith("diff --git "):
            keep = " b/tests/" in line or line.rsplit("/", 1)[-1].startswith("test_")
        if keep:
            out.append(line)
    return "\n".join(out)


def answer_lines(episode) -> list[str]:
    """Test lines the developer's repair added (and that were not already there): what a leak would hand the agent."""
    out = []
    for repair in episode.ground_truth.reference_repairs:
        diff = _test_hunks(repair.unified_diff)
        removed = set(changed_lines(diff, "-"))
        out += [line for line in changed_lines(diff, "+") if line not in removed]
    return out


def windows(text: str) -> set[str]:
    return {text[i : i + WINDOW] for i in range(len(text) - WINDOW + 1)}


def test_pilot_episodes_have_reference_repairs_to_check() -> None:
    assert sum(1 for e in EPISODES for line in answer_lines(e)) > 0


@pytest.mark.parametrize("name", sorted(STATIC))
def test_static_prompt_shares_nothing_with_any_reference_repair(name: str) -> None:
    text = STATIC[name]
    for episode in EPISODES:
        for repair in episode.ground_truth.reference_repairs:
            lines = changed_lines(repair.unified_diff, "+") + changed_lines(repair.unified_diff, "-")
            shared = [w for line in lines for w in windows(line) if w in text]
            assert not shared, f"{name} shares {shared[0]!r} with {episode.episode_id}'s reference repair"


@pytest.mark.parametrize("name", sorted(STATIC))
def test_static_prompt_names_no_pilot_test_and_no_known_leak(name: str) -> None:
    text = STATIC[name].lower()
    for token in ("unscoped", "scoped=false", "cached(lru"):
        assert token not in text, f"{name} contains {token!r}"
    names = {t.test_name.split("[", 1)[0] for e in EPISODES for t in e.agent_input.ci_event.failing_tests}
    assert not [n for n in names if n.lower() in text]


@pytest.mark.parametrize("split", ["dev", "test"])
def test_agent_view_never_contains_a_reference_repairs_answer(split: str) -> None:
    # Everything a user prompt is rendered from comes from the agent view (plus the worktree at the CI commit).
    views = {}
    for line in agent_view_path(split).read_text().splitlines():
        if line.strip():
            record = json.loads(line)
            views[record["episode_id"]] = line
    for episode in (e for e in EPISODES if e.episode_id in views):
        # A line the CI change itself shows in a source file (e.g. dev-08's docstring example reused by the developer's
        # new test) is visible by design, not leaked.
        shown = "\n".join(c.unified_diff or "" for c in episode.agent_input.ci_event.changed_files
                          if not c.path.startswith("tests/"))
        leaked = [a for a in answer_lines(episode) if a in views[episode.episode_id] and a not in shown]
        assert not leaked, f"{episode.episode_id}: agent view contains {leaked[0]!r}"
        assert "reference_repair" not in views[episode.episode_id]
