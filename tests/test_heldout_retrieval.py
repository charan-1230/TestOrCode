"""Held-out corpora (72 episodes): no future-commit evidence, unique ids, gold present, BM25 evaluation runs offline."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "heldout" / "eval_retrieval_heldout.py"
pytestmark = pytest.mark.skipif(not (ROOT / "datasets" / "heldout_attrs.jsonl").is_file(), reason="held-out dataset missing")


@pytest.fixture(scope="module")
def ev():
    spec = importlib.util.spec_from_file_location("eval_retrieval_heldout", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_every_corpus_is_cutoff_clean_and_unique(ev) -> None:
    episodes = ev.load_all()
    assert len(episodes) == 72
    for _, e in episodes:
        corpus = e.agent_input.corpus
        ids = [i.evidence_id for i in corpus.items]
        assert len(ids) == len(set(ids)), e.episode_id
        assert corpus.history_cutoff_commit == e.agent_input.ci_event.commit.sha
        assert all(i.commit_index is None or i.commit_index <= corpus.history_cutoff_index for i in corpus.items), e.episode_id
        future = set(e.ground_truth.future_commit_shas or [])
        assert not any(i.commit_sha in future for i in corpus.items if i.commit_sha), e.episode_id


def test_gold_ids_exist_in_the_corpus_and_agent_view_hides_them(ev) -> None:
    for _, e in ev.load_all():
        ids = {i.evidence_id for i in e.agent_input.corpus.items}
        assert ev.gold_of(e) <= ids, e.episode_id
        assert "gold_evidence_ids" not in e.agent_payload().__str__()


def test_message_free_corpus_drops_only_the_ci_message(ev) -> None:
    _, e = next((lib, e) for lib, e in ev.load_all() if ev.gold_of(e))
    full = {i.evidence_id for i in ev.corpus_for(e, "standard").items}
    free = {i.evidence_id for i in ev.corpus_for(e, "msgfree").items}
    assert full - free == {ev.ci_message_id(e)}


def test_bm25_evaluation_runs_offline_and_reports_every_metric(ev) -> None:
    from testorcode.config import Settings

    result = ev.evaluate(Settings.from_env({}))
    rows = [r for r in result["rows"] if r["protocol"] == "standard" and r["method"] == "intent"]
    assert len(rows) == 42  # the gold-bearing episodes
    assert {"recall@1", "recall@3", "recall@5", "recall@10", "hit@5", "precision@5", "stale@5"} <= set(rows[0])


# --- audit regressions: leakage, gold visibility, metric arithmetic ---------------------------------------------------

import json  # noqa: E402
import subprocess  # noqa: E402

HELDOUT_CLONES = {"more_itertools": "more-itertools", "attrs": "attrs", "packaging": "packaging"}
CLONE_ROOT = ROOT.parent.parent / "screening" / "heldout"


def test_agent_view_files_carry_no_ground_truth() -> None:
    from testorcode.schemas.episode_schemas import find_leaked_ground_truth

    for path in sorted((ROOT / "datasets").glob("heldout_*_agent_view.jsonl")):
        for line in path.read_text().splitlines():
            assert not find_leaked_ground_truth(json.loads(line)), path.name


def test_corpus_commits_are_ancestors_of_the_ci_parent(ev) -> None:
    """Every history item is an ancestor of the parent (the git tools' cutoff); only the CI commit itself may be newer."""
    if not CLONE_ROOT.is_dir():
        pytest.skip("held-out clones not available")
    cache: dict[tuple[str, str], set[str]] = {}
    for lib, e in ev.load_all():
        repo = CLONE_ROOT / HELDOUT_CLONES[lib]
        ci = e.agent_input.ci_event
        key = (lib, ci.parent_commit.sha)
        if key not in cache:
            out = subprocess.run(["git", "-C", str(repo), "rev-list", ci.parent_commit.sha], capture_output=True, text=True)
            cache[key] = set(out.stdout.split())
        allowed = cache[key] | {ci.commit.sha}
        for item in e.agent_input.corpus.items:
            assert not item.commit_sha or item.commit_sha in allowed, (e.episode_id, item.evidence_id)


def test_no_reference_repair_text_reaches_the_corpus(ev) -> None:
    for _, e in ev.load_all():
        for repair in e.ground_truth.reference_repairs or []:
            added = [ln[1:].strip() for ln in (repair.unified_diff or "").splitlines()
                     if ln.startswith("+") and not ln.startswith("+++") and len(ln.strip()) > 3]
            triples = ["\n".join(added[i:i + 3]) for i in range(len(added) - 2)]
            for item in e.agent_input.corpus.items:
                body = "\n".join(ln.strip() for ln in (item.text or "").splitlines())
                assert not any(t in body for t in triples), (e.episode_id, item.evidence_id)
        assert all(i.source_type.value != "test_file" for i in e.agent_input.corpus.items)


def test_msg_gold_is_always_the_ci_message_and_visibility_tiers_are_pinned(ev) -> None:
    standard = msgfree = discoverable = 0
    for _, e in ev.load_all():
        gold = ev.gold_of(e)
        assert all(g == ev.ci_message_id(e) for g in gold if g.startswith("msg:")), e.episode_id
        standard += bool(ev.protocol_gold(e, gold, "standard"))
        msgfree += bool(ev.protocol_gold(e, gold, "msgfree"))
        discoverable += bool(ev.protocol_gold(e, gold, "discoverable"))
    # dataset facts behind the message-free sample size: a property of the frozen gold, not of the evaluation
    assert (standard, msgfree, discoverable) == (42, 5, 2)


def test_metrics_match_hand_computation(ev) -> None:
    row = ev.metrics(["a", "b", "g", "c", "d", "e"], {"g", "z"}, {"b", "d"})
    assert row["rank"] == 3
    assert (row["recall@1"], row["recall@3"], row["recall@5"], row["recall@10"]) == (0.0, 0.5, 0.5, 0.5)
    assert (row["hit@1"], row["hit@3"], row["hit@5"]) == (0.0, 1.0, 1.0)
    assert row["precision@5"] == pytest.approx(1 / 5)  # fixed denominator 5
    assert row["stale@5"] == pytest.approx(2 / 5) and row["stale_in_top5"] == 1.0
    short = ev.metrics(["g"], {"g"}, set())
    assert short["precision@5"] == pytest.approx(1 / 5) and short["recall@10"] == 1.0  # not 1.0: only one item came back


def test_bootstrap_is_deterministic_and_brackets_the_mean(ev) -> None:
    diffs = [0.0, 0.5, 0.0, 1.0, 0.5, 0.0, 0.0, 0.5, 1.0, 0.0]
    lo, hi = ev.bootstrap(diffs)
    assert (lo, hi) == ev.bootstrap(diffs) and lo <= sum(diffs) / len(diffs) <= hi
    assert ev.bootstrap([0.0] * 5) == (0.0, 0.0)


def test_stale_text_is_scored_only_for_update_test_functions(ev) -> None:
    from testorcode.evaluation import stale_evidence_ids
    from testorcode.schemas.episode_schemas import RepairDirection

    changed = 0
    for _, e in ev.load_all():
        labels = e.ground_truth.per_test_labels
        fix_only = {lbl.function_qualname for lbl in labels if lbl.gold_direction is RepairDirection.FIX_CODE and lbl.function_qualname}
        stale = ev.stale_for_episode(e)
        upd = {lbl.function_qualname for lbl in labels if lbl.gold_direction is RepairDirection.UPDATE_TEST}
        unfiltered = set().union(*(stale_evidence_ids(e, f) for f in upd | fix_only if f)) if (upd | fix_only) else set()
        changed += stale != unfiltered  # the filter really removes something for at least one episode (if S3 has any)
        assert stale <= unfiltered
    assert changed > 0  # 21 episodes at the time of the audit


def test_withheld_message_is_not_in_the_corpus(ev, monkeypatch) -> None:
    """A condition that withholds the message must not keep the CI message item (msg:<sha12>) in a retained corpus."""
    import dataclasses

    from testorcode import conditions
    from testorcode.schemas.result_schemas import ExperimentCondition

    _, e = ev.load_all()[0]
    view = e.agent_input
    leaky = dataclasses.replace(conditions.profile_for(ExperimentCondition.A2_NO_MEMORY), sees_commit_message=False)
    monkeypatch.setattr(conditions, "profile_for", lambda _c: leaky)
    out = conditions.episode_for_condition(view, ExperimentCondition.A2_NO_MEMORY)
    ids = {i.evidence_id for i in out.corpus.items}
    assert ev.ci_message_id(e) not in ids and len(ids) == len(view.corpus.items) - 1
    assert out.ci_event.commit_message is None
