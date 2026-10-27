"""Retrieval: BM25 behaviour on a synthetic corpus, fusion, queries, tool, memory evidence, metrics, pilot smoke."""

from __future__ import annotations

import pytest

from testorcode.dataset import agent_view_path, load_agent_inputs
from testorcode.errors import BudgetExceededError, HistoryCutoffError, ToolError
from testorcode.retrieval import (
    RetrievalLog,
    SearchRepoTool,
    build_retriever,
    citation_existence_rate,
    hit_rank,
    intent_queries,
    memory_evidence,
    naive_query,
    recall_at_k,
    reciprocal_rank_fusion,
    search_many,
    tokenize,
)
from testorcode.schemas.contracts import MemoryReadResult, MemoryRecordCheck
from testorcode.schemas.episode_schemas import (
    CIEvent,
    CommitRef,
    EvidenceItem,
    EvidenceSourceType,
    ExecutableProbe,
    FailingTest,
    FileChange,
    MemoryRecord,
    RepairDirection,
    RetrievalCorpusSpec,
    ToolName,
)
from testorcode.schemas.result_schemas import ProbeStatus, RetrievalMethod

SHA = "a1b2c3d4e5f60718293a4b5c6d7e8f9012345678"
T = EvidenceSourceType


def item(eid: str, text: str, kind: T = T.DOCUMENTATION, index: int | None = None) -> EvidenceItem:
    return EvidenceItem(evidence_id=eid, source_type=kind, source_ref=f"ref/{eid}", text=text, commit_index=index)


def corpus(items: list[EvidenceItem] | None, cutoff: int = 10) -> RetrievalCorpusSpec:
    return RetrievalCorpusSpec(corpus_id="c", history_cutoff_commit=SHA, history_cutoff_index=cutoff, items=items)


ITEMS = [
    item("msg:1", "rotate_file now keeps the requested number of generations", T.COMMIT_MESSAGE, 3),
    item("doc:1", "rotate files and keep backups of log files"),
    item("ds:1", "Return the cached value for a key", T.DOCSTRING),
    item("ds:2", "Return the cached value for a key", T.DOCSTRING),
    item("other", "unrelated text about sockets"),
]


def ci_event(message: str | None = "fileutils: keep rotate_file generations (#483)") -> CIEvent:
    diff = (
        "--- a/boltons/fileutils.py\n+++ b/boltons/fileutils.py\n"
        "@@ -700,6 +700,9 @@ def rotate_file(filename, *, keep: int = 5):\n"
        "+    if os.path.exists(kept_names[-1]):\n+        os.remove(kept_names[-1])\n-    old_generation = 1\n"
    )
    return CIEvent(
        event_id="ci",
        commit=CommitRef(sha=SHA),
        commit_message=message,
        commit_message_availability="available" if message else "absent",
        changed_files=[FileChange(path="boltons/fileutils.py", change_kind="modified", unified_diff=diff)],
        failing_tests=[
            FailingTest(
                test_id="tests/test_fileutils.py::test_rotate_file_full_rotation",
                test_file="tests/test_fileutils.py",
                test_name="test_rotate_file_full_rotation",
                failure_message="AssertionError: assert not True where exists at 0x7f0000000000",
            )
        ],
    )


def test_tokenize_splits_snake_and_camel_and_keeps_compound() -> None:
    tokens = tokenize("rotate_file CachedFunction x")
    assert {"rotate", "file", "cached", "function", "rotate_file", "cachedfunction"} <= set(tokens)
    assert "x" not in tokens


def test_ranking_is_deterministic_with_id_tie_break() -> None:
    retriever = build_retriever(corpus(ITEMS))
    first = retriever.search("cached value key", top_k=5)
    assert [h.evidence_id for h in first] == [h.evidence_id for h in retriever.search("cached value key", top_k=5)]
    ids = [h.evidence_id for h in first]
    assert ids.index("ds:1") < ids.index("ds:2")  # identical text -> evidence_id order
    assert [h.rank for h in first] == list(range(1, len(first) + 1))
    assert all(h.retrieval_method is RetrievalMethod.BM25 and h.score and h.score > 0 for h in first)


def test_results_are_traceable_to_the_corpus() -> None:
    retriever = build_retriever(corpus(ITEMS))
    (hit, *_) = retriever.search("rotate_file generations", top_k=3)
    assert hit.evidence_id == "msg:1" and hit.source_ref == "ref/msg:1" and hit.source_type is T.COMMIT_MESSAGE
    assert hit.retrieval_query == "rotate_file generations" and "requested number" in (hit.snippet or "")
    assert {h.evidence_id for h in retriever.search("anything rotate sockets", top_k=10)} <= {i.evidence_id for i in ITEMS}


def test_no_match_empty_query_and_top_k() -> None:
    retriever = build_retriever(corpus(ITEMS))
    assert retriever.search("zzzzqqq", top_k=3) == []
    assert retriever.search("   ", top_k=3) == []
    assert len(retriever.search("cached rotate sockets", top_k=1)) == 1
    with pytest.raises(ValueError, match="top_k"):
        retriever.search("x", top_k=0)
    assert build_retriever(corpus([])).search("x", top_k=3) == []


def test_tiny_corpus_still_returns_matches() -> None:
    # BM25Okapi gives a term present in half of 2 documents an IDF of 0; that must not become "nothing found".
    retriever = build_retriever(corpus([item("a", "normalise capitalised output"), item("b", "unrelated sockets text")]))
    assert [h.evidence_id for h in retriever.search("normalise output", top_k=5)] == ["a"]


def test_build_guards() -> None:
    with pytest.raises(NotImplementedError):
        build_retriever(corpus(None))
    with pytest.raises(HistoryCutoffError):
        retriever_items = [item("future", "text", T.COMMIT_MESSAGE, 5)]
        spec = RetrievalCorpusSpec.model_construct(  # bypass the schema check to test the defensive guard
            corpus_id="c", history_cutoff_commit=SHA, history_cutoff_index=3, items=retriever_items
        )
        build_retriever(spec)
    with pytest.raises(ValueError, match="duplicate"):
        build_retriever(RetrievalCorpusSpec.model_construct(corpus_id="c", history_cutoff_commit=SHA, history_cutoff_index=None,
                                                            items=[item("a", "x y"), item("a", "z w")]))


def test_reciprocal_rank_fusion() -> None:
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "a"], ["c"]])
    assert [i for i, _ in fused] == ["b", "a", "c"] or [i for i, _ in fused] == ["a", "b", "c"]
    assert fused[0][1] >= fused[-1][1]
    assert reciprocal_rank_fusion([]) == []
    assert [i for i, _ in reciprocal_rank_fusion([["x"], ["x"]])] == ["x"]


def test_search_many_fuses_and_reranks() -> None:
    retriever = build_retriever(corpus(ITEMS))
    hits = search_many(retriever, ["rotate_file generations", "cached value key"], top_k=3)
    assert len(hits) == 3 and [h.rank for h in hits] == [1, 2, 3]
    assert {"msg:1"} <= {h.evidence_id for h in hits}
    assert len({h.evidence_id for h in hits}) == 3


def test_naive_query_is_the_test_text_and_intent_queries_are_not() -> None:
    event = ci_event()
    assert "test_rotate_file_full_rotation" in naive_query(event)
    queries = intent_queries(event)
    joined = " ".join(queries).lower()
    assert "rotate_file" in joined and "fileutils" in joined
    assert "test_rotate_file_full_rotation" not in joined
    assert "assertionerror" not in joined  # failure text is opt-in (it hurt ranking on the dev split)
    assert any("keep rotate_file generations" in q for q in queries)
    assert not any("keep rotate_file generations" in q for q in intent_queries(event, include_message=False))
    assert intent_queries(ci_event(message=None))  # still has function and failure queries


def test_failure_text_is_opt_in_and_scrubbed() -> None:
    event = ci_event()
    text = " ".join(intent_queries(event, include_failure_text=True)).lower()
    assert "assertionerror" in text and "0x7f" not in text and "test_rotate_file_full_rotation" not in text


def test_group_query_uses_the_group_function_not_every_changed_function() -> None:
    from testorcode.schemas.contracts import FailureGroupDetail
    from testorcode.schemas.episode_schemas import FailureGroup

    event = ci_event()
    group = FailureGroupDetail(
        group=FailureGroup(group_id="g", function_qualname="boltons.fileutils.rotate_file", test_ids=[event.failing_tests[0].test_id]),
        failing_tests=tuple(event.failing_tests),
        affected_files=("boltons/fileutils.py",),
    )
    names = intent_queries(event, group, include_message=False)[0].split()
    assert names[:2] == ["fileutils", "rotate_file"] and "boltons" not in names


def test_search_repo_tool_permission_budget_and_logging() -> None:
    retriever = build_retriever(corpus(ITEMS))
    with pytest.raises(ToolError):
        SearchRepoTool(retriever, allowed_tools=[ToolName.RUN_TESTS])("rotate")
    tool = SearchRepoTool(retriever, allowed_tools=[ToolName.SEARCH_REPO], max_calls=2, top_k=2)
    out = tool("rotate_file generations")
    assert out[0]["evidence_id"] == "msg:1" and set(out[0]) == {"evidence_id", "source_type", "source_ref", "rank", "text"}
    tool("cached value")
    with pytest.raises(BudgetExceededError):
        tool("sockets")
    assert tool.calls == 2 and "msg:1" in tool.log.ids()
    assert tool.log.unknown_citations(["msg:1", "invented"]) == ["invented"]


def test_retrieval_log_keeps_first_sighting() -> None:
    retriever = build_retriever(corpus(ITEMS))
    log = RetrievalLog()
    log.add(retriever.search("rotate", top_k=2))
    first = {r.evidence_id: r.rank for r in log.records()}
    log.add(retriever.search("rotate files backups", top_k=5))
    assert {r.evidence_id: r.rank for r in log.records() if r.evidence_id in first} == first


def memory_check(rid: str, status: ProbeStatus, stale: bool = False, blocked: str | None = None) -> MemoryRecordCheck:
    record = MemoryRecord(
        record_id=rid, repo_id="r", function_qualname="m.f", behaviour_signature="f() keeps N files",
        direction=RepairDirection.UPDATE_TEST, probe=ExecutableProbe(probe_id="p", probe_code="assert 1"),
        approved_at_commit=SHA, flagged_stale=stale,
    )
    return MemoryRecordCheck(record=record, probe_status=status, blocked_reason=blocked)


def test_memory_evidence_only_from_valid_records() -> None:
    read = MemoryReadResult(checks=(
        memory_check("ok", ProbeStatus.PASSED),
        memory_check("failed", ProbeStatus.FAILED, blocked="probe_failed"),
        memory_check("unrun", ProbeStatus.NOT_RUN),
        memory_check("stale", ProbeStatus.PASSED, stale=True),
    ))
    out = memory_evidence(read)
    assert [e.evidence_id for e in out] == ["ok"]
    assert out[0].source_type is T.MEMORY_RECORD and out[0].retrieval_method is RetrievalMethod.MEMORY_LOOKUP
    assert memory_evidence(MemoryReadResult()) == []


def test_metrics() -> None:
    ranked = ["a", "b", "c", "d"]
    assert hit_rank(ranked, {"c", "z"}) == 3 and hit_rank(ranked, {"z"}) is None
    assert recall_at_k(ranked, {"a", "d"}, 3) == 0.5 and recall_at_k(ranked, {"a", "d"}, 4) == 1.0
    assert recall_at_k(ranked, set(), 3) is None  # no gold is not recall 0
    assert citation_existence_rate(["a", "x"], ranked) == 0.5
    assert citation_existence_rate([], ranked) is None


@pytest.mark.parametrize("split", ["dev", "test"])
def test_pilot_corpora_are_searchable_and_cutoff_clean(split: str) -> None:
    for episode in load_agent_inputs(agent_view_path(split)):
        corpus_spec = episode.corpus
        ids = {i.evidence_id for i in corpus_spec.items or []}
        retriever = build_retriever(corpus_spec)
        queries = intent_queries(episode.ci_event, include_message=False)
        assert queries, episode.episode_id
        hits = search_many(retriever, queries, top_k=8) + retriever.search(naive_query(episode.ci_event), top_k=8)
        assert hits and {h.evidence_id for h in hits} <= ids
        assert all(i.commit_index is None or i.commit_index <= corpus_spec.history_cutoff_index for i in corpus_spec.items or [])
        text = " ".join(queries).lower()
        assert not any(t.test_name.lower() in text for t in episode.ci_event.failing_tests)
