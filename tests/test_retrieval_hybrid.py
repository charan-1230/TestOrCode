"""Hybrid retrieval (``retrieval_hybrid``): BM25 + dense fused with RRF, optional rerank, cached embeddings."""

from __future__ import annotations

from pathlib import Path

import pytest

from testorcode.config import Settings
from testorcode.errors import ConfigError
from testorcode.retrieval import Bm25Retriever, build_retriever
from testorcode.retrieval_hybrid import CachedEmbedder, HybridRetriever, build_components
from testorcode.schemas.episode_schemas import EvidenceItem, EvidenceSourceType, RetrievalCorpusSpec
from testorcode.schemas.result_schemas import RetrievalMethod

ITEMS = [
    EvidenceItem(evidence_id="msg:a", source_type=EvidenceSourceType.COMMIT_MESSAGE, source_ref="a",
                 text="cache keys are now scoped to the decorated function"),
    EvidenceItem(evidence_id="msg:b", source_type=EvidenceSourceType.COMMIT_MESSAGE, source_ref="b",
                 text="docs: fix a typo in the readme"),
    EvidenceItem(evidence_id="doc:c", source_type=EvidenceSourceType.DOCUMENTATION, source_ref="c",
                 text="memoization helpers and their options"),
]


class KeywordEmbedder:
    """Deterministic fake: one dimension per vocabulary word ("memo" counts as "cache")."""

    model_id = "fake-embedder"
    VOCAB = ("cache", "scoped", "typo", "readme", "function")

    def __init__(self) -> None:
        self.calls: list[tuple[int, str]] = []

    def embed(self, texts, *, input_type):
        self.calls.append((len(texts), input_type))
        return [[float(w in t.lower() or (w == "cache" and "memo" in t.lower())) for w in self.VOCAB] for t in texts]


class ReverseReranker:
    model_id = "fake-reranker"

    def rerank(self, query, documents):
        return [float(-i) for i in range(len(documents))][::-1]  # last fused candidate first


def test_dense_half_finds_what_keywords_miss() -> None:
    query = "memoization"
    assert [h.evidence_id for h in Bm25Retriever(ITEMS).search(query, top_k=3)] == ["doc:c"]
    hits = HybridRetriever(ITEMS, KeywordEmbedder()).search("cache function", top_k=3)
    assert hits[0].evidence_id == "msg:a" and "doc:c" in [h.evidence_id for h in hits]  # dense: "memo" ~ "cache"
    assert {h.retrieval_method for h in hits} == {RetrievalMethod.HYBRID} and [h.rank for h in hits] == [1, 2, 3][: len(hits)]


def test_document_embeddings_are_computed_once_per_corpus() -> None:
    embedder = KeywordEmbedder()
    retriever = HybridRetriever(ITEMS, embedder)
    retriever.search("cache", top_k=2)
    retriever.search("typo", top_k=2)
    assert embedder.calls == [(3, "document"), (1, "query"), (1, "query")]


def test_reranker_reorders_the_fused_candidates() -> None:
    plain = [h.evidence_id for h in HybridRetriever(ITEMS, KeywordEmbedder()).search("cache scoped", top_k=3)]
    hits = HybridRetriever(ITEMS, KeywordEmbedder(), ReverseReranker()).search("cache scoped", top_k=3)
    assert [h.evidence_id for h in hits] == list(reversed(plain))
    assert {h.retrieval_method for h in hits} == {RetrievalMethod.RERANK}


def test_embedding_cache_persists_on_disk(tmp_path: Path) -> None:
    inner = KeywordEmbedder()
    first = CachedEmbedder(inner, tmp_path).embed(["a cache", "readme"], input_type="document")
    again = CachedEmbedder(KeywordEmbedder(), tmp_path)
    assert again.embed(["a cache", "readme"], input_type="document") == first
    assert again.inner.calls == []  # served from the file


def test_build_retriever_keeps_bm25_by_default_and_needs_a_key_for_voyage() -> None:
    corpus = RetrievalCorpusSpec(corpus_id="c", history_cutoff_commit="a1b2c3d4", items=ITEMS)
    assert isinstance(build_retriever(corpus), Bm25Retriever)
    assert isinstance(build_retriever(corpus, settings=Settings.from_env({})), Bm25Retriever)
    with pytest.raises(ConfigError, match="VOYAGE_API_KEY"):
        build_components(Settings.from_env({"RETRIEVAL_MODE": "hybrid", "EMBEDDING_BACKEND": "voyage"}))


def test_hybrid_settings_build_a_hybrid_retriever(monkeypatch) -> None:
    import testorcode.retrieval_hybrid as hybrid

    monkeypatch.setattr(hybrid, "LocalEmbedder", lambda model: KeywordEmbedder())
    monkeypatch.setattr(hybrid, "_CACHES", {})
    settings = Settings.from_env({"RETRIEVAL_MODE": "hybrid", "EMBEDDING_BACKEND": "local"})
    corpus = RetrievalCorpusSpec(corpus_id="c", history_cutoff_commit="a1b2c3d4", items=ITEMS)
    retriever = build_retriever(corpus, settings=settings)
    assert isinstance(retriever, HybridRetriever) and retriever.search("cache", top_k=1)[0].evidence_id == "msg:a"
