"""Hybrid retrieval (proposal: "BM25 plus code embeddings, optionally reranked"), behind the ``Retriever`` protocol.

``HybridRetriever`` ranks the episode corpus twice, by BM25 (``retrieval.Bm25Retriever``) and by the cosine similarity of
dense embeddings, fuses the two rankings with reciprocal-rank fusion (the same RRF as ``retrieval.search_many``) and,
when a reranker is given, reorders the fused top candidates with it.  Results keep the corpus ids, so citations stay
traceable; ``retrieval_method`` says how an item was ranked (``hybrid`` or ``rerank``).

Backends (``Settings.embedding_backend`` / ``rerank_backend``):

* ``voyage`` - the proposal's choice: ``voyage-code-3`` embeddings and ``rerank-2.5`` over HTTPS (``VOYAGE_API_KEY``).
  Calls go out from the orchestrator only, never from the test sandbox.
* ``local`` - sentence-transformers models on this machine (``all-MiniLM-L6-v2`` and the ``ms-marco-MiniLM-L-6-v2``
  cross-encoder by default): a fallback that needs no key and runs offline once the models are cached.  It is a
  general-text model, not a code model; results with it are reported as such.

Embeddings are deterministic for a fixed model, so they are cached per (model, text) in memory and, with
``embedding_cache_dir``, on disk.  Every condition that retrieves uses the same index; conditions differ only in the
queries they send (B1's naive query, the Analyst's intent queries).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Final, Literal, Protocol

from testorcode.errors import ConfigError, ToolError
from testorcode.logging_config import get_logger
from testorcode.retrieval import RRF_K, Bm25Retriever, _evidence, reciprocal_rank_fusion
from testorcode.schemas.episode_schemas import EvidenceItem
from testorcode.schemas.result_schemas import RetrievalMethod, RetrievedEvidence

logger = get_logger(__name__)

VOYAGE_EMBED_URL: Final[str] = "https://api.voyageai.com/v1/embeddings"
VOYAGE_RERANK_URL: Final[str] = "https://api.voyageai.com/v1/rerank"
VOYAGE_BATCH: Final[int] = 64
EMBED_TEXT_CHARS: Final[int] = 4000  # long items (docs sections) are embedded by their head
RERANK_CANDIDATES: Final[int] = 30  # fused candidates the reranker reorders
CANDIDATE_DEPTH: Final[int] = 50  # how deep each of the two rankings goes before fusion

InputType = Literal["query", "document"]


class Embedder(Protocol):
    model_id: str

    def embed(self, texts: Sequence[str], *, input_type: InputType) -> list[list[float]]: ...


class Reranker(Protocol):
    model_id: str

    def rerank(self, query: str, documents: Sequence[str]) -> list[float]:
        """One relevance score per document (higher = more relevant)."""
        ...


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #


class VoyageEmbedder:
    """``voyage-code-3`` (or another Voyage model) over the REST API."""

    def __init__(self, api_key: str, model: str = "voyage-code-3", *, timeout: float = 60.0) -> None:
        if not api_key:
            raise ConfigError("VOYAGE_API_KEY is required for the voyage embedding backend")
        self._key, self.model_id, self._timeout = api_key, model, timeout

    def embed(self, texts: Sequence[str], *, input_type: InputType) -> list[list[float]]:
        out: list[list[float]] = []
        for start in range(0, len(texts), VOYAGE_BATCH):
            batch = list(texts[start:start + VOYAGE_BATCH])
            data = _post(VOYAGE_EMBED_URL, self._key, self._timeout,
                         {"input": batch, "model": self.model_id, "input_type": input_type})
            rows = sorted(data.get("data", []), key=lambda r: r["index"])
            if len(rows) != len(batch):
                raise ToolError(f"voyage returned {len(rows)} embeddings for {len(batch)} texts")
            out += [r["embedding"] for r in rows]
        return out


class VoyageReranker:
    def __init__(self, api_key: str, model: str = "rerank-2.5", *, timeout: float = 60.0) -> None:
        if not api_key:
            raise ConfigError("VOYAGE_API_KEY is required for the voyage rerank backend")
        self._key, self.model_id, self._timeout = api_key, model, timeout

    def rerank(self, query: str, documents: Sequence[str]) -> list[float]:
        data = _post(VOYAGE_RERANK_URL, self._key, self._timeout,
                     {"query": query, "documents": list(documents), "model": self.model_id})
        scores = [0.0] * len(documents)
        for row in data.get("data", []):
            scores[row["index"]] = float(row["relevance_score"])
        return scores


def _post(url: str, key: str, timeout: float, payload: dict) -> dict:
    import httpx  # installed with the openai SDK

    try:
        response = httpx.post(url, json=payload, timeout=timeout, headers={"Authorization": f"Bearer {key}"})
    except httpx.HTTPError as exc:
        raise ToolError(f"voyage request failed: {type(exc).__name__}") from None
    if response.status_code != 200:
        raise ToolError(f"voyage request failed: HTTP {response.status_code}")
    return response.json()


class LocalEmbedder:
    """A sentence-transformers model on this machine (loaded once, offline if already cached)."""

    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model_id = model
        self._model = None

    def embed(self, texts: Sequence[str], *, input_type: InputType) -> list[list[float]]:
        if self._model is None:
            self._model = _load_local("SentenceTransformer", self.model_id)
        vectors = self._model.encode(list(texts), batch_size=32, show_progress_bar=False, normalize_embeddings=True)
        return [list(map(float, v)) for v in vectors]


class LocalCrossEncoder:
    def __init__(self, model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        self.model_id = model
        self._model = None

    def rerank(self, query: str, documents: Sequence[str]) -> list[float]:
        if self._model is None:
            self._model = _load_local("CrossEncoder", self.model_id)
        return [float(s) for s in self._model.predict([(query, d) for d in documents], show_progress_bar=False)]


def _load_local(kind: str, model: str):
    try:
        import sentence_transformers
    except ImportError:
        raise ConfigError("the local backend needs sentence-transformers (pip install -e .[embeddings])") from None
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    cls = getattr(sentence_transformers, kind)
    try:
        return cls(model, local_files_only=True)  # offline when the model is cached
    except Exception:  # noqa: BLE001 - not cached (or an older library): let it download once
        return cls(model)


class CachedEmbedder:
    """Memoises another embedder per (model, input type, text); with ``cache_dir`` also on disk (one JSON per model)."""

    def __init__(self, inner: Embedder, cache_dir: Path | None = None) -> None:
        self.inner, self.model_id = inner, inner.model_id
        self._path = (Path(cache_dir) / f"{hashlib.sha256(inner.model_id.encode()).hexdigest()[:16]}.json"
                      if cache_dir else None)
        self._cache: dict[str, list[float]] = {}
        if self._path is not None and self._path.is_file():
            self._cache = json.loads(self._path.read_text(encoding="utf-8"))

    def embed(self, texts: Sequence[str], *, input_type: InputType) -> list[list[float]]:
        keys = [hashlib.sha256(f"{input_type}\0{t}".encode()).hexdigest() for t in texts]
        missing = [i for i, k in enumerate(keys) if k not in self._cache]
        if missing:
            vectors = self.inner.embed([texts[i] for i in missing], input_type=input_type)
            for i, vector in zip(missing, vectors, strict=True):
                self._cache[keys[i]] = vector
            if self._path is not None:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                tmp = self._path.with_suffix(".tmp")
                tmp.write_text(json.dumps(self._cache), encoding="utf-8")
                os.replace(tmp, self._path)
        return [self._cache[k] for k in keys]


# --------------------------------------------------------------------------- #
# The retriever
# --------------------------------------------------------------------------- #


def _document_text(item: EvidenceItem) -> str:
    return f"{item.source_ref}\n{item.text or ''}"[:EMBED_TEXT_CHARS]


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


class HybridRetriever:
    """BM25 + dense ranking fused with RRF; optional rerank of the fused top ``RERANK_CANDIDATES``.

    Document embeddings are computed once per corpus (on first search).  Ties are broken by ``evidence_id`` as in BM25,
    so a fixed model gives a deterministic ranking.
    """

    def __init__(self, items: Sequence[EvidenceItem], embedder: Embedder, reranker: Reranker | None = None) -> None:
        self._items = list(items)
        self._by_id = {i.evidence_id: i for i in self._items}
        self._bm25 = Bm25Retriever(self._items)
        self._embedder = embedder
        self._reranker = reranker
        self._vectors: list[list[float]] | None = None

    def __len__(self) -> int:
        return len(self._items)

    def _dense(self, query: str, depth: int) -> list[tuple[str, float]]:
        if not self._items:
            return []
        if self._vectors is None:
            self._vectors = self._embedder.embed([_document_text(i) for i in self._items], input_type="document")
        q = self._embedder.embed([query], input_type="query")[0]
        scored = [(item.evidence_id, _cosine(q, v)) for item, v in zip(self._items, self._vectors, strict=True)]
        return sorted(scored, key=lambda kv: (-kv[1], kv[0]))[:depth]

    def search(self, query: str, *, top_k: int) -> list[RetrievedEvidence]:
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        if not query.strip():
            return []
        keyword = [h.evidence_id for h in self._bm25.search(query, top_k=CANDIDATE_DEPTH)]
        dense = [i for i, _ in self._dense(query, CANDIDATE_DEPTH)]
        fused = reciprocal_rank_fusion([keyword, dense], k=RRF_K)
        method = RetrievalMethod.HYBRID
        if self._reranker is not None and fused:
            head = fused[:RERANK_CANDIDATES]
            scores = self._reranker.rerank(query, [_document_text(self._by_id[i]) for i, _ in head])
            fused = sorted(zip((i for i, _ in head), scores, strict=True), key=lambda kv: (-kv[1], kv[0]))
            method = RetrievalMethod.RERANK
        return [_evidence(self._by_id[i], query, rank, score, method) for rank, (i, score) in enumerate(fused[:top_k], 1)]


def build_components(settings) -> tuple[Embedder, Reranker | None]:
    """The embedder (cached) and optional reranker that ``settings`` selects.

    Raises:
        ConfigError: a Voyage backend without ``VOYAGE_API_KEY``, or the local backend without sentence-transformers.
    """
    key = settings.voyage_api_key.get_secret_value() if settings.voyage_api_key is not None else ""
    if settings.embedding_backend == "voyage":
        embedder: Embedder = VoyageEmbedder(key, settings.voyage_embedding_model)
    else:
        embedder = LocalEmbedder(settings.local_embedding_model)
    reranker: Reranker | None = None
    if settings.rerank_backend == "voyage":
        reranker = VoyageReranker(key, settings.voyage_rerank_model)
    elif settings.rerank_backend == "local":
        # One loaded cross-encoder per process (loading it per episode costs seconds each time).
        reranker = _RERANKERS.setdefault(settings.local_rerank_model, LocalCrossEncoder(settings.local_rerank_model))
    return _shared_cache(embedder, settings.embedding_cache_dir), reranker


_CACHES: dict[tuple[str, str | None], CachedEmbedder] = {}
_RERANKERS: dict[str, LocalCrossEncoder] = {}


def _shared_cache(embedder: Embedder, cache_dir: Path | None) -> CachedEmbedder:
    """One cache per (model, directory) per process: every episode of an experiment reuses the corpus embeddings."""
    key = (embedder.model_id, str(cache_dir) if cache_dir else None)
    if key not in _CACHES:
        _CACHES[key] = CachedEmbedder(embedder, cache_dir)
    return _CACHES[key]
