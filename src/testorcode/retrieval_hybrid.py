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
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Final, Literal, Protocol

from testorcode.errors import ConfigError, ToolError
from testorcode.logging_config import get_logger
from testorcode.retrieval import RRF_K, Bm25Retriever, _beyond, _evidence, reciprocal_rank_fusion
from testorcode.schemas.episode_schemas import EvidenceItem
from testorcode.schemas.result_schemas import RetrievalMethod, RetrievedEvidence

logger = get_logger(__name__)

VOYAGE_EMBED_URL: Final[str] = "https://api.voyageai.com/v1/embeddings"
VOYAGE_RERANK_URL: Final[str] = "https://api.voyageai.com/v1/rerank"
EMBED_TEXT_CHARS: Final[int] = 1500  # long items (docs sections) are embedded by their head
CHARS_PER_TOKEN: Final[float] = 3.0  # conservative estimate for rate limiting (code tokenises densely)
VOYAGE_RPM: Final[int] = 3  # per key (free tier)
VOYAGE_TPM: Final[int] = 9000  # per key; the limit is 10k, keep headroom for the estimate
RERANK_CANDIDATES: Final[int] = 30  # fused candidates the reranker reorders
CANDIDATE_DEPTH: Final[int] = 50  # how deep each of the two rankings goes before fusion

InputType = Literal["query", "document"]


def estimate_tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / CHARS_PER_TOKEN))


class Embedder(Protocol):
    model_id: str

    def embed(self, texts: Sequence[str], *, input_type: InputType) -> list[list[float]]: ...


class Reranker(Protocol):
    model_id: str

    def rerank(self, query: str, documents: Sequence[str]) -> list[float]:
        """One relevance score per document (higher = more relevant)."""
        ...


# --------------------------------------------------------------------------- #
# Rate-limited Voyage access
# --------------------------------------------------------------------------- #


class RateLimited(ToolError):
    """HTTP 429 from Voyage: that key is over its limit for now."""


class VoyageKeyPool:
    """Round-robins several API keys, each with its own rolling-minute request and token budget.

    ``acquire(tokens)`` returns the key that can serve a call soonest, sleeping until its window has room, and records
    the spend.  ``cool_down(key)`` benches a key after a 429.  ``clock``/``sleep`` are injectable for tests.
    """

    def __init__(self, keys: Sequence[str], *, rpm: int = VOYAGE_RPM, tpm: int = VOYAGE_TPM,
                 clock=time.monotonic, sleep=time.sleep) -> None:
        if not keys:
            raise ConfigError("no Voyage API key (set VOYAGE_API_KEY1..5 or VOYAGE_API_KEY)")
        self._keys, self._rpm, self._tpm, self._clock, self._sleep = list(keys), rpm, tpm, clock, sleep
        self._spend: dict[str, list[tuple[float, int]]] = {k: [] for k in self._keys}
        self._benched: dict[str, float] = {}

    def __len__(self) -> int:
        return len(self._keys)

    def _ready_at(self, key: str, tokens: int, now: float) -> float:
        window = [(t, n) for t, n in self._spend[key] if t > now - 60.0]
        self._spend[key] = window
        ready = self._benched.get(key, now)
        if len(window) >= self._rpm:
            ready = max(ready, window[len(window) - self._rpm][0] + 60.0)
        used = sum(n for _, n in window)
        for t, n in window:  # oldest first: wait until enough old spend has aged out of the window
            if used + tokens <= self._tpm:
                break
            used -= n
            ready = max(ready, t + 60.0)
        return ready

    def acquire(self, tokens: int) -> str:
        tokens = min(tokens, self._tpm)
        now = self._clock()
        ready = {k: self._ready_at(k, tokens, now) for k in self._keys}
        key = min(self._keys, key=lambda k: (ready[k], len(self._spend[k])))
        wait = ready[key] - now
        if wait > 0:
            logger.info("voyage rate limit: waiting %.1fs", wait)
            self._sleep(wait)
        self._spend[key].append((self._clock(), tokens))
        return key

    def cool_down(self, key: str, seconds: float = 65.0) -> None:
        self._benched[key] = self._clock() + seconds


def _post(url: str, pool: VoyageKeyPool, timeout: float, payload: dict, tokens: int, *, attempts: int = 8) -> dict:
    import httpx  # installed with the openai SDK

    failures = 0
    for _ in range(attempts * 2):
        key = pool.acquire(tokens)
        try:
            response = httpx.post(url, json=payload, timeout=timeout, headers={"Authorization": f"Bearer {key}"})
        except httpx.HTTPError as exc:  # connection drops and timeouts are transient: back off and retry
            failures += 1
            if failures >= attempts:
                raise ToolError(f"voyage request failed: {type(exc).__name__}") from None
            time.sleep(min(5.0 * failures, 30.0))
            continue
        if response.status_code == 429:
            pool.cool_down(key)
            continue
        if response.status_code != 200:
            raise ToolError(f"voyage request failed: HTTP {response.status_code}")
        return response.json()
    raise RateLimited("voyage rate limit persisted across every key")


def _as_pool(keys: str | Sequence[str] | VoyageKeyPool) -> VoyageKeyPool:
    if isinstance(keys, VoyageKeyPool):
        return keys
    return VoyageKeyPool([keys] if isinstance(keys, str) else list(keys))


class VoyageEmbedder:
    """``voyage-code-3`` (or another Voyage model) over the REST API, spread across the key pool."""

    def __init__(self, keys: str | Sequence[str] | VoyageKeyPool, model: str = "voyage-code-3", *,
                 timeout: float = 60.0) -> None:
        self._pool, self.model_id, self._timeout = _as_pool(keys), model, timeout

    def embed(self, texts: Sequence[str], *, input_type: InputType) -> list[list[float]]:
        out: list[list[float]] = []
        batch: list[str] = []
        spent = 0

        def flush() -> None:
            if not batch:
                return
            data = _post(VOYAGE_EMBED_URL, self._pool, self._timeout,
                         {"input": batch, "model": self.model_id, "input_type": input_type}, spent)
            rows = sorted(data.get("data", []), key=lambda r: r["index"])
            if len(rows) != len(batch):
                raise ToolError(f"voyage returned {len(rows)} embeddings for {len(batch)} texts")
            out.extend(r["embedding"] for r in rows)
            batch.clear()

        for text in texts:
            cost = estimate_tokens(text)
            if batch and spent + cost > VOYAGE_TPM:
                flush()
                spent = 0
            batch.append(text)
            spent += cost
        flush()
        return out


class VoyageReranker:
    def __init__(self, keys: str | Sequence[str] | VoyageKeyPool, model: str = "rerank-2.5", *,
                 timeout: float = 60.0) -> None:
        self._pool, self.model_id, self._timeout = _as_pool(keys), model, timeout

    def rerank(self, query: str, documents: Sequence[str]) -> list[float]:
        tokens = estimate_tokens(query) * len(documents) + sum(estimate_tokens(d) for d in documents)
        data = _post(VOYAGE_RERANK_URL, self._pool, self._timeout,
                     {"query": query, "documents": list(documents), "model": self.model_id}, tokens)
        scores = [0.0] * len(documents)
        for row in data.get("data", []):
            scores[row["index"]] = float(row["relevance_score"])
        return scores


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
    """Memoises another embedder per (model, input type, text); with ``cache_dir`` also on disk.

    The disk cache is an append-only JSONL file per model, written after every chunk of ``CACHE_CHUNK`` texts, so a long
    rate-limited run is resumable and loses at most one chunk.
    """

    CACHE_CHUNK = 16

    def __init__(self, inner: Embedder, cache_dir: Path | None = None) -> None:
        self.inner, self.model_id = inner, inner.model_id
        self._path = (Path(cache_dir) / f"{hashlib.sha256(inner.model_id.encode()).hexdigest()[:16]}.jsonl"
                      if cache_dir else None)
        self._cache: dict[str, list[float]] = {}
        if self._path is not None and self._path.is_file():
            for line in self._path.read_text(encoding="utf-8").splitlines():
                try:
                    row = json.loads(line)
                except ValueError:
                    continue  # a torn last line from an interrupted write
                self._cache[row["k"]] = row["v"]

    @staticmethod
    def _key(text: str, input_type: str) -> str:
        return hashlib.sha256(f"{input_type}\0{text}".encode()).hexdigest()

    def has(self, text: str, input_type: InputType) -> bool:
        return self._key(text, input_type) in self._cache

    def embed(self, texts: Sequence[str], *, input_type: InputType) -> list[list[float]]:
        keys = [self._key(t, input_type) for t in texts]
        missing = list(dict.fromkeys(i for i, k in enumerate(keys) if k not in self._cache))
        for start in range(0, len(missing), self.CACHE_CHUNK):
            chunk = [i for i in missing[start:start + self.CACHE_CHUNK] if keys[i] not in self._cache]
            if not chunk:
                continue
            vectors = self.inner.embed([texts[i] for i in chunk], input_type=input_type)
            fresh = {keys[i]: [round(float(x), 6) for x in v] for i, v in zip(chunk, vectors, strict=True)}
            self._cache.update(fresh)
            if self._path is not None:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                with self._path.open("a", encoding="utf-8") as fh:
                    fh.writelines(json.dumps({"k": k, "v": v}) + "\n" for k, v in fresh.items())
        return [self._cache[k] for k in keys]


# --------------------------------------------------------------------------- #
# The retriever
# --------------------------------------------------------------------------- #


def _document_text(item: EvidenceItem) -> str:
    return f"{item.source_ref}\n{item.text or ''}"[:EMBED_TEXT_CHARS]


def _unit_rows(vectors: Sequence[Sequence[float]]):
    import numpy as np

    m = np.asarray(vectors, dtype=np.float64)
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.where(norms == 0, 1.0, norms)


class HybridRetriever:
    """BM25 + dense ranking fused with RRF; optional rerank of the fused top ``RERANK_CANDIDATES``.

    Document embeddings are computed once per corpus (on first search).  Ties are broken by ``evidence_id`` as in BM25,
    so a fixed model gives a deterministic ranking.  Each hit carries ``component_ranks`` (bm25 / dense).
    """

    def __init__(self, items: Sequence[EvidenceItem], embedder: Embedder, reranker: Reranker | None = None, *,
                 cutoff_index: int | None = None) -> None:
        self._items = [i for i in items if not _beyond(i, cutoff_index)]
        self._by_id = {i.evidence_id: i for i in self._items}
        self._bm25 = Bm25Retriever(self._items)
        self._embedder = embedder
        self._reranker = reranker
        self._matrix = None

    def __len__(self) -> int:
        return len(self._items)

    def _dense(self, query: str, depth: int) -> list[tuple[str, float]]:
        if not self._items:
            return []
        if self._matrix is None:
            self._matrix = _unit_rows(self._embedder.embed([_document_text(i) for i in self._items], input_type="document"))
        q = _unit_rows(self._embedder.embed([query], input_type="query"))[0]
        sims = self._matrix @ q
        scored = [(item.evidence_id, float(sim)) for item, sim in zip(self._items, sims, strict=True)]
        return sorted(scored, key=lambda kv: (-kv[1], kv[0]))[:depth]

    def search(self, query: str, *, top_k: int) -> list[RetrievedEvidence]:
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        if not query.strip():
            return []
        keyword = [h.evidence_id for h in self._bm25.search(query, top_k=CANDIDATE_DEPTH)]
        dense = [i for i, _ in self._dense(query, CANDIDATE_DEPTH)]
        ranks = {"bm25": {i: r for r, i in enumerate(keyword, 1)}, "dense": {i: r for r, i in enumerate(dense, 1)}}
        fused = reciprocal_rank_fusion([keyword, dense], k=RRF_K)
        method = RetrievalMethod.HYBRID
        if self._reranker is not None and fused:
            head = fused[:RERANK_CANDIDATES]
            scores = self._reranker.rerank(query, [_document_text(self._by_id[i]) for i, _ in head])
            fused = sorted(zip((i for i, _ in head), scores, strict=True), key=lambda kv: (-kv[1], kv[0]))
            method = RetrievalMethod.RERANK
        out = []
        for rank, (i, score) in enumerate(fused[:top_k], 1):
            hit = _evidence(self._by_id[i], query, rank, score, method)
            component = {name: table[i] for name, table in ranks.items() if i in table}
            out.append(hit.model_copy(update={"component_ranks": component or None}))
        return out


def retrieval_config_id(settings) -> str:
    """A stable label for the retrieval setup, recorded on each run (``ExperimentRunRecord.retrieval_config_version``)."""
    if getattr(settings, "retrieval_mode", "bm25") != "hybrid":
        return "bm25-okapi/rrf60"
    emb = settings.voyage_embedding_model if settings.embedding_backend == "voyage" else settings.local_embedding_model
    rer = {"none": "", "voyage": f"+{settings.voyage_rerank_model}", "local": f"+{settings.local_rerank_model}"}[settings.rerank_backend]
    return f"hybrid:bm25+{emb}{rer}/rrf{RRF_K}/depth{CANDIDATE_DEPTH}/rerank{RERANK_CANDIDATES}"


def build_components(settings) -> tuple[Embedder, Reranker | None]:
    """The embedder (cached) and optional reranker that ``settings`` selects.

    Raises:
        ConfigError: a Voyage backend without ``VOYAGE_API_KEY``, or the local backend without sentence-transformers.
    """
    if "voyage" in (settings.embedding_backend, settings.rerank_backend):
        pool = _make_pool(settings)
    if settings.embedding_backend == "voyage":
        embedder: Embedder = VoyageEmbedder(pool, settings.voyage_embedding_model)
    else:
        embedder = LocalEmbedder(settings.local_embedding_model)
    reranker: Reranker | None = None
    if settings.rerank_backend == "voyage":
        reranker = VoyageReranker(pool, settings.voyage_rerank_model)
    elif settings.rerank_backend == "local":
        # One loaded cross-encoder per process (loading it per episode costs seconds each time).
        reranker = _RERANKERS.setdefault(settings.local_rerank_model, LocalCrossEncoder(settings.local_rerank_model))
    return _shared_cache(embedder, settings.embedding_cache_dir), reranker


_POOLS: dict[tuple[str, ...], VoyageKeyPool | None] = {}


def _make_pool(settings) -> VoyageKeyPool:
    """One shared pool per key set per process, so the rate limiter sees every call."""
    keys = tuple(settings.voyage_api_keys())
    if not keys:
        raise ConfigError("the voyage backend needs VOYAGE_API_KEY1..5 or VOYAGE_API_KEY")
    if _POOLS.get(keys) is None:
        _POOLS[keys] = VoyageKeyPool(keys)
    return _POOLS[keys]


_CACHES: dict[tuple[str, str | None], CachedEmbedder] = {}
_RERANKERS: dict[str, LocalCrossEncoder] = {}


def _shared_cache(embedder: Embedder, cache_dir: Path | None) -> CachedEmbedder:
    """One cache per (model, directory) per process: every episode of an experiment reuses the corpus embeddings."""
    key = (embedder.model_id, str(cache_dir) if cache_dir else None)
    if key not in _CACHES:
        _CACHES[key] = CachedEmbedder(embedder, cache_dir)
    return _CACHES[key]
