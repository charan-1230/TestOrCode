"""Evidence retrieval over the episode's repository corpus (BM25 first).

The corpus is the materialised ``RetrievalCorpusSpec.items`` of an episode (commit messages, docs, changelog sections, docstrings;
all at or before the current commit).  Nothing newer than ``history_cutoff_index`` is ever indexed.  Results are
``RetrievedEvidence`` whose ``evidence_id`` and ``source_ref`` come from the corpus, so a citation is always traceable.

Pieces, from low to high level:

* ``Bm25Retriever`` / ``build_retriever``: deterministic keyword search (score desc, then ``evidence_id`` asc).
* ``naive_query`` (B1, "standard RAG": the failing test's own text) and ``intent_queries`` (Full/A2: changed functions,
  behaviour words, failure text without test ids), fused with ``search_many`` / ``reciprocal_rank_fusion``.
* ``RetrievalLog`` (what a run retrieved, for ``ExperimentRunRecord.retrieved_evidence``) and ``SearchRepoTool`` (the
  ``search_repo`` agent tool with permission, budget and logging).
* ``memory_evidence``: only memory records whose probe passed become evidence.
* ``hit_rank`` / ``recall_at_k`` / ``citation_existence_rate``: pure metric helpers.

Deferred, behind the same ``Retriever`` protocol: embeddings, reranking, building the corpus from a worktree.
"""

from __future__ import annotations

import keyword
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Protocol

from rank_bm25 import BM25Okapi

from testorcode.errors import BudgetExceededError, HistoryCutoffError, ToolError
from testorcode.schemas.contracts import FailureGroupDetail, MemoryReadResult, unknown_evidence_ids
from testorcode.schemas.episode_schemas import (
    CIEvent,
    EvidenceItem,
    EvidenceSourceType,
    RetrievalCorpusSpec,
    ToolName,
)
from testorcode.schemas.result_schemas import RetrievalMethod, RetrievedEvidence

SNIPPET_CHARS = 600
RRF_K = 60
BEHAVIOUR_TERMS = 15
_STOP_IDENTIFIERS = frozenset(keyword.kwlist) | {"self", "args", "kwargs", "none", "true", "false", "return", "assert"}


class Retriever(Protocol):
    def search(self, query: str, *, top_k: int) -> list[RetrievedEvidence]:
        """Ranked evidence, best first, ``rank`` starting at 1.  Empty list = nothing found."""
        ...


# --------------------------------------------------------------------------- #
# BM25 retriever
# --------------------------------------------------------------------------- #

_CAMEL = re.compile(r"([a-z0-9])([A-Z])")
_WORD = re.compile(r"[A-Za-z0-9]+")
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")


def tokenize(text: str) -> list[str]:
    """Lowercase word pieces (snake_case and camelCase split) plus whole compound identifiers."""
    pieces = [t.lower() for t in _WORD.findall(_CAMEL.sub(r"\1 \2", text)) if len(t) > 1]
    wholes = [w.lower() for w in _IDENT.findall(text) if "_" in w.strip("_") or _CAMEL.search(w)]
    return pieces + wholes


def _evidence(item: EvidenceItem, query: str, rank: int, score: float, method: RetrievalMethod) -> RetrievedEvidence:
    return RetrievedEvidence(
        evidence_id=item.evidence_id,
        source_type=item.source_type,
        source_ref=item.source_ref,
        commit_sha=item.commit_sha,
        line_start=item.line_start,
        line_end=item.line_end,
        snippet=(item.text or "")[:SNIPPET_CHARS] or None,
        retrieval_query=query,
        retrieval_method=method,
        rank=rank,
        score=float(score),
    )


class Bm25Retriever:
    """BM25Okapi over ``text`` + ``source_ref`` of each item.  Deterministic; no network.

    Items sharing no query term are never returned; everything else is ranked by score, then ``evidence_id``.
    """

    def __init__(self, items: Sequence[EvidenceItem]) -> None:
        self._items = list(items)
        docs = [tokenize(f"{i.text or ''} {i.source_ref}") for i in self._items]
        self._token_sets = [set(d) for d in docs]
        self._index = BM25Okapi(docs) if docs else None

    def __len__(self) -> int:
        return len(self._items)

    def search(self, query: str, *, top_k: int) -> list[RetrievedEvidence]:
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        tokens = tokenize(query)
        if not tokens or self._index is None:
            return []
        scores = self._index.get_scores(tokens)
        order = sorted(range(len(self._items)), key=lambda i: (-scores[i], self._items[i].evidence_id))
        # Keep any item that shares a query term.  Do not drop by score: BM25Okapi gives a term found in half of the
        # documents an IDF of exactly 0, which on a tiny corpus would turn a real match into "nothing found".
        wanted = set(tokens)
        hits = [i for i in order if self._token_sets[i] & wanted][:top_k]
        return [_evidence(self._items[i], query, rank, scores[i], RetrievalMethod.BM25) for rank, i in enumerate(hits, 1)]


def build_retriever(corpus: RetrievalCorpusSpec, worktree_path: Path | None = None) -> Retriever:
    """Index the episode's materialised corpus.

    Raises:
        NotImplementedError: ``corpus.items is None`` (building from a worktree is deferred).
        HistoryCutoffError: an item lies beyond ``history_cutoff_index``.
        ValueError: duplicate ``evidence_id``.
    """
    if corpus.items is None:
        raise NotImplementedError("building the corpus from a worktree is deferred; pass a materialised corpus")
    seen: set[str] = set()
    for item in corpus.items:
        if item.evidence_id in seen:
            raise ValueError(f"duplicate evidence_id {item.evidence_id!r}")
        seen.add(item.evidence_id)
        if (
            corpus.history_cutoff_index is not None
            and item.commit_index is not None
            and item.commit_index > corpus.history_cutoff_index
        ):
            raise HistoryCutoffError(f"{item.evidence_id} is newer than the history cutoff")
    return Bm25Retriever(corpus.items)


# --------------------------------------------------------------------------- #
# Fusion and queries
# --------------------------------------------------------------------------- #


def reciprocal_rank_fusion(rankings: Iterable[Sequence[str]], k: int = RRF_K) -> list[tuple[str, float]]:
    """Fuse ranked id lists: score = sum 1/(k + rank).  Sorted by score desc, then id asc."""
    scores: Counter[str] = Counter()
    for ranking in rankings:
        for rank, evidence_id in enumerate(ranking, 1):
            scores[evidence_id] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


def search_many(retriever: Retriever, queries: Sequence[str], *, top_k: int, depth: int | None = None) -> list[RetrievedEvidence]:
    """Run several queries, fuse with RRF, return the best ``top_k`` (ranks from 1, ``score`` = fused score)."""
    depth = depth or max(top_k * 3, 20)
    by_id: dict[str, RetrievedEvidence] = {}
    rankings: list[list[str]] = []
    for query in queries:
        hits = retriever.search(query, top_k=depth)
        rankings.append([h.evidence_id for h in hits])
        for hit in hits:
            by_id.setdefault(hit.evidence_id, hit)  # keeps the query that ranked it best first
    fused = reciprocal_rank_fusion(rankings)[:top_k]
    return [by_id[i].model_copy(update={"rank": r, "score": s}) for r, (i, s) in enumerate(fused, 1)]


def _diff_text(ci_event: CIEvent, paths: Iterable[str] | None = None) -> str:
    wanted = set(paths) if paths else None
    return "\n".join(c.unified_diff or "" for c in ci_event.changed_files if wanted is None or c.path in wanted)


def _changed_lines(diff: str) -> list[str]:
    return [ln[1:] for ln in diff.splitlines() if ln[:1] in ("+", "-") and ln[:3] not in ("+++", "---")]


def _test_words(ci_event: CIEvent, group: FailureGroupDetail | None) -> set[str]:
    tests = group.failing_tests if group else ci_event.failing_tests
    words: set[str] = set()
    for t in tests:
        words.update(tokenize(f"{t.test_id} {t.test_file} {t.test_name}"))
    return words


def naive_query(ci_event: CIEvent) -> str:
    """B1 'standard RAG': the failing test's own text (id, message, traceback head)."""
    parts = []
    for t in ci_event.failing_tests[:3]:
        parts += [t.test_id, t.failure_message or "", (t.traceback or "")[:500]]
    return "\n".join(p for p in parts if p)


def intent_queries(
    ci_event: CIEvent,
    group: FailureGroupDetail | None = None,
    *,
    include_message: bool = True,
    include_failure_text: bool = False,
) -> list[str]:
    """Queries aimed at change intent (Full / A2).  Never contains test ids or test names.

    1. changed functions/classes and modules: the group's own function if it has one, otherwise every function named in the
       diff hunk headers and ``def``/``class`` lines;
    2. behaviour words: the most frequent identifiers on changed source lines;
    3. optional (``include_failure_text``): failure text without test ids, paths or memory addresses.  Off by default: on the
       dev split it pulled test and symptom text up the ranking and made gold-evidence ranks worse;
    4. the commit subject, only if the message is visible and ``include_message``.
    """
    paths = list(group.affected_files) if group and group.affected_files else None
    diff = _diff_text(ci_event, paths)
    test_words = _test_words(ci_event, group)

    names: list[str] = []
    qualname = group.group.function_qualname if group else None
    if qualname:
        names += [part for part in qualname.split(".") if part != "boltons"]
    else:
        names += re.findall(r"^@@[^@]*@@\s*(?:def|class)\s+([A-Za-z_]\w*)", diff, flags=re.MULTILINE)
        names += re.findall(r"^[+-]\s*(?:async\s+)?(?:def|class)\s+([A-Za-z_]\w*)", diff, flags=re.MULTILINE)
    names += [Path(c.path).stem for c in ci_event.changed_files if paths is None or c.path in paths]
    queries = [" ".join(dict.fromkeys(names))]

    idents = Counter(
        w.lower() for line in _changed_lines(diff) for w in _IDENT.findall(line) if w.lower() not in _STOP_IDENTIFIERS
    )
    behaviour = [w for w, _ in idents.most_common(BEHAVIOUR_TERMS * 2) if not set(tokenize(w)) <= test_words][:BEHAVIOUR_TERMS]
    queries.append(" ".join(behaviour))

    if include_failure_text:
        failures = []
        for t in group.failing_tests if group else ci_event.failing_tests:
            text = re.sub(r"0x[0-9a-fA-F]+|(?:/[\w.\-]+){2,}|[\w/]+\.py::\S+", " ", t.failure_message or "")
            failures.append(" ".join(w for w in tokenize(text) if w not in test_words))
        queries.append(" ".join(failures))

    if include_message and ci_event.commit_message:
        queries.append(ci_event.commit_message.splitlines()[0])
    return [q for q in queries if q.strip()]


# --------------------------------------------------------------------------- #
# Run logging, the search_repo tool, memory evidence
# --------------------------------------------------------------------------- #


class RetrievalLog:
    """Everything a run retrieved (first sighting of each id wins).  Feeds the run record and citation checks."""

    def __init__(self) -> None:
        self._seen: dict[str, RetrievedEvidence] = {}

    def add(self, hits: Iterable[RetrievedEvidence]) -> None:
        for hit in hits:
            self._seen.setdefault(hit.evidence_id, hit)

    def ids(self) -> set[str]:
        return set(self._seen)

    def records(self) -> list[RetrievedEvidence]:
        return list(self._seen.values())

    def unknown_citations(self, cited: Iterable[str]) -> list[str]:
        """Cited ids this run never retrieved (invented, or retrieved by nobody)."""
        return unknown_evidence_ids(cited, self._seen)


class SearchRepoTool:
    """The ``search_repo`` tool for agents: permission check, call budget, logging, compact output.

    Raises:
        ToolError: ``search_repo`` is not in ``allowed_tools``.
        BudgetExceededError: more than ``max_calls`` searches.
    """

    def __init__(
        self,
        retriever: Retriever,
        *,
        allowed_tools: Iterable[ToolName],
        top_k: int = 8,
        max_calls: int | None = None,
        log: RetrievalLog | None = None,
    ) -> None:
        self._retriever = retriever
        self._allowed = ToolName.SEARCH_REPO in set(allowed_tools)
        self._top_k = top_k
        self._max_calls = max_calls
        self.calls = 0
        self.log = log if log is not None else RetrievalLog()

    def __call__(self, query: str, top_k: int | None = None) -> list[dict[str, object]]:
        if not self._allowed:
            raise ToolError("search_repo is not permitted for this episode")
        if self._max_calls is not None and self.calls >= self._max_calls:
            raise BudgetExceededError(f"search_repo budget of {self._max_calls} calls exhausted")
        self.calls += 1
        hits = self._retriever.search(query, top_k=top_k or self._top_k)
        self.log.add(hits)
        return [
            {
                "evidence_id": h.evidence_id,
                "source_type": h.source_type.value,
                "source_ref": h.source_ref,
                "rank": h.rank,
                "text": h.snippet or "",
            }
            for h in hits
        ]


def memory_evidence(read: MemoryReadResult) -> list[RetrievedEvidence]:
    """Evidence from memory: only records whose probe passed on the current code and that are not stale."""
    return [
        RetrievedEvidence(
            evidence_id=r.record_id,
            source_type=EvidenceSourceType.MEMORY_RECORD,
            source_ref=r.record_id,
            commit_sha=r.approved_at_commit,
            snippet=r.behaviour_signature[:SNIPPET_CHARS],
            retrieval_method=RetrievalMethod.MEMORY_LOOKUP,
            rank=rank,
        )
        for rank, r in enumerate(read.usable, 1)
    ]


# --------------------------------------------------------------------------- #
# Metrics (pure)
# --------------------------------------------------------------------------- #


def hit_rank(ranked_ids: Sequence[str], gold_ids: Iterable[str]) -> int | None:
    """1-based rank of the best-ranked gold item, or None if no gold item was retrieved."""
    gold = set(gold_ids)
    return next((r for r, i in enumerate(ranked_ids, 1) if i in gold), None)


def recall_at_k(ranked_ids: Sequence[str], gold_ids: Iterable[str], k: int) -> float | None:
    """Fraction of gold items in the top ``k``; None when there is no gold (not annotated or none exists)."""
    gold = set(gold_ids)
    if not gold:
        return None
    return len(gold & set(ranked_ids[:k])) / len(gold)


def citation_existence_rate(cited: Iterable[str], retrieved_ids: Iterable[str]) -> float | None:
    """Share of cited ids that were actually retrieved; None when nothing was cited.  (Whether a cited item
    *supports* the claim needs a judge and lives in evaluation.)"""
    cited = list(dict.fromkeys(cited))
    if not cited:
        return None
    return 1 - len(unknown_evidence_ids(cited, retrieved_ids)) / len(cited)
