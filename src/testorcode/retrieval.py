"""Evidence retrieval over the episode's repository corpus.

The corpus is built from the pinned repository at the episode's current commit.
Anything newer than ``RetrievalCorpusSpec.history_cutoff_commit`` must never be
indexed (``HistoryCutoffError`` if a request reaches past it).  Results are
``RetrievedEvidence`` whose ``evidence_id`` and ``source_ref`` come from the
corpus, so a citation can always be traced to a real source.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from testorcode.schemas.episode_schemas import RetrievalCorpusSpec
from testorcode.schemas.result_schemas import RetrievedEvidence


class Retriever(Protocol):
    def search(self, query: str, *, top_k: int) -> list[RetrievedEvidence]:
        """Ranked evidence, best first, ``rank`` starting at 1.  Empty list = nothing found."""
        ...


def build_retriever(corpus: RetrievalCorpusSpec, worktree_path: Path) -> Retriever:
    """Index docs, docstrings, changelogs, history and scoped memory up to the cutoff.

    Raises:
        HistoryCutoffError: the corpus would include commits after the cutoff.
        ToolError: the repository cannot be read.
    """
    raise NotImplementedError("BM25 retrieval is not implemented yet")
