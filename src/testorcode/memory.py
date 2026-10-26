"""Scoped change memory.

Read: ``MemoryStore.read`` returns every record for a function with the status of
its executable probe on the *current* code.  Only ``MemoryReadResult.usable``
records may be cited as evidence; stale or failing records are returned but marked.

Write: ``MemoryStore.write`` requires a ``MemoryApproval``, which only the
Integrity Guard's workflow builds after a Guard APPROVE and maintainer approval.
Without one it raises ``MemoryAuthorizationError``.  This is a convention
enforced by the store, not a security boundary: Python cannot stop other code
from constructing an approval, so the orchestrator must not hand the write
path to any agent except the Integrity Guard.
"""

from __future__ import annotations

from typing import Protocol

from testorcode.schemas.contracts import EpisodeContext, MemoryApproval, MemoryReadResult
from testorcode.schemas.episode_schemas import MemoryRecord


class MemoryStore(Protocol):
    def read(self, context: EpisodeContext, function_qualname: str) -> MemoryReadResult:
        """Records for ``function_qualname`` with probes run against the current code."""
        ...

    def write(self, record: MemoryRecord, approval: MemoryApproval) -> None:
        """Persist ``record``.

        Raises:
            MemoryAuthorizationError: ``approval`` is for another record or invalid.
        """
        ...


class JsonlMemoryStore:
    """Planned JSONL-backed store."""

    def read(self, context: EpisodeContext, function_qualname: str) -> MemoryReadResult:
        raise NotImplementedError("memory reads are not implemented yet")

    def write(self, record: MemoryRecord, approval: MemoryApproval) -> None:
        raise NotImplementedError("memory writes are not implemented yet")
