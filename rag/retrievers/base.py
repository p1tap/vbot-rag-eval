"""Common document-ranking contracts for benchmark retrieval experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence


@dataclass(frozen=True)
class RetrievalDocument:
    id: str
    title: str
    text: str


@dataclass(frozen=True)
class RetrievalHit:
    document: RetrievalDocument
    score: float
    rank: int


class Retriever(Protocol):
    """Rank a supplied candidate corpus for one query.

    Candidate scope is intentional: some public tasks supply a per-query
    corpus, while others require a separately built global corpus.
    """

    name: str

    def rank(
        self,
        query: str,
        documents: Sequence[RetrievalDocument],
        k: int,
    ) -> list[RetrievalHit]: ...

