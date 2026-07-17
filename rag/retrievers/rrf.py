"""Reciprocal-rank fusion for retriever experiment results."""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

from .base import RetrievalHit


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[RetrievalHit]],
    *,
    k: int,
    rank_constant: int = 60,
) -> list[RetrievalHit]:
    if k < 1:
        raise ValueError("retrieval k must be at least one")
    if rank_constant < 1:
        raise ValueError("RRF rank constant must be positive")
    scores: dict[str, float] = defaultdict(float)
    documents = {}
    for ranking in rankings:
        for hit in ranking:
            documents[hit.document.id] = hit.document
            scores[hit.document.id] += 1.0 / (rank_constant + hit.rank)
    order = sorted(scores, key=lambda document_id: (-scores[document_id], document_id))
    return [
        RetrievalHit(documents[document_id], scores[document_id], rank)
        for rank, document_id in enumerate(order[:k], start=1)
    ]

