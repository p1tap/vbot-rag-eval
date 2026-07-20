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
    weights: Sequence[float] | None = None,
) -> list[RetrievalHit]:
    if k < 1:
        raise ValueError("retrieval k must be at least one")
    if rank_constant < 1:
        raise ValueError("RRF rank constant must be positive")
    resolved_weights = [1.0] * len(rankings) if weights is None else list(weights)
    if len(resolved_weights) != len(rankings):
        raise ValueError("RRF weights must match the number of rankings")
    if any(weight < 0 for weight in resolved_weights) or not any(resolved_weights):
        raise ValueError("RRF weights must be non-negative with at least one positive")
    scores: dict[str, float] = defaultdict(float)
    documents = {}
    for ranking, weight in zip(rankings, resolved_weights):
        for hit in ranking:
            documents[hit.document.id] = hit.document
            scores[hit.document.id] += weight / (rank_constant + hit.rank)
    order = sorted(scores, key=lambda document_id: (-scores[document_id], document_id))
    return [
        RetrievalHit(documents[document_id], scores[document_id], rank)
        for rank, document_id in enumerate(order[:k], start=1)
    ]
