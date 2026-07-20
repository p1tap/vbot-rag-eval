"""Dependency-free Okapi BM25 for bounded benchmark candidate sets."""

from __future__ import annotations

from collections import Counter
from math import log
import re
from typing import Sequence

from .base import RetrievalDocument, RetrievalHit


TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.casefold())


class BM25Retriever:
    name = "bm25"

    def __init__(self, *, k1: float = 1.2, b: float = 0.75) -> None:
        if k1 <= 0:
            raise ValueError("BM25 k1 must be positive")
        if not 0 <= b <= 1:
            raise ValueError("BM25 b must be between zero and one")
        self.k1 = k1
        self.b = b

    def rank(
        self,
        query: str,
        documents: Sequence[RetrievalDocument],
        k: int,
    ) -> list[RetrievalHit]:
        if k < 1:
            raise ValueError("retrieval k must be at least one")
        if not documents:
            return []
        query_terms = list(dict.fromkeys(tokenize(query)))
        term_counts = [Counter(tokenize(f"{doc.title} {doc.text}")) for doc in documents]
        lengths = [sum(counts.values()) for counts in term_counts]
        average_length = sum(lengths) / len(lengths) if lengths else 1.0
        document_frequency = {
            term: sum(term in counts for counts in term_counts) for term in query_terms
        }

        scored: list[tuple[float, str, RetrievalDocument]] = []
        for document, counts, length in zip(documents, term_counts, lengths):
            score = 0.0
            for term in query_terms:
                frequency = counts.get(term, 0)
                if not frequency:
                    continue
                df = document_frequency[term]
                inverse_document_frequency = log(
                    1 + (len(documents) - df + 0.5) / (df + 0.5)
                )
                length_norm = 1 - self.b + self.b * length / max(average_length, 1.0)
                score += inverse_document_frequency * (
                    frequency * (self.k1 + 1)
                ) / (frequency + self.k1 * length_norm)
            scored.append((score, document.id, document))

        scored.sort(key=lambda item: (-item[0], item[1]))
        return [
            RetrievalHit(document=document, score=score, rank=rank)
            for rank, (score, _, document) in enumerate(scored[:k], start=1)
        ]

