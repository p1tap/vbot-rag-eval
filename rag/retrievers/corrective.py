"""Bounded corrective retrieval with auditable, fail-closed outcomes."""

from __future__ import annotations

from dataclasses import dataclass
import math
from time import perf_counter
from typing import Callable, Protocol, Sequence

from .base import RetrievalDocument, RetrievalHit, Retriever


@dataclass(frozen=True)
class ConfidenceAssessment:
    accepted: bool
    reason: str


class ConfidenceEvaluator(Protocol):
    def assess(self, hits: Sequence[RetrievalHit]) -> ConfidenceAssessment: ...


@dataclass(frozen=True)
class MinScoreConfidenceEvaluator:
    """Simple calibrated boundary for one retriever's score distribution."""

    min_hits: int
    min_top_score: float
    min_margin: float = 0.0

    def __post_init__(self) -> None:
        values = (self.min_top_score, self.min_margin)
        if self.min_hits < 1 or any(not math.isfinite(value) for value in values):
            raise ValueError("confidence evaluator parameters are invalid")
        if self.min_margin < 0:
            raise ValueError("confidence margin cannot be negative")

    def assess(self, hits: Sequence[RetrievalHit]) -> ConfidenceAssessment:
        validate_ranking(hits)
        if len(hits) < self.min_hits:
            return ConfidenceAssessment(False, "too_few_hits")
        if hits[0].score < self.min_top_score:
            return ConfidenceAssessment(False, "top_score_below_threshold")
        if len(hits) > 1 and hits[0].score - hits[1].score < self.min_margin:
            return ConfidenceAssessment(False, "top_score_margin_below_threshold")
        return ConfidenceAssessment(True, "confidence_gate_passed")


@dataclass(frozen=True)
class RetrievalAttempt:
    stage: str
    outcome: str
    reason: str
    elapsed_ms: float
    document_ids: tuple[str, ...]
    error_type: str | None = None


@dataclass(frozen=True)
class CorrectiveRetrievalResult:
    status: str
    hits: tuple[RetrievalHit, ...]
    attempts: tuple[RetrievalAttempt, ...]


def validate_ranking(hits: Sequence[RetrievalHit]) -> None:
    ids = []
    for expected_rank, hit in enumerate(hits, start=1):
        if not isinstance(hit, RetrievalHit):
            raise ValueError("retriever returned a non-hit value")
        if hit.rank != expected_rank:
            raise ValueError("retriever returned non-contiguous ranks")
        if not math.isfinite(float(hit.score)):
            raise ValueError("retriever returned a non-finite score")
        if not hit.document.id:
            raise ValueError("retriever returned an empty document ID")
        ids.append(hit.document.id)
    if len(ids) != len(set(ids)):
        raise ValueError("retriever returned duplicate documents")


class CorrectiveRetriever:
    """Try primary retrieval once, then one approved-corpus fallback."""

    name = "corrective_retriever"

    def __init__(
        self,
        *,
        primary: Retriever,
        fallback: Retriever,
        primary_evaluator: ConfidenceEvaluator,
        fallback_evaluator: ConfidenceEvaluator,
        query_rewriter: Callable[[str], str] | None = None,
    ) -> None:
        self.primary = primary
        self.fallback = fallback
        self.primary_evaluator = primary_evaluator
        self.fallback_evaluator = fallback_evaluator
        self.query_rewriter = query_rewriter or (lambda query: query)

    def retrieve(
        self,
        query: str,
        documents: Sequence[RetrievalDocument],
        k: int,
    ) -> CorrectiveRetrievalResult:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("corrective retrieval query must be non-empty")
        if k < 1:
            raise ValueError("retrieval k must be at least one")
        attempts = []
        primary_hits, primary_attempt = self._attempt(
            "primary",
            self.primary,
            self.primary_evaluator,
            query,
            documents,
            k,
        )
        attempts.append(primary_attempt)
        if primary_hits is not None:
            return CorrectiveRetrievalResult(
                "primary",
                tuple(primary_hits),
                tuple(attempts),
            )
        try:
            rewritten = self.query_rewriter(query)
            if not isinstance(rewritten, str) or not rewritten.strip():
                raise ValueError("query rewriter returned an empty query")
            if len(rewritten) > 2048:
                raise ValueError("query rewriter exceeded the length boundary")
        except Exception as error:
            attempts.append(
                RetrievalAttempt(
                    stage="rewrite",
                    outcome="error",
                    reason="query_rewrite_failed",
                    elapsed_ms=0.0,
                    document_ids=(),
                    error_type=type(error).__name__,
                )
            )
            return CorrectiveRetrievalResult("abstain", (), tuple(attempts))
        fallback_hits, fallback_attempt = self._attempt(
            "fallback",
            self.fallback,
            self.fallback_evaluator,
            rewritten,
            documents,
            k,
        )
        attempts.append(fallback_attempt)
        if fallback_hits is not None:
            return CorrectiveRetrievalResult(
                "fallback",
                tuple(fallback_hits),
                tuple(attempts),
            )
        return CorrectiveRetrievalResult("abstain", (), tuple(attempts))

    @staticmethod
    def _attempt(
        stage: str,
        retriever: Retriever,
        evaluator: ConfidenceEvaluator,
        query: str,
        documents: Sequence[RetrievalDocument],
        k: int,
    ) -> tuple[list[RetrievalHit] | None, RetrievalAttempt]:
        started = perf_counter()
        try:
            hits = retriever.rank(query, documents, k)
            validate_ranking(hits)
            assessment = evaluator.assess(hits)
            elapsed_ms = (perf_counter() - started) * 1000
            attempt = RetrievalAttempt(
                stage=stage,
                outcome="accepted" if assessment.accepted else "insufficient",
                reason=assessment.reason,
                elapsed_ms=elapsed_ms,
                document_ids=tuple(hit.document.id for hit in hits),
            )
            return (list(hits), attempt) if assessment.accepted else (None, attempt)
        except Exception as error:
            return None, RetrievalAttempt(
                stage=stage,
                outcome="error",
                reason="retriever_or_confidence_error",
                elapsed_ms=(perf_counter() - started) * 1000,
                document_ids=(),
                error_type=type(error).__name__,
            )
