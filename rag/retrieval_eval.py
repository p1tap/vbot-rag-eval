"""Benchmark-scope-aware document retrieval evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from rag.retrievers import RetrievalDocument, Retriever


@dataclass(frozen=True)
class CaseRetrievalResult:
    case_id: str
    relevant_document_ids: frozenset[str]
    ranked_document_ids: tuple[str, ...]


def documents_from_case(case: dict[str, Any]) -> list[RetrievalDocument]:
    return [
        RetrievalDocument(
            id=document["id"],
            title=document["title"],
            text=" ".join(document["sentences"]),
        )
        for document in case["documents"]
    ]


def evaluate_case(
    case: dict[str, Any], retriever: Retriever, *, max_k: int
) -> CaseRetrievalResult:
    documents = documents_from_case(case)
    relevant = frozenset(
        evidence["document_id"] for evidence in case["supporting_evidence"]
    )
    available = {document.id for document in documents}
    missing = relevant - available
    if missing:
        raise ValueError(f"{case['id']} has evidence outside its candidate scope")
    hits = retriever.rank(case["query"], documents, min(max_k, len(documents)))
    ranked = tuple(hit.document.id for hit in hits)
    if len(ranked) != len(set(ranked)):
        raise ValueError(f"retriever returned duplicate documents for {case['id']}")
    return CaseRetrievalResult(case["id"], relevant, ranked)


def aggregate_results(
    results: Sequence[CaseRetrievalResult], ks: Sequence[int]
) -> dict[str, dict[str, float]]:
    answerable = [result for result in results if result.relevant_document_ids]
    if not answerable:
        raise ValueError("retrieval evaluation has no answerable cases")
    metrics: dict[str, dict[str, float]] = {}
    for k in ks:
        any_hits = 0
        all_hits = 0
        evidence_recall = 0.0
        reciprocal_rank = 0.0
        for result in answerable:
            retrieved = result.ranked_document_ids[:k]
            retrieved_set = set(retrieved)
            overlap = result.relevant_document_ids & retrieved_set
            any_hits += bool(overlap)
            all_hits += result.relevant_document_ids <= retrieved_set
            evidence_recall += len(overlap) / len(result.relevant_document_ids)
            first_relevant = next(
                (
                    rank
                    for rank, document_id in enumerate(retrieved, start=1)
                    if document_id in result.relevant_document_ids
                ),
                None,
            )
            if first_relevant is not None:
                reciprocal_rank += 1.0 / first_relevant
        denominator = len(answerable)
        metrics[str(k)] = {
            "case_recall_any": any_hits / denominator,
            "case_recall_all": all_hits / denominator,
            "evidence_recall": evidence_recall / denominator,
            "mrr": reciprocal_rank / denominator,
        }
    return metrics

