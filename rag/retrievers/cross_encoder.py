"""Optional cross-encoder candidate reranking.

The accepted retriever remains E5. This module reranks only a bounded first-
stage candidate list and is loaded lazily so deterministic unit tests and the
production service do not initialize an additional model.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Protocol, Sequence

from config import RERANK_MODEL, RERANK_MODEL_REVISION

from .base import RetrievalDocument, RetrievalHit, Retriever


class PairScorer(Protocol):
    """Score query-document pairs on one comparable numeric scale."""

    name: str

    def score(
        self, query: str, documents: Sequence[RetrievalDocument]
    ) -> list[float]: ...


def resolve_device(requested: str) -> str:
    import torch

    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA reranking was requested but CUDA is unavailable")
    return requested


@lru_cache(maxsize=4)
def _load_model(model_name: str, revision: str, device: str):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        revision=revision,
    )
    model.eval()
    model.to(device)
    return tokenizer, model


class CrossEncoderScorer:
    """Pinned local cross-encoder with bounded, auditable inference settings."""

    name = "cross_encoder"

    def __init__(
        self,
        *,
        model_name: str = RERANK_MODEL,
        model_revision: str = RERANK_MODEL_REVISION,
        device: str = "auto",
        batch_size: int = 32,
        max_length: int = 512,
        precision: str = "float32",
    ) -> None:
        if batch_size < 1:
            raise ValueError("reranker batch size must be positive")
        if max_length < 8:
            raise ValueError("reranker max length must be at least eight")
        if not model_revision:
            raise ValueError("reranker model revision must be pinned")
        if precision not in {"float32", "float16", "bfloat16"}:
            raise ValueError("unsupported reranker precision")
        if precision != "float32" and not resolve_device(device).startswith("cuda"):
            raise ValueError("reduced reranker precision requires CUDA")
        self.model_name = model_name
        self.model_revision = model_revision
        self.device = resolve_device(device)
        self.batch_size = batch_size
        self.max_length = max_length
        self.precision = precision

    def score(
        self, query: str, documents: Sequence[RetrievalDocument]
    ) -> list[float]:
        import torch

        if not documents:
            return []
        tokenizer, model = _load_model(
            self.model_name,
            self.model_revision,
            self.device,
        )
        values: list[float] = []
        for offset in range(0, len(documents), self.batch_size):
            batch = documents[offset : offset + self.batch_size]
            encoded = tokenizer(
                [query] * len(batch),
                [f"{document.title}. {document.text}" for document in batch],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            autocast_dtype = {
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
            }.get(self.precision)
            with torch.inference_mode(), torch.autocast(
                device_type="cuda",
                dtype=autocast_dtype or torch.float16,
                enabled=autocast_dtype is not None,
            ):
                logits = model(**encoded).logits.reshape(-1)
            values.extend(float(value) for value in logits.detach().cpu())
        if len(values) != len(documents):
            raise RuntimeError("cross-encoder returned a mismatched score count")
        return values


class CrossEncoderReranker:
    """Rerank a bounded result set from an injected first-stage retriever."""

    def __init__(
        self,
        first_stage: Retriever,
        scorer: PairScorer,
        *,
        candidate_k: int = 20,
    ) -> None:
        if candidate_k < 1:
            raise ValueError("reranker candidate depth must be positive")
        self.first_stage = first_stage
        self.scorer = scorer
        self.candidate_k = candidate_k
        self.name = f"{first_stage.name}_then_{scorer.name}"

    def rank(
        self,
        query: str,
        documents: Sequence[RetrievalDocument],
        k: int,
    ) -> list[RetrievalHit]:
        if k < 1:
            raise ValueError("retrieval k must be at least one")
        depth = min(len(documents), max(k, self.candidate_k))
        candidates = self.first_stage.rank(query, documents, depth)
        scores = self.scorer.score(
            query,
            [candidate.document for candidate in candidates],
        )
        if len(scores) != len(candidates):
            raise ValueError("pair scorer returned a mismatched score count")
        ranked = sorted(
            zip(candidates, scores),
            key=lambda item: (-float(item[1]), item[0].rank, item[0].document.id),
        )
        return [
            RetrievalHit(candidate.document, float(score), rank)
            for rank, (candidate, score) in enumerate(ranked[:k], start=1)
        ]
