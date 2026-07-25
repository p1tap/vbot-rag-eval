"""Long-context embedding followed by stable chunk-span pooling."""

from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
from typing import Sequence

import numpy as np

from config import (
    LATE_CHUNK_CODE_REVISION,
    LATE_CHUNK_MODEL,
    LATE_CHUNK_MODEL_REVISION,
)


def assemble_document(chunks: Sequence[dict]) -> tuple[str, list[tuple[int, int]]]:
    """Create a deterministic document view and character span per chunk."""
    if not chunks:
        raise ValueError("late chunking requires at least one chunk")
    doc_ids = {chunk.get("doc_id") for chunk in chunks}
    if len(doc_ids) != 1:
        raise ValueError("late chunking document assembly mixed document IDs")
    parts: list[str] = []
    spans: list[tuple[int, int]] = []
    cursor = 0
    for chunk in chunks:
        heading = " ".join(str(chunk.get("heading", "")).split())
        text = str(chunk.get("text", "")).strip()
        if not heading or not text:
            raise ValueError("late chunking encountered an empty heading or chunk")
        if parts:
            separator = "\n\n"
            parts.append(separator)
            cursor += len(separator)
        piece = f"{heading}\n{text}"
        start = cursor
        parts.append(piece)
        cursor += len(piece)
        spans.append((start, cursor))
    return "".join(parts), spans


def token_indexes_for_span(
    offsets: Sequence[Sequence[int]],
    span: tuple[int, int],
) -> list[int]:
    start, end = span
    if start < 0 or end <= start:
        raise ValueError("late chunking span is invalid")
    indexes = []
    for index, pair in enumerate(offsets):
        token_start, token_end = int(pair[0]), int(pair[1])
        if token_end <= token_start:
            continue
        if token_start < end and token_end > start:
            indexes.append(index)
    if not indexes:
        raise ValueError("late chunking span did not align to any tokens")
    return indexes


def resolve_device(requested: str) -> str:
    import torch

    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA late chunking was requested but CUDA is unavailable")
    return requested


@lru_cache(maxsize=2)
def _load_model(
    model_name: str,
    model_revision: str,
    code_revision: str,
    device: str,
):
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        revision=model_revision,
        use_fast=True,
    )
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("late chunking requires a fast tokenizer with offsets")
    model = AutoModel.from_pretrained(
        model_name,
        revision=model_revision,
        code_revision=code_revision,
        trust_remote_code=True,
    )
    model.eval()
    model.to(device)
    return tokenizer, model


class LateChunkingEmbedder:
    """Embed full document views, then pool token states into stable chunks."""

    def __init__(
        self,
        *,
        model_name: str = LATE_CHUNK_MODEL,
        model_revision: str = LATE_CHUNK_MODEL_REVISION,
        code_revision: str = LATE_CHUNK_CODE_REVISION,
        device: str = "auto",
        max_length: int = 8192,
        query_max_length: int = 512,
    ) -> None:
        if not model_revision or not code_revision:
            raise ValueError("late chunking model and code revisions must be pinned")
        if max_length < 512 or query_max_length < 8:
            raise ValueError("late chunking token limits are invalid")
        self.model_name = model_name
        self.model_revision = model_revision
        self.code_revision = code_revision
        self.device = resolve_device(device)
        self.max_length = max_length
        self.query_max_length = query_max_length

    def _runtime(self):
        return _load_model(
            self.model_name,
            self.model_revision,
            self.code_revision,
            self.device,
        )

    def embed_chunks(self, chunks: Sequence[dict]) -> np.ndarray:
        import torch

        tokenizer, model = self._runtime()
        by_document: dict[str, list[tuple[int, dict]]] = defaultdict(list)
        for position, chunk in enumerate(chunks):
            by_document[str(chunk.get("doc_id"))].append((position, chunk))
        output: list[np.ndarray | None] = [None] * len(chunks)
        for document_id in sorted(by_document):
            positioned = by_document[document_id]
            text, spans = assemble_document([chunk for _, chunk in positioned])
            encoded = tokenizer(
                text,
                return_offsets_mapping=True,
                return_tensors="pt",
                truncation=False,
            )
            if encoded["input_ids"].shape[1] > self.max_length:
                raise ValueError(
                    f"late chunking document exceeds {self.max_length} tokens: "
                    f"{document_id}"
                )
            offsets = encoded.pop("offset_mapping")[0].tolist()
            inputs = {key: value.to(self.device) for key, value in encoded.items()}
            with torch.inference_mode():
                hidden = model(**inputs).last_hidden_state[0]
            for (position, _), span in zip(positioned, spans):
                indexes = token_indexes_for_span(offsets, span)
                vector = hidden[indexes].mean(dim=0)
                vector = torch.nn.functional.normalize(vector, p=2, dim=0)
                output[position] = vector.detach().cpu().numpy().astype(np.float32)
        if any(vector is None for vector in output):
            raise RuntimeError("late chunking did not emit every chunk vector")
        return np.stack(output)

    def embed_queries(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        return self._embed_independent(
            texts,
            batch_size=batch_size,
            max_length=self.query_max_length,
        )

    def embed_passages(
        self, texts: Sequence[str], batch_size: int = 32, max_length: int = 512
    ) -> np.ndarray:
        return self._embed_independent(
            texts,
            batch_size=batch_size,
            max_length=max_length,
        )

    def _embed_independent(
        self,
        texts: Sequence[str],
        *,
        batch_size: int,
        max_length: int,
    ) -> np.ndarray:
        import torch

        if not texts:
            raise ValueError("embedding requires at least one input")
        if batch_size < 1 or max_length < 8:
            raise ValueError("embedding batch size or token limit is invalid")
        tokenizer, model = self._runtime()
        output = []
        for offset in range(0, len(texts), batch_size):
            batch = list(texts[offset : offset + batch_size])
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            inputs = {key: value.to(self.device) for key, value in encoded.items()}
            with torch.inference_mode():
                hidden = model(**inputs).last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            vectors = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            vectors = torch.nn.functional.normalize(vectors, p=2, dim=1)
            output.append(vectors.detach().cpu().numpy().astype(np.float32))
        return np.vstack(output)
