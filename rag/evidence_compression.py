"""Deterministic query-focused compression for retrieved evidence documents."""

from __future__ import annotations

import re
from collections.abc import Sequence

import numpy as np

from config import EMBED_MODEL, EMBED_MODEL_REVISION


SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|[\r\n]+")
SEMANTIC_E5_CONFIG = {
    "model": EMBED_MODEL,
    "model_revision": EMBED_MODEL_REVISION,
    "scoring": "query_passage_cosine",
    "title_in_passage": True,
    "neighbor_radius": 1,
    "max_seed_sentences": 12,
}


def document_sentences(document: dict) -> list[str]:
    sentences = [
        str(item).strip() for item in document.get("sentences", []) if str(item).strip()
    ]
    complete = " ".join(sentences)
    if len(sentences) == 1:
        split = [
            item.strip() for item in SENTENCE_SPLIT.split(complete) if item.strip()
        ]
        if len(split) > 1:
            sentences = split
    return sentences


def _lexical_scores(query: str, sentences: Sequence[str]) -> list[float]:
    query_terms = set(re.findall(r"\w+", query.casefold()))
    scores = []
    for sentence in sentences:
        terms = set(re.findall(r"\w+", sentence.casefold()))
        scores.append(sum(1 + len(term) / 20 for term in query_terms & terms))
    return scores


def _render_ranked(
    sentences: Sequence[str],
    scores: Sequence[float],
    max_chars: int,
    *,
    neighbor_radius: int = 1,
    max_seed_sentences: int = 12,
) -> str:
    if len(sentences) != len(scores):
        raise ValueError("sentence and score counts differ")
    if max_chars < 1:
        raise ValueError("max_chars must be positive")
    if not sentences:
        return ""

    ranked = sorted(range(len(sentences)), key=lambda index: (-scores[index], index))
    priority: dict[int, tuple[int, int]] = {}
    for seed_rank, index in enumerate(ranked[:max_seed_sentences]):
        priority.setdefault(index, (seed_rank, 0))
        for distance in range(1, neighbor_radius + 1):
            for neighbor in (index - distance, index + distance):
                if 0 <= neighbor < len(sentences):
                    priority.setdefault(neighbor, (seed_rank, distance))

    selected: set[int] = set()
    used = 0
    for index in sorted(priority, key=lambda item: (*priority[item], item)):
        piece = f"[{index}] {sentences[index]}"
        added = len(piece) + (1 if selected else 0)
        if used + added <= max_chars:
            selected.add(index)
            used += added
    if not selected:
        index = ranked[0]
        return f"[{index}] {sentences[index]}"[:max_chars]
    return " ".join(f"[{index}] {sentences[index]}" for index in sorted(selected))


def compress_lexical(query: str, document: dict, max_chars: int) -> str:
    sentences = document_sentences(document)
    complete = " ".join(sentences)
    if len(complete) <= max_chars:
        return complete
    ranked = [
        (-score, index) for index, score in enumerate(_lexical_scores(query, sentences))
    ]
    selected = set()
    for _, index in sorted(ranked)[:12]:
        selected.update(range(max(0, index - 1), min(len(sentences), index + 2)))
    pieces = []
    length = 0
    for index in sorted(selected):
        piece = f"[{index}] {sentences[index]}"
        if pieces and length + len(piece) + 1 > max_chars:
            continue
        pieces.append(piece)
        length += len(piece) + 1
    return " ".join(pieces)[:max_chars]


def compress_documents(
    query: str,
    documents: Sequence[dict],
    max_chars: int,
    *,
    mode: str,
) -> list[str]:
    if mode == "lexical":
        return [compress_lexical(query, document, max_chars) for document in documents]
    if mode != "semantic_e5":
        raise ValueError(f"unknown evidence compression mode: {mode}")

    sentences_by_document = [document_sentences(document) for document in documents]
    rendered = [" ".join(sentences) for sentences in sentences_by_document]
    long_indexes = [
        index for index, text in enumerate(rendered) if len(text) > max_chars
    ]
    if not long_indexes:
        return rendered

    passage_texts = []
    passage_locations = []
    for document_index in long_indexes:
        title = str(documents[document_index].get("title", "")).strip()
        for sentence_index, sentence in enumerate(
            sentences_by_document[document_index]
        ):
            passage_texts.append(f"{title}. {sentence}" if title else sentence)
            passage_locations.append((document_index, sentence_index))

    # Import lazily so lexical evaluation and unit tests do not load Torch.
    from rag.embed import embed_passages, embed_queries

    query_vector = embed_queries(
        [query],
        model_name=EMBED_MODEL,
        model_revision=EMBED_MODEL_REVISION,
    )[0]
    passage_vectors = embed_passages(
        passage_texts,
        model_name=EMBED_MODEL,
        model_revision=EMBED_MODEL_REVISION,
    )
    semantic_scores = passage_vectors @ np.asarray(query_vector)
    scores_by_document = {
        index: [0.0] * len(sentences_by_document[index]) for index in long_indexes
    }
    for score, (document_index, sentence_index) in zip(
        semantic_scores, passage_locations, strict=True
    ):
        scores_by_document[document_index][sentence_index] = float(score)

    for index in long_indexes:
        rendered[index] = _render_ranked(
            sentences_by_document[index],
            scores_by_document[index],
            max_chars,
            neighbor_radius=SEMANTIC_E5_CONFIG["neighbor_radius"],
            max_seed_sentences=SEMANTIC_E5_CONFIG["max_seed_sentences"],
        )
    return rendered


__all__ = [
    "SEMANTIC_E5_CONFIG",
    "compress_documents",
    "compress_lexical",
    "document_sentences",
]
