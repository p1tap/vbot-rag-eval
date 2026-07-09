"""Corpus → section chunks → embedded numpy index.

Chunking is heading-aware: each markdown section (`#`/`##`/`###`) becomes a
chunk, and sections longer than CHUNK_MAX_WORDS are split into overlapping
windows. Every chunk keeps a stable `heading_key` (doc + slug). The golden
set references those keys, NOT chunk indices — so re-chunking (the whole
point of a chunk-size PR) never invalidates the relevance labels.
"""
import json
import re
from pathlib import Path

import numpy as np

from config import CHUNK_MAX_WORDS, CHUNK_OVERLAP_WORDS, EMBED_MODEL
from rag.embed import embed_passages

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "corpus"
INDEX = ROOT / "index"


def slug(text):
    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9]+", "-", text.lower())).strip("-")


def heading_key(doc_id, heading):
    return f"{doc_id}::{slug(heading)}"


def _windows(words, max_words, overlap):
    if len(words) <= max_words:
        return [" ".join(words)]
    step = max(1, max_words - overlap)
    out = []
    for i in range(0, len(words), step):
        out.append(" ".join(words[i : i + max_words]))
        if i + max_words >= len(words):
            break
    return out


def _sections(text):
    """Yield (heading, body) pairs. Content before the first heading is
    attached to a synthetic 'intro' heading. Lines inside fenced code blocks
    are never treated as headings — shell comments (`# ...`) are not sections."""
    heading, buf, in_fence = "intro", [], False
    for line in text.splitlines():
        if line.startswith("<!--"):  # provenance comment
            continue
        if re.match(r"^\s*(```|~~~)", line):
            in_fence = not in_fence
            buf.append(line)
            continue
        m = None if in_fence else re.match(r"^#{1,4}\s+(.*)$", line)
        if m:
            if buf:
                yield heading, "\n".join(buf).strip()
            heading, buf = m.group(1).strip(), []
        else:
            buf.append(line)
    if buf:
        yield heading, "\n".join(buf).strip()


def load_chunks():
    chunks = []
    for path in sorted(CORPUS.glob("*.md")):
        doc_id = path.stem
        text = path.read_text(encoding="utf-8")
        for heading, body in _sections(text):
            if not body.strip():
                continue
            for wi, window in enumerate(_windows(body.split(), CHUNK_MAX_WORDS, CHUNK_OVERLAP_WORDS)):
                chunks.append({
                    "id": f"{heading_key(doc_id, heading)}::{wi}",
                    "doc_id": doc_id,
                    "heading": heading,
                    "heading_key": heading_key(doc_id, heading),
                    "text": window,
                })
    return chunks


def build_index():
    chunks = load_chunks()
    # embed with heading as context — retrieval keys off both topic and body
    passages = [f"{c['heading']}. {c['text']}" for c in chunks]
    vecs = embed_passages(passages)
    INDEX.mkdir(exist_ok=True)
    np.save(INDEX / "embeddings.npy", vecs)
    (INDEX / "chunks.jsonl").write_text(
        "\n".join(json.dumps(c, ensure_ascii=False) for c in chunks), encoding="utf-8")
    (INDEX / "meta.json").write_text(json.dumps({
        "embed_model": EMBED_MODEL, "chunk_max_words": CHUNK_MAX_WORDS,
        "chunk_overlap_words": CHUNK_OVERLAP_WORDS, "n_chunks": len(chunks),
    }, indent=2), encoding="utf-8")
    return chunks
