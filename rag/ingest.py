"""Corpus → section chunks → embedded numpy index.

Chunking is heading-aware: each markdown section (`#`/`##`/`###`) becomes a
chunk, and sections longer than CHUNK_MAX_WORDS are split into overlapping
windows. Every chunk keeps a stable `heading_key` (doc + slug). The golden
set references those keys, NOT chunk indices — so re-chunking (the whole
point of a chunk-size PR) never invalidates the relevance labels.
"""
import json
import hashlib
import re
from pathlib import Path

import numpy as np

from config import (
    CHUNK_MAX_WORDS,
    CHUNK_OVERLAP_WORDS,
    EMBED_MODEL,
    EMBED_MODEL_REVISION,
)
from rag.embed import embed_passages

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "corpus"
INDEX = ROOT / "index"
MANIFEST = CORPUS / "manifest.json"


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


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_document_contexts(manifest_path=MANIFEST):
    """Load only hash-bound, ingestion-approved document metadata."""
    manifest_path = Path(manifest_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    contexts = {}
    for document in payload.get("documents", []):
        document_id = document.get("document_id")
        relative_path = document.get("path")
        if not isinstance(document_id, str) or not document_id:
            raise ValueError("corpus manifest contains an invalid document ID")
        if not isinstance(relative_path, str) or not relative_path:
            raise ValueError(f"manifest path is invalid for document {document_id}")
        if document_id in contexts:
            raise ValueError(f"corpus manifest duplicates document {document_id}")
        if document.get("ingestion", {}).get("allowed") is not True:
            continue
        path = ROOT / relative_path
        if not path.is_file() or path.parent != CORPUS:
            raise ValueError(f"manifest path is invalid for document {document_id}")
        if _sha256(path) != document.get("content_sha256"):
            raise ValueError(f"manifest content hash drifted for document {document_id}")
        source = document.get("source", {})
        contexts[document_id] = {
            "document_id": document_id,
            "repository": source.get("declared_repository") or "unknown",
            "source_path": source.get("declared_path") or "unknown",
            "status": document.get("status") or "unknown",
            "access_class": document.get("ingestion", {}).get("access_class")
            or "unknown",
        }
    return contexts, payload


def provenance_context_header(chunk, document_context):
    """Render a bounded factual header without LLM-authored assertions."""
    if chunk["doc_id"] != document_context["document_id"]:
        raise ValueError("chunk and document context IDs do not match")
    fields = (
        ("Document", document_context["document_id"]),
        ("Repository", document_context["repository"]),
        ("Source path", document_context["source_path"]),
        ("Status", document_context["status"]),
        ("Access", document_context["access_class"]),
        ("Section", chunk["heading"]),
    )
    values = []
    for label, raw_value in fields:
        value = " ".join(str(raw_value).split())
        if not value or len(value) > 200:
            raise ValueError(f"context field is invalid: {label}")
        values.append(f"{label}: {value}")
    return "[" + " | ".join(values) + "]"


def load_chunks(*, context_mode="heading", manifest_path=MANIFEST):
    if context_mode not in {"heading", "provenance_header"}:
        raise ValueError(f"unsupported context mode: {context_mode}")
    document_contexts = {}
    if context_mode == "provenance_header":
        document_contexts, _ = load_document_contexts(manifest_path)
    chunks = []
    for path in sorted(CORPUS.glob("*.md")):
        doc_id = path.stem
        text = path.read_text(encoding="utf-8")
        for heading, body in _sections(text):
            if not body.strip():
                continue
            for wi, window in enumerate(_windows(body.split(), CHUNK_MAX_WORDS, CHUNK_OVERLAP_WORDS)):
                chunk = {
                    "id": f"{heading_key(doc_id, heading)}::{wi}",
                    "doc_id": doc_id,
                    "heading": heading,
                    "heading_key": heading_key(doc_id, heading),
                    "text": window,
                }
                if context_mode == "provenance_header":
                    if doc_id not in document_contexts:
                        raise ValueError(
                            f"ingested document is absent from approved manifest: {doc_id}"
                        )
                    chunk["context_header"] = provenance_context_header(
                        chunk, document_contexts[doc_id]
                    )
                chunks.append(chunk)
    return chunks


def build_index(*, index_dir=INDEX, context_mode="heading", manifest_path=MANIFEST):
    index_dir = Path(index_dir)
    chunks = load_chunks(context_mode=context_mode, manifest_path=manifest_path)
    passages = [
        (
            f"{c['context_header']}\n{c['heading']}. {c['text']}"
            if context_mode == "provenance_header"
            else f"{c['heading']}. {c['text']}"
        )
        for c in chunks
    ]
    vecs = embed_passages(passages)
    index_dir.mkdir(parents=True, exist_ok=True)
    np.save(index_dir / "embeddings.npy", vecs)
    (index_dir / "chunks.jsonl").write_text(
        "\n".join(json.dumps(c, ensure_ascii=False) for c in chunks), encoding="utf-8")
    metadata = {
        "embed_model": EMBED_MODEL, "embed_model_revision": EMBED_MODEL_REVISION,
        "chunk_max_words": CHUNK_MAX_WORDS,
        "chunk_overlap_words": CHUNK_OVERLAP_WORDS, "n_chunks": len(chunks),
    }
    if context_mode != "heading":
        manifest_payload = json.loads(
            Path(manifest_path).read_text(encoding="utf-8")
        )
        metadata.update(
            {
                "context_mode": context_mode,
                "corpus_fingerprint_sha256": manifest_payload.get(
                    "fingerprint_sha256"
                ),
            }
        )
    (index_dir / "meta.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    return chunks
