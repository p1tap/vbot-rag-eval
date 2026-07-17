"""Load the numpy index and rank chunks by cosine similarity to a query."""
import json
from pathlib import Path

import numpy as np

from config import TOP_K
from rag.embed import embed_queries

ROOT = Path(__file__).resolve().parent.parent
INDEX = ROOT / "index"


def load_index(index_dir=None):
    directory = Path(index_dir) if index_dir is not None else INDEX
    meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
    vecs = np.load(directory / "embeddings.npy", allow_pickle=False)
    chunks = [
        json.loads(line)
        for line in (directory / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if vecs.ndim != 2 or not vecs.shape[1]:
        raise ValueError("index embeddings must be a non-empty two-dimensional matrix")
    if len(chunks) != len(vecs) or meta.get("n_chunks") != len(chunks):
        raise ValueError("index metadata, chunks, and embeddings count mismatch")
    chunk_ids = [chunk.get("id") for chunk in chunks]
    if any(not isinstance(chunk_id, str) or not chunk_id for chunk_id in chunk_ids):
        raise ValueError("index chunk IDs are missing")
    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError("index chunk IDs are duplicated")
    if not np.isfinite(vecs).all():
        raise ValueError("index embeddings contain non-finite values")
    norms = np.linalg.norm(vecs, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-4):
        raise ValueError("index embeddings are not L2-normalized")
    return vecs, chunks


def retrieve(query, k=None, index=None, *, unique_headings=False):
    k = k or TOP_K
    vecs, chunks = index or load_index()
    q = embed_queries([query])[0]
    scores = vecs @ q  # both L2-normalized → cosine
    order = np.argsort(-scores)
    if unique_headings:
        selected = []
        seen = set()
        for item in order:
            heading_key = chunks[item]["heading_key"]
            if heading_key in seen:
                continue
            seen.add(heading_key)
            selected.append(item)
            if len(selected) == k:
                break
        order = selected
    else:
        order = order[:k]
    return [{**chunks[i], "score": float(scores[i])} for i in order]
