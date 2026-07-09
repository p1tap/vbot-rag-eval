"""Load the numpy index and rank chunks by cosine similarity to a query."""
import json
from pathlib import Path

import numpy as np

from config import TOP_K
from rag.embed import embed_queries

ROOT = Path(__file__).resolve().parent.parent
INDEX = ROOT / "index"


def load_index():
    vecs = np.load(INDEX / "embeddings.npy")
    chunks = [json.loads(l) for l in (INDEX / "chunks.jsonl").read_text(encoding="utf-8").splitlines()]
    return vecs, chunks


def retrieve(query, k=None, index=None):
    k = k or TOP_K
    vecs, chunks = index or load_index()
    q = embed_queries([query])[0]
    scores = vecs @ q  # both L2-normalized → cosine
    order = np.argsort(-scores)[:k]
    return [{**chunks[i], "score": float(scores[i])} for i in order]
