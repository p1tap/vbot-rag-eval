"""Local embeddings via E5 — no API, reproducible, CI-free.

E5 is an asymmetric retriever: queries and passages get different prefixes
("query:" / "passage:") and the model is mean-pooled + L2-normalized, so a
dot product is cosine similarity. Rolling this by hand (rather than pulling
sentence-transformers) keeps the dependency surface to torch+transformers —
the same stack the fine-tuning lab already uses — and makes the pooling
explicit instead of hidden behind a wrapper.
"""
from functools import lru_cache

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from config import EMBED_MODEL


@lru_cache(maxsize=2)
def _load(model_name):
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    return tok, model


def _mean_pool(last_hidden, attention_mask):
    mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
    summed = (last_hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def _encode(texts, prefix, model_name=None, batch_size=16):
    tok, model = _load(model_name or EMBED_MODEL)
    out = []
    for i in range(0, len(texts), batch_size):
        batch = [f"{prefix}{t}" for t in texts[i : i + batch_size]]
        enc = tok(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
        with torch.no_grad():
            hidden = model(**enc).last_hidden_state
        vecs = _mean_pool(hidden, enc["attention_mask"])
        vecs = torch.nn.functional.normalize(vecs, p=2, dim=1)
        out.append(vecs.cpu().numpy())
    return np.vstack(out).astype(np.float32)


def embed_passages(texts, model_name=None):
    return _encode(texts, "passage: ", model_name)


def embed_queries(texts, model_name=None):
    return _encode(texts, "query: ", model_name)
