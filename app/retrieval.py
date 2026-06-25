"""Hybrid retrieval: dense vector search fused with BM25 keyword search.

Vector search captures meaning; BM25 captures exact terms (course codes, names,
"Wi-Fi") that embeddings can miss. We combine their rankings with Reciprocal
Rank Fusion (RRF) — robust and parameter-light.

Returns (contexts, relevance):
  - contexts: top-k chunks ordered by fused rank, each carrying its *vector
    cosine* as `score` (so downstream grounding/citation is consistent).
  - relevance: the max vector cosine over all chunks — the signal used to tell a
    real question from a greeting/off-topic message.
"""
from __future__ import annotations

import re

import numpy as np

from . import store
from .config import settings

_WORD = re.compile(r"[a-z0-9]+")


def _tok(s: str) -> list[str]:
    return _WORD.findall(s.lower())


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def hybrid_retrieve(query_vec: np.ndarray, query_text: str, k: int | None = None):
    k = k or settings.RAG_TOP_K
    chunks = store.all_kb_chunks()
    if not chunks:
        return [], 0.0

    # Dense: cosine similarity (stored chunk vectors are already normalized).
    qv = _unit(query_vec)
    mat = np.vstack([c["vec"] for c in chunks])
    vec_sims = mat @ qv
    vec_order = list(np.argsort(-vec_sims))

    # Sparse: BM25 keyword scores.
    from rank_bm25 import BM25Okapi

    bm25 = BM25Okapi([_tok(c["text"]) for c in chunks])
    bm_scores = bm25.get_scores(_tok(query_text))
    bm_order = list(np.argsort(-bm_scores))

    # Reciprocal Rank Fusion.
    rrf: dict[int, float] = {}
    for rank, idx in enumerate(vec_order):
        rrf[int(idx)] = rrf.get(int(idx), 0.0) + 1.0 / (settings.RRF_K + rank + 1)
    for rank, idx in enumerate(bm_order):
        rrf[int(idx)] = rrf.get(int(idx), 0.0) + 1.0 / (settings.RRF_K + rank + 1)

    fused = sorted(rrf, key=lambda i: -rrf[i])[:k]
    contexts = [
        {"text": chunks[i]["text"], "source": chunks[i]["source"], "score": float(vec_sims[i])}
        for i in fused
    ]
    return contexts, float(vec_sims.max())
