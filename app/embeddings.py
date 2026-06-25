"""Local, free sentence embeddings (CPU).

The same embedding space serves BOTH:
  - RAG retrieval (question -> knowledge-base chunks), and
  - the semantic frequency engine (question -> question cluster).

sentence-transformers is imported lazily so that modules which only touch the
store/clustering math (e.g. unit tests with hand-made vectors) don't pay the
torch import cost.
"""
from __future__ import annotations

import numpy as np

from .config import settings

_model = None


def _get_model():
    global _model
    if _model is None:
        # Imported here (not at module top) on purpose — see module docstring.
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(settings.EMBED_MODEL)
    return _model


def embed(texts: list[str]) -> np.ndarray:
    """Embed a list of strings. Returns an (n, dim) L2-normalized float32 array.

    Because vectors are unit-length, cosine similarity == dot product, which
    keeps the vector search and clustering code trivial.
    """
    model = _get_model()
    vecs = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return vecs.astype(np.float32)


def embed_one(text: str) -> np.ndarray:
    """Embed a single string -> (dim,) normalized float32 vector."""
    return embed([text])[0]


def dim() -> int:
    return int(_get_model().get_sentence_embedding_dimension())
