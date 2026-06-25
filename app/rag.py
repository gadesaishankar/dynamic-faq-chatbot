"""Retrieval-Augmented Generation: question -> retrieve KB chunks -> answer."""
from __future__ import annotations

import numpy as np

from . import embeddings, llm, store
from .config import settings


def retrieve(query_vec: np.ndarray, k: int | None = None) -> list[dict]:
    return store.search_chunks(query_vec, k or settings.RAG_TOP_K)


def answer(question: str, history: list[dict] | None = None) -> dict:
    """Embed the question once, reuse the vector for retrieval AND clustering."""
    query_vec = embeddings.embed_one(question)
    contexts = retrieve(query_vec)
    text, llm_used = llm.generate_answer(question, contexts, history=history)
    return {
        "query_vec": query_vec,
        "answer": text,
        "citations": contexts,
        "llm_used": llm_used,
    }
