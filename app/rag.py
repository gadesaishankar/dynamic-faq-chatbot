"""Retrieval-Augmented Generation orchestration.

Per question: (optionally) rewrite a follow-up into a standalone query, check the
semantic answer cache, run hybrid retrieval, then generate a grounded answer.
The `prepare` step is shared by the streaming and non-streaming endpoints.
"""
from __future__ import annotations

import numpy as np

from . import embeddings, llm, retrieval, store
from .config import settings


def prepare(question: str, history: list[dict] | None = None) -> dict:
    """Embed, (rewrite), cache-check, and retrieve — everything before generation.

    Returns: query_vec (of the ORIGINAL question, for clustering/logging/cache),
    contexts, relevance, and a cached answer if there's a near-duplicate hit.
    """
    query_vec = embeddings.embed_one(question)

    # Follow-up rewriting improves retrieval for things like "and on weekends?".
    search_text, search_vec = question, query_vec
    if history and settings.ENABLE_QUERY_REWRITE:
        rewritten = llm.rewrite_query(question, history)
        if rewritten and rewritten.lower() != question.lower():
            search_text = rewritten
            search_vec = embeddings.embed_one(rewritten)

    # Semantic cache: only for first-turn questions (follow-ups depend on history).
    cached = None
    if settings.ENABLE_ANSWER_CACHE and not history:
        cached = store.cache_lookup(query_vec, settings.CACHE_SIM_THRESHOLD)

    contexts, relevance = retrieval.hybrid_retrieve(search_vec, search_text)
    return {
        "query_vec": query_vec,
        "contexts": contexts,
        "relevance": relevance,
        "cached": cached,
    }


def answer(question: str, history: list[dict] | None = None) -> dict:
    p = prepare(question, history)

    if p["cached"] is not None:
        cits = p["cached"]["citations"]
        score = cits[0]["score"] if cits else 1.0
        return {
            "query_vec": p["query_vec"],
            "answer": p["cached"]["answer"],
            "citations": cits,
            "llm_used": False,
            "score": float(score),
            "cache_hit": True,
        }

    text, llm_used = llm.generate_answer(question, p["contexts"], history=history)
    _maybe_cache(question, p, text, history)
    return {
        "query_vec": p["query_vec"],
        "answer": text,
        "citations": p["contexts"],
        "llm_used": llm_used,
        "score": p["relevance"],
        "cache_hit": False,
    }


def _maybe_cache(question: str, p: dict, text: str, history) -> None:
    """Cache only confident, first-turn answers (don't cache greetings)."""
    if (
        settings.ENABLE_ANSWER_CACHE
        and not history
        and p["relevance"] >= settings.RELEVANCE_THRESHOLD
    ):
        store.cache_store(question, p["query_vec"], text, p["contexts"])
