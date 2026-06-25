"""Retrieval returns the most similar chunk first (vector search sanity check)."""
import numpy as np

from app import store
from tests.conftest import unit


def test_search_chunks_orders_by_similarity(temp_db):
    items = [
        ("about registration", "academics.md"),
        ("about the library", "campus.md"),
        ("about fees", "fees.md"),
    ]
    embeddings = np.vstack(
        [unit([1, 0, 0, 0]), unit([0, 1, 0, 0]), unit([0, 0, 1, 0])]
    )
    store.add_chunks(items, embeddings)

    # Query closest to the "library" chunk.
    results = store.search_chunks(unit([0.05, 0.99, 0, 0]), k=3)
    assert results[0]["source"] == "campus.md"
    assert results[0]["score"] > results[1]["score"]


def test_search_chunks_empty_kb_returns_empty(temp_db):
    assert store.search_chunks(unit([1, 0, 0, 0]), k=3) == []
