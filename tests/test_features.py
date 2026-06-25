"""Tests for v2 features: hybrid retrieval, answer cache, feedback, content gaps."""
import numpy as np
import pytest

from app import analytics, embeddings, rag, retrieval, store
from tests.conftest import unit


def test_hybrid_retrieval_finds_relevant_chunk(temp_db):
    pytest.importorskip("rank_bm25")
    items = [
        ("How to register for courses through the student portal", "acad.md"),
        ("Library hours and weekend timings", "camp.md"),
    ]
    embs = np.vstack([unit([1, 0, 0, 0]), unit([0, 1, 0, 0])])
    store.add_chunks(items, embs)

    contexts, relevance = retrieval.hybrid_retrieve(unit([0.98, 0.02, 0, 0]), "register for courses")
    assert contexts[0]["source"] == "acad.md"
    assert relevance > 0.9


def test_answer_cache_hit_and_miss(temp_db):
    v = unit([1, 0, 0, 0])
    assert store.cache_lookup(v, 0.93) is None  # empty
    store.cache_store("q", v, "cached answer", [{"text": "t", "source": "s", "score": 0.6}])
    hit = store.cache_lookup(unit([0.999, 0.001, 0, 0]), 0.93)
    assert hit and hit["answer"] == "cached answer"
    assert store.cache_lookup(unit([0, 1, 0, 0]), 0.93) is None  # different question


def test_feedback_aggregation(temp_db):
    cid = store.create_cluster(unit([1, 0, 0, 0]), "q")
    l1 = store.add_query_log("q", unit([1, 0, 0, 0]), cid, "a", score=0.6)
    l2 = store.add_query_log("q2", unit([1, 0, 0, 0]), cid, "a", score=0.6)
    store.add_feedback(l1, 1)
    store.add_feedback(l2, -1)

    stats = store.feedback_stats()
    assert stats["count"] == 2 and abs(stats["helpful_rate"] - 0.5) < 1e-9
    assert store.feedback_by_cluster()[cid] == {"up": 1, "down": 1}


def test_content_gap_flags_low_coverage(temp_db):
    cid = store.create_cluster(unit([1, 0, 0, 0]), "is there a hostel on campus")
    for _ in range(3):  # asked often...
        store.add_query_log("is there a hostel", unit([1, 0, 0, 0]), cid, "ans", score=0.10)  # ...but KB barely covers it
    gaps = analytics.content_gaps(min_asks=2)
    assert any(g["cluster_id"] == cid and "barely covers" in g["reason"] for g in gaps)


def test_rag_answer_uses_cache_second_time(temp_db, monkeypatch):
    pytest.importorskip("rank_bm25")
    qv = unit([1, 0, 0, 0])
    monkeypatch.setattr(embeddings, "embed_one", lambda t: qv)  # deterministic, no model
    store.add_chunks([("Register via the student portal during the window.", "acad.md")], np.vstack([qv]))

    r1 = rag.answer("how do I register?")
    assert r1["cache_hit"] is False and r1["answer"]
    r2 = rag.answer("how do I register?")
    assert r2["cache_hit"] is True and r2["answer"] == r1["answer"]
