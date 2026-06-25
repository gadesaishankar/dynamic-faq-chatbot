"""The core guarantee: semantically similar questions collapse into ONE cluster.

These tests use hand-made unit vectors to stand in for embeddings, so the
threshold/grouping logic is verified deterministically without the model.
"""
import numpy as np
import pytest

from app import clustering, faq, store
from app.config import settings
from tests.conftest import unit


def _near(base, jitter, rng):
    """A unit vector close to `base` (high cosine similarity)."""
    v = np.array(base, dtype=np.float32) + jitter * rng.standard_normal(len(base)).astype(np.float32)
    return v / np.linalg.norm(v)


def test_paraphrases_collapse_into_one_cluster(temp_db):
    rng = np.random.default_rng(0)
    # 8 "paraphrases" of intent A: all near the same direction.
    for _ in range(8):
        v = _near([1, 0, 0, 0], 0.05, rng)
        clustering.assign_realtime("register?", v)

    assert store.count_clusters() == 1
    c = store.all_clusters()[0]
    assert c["member_count"] == 8


def test_different_intent_makes_new_cluster(temp_db):
    rng = np.random.default_rng(1)
    for _ in range(5):
        clustering.assign_realtime("A", _near([1, 0, 0, 0], 0.05, rng))
    for _ in range(3):
        clustering.assign_realtime("B", _near([0, 1, 0, 0], 0.05, rng))

    assert store.count_clusters() == 2
    counts = sorted(c["member_count"] for c in store.all_clusters())
    assert counts == [3, 5]


def test_realtime_assignment_returns_new_flag(temp_db):
    cid1, new1 = clustering.assign_realtime("first", unit([1, 0, 0, 0]))
    cid2, new2 = clustering.assign_realtime("second", unit([0.99, 0.01, 0, 0]))
    assert new1 is True
    assert new2 is False
    assert cid1 == cid2


def test_faq_ranks_by_frequency(temp_db):
    rng = np.random.default_rng(2)
    # Both intents above the FAQ_MIN_ASKS threshold (default 3).
    for _ in range(6):
        v = _near([1, 0, 0, 0], 0.04, rng)
        cid, _ = clustering.assign_realtime("popular", v)
        store.add_query_log("popular", v, cid, "ans")
    for _ in range(4):
        v = _near([0, 1, 0, 0], 0.04, rng)
        cid, _ = clustering.assign_realtime("less", v)
        store.add_query_log("less", v, cid, "ans")

    items = faq.build_faq(top_n=5)["items"]
    assert len(items) == 2
    assert items[0]["ask_count"] == 6  # most-asked first
    assert items[1]["ask_count"] == 4


def test_faq_threshold_excludes_rare_questions(temp_db):
    rng = np.random.default_rng(5)
    # 5 asks of a popular intent (>= 3) ...
    for _ in range(5):
        v = _near([1, 0, 0, 0], 0.04, rng)
        cid, _ = clustering.assign_realtime("popular", v)
        store.add_query_log("popular", v, cid, "ans")
    # ... and a one-off question (1 ask, below threshold) must NOT appear.
    v = _near([0, 1, 0, 0], 0.04, rng)
    cid, _ = clustering.assign_realtime("one-off", v)
    store.add_query_log("one-off", v, cid, "ans")

    items = faq.build_faq(top_n=10)["items"]
    assert len(items) == 1
    assert items[0]["ask_count"] == 5


def test_batch_recluster_groups_two_intents(temp_db):
    pytest.importorskip("sklearn", reason="scikit-learn not installed")
    rng = np.random.default_rng(3)
    # Log queries for two well-separated intents directly.
    for _ in range(5):
        store.add_query_log("A", _near([1, 0, 0, 0], 0.03, rng), None, "ans")
    for _ in range(4):
        store.add_query_log("B", _near([0, 1, 0, 0], 0.03, rng), None, "ans")

    result = clustering.recluster_batch(generate_canonical=False)
    assert result["queries_processed"] == 9
    assert result["clusters_after"] == 2
