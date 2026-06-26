"""MongoDB backend exercised against an in-memory Mongo (mongomock) — proves the
backend works before it ever touches a real Atlas cluster. Skipped if mongomock
isn't installed."""
import numpy as np
import pytest

from app import store
from app.config import settings
from tests.conftest import unit


@pytest.fixture()
def mongo_db(monkeypatch):
    mongomock = pytest.importorskip("mongomock")
    import app.stores.mongo_backend as mb

    monkeypatch.setattr(settings, "STORE_BACKEND", "mongodb")
    monkeypatch.setattr(settings, "MONGODB_URI", "mongodb://localhost")
    monkeypatch.setattr(mb, "MongoClient", mongomock.MongoClient)
    mb._client = None
    mb._db = None
    store._reset()
    store.init_db()
    yield store
    mb._client = None
    mb._db = None
    store._reset()


def test_mongo_backend_end_to_end(mongo_db):
    s = mongo_db

    # knowledge base + vector search
    s.add_chunks(
        [("register via the portal", "acad.md"), ("library hours", "camp.md")],
        np.vstack([unit([1, 0, 0, 0]), unit([0, 1, 0, 0])]),
    )
    res = s.search_chunks(unit([0.99, 0.01, 0, 0]), 2)
    assert res[0]["source"] == "acad.md" and res[0]["score"] > 0.9
    assert len(s.all_kb_chunks()) == 2

    # clusters + integer ids
    cid = s.create_cluster(unit([1, 0, 0, 0]), "how to register")
    assert isinstance(cid, int) and s.count_clusters() == 1
    nc = s.nearest_cluster(unit([1, 0, 0, 0]))
    assert nc[0] == cid and nc[1] > 0.9
    s.assign_to_cluster(cid, unit([1, 0, 0, 0]), "another phrasing")
    assert s.get_cluster(cid)["member_count"] == 2

    # query logs + feedback aggregation through cluster
    lid = s.add_query_log("how to register", unit([1, 0, 0, 0]), cid, "ans", score=0.6)
    assert s.total_query_count() == 1
    s.add_feedback(lid, 1)
    assert s.feedback_stats()["count"] == 1
    assert s.feedback_by_cluster()[cid] == {"up": 1, "down": 0}
    assert cid in s.avg_score_by_cluster()

    # answer cache
    assert s.cache_lookup(unit([1, 0, 0, 0]), 0.93) is None
    s.cache_store("q", unit([1, 0, 0, 0]), "cached", [{"text": "t", "source": "s", "score": 0.6}])
    hit = s.cache_lookup(unit([0.999, 0.001, 0, 0]), 0.93)
    assert hit and hit["answer"] == "cached"

    # batch replace
    mapping = s.replace_clusters([
        {"centroid": unit([1, 0, 0, 0]), "member_count": 3, "representative_queries": ["a", "b"]}
    ])
    assert len(mapping) == 1 and s.count_clusters() == 1
