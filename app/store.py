"""Storage facade — delegates to a pluggable backend (config.STORE_BACKEND).

The rest of the app imports `store` and calls these functions; swapping
SQLite ↔ MongoDB is a one-line config change and touches nothing else. This is
the seam the architecture was designed around.
"""
from __future__ import annotations

from .config import settings

_backend = None


def _b():
    global _backend
    if _backend is None:
        # Use MongoDB only when explicitly selected AND a URI is configured —
        # otherwise fall back to SQLite so a misconfig can't crash startup.
        if settings.STORE_BACKEND == "mongodb" and settings.MONGODB_URI:
            from .stores import mongo_backend as _backend  # noqa: F811
        else:
            from .stores import sqlite_backend as _backend  # noqa: F811
    return _backend


def _reset() -> None:
    """Force re-selection of the backend (used by tests)."""
    global _backend
    _backend = None


def init_db():
    return _b().init_db()


def clear_kb():
    return _b().clear_kb()


def add_chunks(items, embeddings):
    return _b().add_chunks(items, embeddings)


def search_chunks(query_vec, k):
    return _b().search_chunks(query_vec, k)


def all_kb_chunks():
    return _b().all_kb_chunks()


def add_query_log(text, vec, cluster_id, answer, score=0.0, cache_hit=False):
    return _b().add_query_log(text, vec, cluster_id, answer, score, cache_hit)


def all_query_logs():
    return _b().all_query_logs()


def set_query_cluster(query_id, cluster_id):
    return _b().set_query_cluster(query_id, cluster_id)


def recent_count_for_cluster(cluster_id, window_days):
    return _b().recent_count_for_cluster(cluster_id, window_days)


def total_query_count():
    return _b().total_query_count()


def count_below_score(threshold):
    return _b().count_below_score(threshold)


def cache_hit_rate():
    return _b().cache_hit_rate()


def avg_score_by_cluster():
    return _b().avg_score_by_cluster()


def add_feedback(log_id, vote):
    return _b().add_feedback(log_id, vote)


def feedback_stats():
    return _b().feedback_stats()


def feedback_by_cluster():
    return _b().feedback_by_cluster()


def cache_lookup(vec, threshold):
    return _b().cache_lookup(vec, threshold)


def cache_store(question, vec, answer, citations):
    return _b().cache_store(question, vec, answer, citations)


def nearest_cluster(vec):
    return _b().nearest_cluster(vec)


def create_cluster(vec, seed_text):
    return _b().create_cluster(vec, seed_text)


def assign_to_cluster(cluster_id, vec, text):
    return _b().assign_to_cluster(cluster_id, vec, text)


def get_cluster(cluster_id):
    return _b().get_cluster(cluster_id)


def all_clusters():
    return _b().all_clusters()


def count_clusters():
    return _b().count_clusters()


def set_canonical(cluster_id, question, answer):
    return _b().set_canonical(cluster_id, question, answer)


def replace_clusters(new_clusters):
    return _b().replace_clusters(new_clusters)
