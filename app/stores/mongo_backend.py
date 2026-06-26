"""MongoDB storage backend (e.g. MongoDB Atlas free tier — persistent).

Same interface as sqlite_backend. Documents live in MongoDB; vector similarity
is still computed in-process with numpy (load vectors, dot product) — simple,
portable, and plenty fast at prototype scale. Integer ids (cluster_id, log_id)
are issued from a `counters` collection so they stay stable and human-friendly.

Embeddings are stored as plain float lists; citations as native arrays.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone, timedelta

import numpy as np
from pymongo import MongoClient

from ..config import settings

_client = None
_db = None


def _get_db():
    global _client, _db
    if _db is None:
        _client = MongoClient(settings.MONGODB_URI)
        _db = _client[settings.MONGODB_DB]
    return _db


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float32)
    n = np.linalg.norm(vec)
    return vec / n if n > 0 else vec


def _vec(lst) -> np.ndarray:
    return np.asarray(lst, dtype=np.float32)


def _next_id(name: str) -> int:
    db = _get_db()
    db.counters.update_one({"_id": name}, {"$inc": {"seq": 1}}, upsert=True)
    return int(db.counters.find_one({"_id": name})["seq"])


def init_db() -> None:
    db = _get_db()
    try:
        db.query_logs.create_index("cluster_id")
        db.query_logs.create_index("ts")
        db.feedback.create_index("log_id")
    except Exception:
        pass


# --- knowledge base ---------------------------------------------------------

def clear_kb() -> None:
    _get_db().kb_chunks.delete_many({})


def add_chunks(items: list[tuple[str, str]], embeddings: np.ndarray) -> int:
    db = _get_db()
    docs = [
        {"_id": _next_id("kb_chunks"), "text": t, "source": s,
         "embedding": np.asarray(embeddings[i], dtype=np.float32).tolist()}
        for i, (t, s) in enumerate(items)
    ]
    if docs:
        db.kb_chunks.insert_many(docs)
    return len(items)


def search_chunks(query_vec: np.ndarray, k: int) -> list[dict]:
    docs = list(_get_db().kb_chunks.find({}, {"text": 1, "source": 1, "embedding": 1}))
    if not docs:
        return []
    mat = np.vstack([_vec(d["embedding"]) for d in docs])
    sims = mat @ _normalize(query_vec)
    order = np.argsort(-sims)[:k]
    return [
        {"text": docs[i]["text"], "source": docs[i]["source"], "score": float(sims[i])}
        for i in order
    ]


def all_kb_chunks() -> list[dict]:
    docs = list(_get_db().kb_chunks.find({}, {"text": 1, "source": 1, "embedding": 1}))
    return [{"text": d["text"], "source": d["source"], "vec": _vec(d["embedding"])} for d in docs]


# --- query logs -------------------------------------------------------------

def add_query_log(text, vec, cluster_id, answer, score=0.0, cache_hit=False) -> int:
    qid = _next_id("query_logs")
    _get_db().query_logs.insert_one({
        "_id": qid, "text": text,
        "embedding": np.asarray(vec, dtype=np.float32).tolist(),
        "ts": _now(), "cluster_id": cluster_id, "answer": answer,
        "score": float(score), "cache_hit": int(cache_hit),
    })
    return qid


def all_query_logs() -> list[dict]:
    docs = _get_db().query_logs.find({}, {"text": 1, "embedding": 1}).sort("_id", 1)
    return [{"id": d["_id"], "text": d["text"], "vec": _vec(d["embedding"])} for d in docs]


def set_query_cluster(query_id: int, cluster_id: int) -> None:
    _get_db().query_logs.update_one({"_id": query_id}, {"$set": {"cluster_id": cluster_id}})


def recent_count_for_cluster(cluster_id: int, window_days: int) -> int:
    q = {"cluster_id": cluster_id}
    if window_days and window_days > 0:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=window_days)).isoformat()
        q["ts"] = {"$gte": cutoff}
    return _get_db().query_logs.count_documents(q)


def total_query_count() -> int:
    return _get_db().query_logs.count_documents({})


def count_below_score(threshold: float) -> int:
    return _get_db().query_logs.count_documents({"score": {"$lt": threshold}})


def cache_hit_rate() -> float:
    db = _get_db()
    total = db.query_logs.count_documents({})
    if not total:
        return 0.0
    return db.query_logs.count_documents({"cache_hit": 1}) / total


def avg_score_by_cluster() -> dict[int, float]:
    docs = _get_db().query_logs.find(
        {"cluster_id": {"$ne": None}}, {"cluster_id": 1, "score": 1}
    )
    agg: dict[int, list] = defaultdict(list)
    for d in docs:
        agg[int(d["cluster_id"])].append(float(d.get("score", 0.0)))
    return {cid: sum(v) / len(v) for cid, v in agg.items()}


# --- feedback ---------------------------------------------------------------

def add_feedback(log_id: int, vote: int) -> None:
    _get_db().feedback.insert_one(
        {"_id": _next_id("feedback"), "log_id": log_id, "vote": 1 if vote >= 0 else -1, "ts": _now()}
    )


def feedback_stats() -> dict:
    db = _get_db()
    total = db.feedback.count_documents({})
    up = db.feedback.count_documents({"vote": {"$gt": 0}})
    return {"count": total, "helpful_rate": (up / total) if total else None}


def feedback_by_cluster() -> dict[int, dict]:
    db = _get_db()
    fb = list(db.feedback.find({}, {"log_id": 1, "vote": 1}))
    if not fb:
        return {}
    log_ids = list({f["log_id"] for f in fb})
    cluster_of = {
        d["_id"]: d.get("cluster_id")
        for d in db.query_logs.find({"_id": {"$in": log_ids}}, {"cluster_id": 1})
    }
    res: dict[int, dict] = defaultdict(lambda: {"up": 0, "down": 0})
    for f in fb:
        cid = cluster_of.get(f["log_id"])
        if cid is None:
            continue
        res[int(cid)]["up" if f["vote"] > 0 else "down"] += 1
    return dict(res)


# --- answer cache -----------------------------------------------------------

def cache_lookup(vec: np.ndarray, threshold: float) -> dict | None:
    docs = list(_get_db().answer_cache.find({}, {"answer": 1, "citations": 1, "embedding": 1}))
    if not docs:
        return None
    mat = np.vstack([_vec(d["embedding"]) for d in docs])
    sims = mat @ _normalize(vec)
    best = int(np.argmax(sims))
    if float(sims[best]) >= threshold:
        return {"answer": docs[best]["answer"], "citations": docs[best]["citations"]}
    return None


def cache_store(question: str, vec: np.ndarray, answer: str, citations: list[dict]) -> None:
    _get_db().answer_cache.insert_one({
        "_id": _next_id("answer_cache"), "question": question,
        "embedding": _normalize(vec).tolist(), "answer": answer,
        "citations": citations, "ts": _now(),
    })


# --- clusters ---------------------------------------------------------------

def nearest_cluster(vec: np.ndarray) -> tuple[int, float] | None:
    docs = list(_get_db().clusters.find({}, {"centroid": 1}))
    if not docs:
        return None
    mat = np.vstack([_vec(d["centroid"]) for d in docs])
    sims = mat @ _normalize(vec)
    best = int(np.argmax(sims))
    return int(docs[best]["_id"]), float(sims[best])


def create_cluster(vec: np.ndarray, seed_text: str) -> int:
    cid = _next_id("clusters")
    now = _now()
    _get_db().clusters.insert_one({
        "_id": cid, "centroid": _normalize(vec).tolist(), "member_count": 1,
        "canonical_question": None, "canonical_answer": None,
        "representative_queries": [seed_text], "created_at": now, "last_updated": now,
    })
    return cid


def assign_to_cluster(cluster_id: int, vec: np.ndarray, text: str) -> None:
    db = _get_db()
    d = db.clusters.find_one({"_id": cluster_id})
    if d is None:
        return
    centroid = _vec(d["centroid"])
    count = int(d["member_count"])
    new_centroid = _normalize((centroid * count + _normalize(vec)) / (count + 1))
    reps = d.get("representative_queries", [])
    if text not in reps:
        reps = (reps + [text])[-settings.MAX_REPRESENTATIVES:]
    db.clusters.update_one(
        {"_id": cluster_id},
        {"$set": {"centroid": new_centroid.tolist(), "member_count": count + 1,
                  "representative_queries": reps, "last_updated": _now()}},
    )


def get_cluster(cluster_id: int) -> dict | None:
    d = _get_db().clusters.find_one({"_id": cluster_id})
    return _doc_to_cluster(d) if d else None


def all_clusters() -> list[dict]:
    return [_doc_to_cluster(d) for d in _get_db().clusters.find({})]


def count_clusters() -> int:
    return _get_db().clusters.count_documents({})


def set_canonical(cluster_id: int, question: str, answer: str) -> None:
    _get_db().clusters.update_one(
        {"_id": cluster_id},
        {"$set": {"canonical_question": question, "canonical_answer": answer}},
    )


def replace_clusters(new_clusters: list[dict]) -> dict[int, int]:
    db = _get_db()
    db.clusters.delete_many({})
    now = _now()
    index_to_id: dict[int, int] = {}
    for i, c in enumerate(new_clusters):
        cid = _next_id("clusters")
        db.clusters.insert_one({
            "_id": cid, "centroid": _normalize(c["centroid"]).tolist(),
            "member_count": int(c["member_count"]), "canonical_question": None,
            "canonical_answer": None, "representative_queries": c["representative_queries"],
            "created_at": now, "last_updated": now,
        })
        index_to_id[i] = cid
    return index_to_id


def _doc_to_cluster(d: dict) -> dict:
    return {
        "id": int(d["_id"]),
        "centroid": _vec(d["centroid"]),
        "member_count": int(d["member_count"]),
        "canonical_question": d.get("canonical_question"),
        "canonical_answer": d.get("canonical_answer"),
        "representative_queries": d.get("representative_queries", []),
        "created_at": d.get("created_at"),
        "last_updated": d.get("last_updated"),
    }
