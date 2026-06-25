"""Local storage: SQLite for rows + numpy for vector similarity.

Why local-first instead of MongoDB Atlas? The brief is a $0 prototype. SQLite +
brute-force numpy cosine needs no account, no network, no extra service, and is
plenty fast for prototype volumes (thousands of vectors). The access functions
below are the seam where a MongoDB Atlas Vector Search backend would slot in
later without touching the rest of the app.

All stored vectors are L2-normalized (see embeddings.py), so cosine similarity
is a plain dot product.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np

from .config import settings

_local = threading.local()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_conn() -> sqlite3.Connection:
    """One connection per thread (SQLite connections aren't thread-safe)."""
    conn = getattr(_local, "conn", None)
    if conn is None:
        Path(settings.DB_PATH).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(settings.DB_PATH)
        conn.row_factory = sqlite3.Row
        _local.conn = conn
    return conn


def init_db() -> None:
    conn = get_conn()
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS kb_chunks (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            text      TEXT NOT NULL,
            source    TEXT NOT NULL,
            embedding BLOB NOT NULL
        );

        CREATE TABLE IF NOT EXISTS query_logs (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            text       TEXT NOT NULL,
            embedding  BLOB NOT NULL,
            ts         TEXT NOT NULL,
            cluster_id INTEGER,
            answer     TEXT
        );

        CREATE TABLE IF NOT EXISTS clusters (
            id                   INTEGER PRIMARY KEY AUTOINCREMENT,
            centroid             BLOB NOT NULL,
            member_count         INTEGER NOT NULL DEFAULT 0,
            canonical_question   TEXT,
            canonical_answer     TEXT,
            representative_queries TEXT NOT NULL DEFAULT '[]',
            created_at           TEXT NOT NULL,
            last_updated         TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_query_logs_cluster ON query_logs(cluster_id);
        CREATE INDEX IF NOT EXISTS idx_query_logs_ts ON query_logs(ts);
        """
    )
    conn.commit()


# --- vector (de)serialization ----------------------------------------------

def _to_blob(vec: np.ndarray) -> bytes:
    return np.asarray(vec, dtype=np.float32).tobytes()


def _from_blob(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


def _normalize(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float32)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


# --- knowledge base ---------------------------------------------------------

def clear_kb() -> None:
    conn = get_conn()
    conn.execute("DELETE FROM kb_chunks")
    conn.commit()


def add_chunks(items: list[tuple[str, str]], embeddings: np.ndarray) -> int:
    """items: list of (text, source). embeddings: (n, dim) aligned with items."""
    conn = get_conn()
    conn.executemany(
        "INSERT INTO kb_chunks (text, source, embedding) VALUES (?, ?, ?)",
        [(t, s, _to_blob(embeddings[i])) for i, (t, s) in enumerate(items)],
    )
    conn.commit()
    return len(items)


def search_chunks(query_vec: np.ndarray, k: int) -> list[dict]:
    conn = get_conn()
    rows = conn.execute("SELECT text, source, embedding FROM kb_chunks").fetchall()
    if not rows:
        return []
    mat = np.vstack([_from_blob(r["embedding"]) for r in rows])
    sims = mat @ _normalize(query_vec)
    order = np.argsort(-sims)[:k]
    return [
        {"text": rows[i]["text"], "source": rows[i]["source"], "score": float(sims[i])}
        for i in order
    ]


# --- query logs -------------------------------------------------------------

def add_query_log(text: str, vec: np.ndarray, cluster_id: int | None, answer: str) -> int:
    conn = get_conn()
    cur = conn.execute(
        "INSERT INTO query_logs (text, embedding, ts, cluster_id, answer) "
        "VALUES (?, ?, ?, ?, ?)",
        (text, _to_blob(vec), _now(), cluster_id, answer),
    )
    conn.commit()
    return int(cur.lastrowid)


def all_query_logs() -> list[dict]:
    conn = get_conn()
    rows = conn.execute(
        "SELECT id, text, embedding FROM query_logs ORDER BY id"
    ).fetchall()
    return [
        {"id": r["id"], "text": r["text"], "vec": _from_blob(r["embedding"])}
        for r in rows
    ]


def set_query_cluster(query_id: int, cluster_id: int) -> None:
    conn = get_conn()
    conn.execute(
        "UPDATE query_logs SET cluster_id = ? WHERE id = ?", (cluster_id, query_id)
    )
    conn.commit()


def recent_count_for_cluster(cluster_id: int, window_days: int) -> int:
    conn = get_conn()
    if window_days and window_days > 0:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=window_days)).isoformat()
        row = conn.execute(
            "SELECT COUNT(*) AS c FROM query_logs WHERE cluster_id = ? AND ts >= ?",
            (cluster_id, cutoff),
        ).fetchone()
    else:
        row = conn.execute(
            "SELECT COUNT(*) AS c FROM query_logs WHERE cluster_id = ?",
            (cluster_id,),
        ).fetchone()
    return int(row["c"])


def total_query_count() -> int:
    conn = get_conn()
    return int(conn.execute("SELECT COUNT(*) AS c FROM query_logs").fetchone()["c"])


# --- clusters ---------------------------------------------------------------

def nearest_cluster(vec: np.ndarray) -> tuple[int, float] | None:
    """Return (cluster_id, cosine_similarity) of the closest cluster, or None."""
    conn = get_conn()
    rows = conn.execute("SELECT id, centroid FROM clusters").fetchall()
    if not rows:
        return None
    mat = np.vstack([_from_blob(r["centroid"]) for r in rows])
    sims = mat @ _normalize(vec)
    best = int(np.argmax(sims))
    return int(rows[best]["id"]), float(sims[best])


def create_cluster(vec: np.ndarray, seed_text: str) -> int:
    conn = get_conn()
    now = _now()
    cur = conn.execute(
        "INSERT INTO clusters (centroid, member_count, representative_queries, "
        "created_at, last_updated) VALUES (?, 1, ?, ?, ?)",
        (_to_blob(_normalize(vec)), json.dumps([seed_text]), now, now),
    )
    conn.commit()
    return int(cur.lastrowid)


def assign_to_cluster(cluster_id: int, vec: np.ndarray, text: str) -> None:
    """Incremental update: running-mean centroid + bump count + keep examples."""
    conn = get_conn()
    row = conn.execute(
        "SELECT centroid, member_count, representative_queries FROM clusters WHERE id = ?",
        (cluster_id,),
    ).fetchone()
    if row is None:
        return
    centroid = _from_blob(row["centroid"])
    count = int(row["member_count"])
    new_centroid = _normalize((centroid * count + _normalize(vec)) / (count + 1))

    reps = json.loads(row["representative_queries"])
    if text not in reps:
        reps.append(text)
        reps = reps[-settings.MAX_REPRESENTATIVES:]

    conn.execute(
        "UPDATE clusters SET centroid = ?, member_count = ?, "
        "representative_queries = ?, last_updated = ? WHERE id = ?",
        (_to_blob(new_centroid), count + 1, json.dumps(reps), _now(), cluster_id),
    )
    conn.commit()


def get_cluster(cluster_id: int) -> dict | None:
    conn = get_conn()
    r = conn.execute("SELECT * FROM clusters WHERE id = ?", (cluster_id,)).fetchone()
    return _cluster_row_to_dict(r) if r else None


def all_clusters() -> list[dict]:
    conn = get_conn()
    rows = conn.execute("SELECT * FROM clusters").fetchall()
    return [_cluster_row_to_dict(r) for r in rows]


def count_clusters() -> int:
    conn = get_conn()
    return int(conn.execute("SELECT COUNT(*) AS c FROM clusters").fetchone()["c"])


def set_canonical(cluster_id: int, question: str, answer: str) -> None:
    conn = get_conn()
    conn.execute(
        "UPDATE clusters SET canonical_question = ?, canonical_answer = ? WHERE id = ?",
        (question, answer, cluster_id),
    )
    conn.commit()


def replace_clusters(new_clusters: list[dict]) -> dict[int, int]:
    """Wipe and rebuild the clusters table from a batch run.

    Each item: {centroid, member_count, representative_queries, query_ids}.
    Returns a map {temp_index -> new_cluster_id} so callers can reassign logs.
    """
    conn = get_conn()
    conn.execute("DELETE FROM clusters")
    now = _now()
    index_to_id: dict[int, int] = {}
    for i, c in enumerate(new_clusters):
        cur = conn.execute(
            "INSERT INTO clusters (centroid, member_count, representative_queries, "
            "created_at, last_updated) VALUES (?, ?, ?, ?, ?)",
            (
                _to_blob(_normalize(c["centroid"])),
                int(c["member_count"]),
                json.dumps(c["representative_queries"]),
                now,
                now,
            ),
        )
        index_to_id[i] = int(cur.lastrowid)
    conn.commit()
    return index_to_id


def _cluster_row_to_dict(r: sqlite3.Row) -> dict:
    return {
        "id": int(r["id"]),
        "centroid": _from_blob(r["centroid"]),
        "member_count": int(r["member_count"]),
        "canonical_question": r["canonical_question"],
        "canonical_answer": r["canonical_answer"],
        "representative_queries": json.loads(r["representative_queries"]),
        "created_at": r["created_at"],
        "last_updated": r["last_updated"],
    }
