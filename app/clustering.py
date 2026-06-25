"""The semantic frequency engine (HYBRID).

The whole point of this project: measure "most frequently asked" SEMANTICALLY,
so that "How do I register for courses?", "what's the enrollment process", and
"how to sign up for classes" count as ONE question asked three times.

Two cooperating paths:
  1. REAL-TIME (assign_realtime): on every question, snap it to the nearest
     existing cluster centroid if cosine >= SIM_THRESHOLD; otherwise open a new
     cluster. Instant counts, no batch job needed.
  2. BATCH (recluster_batch): periodically re-cluster ALL logged questions from
     scratch with agglomerative clustering (no preset k). This corrects the
     drift that greedy real-time assignment accumulates — merging clusters that
     should be one and splitting ones that crept together.
"""
from __future__ import annotations

import numpy as np

from . import llm, store
from .config import settings


def assign_realtime(text: str, vec: np.ndarray) -> tuple[int, bool]:
    """Snap a question to a cluster (or create one). Returns (cluster_id, is_new)."""
    nearest = store.nearest_cluster(vec)
    if nearest is not None and nearest[1] >= settings.SIM_THRESHOLD:
        cluster_id = nearest[0]
        store.assign_to_cluster(cluster_id, vec, text)
        return cluster_id, False
    cluster_id = store.create_cluster(vec, text)
    return cluster_id, True


def recluster_batch(generate_canonical: bool = True) -> dict:
    """Rebuild all clusters from the full query log; regenerate canonical Q/A.

    Uses agglomerative clustering with average linkage on cosine distance and a
    distance threshold derived from SIM_THRESHOLD (no preset number of clusters,
    handles arbitrary cluster shapes). This is the Windows-friendly, wheel-
    installable stand-in for HDBSCAN and serves the same purpose: merge drift.
    """
    before = store.count_clusters()
    logs = store.all_query_logs()
    if not logs:
        return {
            "clusters_before": before,
            "clusters_after": 0,
            "queries_processed": 0,
            "canonical_generated": 0,
        }

    mat = np.vstack([_unit(l["vec"]) for l in logs])
    labels = _agglomerative_labels(mat)

    # Build new clusters grouped by label.
    groups: dict[int, list[int]] = {}
    for idx, label in enumerate(labels):
        groups.setdefault(int(label), []).append(idx)

    new_clusters: list[dict] = []
    group_query_ids: list[list[int]] = []
    for _, idxs in sorted(groups.items()):
        centroid = _unit(mat[idxs].mean(axis=0))
        reps = [logs[i]["text"] for i in idxs][-settings.MAX_REPRESENTATIVES:]
        new_clusters.append(
            {
                "centroid": centroid,
                "member_count": len(idxs),
                "representative_queries": reps,
            }
        )
        group_query_ids.append([logs[i]["id"] for i in idxs])

    index_to_id = store.replace_clusters(new_clusters)

    # Reassign each query log to its rebuilt cluster.
    for gi, query_ids in enumerate(group_query_ids):
        cid = index_to_id[gi]
        for qid in query_ids:
            store.set_query_cluster(qid, cid)

    canonical_generated = 0
    if generate_canonical:
        canonical_generated = _generate_canonical_for_top(index_to_id, new_clusters)

    return {
        "clusters_before": before,
        "clusters_after": len(new_clusters),
        "queries_processed": len(logs),
        "canonical_generated": canonical_generated,
    }


def ensure_canonical(cluster_id: int) -> dict | None:
    """Lazily generate + cache canonical Q/A for one cluster if it's missing."""
    cluster = store.get_cluster(cluster_id)
    if cluster is None:
        return None
    if cluster["canonical_question"] and cluster["canonical_answer"]:
        return cluster

    reps = cluster["representative_queries"]
    if not reps:
        return cluster
    contexts = store.search_chunks(cluster["centroid"], settings.RAG_TOP_K)
    question, answer, used = llm.synthesize_canonical(reps, contexts)
    # Only PERSIST an LLM-generated title. If the LLM was unavailable/rate-limited
    # (used=False), don't cache the fallback — leave it empty so we retry later
    # rather than poisoning the FAQ with a verbatim user phrasing forever.
    if used:
        store.set_canonical(cluster_id, question, answer)
        cluster["canonical_question"] = question
        cluster["canonical_answer"] = answer
    return cluster


# --- internals --------------------------------------------------------------

def _generate_canonical_for_top(index_to_id: dict[int, int], new_clusters: list[dict]) -> int:
    """Generate canonical Q/A for clusters big enough to matter, top-N by size."""
    sized = [
        (i, c) for i, c in enumerate(new_clusters)
        if c["member_count"] >= settings.CANONICAL_MIN_COUNT
    ]
    sized.sort(key=lambda x: -x[1]["member_count"])
    generated = 0
    for i, cluster in sized[: settings.FAQ_TOP_N]:
        cid = index_to_id[i]
        contexts = store.search_chunks(cluster["centroid"], settings.RAG_TOP_K)
        question, answer, used = llm.synthesize_canonical(
            cluster["representative_queries"], contexts
        )
        # Persist only real LLM titles (see ensure_canonical for the rationale).
        if used:
            store.set_canonical(cid, question, answer)
            generated += 1
    return generated


def _unit(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float32)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def _agglomerative_labels(mat: np.ndarray) -> np.ndarray:
    """Cluster rows of `mat` (unit vectors). Returns an integer label per row.

    distance_threshold = 1 - SIM_THRESHOLD on cosine distance: pairs closer than
    that get merged. No preset cluster count.
    """
    n = mat.shape[0]
    if n == 1:
        return np.array([0])

    from sklearn.cluster import AgglomerativeClustering

    model = AgglomerativeClustering(
        n_clusters=None,
        metric="cosine",
        linkage="average",
        distance_threshold=1.0 - settings.SIM_THRESHOLD,
    )
    return model.fit_predict(mat)
