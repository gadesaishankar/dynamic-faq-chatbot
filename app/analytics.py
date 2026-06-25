"""Insights for the product owner: content gaps + usage analytics.

Content gaps are the headline feature: combine the FREQUENCY signal (the
clustering engine) with a QUALITY signal (low KB relevance and/or 👎 feedback) to
surface "asked a lot, answered poorly" — i.e. exactly what docs to write next.
"""
from __future__ import annotations

from . import store
from .config import settings


def _title(cluster: dict) -> str:
    if cluster["canonical_question"]:
        return cluster["canonical_question"]
    reps = cluster["representative_queries"]
    return sorted(reps, key=len)[len(reps) // 2] if reps else "(question)"


def _freq(cluster: dict) -> int:
    recent = store.recent_count_for_cluster(cluster["id"], settings.RECENCY_WINDOW_DAYS)
    return recent if recent > 0 else cluster["member_count"]


def content_gaps(min_asks: int = 2) -> list[dict]:
    avg_scores = store.avg_score_by_cluster()
    fb = store.feedback_by_cluster()
    gaps = []
    for c in store.all_clusters():
        count = _freq(c)
        if count < min_asks:
            continue
        avg_rel = avg_scores.get(c["id"], 0.0)
        votes = fb.get(c["id"], {"up": 0, "down": 0})
        total_votes = votes["up"] + votes["down"]
        helpful = (votes["up"] / total_votes) if total_votes else None

        reasons = []
        if avg_rel < settings.RELEVANCE_THRESHOLD:
            reasons.append("knowledge base barely covers this")
        if helpful is not None and helpful < 0.5:
            reasons.append("users rated answers unhelpful")
        elif votes["down"] >= 2:
            reasons.append("repeated negative feedback")
        if not reasons:
            continue

        gaps.append(
            {
                "cluster_id": c["id"],
                "question": _title(c),
                "ask_count": count,
                "avg_relevance": round(avg_rel, 3),
                "helpful_rate": (round(helpful, 3) if helpful is not None else None),
                "down_votes": votes["down"],
                "reason": "; ".join(reasons),
            }
        )
    gaps.sort(key=lambda g: (-g["ask_count"], g["avg_relevance"]))
    return gaps


def overview() -> dict:
    clusters = store.all_clusters()
    total_q = store.total_query_count()
    fb = store.feedback_stats()
    low = store.count_below_score(settings.RELEVANCE_THRESHOLD)
    avg_scores = store.avg_score_by_cluster()

    ranked = sorted(clusters, key=lambda c: -c["member_count"])
    top = [{"question": _title(c), "ask_count": c["member_count"]} for c in ranked[:8]]

    unanswered = [
        {"question": _title(c), "ask_count": c["member_count"]}
        for c in ranked
        if c["member_count"] >= 2
        and avg_scores.get(c["id"], 0.0) < settings.RELEVANCE_THRESHOLD
    ][:8]

    return {
        "total_questions": total_q,
        "total_clusters": len(clusters),
        "cache_hit_rate": round(store.cache_hit_rate(), 3),
        "feedback_count": fb["count"],
        "helpful_rate": (round(fb["helpful_rate"], 3) if fb["helpful_rate"] is not None else None),
        "low_confidence_rate": round(low / total_q, 3) if total_q else 0.0,
        "top_questions": top,
        "unanswered": unanswered,
    }
