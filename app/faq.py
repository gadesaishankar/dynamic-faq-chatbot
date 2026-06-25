"""Build the dynamic FAQ from the semantic clusters.

Two gates so the FAQ stays clean and useful:
  1. FREQUENCY THRESHOLD (FAQ_MIN_ASKS): a question must be asked enough times
     (semantically grouped) to appear at all — one-off questions are excluded.
  2. LLM TITLES: for the questions that qualify, show a clean, GENERIC title the
     LLM derives from the intent (not any user's raw wording). Raw phrasings are
     surfaced separately as "examples".

Ranking signal: asks within the recency window (so trending questions rise),
falling back to all-time count.
"""
from __future__ import annotations

from . import clustering, store
from .config import settings


def build_faq(top_n: int | None = None, min_asks: int | None = None) -> dict:
    top_n = top_n or settings.FAQ_TOP_N
    min_asks = settings.FAQ_MIN_ASKS if min_asks is None else min_asks

    clusters = store.all_clusters()

    ranked = []
    for c in clusters:
        recent = store.recent_count_for_cluster(c["id"], settings.RECENCY_WINDOW_DAYS)
        score = recent if recent > 0 else c["member_count"]
        # Frequency threshold: skip questions that aren't asked often enough.
        if score < min_asks:
            continue
        ranked.append((score, recent, c))

    ranked.sort(key=lambda x: (-x[0], -x[2]["member_count"]))
    top = ranked[:top_n]

    items = []
    for score, recent, c in top:
        # Generate a clean, generic LLM title for shown items if missing.
        if not c["canonical_question"]:
            refreshed = clustering.ensure_canonical(c["id"])
            if refreshed:
                c = refreshed

        question = c["canonical_question"] or _fallback_question(c)
        answer = c["canonical_answer"] or ""
        items.append(
            {
                "cluster_id": c["id"],
                "question": question,
                "answer": answer,
                "ask_count": score,
                "total_count": c["member_count"],
                "examples": c["representative_queries"][:3],
            }
        )

    return {"items": items, "generated_from_queries": store.total_query_count()}


def _fallback_question(cluster: dict) -> str:
    """Only used when the LLM is unavailable — pick a representative phrasing."""
    reps = cluster["representative_queries"]
    if not reps:
        return "(question)"
    return sorted(reps, key=len)[len(reps) // 2]
