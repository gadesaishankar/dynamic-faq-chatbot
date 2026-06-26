"""Pydantic request/response models for the API."""
from __future__ import annotations

from pydantic import BaseModel


class Turn(BaseModel):
    role: str  # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    question: str
    # Prior conversation turns (excluding the current question) for multi-turn
    # context. The frontend keeps these in memory for the session.
    history: list[Turn] | None = None


class Citation(BaseModel):
    text: str
    source: str
    score: float


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation]
    cluster_id: int
    new_cluster: bool
    llm_used: bool
    log_id: int            # id of this logged turn — used to attach feedback
    confidence: str        # "high" | "low" (relevance-based)
    cache_hit: bool        # answer served from the semantic cache


class FeedbackRequest(BaseModel):
    log_id: int
    vote: int  # 1 = 👍 helpful, -1 = 👎 not helpful


class FaqItem(BaseModel):
    cluster_id: int
    question: str
    answer: str
    ask_count: int
    total_count: int
    examples: list[str]


class FaqResponse(BaseModel):
    items: list[FaqItem]
    generated_from_queries: int


class CategoryQuestion(BaseModel):
    cluster_id: int
    question: str
    ask_count: int


class CategoryGroup(BaseModel):
    category: str
    total_asks: int
    questions: list[CategoryQuestion]


class CategoriesResponse(BaseModel):
    categories: list[CategoryGroup]


class ContentGap(BaseModel):
    cluster_id: int
    question: str
    ask_count: int
    avg_relevance: float    # how well the KB covers it (low = gap)
    helpful_rate: float | None  # 👍 / total feedback, or null if no feedback
    down_votes: int
    reason: str             # why it's flagged


class ContentGapsResponse(BaseModel):
    gaps: list[ContentGap]


class AnalyticsResponse(BaseModel):
    total_questions: int
    total_clusters: int
    cache_hit_rate: float
    feedback_count: int
    helpful_rate: float | None
    low_confidence_rate: float
    top_questions: list[dict]   # {question, ask_count}
    unanswered: list[dict]      # frequent + low-relevance {question, ask_count}


class AddKbRequest(BaseModel):
    filename: str
    text: str


class IngestResponse(BaseModel):
    chunks_indexed: int
    sources: list[str]


class ReclusterResponse(BaseModel):
    clusters_before: int
    clusters_after: int
    queries_processed: int
    canonical_generated: int
