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
    # True when this question opened a brand-new semantic cluster.
    new_cluster: bool
    llm_used: bool


class FaqItem(BaseModel):
    cluster_id: int
    question: str
    answer: str
    ask_count: int          # asks within the recency window (the ranking signal)
    total_count: int        # all-time asks for this intent
    examples: list[str]     # a few real phrasings users typed


class FaqResponse(BaseModel):
    items: list[FaqItem]
    generated_from_queries: int


class IngestResponse(BaseModel):
    chunks_indexed: int
    sources: list[str]


class ReclusterResponse(BaseModel):
    clusters_before: int
    clusters_after: int
    queries_processed: int
    canonical_generated: int
