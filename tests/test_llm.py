"""Greeting / relevance handling in generate_answer (LLM disabled in tests, so
the deterministic fallbacks are exercised)."""
from app import llm
from app.config import settings


def test_greeting_gets_friendly_reply_not_kb_dump(temp_db):
    # "hi" retrieves an irrelevant chunk (low score) -> conversational fallback,
    # NOT the raw knowledge-base text.
    contexts = [{"text": "Tuition fees are due before registration.", "source": "fees.md", "score": 0.05}]
    answer, used = llm.generate_answer("hi", contexts)
    assert used is False
    assert answer == llm.GREETING_FALLBACK
    assert "Tuition fees" not in answer


def test_no_context_message_is_conversational(temp_db):
    answer, used = llm.generate_answer("hello there", [])
    assert answer == llm.GREETING_FALLBACK


def test_relevant_question_uses_kb_chunk(temp_db):
    contexts = [{"text": "Register via the student portal.", "source": "acad.md", "score": 0.62}]
    answer, used = llm.generate_answer("how do I register?", contexts)
    assert used is False  # LLM disabled in tests
    assert answer == "Register via the student portal."  # extractive, grounded
