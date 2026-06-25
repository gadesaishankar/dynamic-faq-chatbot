"""Central configuration. All values are env-overridable (see .env.example)."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Project root = parent of the `app` package.
BASE_DIR = Path(__file__).resolve().parent.parent

# Load .env if present (no error if it's missing).
load_dotenv(BASE_DIR / ".env")


def _get(name: str, default: str) -> str:
    val = os.getenv(name)
    return val if val not in (None, "") else default


def _get_int(name: str, default: int) -> int:
    try:
        return int(_get(name, str(default)))
    except ValueError:
        return default


def _get_float(name: str, default: float) -> float:
    try:
        return float(_get(name, str(default)))
    except ValueError:
        return default


class Settings:
    # --- LLM for answer generation (OPTIONAL) ---
    # provider: "gemini" | "openrouter" | "anthropic" | "none"
    LLM_PROVIDER: str = _get("LLM_PROVIDER", "openrouter").lower()

    # Google Gemini via its OpenAI-compatible endpoint. Free API key (free tier)
    # from https://aistudio.google.com/apikey — more reliable than OpenRouter free.
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "").strip()
    GEMINI_MODEL: str = _get("GEMINI_MODEL", "gemini-2.5-flash")
    GEMINI_BASE_URL: str = _get(
        "GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/"
    )

    # OpenRouter: OpenAI-compatible gateway to open-source models. Many ":free"
    # models cost $0 and run on OpenRouter's servers (nothing heavy runs locally).
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "").strip()
    OPENROUTER_MODEL: str = _get(
        "OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct:free"
    )
    OPENROUTER_BASE_URL: str = _get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    # Optional attribution headers OpenRouter uses for its rankings.
    OPENROUTER_REFERER: str = _get("OPENROUTER_REFERER", "http://localhost")
    OPENROUTER_TITLE: str = _get("OPENROUTER_TITLE", "Dynamic FAQ Chatbot")

    # Anthropic (only used when LLM_PROVIDER=anthropic)
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "").strip()
    CLAUDE_MODEL: str = _get("CLAUDE_MODEL", "claude-haiku-4-5")

    # --- Semantic frequency engine ---
    # Tuned for the default all-MiniLM-L6-v2 question embeddings: ~0.45 collapses
    # paraphrases into one cluster while keeping distinct intents apart. Raise for
    # stricter (more) clusters, lower for looser grouping.
    SIM_THRESHOLD: float = _get_float("SIM_THRESHOLD", 0.45)
    FAQ_TOP_N: int = _get_int("FAQ_TOP_N", 8)
    RECENCY_WINDOW_DAYS: int = _get_int("RECENCY_WINDOW_DAYS", 30)
    # A question must be asked at least this many times (semantically) to appear
    # on the FAQ page — keeps one-off questions out.
    FAQ_MIN_ASKS: int = _get_int("FAQ_MIN_ASKS", 3)
    CANONICAL_MIN_COUNT: int = _get_int("CANONICAL_MIN_COUNT", 3)
    # How many recent conversation turns to send to the LLM for follow-up context.
    MAX_HISTORY_TURNS: int = _get_int("MAX_HISTORY_TURNS", 6)
    # Max member queries kept on a cluster for canonical synthesis / display.
    MAX_REPRESENTATIVES: int = _get_int("MAX_REPRESENTATIVES", 12)

    # --- Retrieval ---
    RAG_TOP_K: int = _get_int("RAG_TOP_K", 4)

    # --- Models / paths ---
    EMBED_MODEL: str = _get("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    DB_PATH: str = _get("DB_PATH", str(BASE_DIR / "data" / "faq.db"))

    # --- Ingestion ---
    SOURCES_DIR: str = _get("SOURCES_DIR", str(BASE_DIR / "data" / "sources"))
    CHUNK_CHARS: int = _get_int("CHUNK_CHARS", 800)

    WEB_DIR: Path = BASE_DIR / "web"

    @property
    def llm_enabled(self) -> bool:
        if self.LLM_PROVIDER == "gemini":
            return bool(self.GEMINI_API_KEY)
        if self.LLM_PROVIDER == "openrouter":
            return bool(self.OPENROUTER_API_KEY)
        if self.LLM_PROVIDER == "anthropic":
            return bool(self.ANTHROPIC_API_KEY)
        return False


settings = Settings()
