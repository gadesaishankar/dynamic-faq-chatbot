"""LLM wrapper for answer generation — OPTIONAL and provider-pluggable.

Providers (config.LLM_PROVIDER):
  - "openrouter" (default): OpenAI-compatible gateway to open-source models.
    Use a ":free" model for $0. All compute runs on OpenRouter's servers, so
    nothing heavy runs on the user's machine (good for low-RAM laptops).
  - "anthropic": Claude (paid).
  - "none" / no key: extractive fallback.

If no key is configured (or any call fails), every function degrades gracefully
to a free, no-network fallback so the whole app — RAG + the semantic frequency
engine — still works at $0. The LLM only *improves* answer/FAQ phrasing.
"""
from __future__ import annotations

from .config import settings

_client = None


def _get_client():
    """Lazily build an OpenAI-compatible client (used for OpenRouter and Gemini —
    both speak the OpenAI Chat Completions API)."""
    global _client
    if _client is None:
        from openai import OpenAI

        if settings.LLM_PROVIDER == "gemini":
            _client = OpenAI(
                base_url=settings.GEMINI_BASE_URL,
                api_key=settings.GEMINI_API_KEY,
            )
        else:  # openrouter
            _client = OpenAI(
                base_url=settings.OPENROUTER_BASE_URL,
                api_key=settings.OPENROUTER_API_KEY,
            )
    return _client


def _get_anthropic():
    global _client
    if _client is None:
        import anthropic

        _client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
    return _client


def _chat(system: str, messages: list[dict], max_tokens: int) -> str | None:
    """Multi-turn completion. `messages` is a list of {role, content} ending with
    the current user turn. Returns text, or None to trigger the fallback."""
    if not settings.llm_enabled:
        return None
    try:
        if settings.LLM_PROVIDER == "anthropic":
            resp = _get_anthropic().messages.create(
                model=settings.CLAUDE_MODEL,
                max_tokens=max_tokens,
                system=system,
                messages=messages,
            )
            return "".join(b.text for b in resp.content if b.type == "text").strip()

        # OpenAI-compatible providers: gemini or openrouter
        if settings.LLM_PROVIDER == "gemini":
            model = settings.GEMINI_MODEL
            extra_headers = None
        else:  # openrouter
            model = settings.OPENROUTER_MODEL
            extra_headers = {
                "HTTP-Referer": settings.OPENROUTER_REFERER,
                "X-Title": settings.OPENROUTER_TITLE,
            }
        resp = _get_client().chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "system", "content": system}, *messages],
            extra_headers=extra_headers,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception:
        # Never let an LLM hiccup break the app — fall back to extractive.
        return None


def _context_block(contexts: list[dict]) -> str:
    return "\n\n".join(
        f"[{i + 1}] (source: {c['source']})\n{c['text']}"
        for i, c in enumerate(contexts)
    )


GREETING_FALLBACK = (
    "Hi! I'm the department FAQ assistant. You can ask me about courses, "
    "registration, fees, library hours, scholarships, Wi-Fi, and more."
)


def generate_answer(
    question: str, contexts: list[dict], history: list[dict] | None = None
) -> tuple[str, bool]:
    """Return (answer, llm_used).

    Factual answers are grounded strictly in `contexts`, but greetings / small
    talk / off-topic messages (no relevant context) get a friendly reply instead
    of a forced "I don't know". `history` is prior turns for follow-ups.
    """
    relevant = bool(contexts) and contexts[0]["score"] >= settings.RELEVANCE_THRESHOLD

    system = (
        "You are a friendly assistant for a college department's FAQ desk. "
        "If the user greets you or makes small talk, reply warmly in one short "
        "sentence and invite them to ask about the department (courses, "
        "registration, fees, library, etc.). "
        "For actual questions, answer ONLY from the provided context and cite "
        "sources inline like [1], [2]. If no relevant context is provided, do "
        "not invent facts — briefly say what you can help with, or suggest "
        "contacting the department office. Keep replies concise (1-4 sentences)."
    )
    turns = list(history or [])[-settings.MAX_HISTORY_TURNS:]
    if relevant:
        content = f"Context:\n{_context_block(contexts)}\n\nUser message: {question}"
    else:
        content = f"(No relevant FAQ context found for this message.)\n\nUser message: {question}"
    turns.append({"role": "user", "content": content})

    text = _chat(system, turns, max_tokens=512)
    if text:
        return text, True

    # Fallbacks when the LLM is unavailable:
    if relevant:
        return contexts[0]["text"].strip(), False  # extractive: best chunk
    return GREETING_FALLBACK, False                 # conversational default


def synthesize_canonical(
    example_questions: list[str], contexts: list[dict]
) -> tuple[str, str, bool]:
    """From several real phrasings of one intent, produce a clean canonical
    FAQ question + a concise grounded answer.

    Returns (question, answer, llm_used).
    """
    fallback_q = _pick_representative(example_questions)
    fallback_a = contexts[0]["text"].strip() if contexts else ""

    system = (
        "You write FAQ entries for a college department. You are given several "
        "real user questions that all share the SAME underlying intent, phrased "
        "differently. Do NOT copy any user's wording. Instead, extract the "
        "shared intent and write a single clean, GENERIC FAQ question that "
        "captures it (well-formed, neutral, no slang/typos), then a concise "
        "answer grounded ONLY in the provided context. Format EXACTLY as:\n"
        "Q: <one clear generic question>\nA: <concise answer>"
    )
    user = (
        "User questions (same intent, different phrasings):\n"
        + "\n".join(f"- {q}" for q in example_questions)
        + f"\n\nContext:\n{_context_block(contexts)}"
    )
    text = _chat(system, [{"role": "user", "content": user}], max_tokens=400)
    if text:
        q, a = _parse_qa(text)
        return (q or fallback_q), (a or fallback_a), True
    return fallback_q, fallback_a, False


def _pick_representative(questions: list[str]) -> str:
    """Heuristic: the medium-length phrasing is usually the clearest."""
    if not questions:
        return ""
    ordered = sorted(questions, key=len)
    return ordered[len(ordered) // 2]


def _parse_qa(text: str) -> tuple[str, str]:
    q, a = "", ""
    for line in text.splitlines():
        s = line.strip()
        if s.lower().startswith("q:"):
            q = s[2:].strip()
        elif s.lower().startswith("a:"):
            a = s[2:].strip()
        elif a:
            a += " " + s
    return q, a
