# Architecture & Design — Dynamic FAQ Chatbot

A deep dive into how the AI system works: how it produces a reply, the
self-updating FAQ engine, every metric it tracks, and **every threshold value
with the reasoning (and measurements) behind it.**

- [1. What it is](#1-what-it-is)
- [2. System architecture](#2-system-architecture)
- [3. How a reply is produced (request lifecycle)](#3-how-a-reply-is-produced-request-lifecycle)
- [4. The self-updating FAQ engine](#4-the-self-updating-faq-engine)
- [5. Thresholds & parameters — values and why](#5-thresholds--parameters--values-and-why)
- [6. Metrics](#6-metrics)
- [7. Guardrails & safety](#7-guardrails--safety)
- [8. Cost & latency design](#8-cost--latency-design)
- [9. Tech stack & data model](#9-tech-stack--data-model)
- [10. Key design decisions & trade-offs](#10-key-design-decisions--trade-offs)

---

## 1. What it is

A Retrieval-Augmented Generation (RAG) chatbot for a college department's FAQs
whose **FAQ page writes itself** from what users actually ask — where "most
asked" is measured **semantically** (paraphrases of one question count as one).
It runs entirely on free/open-source components plus a free Gemini API tier.

Two cooperating jobs:
1. **Answer questions** accurately, grounded in a knowledge base.
2. **Learn from the questions** to keep the FAQ — and a content-gap report —
   continuously up to date.

---

## 2. System architecture

```
                        ┌──────────────────────────────────────────┐
   Browser (SPA)        │  Chat tab  │  Dynamic FAQ tab │ Insights  │
   one page, 3 tabs     └──────┬───────────────┬───────────────┬────┘
                               │ POST /chat[/stream]            │ GET /analytics
                               │ POST /feedback │ GET /faq      │ GET /admin/content-gaps
                               ▼                ▼               ▼
                        ┌──────────────────── FastAPI (app/main.py) ──────────────────┐
                        │  rag │ retrieval │ clustering │ faq │ analytics │ llm        │
                        └──────┬───────────────┬───────────────┬───────────┬──────────┘
                               │               │               │           │
              embeddings (local MiniLM, CPU)   │               │      LLM gateway
              ─ used for retrieval AND clustering AND cache ─   │     gemini│openrouter│anthropic│none
                               │               │               │           │
                               ▼               ▼               ▼           ▼
                        ┌───────────────── SQLite + numpy cosine (app/store.py) ──────┐
                        │  kb_chunks │ query_logs │ clusters │ feedback │ answer_cache │
                        └─────────────────────────────────────────────────────────────┘
                                                                            │
                                                          external (only paid-ish piece)
                                                                Google Gemini API (free tier)
```

**The left side is 100% local and free** (embeddings + SQLite); **the right side
is the only external dependency** (Gemini). Remove Gemini and the app still runs
— answers just become extractive (top retrieved chunk).

| Module | Responsibility |
|---|---|
| `main.py` | FastAPI routes, streaming, request validation, post-processing |
| `embeddings.py` | Local `all-MiniLM-L6-v2`; one embedding space for retrieval + clustering + cache |
| `retrieval.py` | Hybrid retrieval: dense vectors + BM25 keyword, fused with RRF |
| `rag.py` | Per-request orchestration: rewrite → cache → retrieve → generate → cache |
| `llm.py` | Provider-pluggable LLM (generation, streaming, query rewriting, FAQ titles) |
| `clustering.py` | Semantic frequency engine: real-time grouping + nightly batch |
| `faq.py` | Builds the ranked, thresholded, LLM-titled FAQ |
| `analytics.py` | Usage metrics + the content-gap report |
| `store.py` | SQLite tables + brute-force numpy cosine search |

---

## 3. How a reply is produced (request lifecycle)

A single `POST /chat` (the `/chat/stream` path is identical, just streamed):

```
question + history
   │
   1. clean input (strip, cap 1000 chars)
   │
   2. embed the ORIGINAL question  ──► query_vec (384-dim, L2-normalized)
   │      (this vector is reused for clustering, logging, and the cache key)
   │
   3. follow-up rewrite?  if history & LLM on:
   │      "and on weekends?" ──LLM──► "library hours on weekends"  ──► search_vec
   │
   4. cache check (first-turn only):
   │      nearest cached question, cosine ≥ 0.93 ?  ── HIT ──► return cached answer (no LLM)
   │
   5. hybrid retrieval (search_vec, search_text):
   │      vector cosine ranking  ⨁  BM25 keyword ranking  ──RRF──► top-4 chunks
   │      relevance = max vector cosine over all chunks
   │
   6. relevance gate:  relevant = relevance ≥ 0.25
   │
   7. generate:
   │      LLM on  ──► grounded answer + [1][2] citations   (relevant)
   │                  friendly greeting / "I can help with…" (not relevant)
   │      LLM off/fails ──► extractive top chunk (relevant) | friendly default (not)
   │
   8. cache store (first-turn & relevant)
   │
   9. real-time cluster assignment (centroid cosine ≥ 0.45 → join, else new cluster)
   │
   10. log to query_logs (text, embedding, score=relevance, cache_hit)
   │
   ▼
answer + citations + {log_id, cluster_id, new_cluster, llm_used, confidence, cache_hit}
```

**Why the original question is embedded separately from the rewritten one:**
clustering and the cache should key off *what the user actually asked* (so the
FAQ reflects real phrasings), while retrieval benefits from the
context-resolved, standalone query.

**Streaming (`/chat/stream`)** does steps 2–6 up front, then streams the answer
tokens as Server-Sent Events, and finally emits a `meta` event with
`log_id`/`cluster_id`/`citations` so the UI can attach 👍/👎. If the LLM can't
stream (e.g. rate-limited), it falls back to a single full answer — the client
never breaks.

---

## 4. The self-updating FAQ engine

The novel part. "Most frequently asked" is measured by **meaning**, not text.

**Real-time grouping** (every question): embed it, find the nearest existing
cluster centroid; if cosine ≥ `SIM_THRESHOLD` (0.45) it joins that cluster and
the centroid updates as a running mean — else it seeds a new cluster. Instant
counts, no batch needed.

**Nightly batch re-clustering** (`scripts/recluster.py`, also auto at 03:00):
re-clusters *all* logged questions from scratch with **agglomerative clustering**
(average linkage, cosine, no preset *k*). This corrects the drift greedy
real-time assignment accumulates and regenerates a clean **LLM-generated generic
title** per top cluster.

**The FAQ page** (`GET /faq`): keep clusters asked ≥ `FAQ_MIN_ASKS` (3) times,
rank by recency-weighted frequency (asks in the last 30 days), show the LLM
title + cached answer + example phrasings.

**Content-gap report** (`GET /admin/content-gaps`) — the headline insight:
combine the **frequency** signal with a **quality** signal (low KB relevance
and/or 👎 feedback) to surface *"asked a lot, answered poorly"* — i.e. exactly
what documentation to write next.

---

## 5. Thresholds & parameters — values and why

Every value is env-overridable (see `.env.example`). The three similarity
thresholds are the important ones, and they were **chosen from measurements**,
not guessed.

| Parameter | Value | Controls | Why this value |
|---|---|---|---|
| `RELEVANCE_THRESHOLD` | **0.25** | question vs. greeting/off-topic | **Measured:** greetings ("hi", "thanks") score **≤ 0.12** top-chunk cosine; real questions score **≥ 0.34** (even "wifi password", which is in the KB, scored 0.34). 0.25 sits cleanly in the gap → greetings get a friendly reply, real questions get grounded answers. |
| `SIM_THRESHOLD` | **0.45** | real-time semantic clustering | **Measured:** with `all-MiniLM-L6-v2`, paraphrases of one intent average **0.61** cosine (min ~0.30); *different* intents top out around **0.44**. A sweep (0.40/0.45/0.50/0.55) showed **0.45** collapses 8 registration paraphrases into **one** cluster while keeping registration/library/fees separate (zero cross-contamination). The original default of 0.78 was far too high and left every paraphrase in its own cluster. |
| `CACHE_SIM_THRESHOLD` | **0.93** | reuse a cached answer | Deliberately **strict**. Reuse must be *safe* — only a near-identical question (same intent, trivially different wording/punctuation) should get a recycled answer. At 0.45 it would wrongly serve a "library hours" answer to "library books". 0.93 ≈ "essentially the same question." |
| `FAQ_MIN_ASKS` | **3** | min asks to appear on the FAQ | Keeps one-off questions out of the FAQ so it reflects genuinely *frequent* questions, not noise. |
| `RAG_TOP_K` | **4** | chunks retrieved per query | Enough context for grounded answers without burying the model in irrelevant text (the KB chunks are ~800 chars). |
| `RRF_K` | **60** | reciprocal rank fusion constant | Standard RRF default; dampens the weight of lower-ranked results when fusing vector + BM25. |
| `RECENCY_WINDOW_DAYS` | **30** | FAQ ranking window | Ranks by asks in the trailing 30 days so *currently trending* questions rise, not all-time history. |
| `MAX_HISTORY_TURNS` | **6** | multi-turn context sent to LLM | Enough for follow-ups; bounded so prompts stay small/cheap. |
| `CANONICAL_MIN_COUNT` | **3** | when to spend an LLM call on a title | Only generate a polished FAQ title once a cluster is clearly worth showing. |
| `CHUNK_CHARS` | **800** | KB chunk size | Paragraph-ish chunks: specific enough to be relevant, large enough to carry a complete answer. |

> **The two-threshold insight** (a sharp thing to be able to explain): the system
> runs **two different similarity jobs on the same embedding space**.
> **Clustering matches loosely at 0.45** — group paraphrases and *count*
> frequency. **The cache matches strictly at 0.93** — safely *reuse* an exact
> answer. Same model, two thresholds tuned for opposite goals (recall vs.
> precision).

---

## 6. Metrics

**Offline / eval** (`python -m scripts.eval`, golden set in `data/eval/golden.json`):

| Metric | Definition |
|---|---|
| **Retrieval recall** | Did the expected source document make it into the top-k retrieved chunks? |
| **Keyword coverage** | Does the answer contain the expected facts (golden keywords)? |
| **Faithfulness** (LLM-judge) | Is the answer *fully supported* by the retrieved context? Scored YES/NO by the LLM. Catches hallucination. |

**Online / live** (`GET /analytics`, shown on the Insights tab):

| Metric | Definition / why it matters |
|---|---|
| **Helpful rate** | 👍 / total feedback — the core quality signal users give you. |
| **Cache hit rate** | % of questions served from the answer cache (no LLM call) — cost + latency + rate-limit relief. |
| **Low-confidence rate** | % of questions whose top relevance < 0.25 — proxy for off-topic / out-of-scope traffic and KB gaps. |
| **Top questions** | Most-asked clusters (volume). |
| **Unanswered** | Frequent questions with low KB relevance — KB gaps to fill. |
| **Content gaps** | Frequency × low relevance/👎 → ranked "what to document next". |

These map to the product's north-star: **self-service deflection** (answer
without a human), measured via helpful rate and falling low-confidence/unanswered
rates over time.

---

## 7. Guardrails & safety

- **Strict grounding** — the system prompt forces answers to come *only* from
  retrieved context; otherwise the bot says it doesn't have the info.
- **Honest "I don't know"** — no relevant context → it states scope / suggests
  contacting the office instead of inventing facts.
- **Prompt-injection resistance** — the prompt instructs the model to treat
  retrieved context and user text as **data, not instructions** (so a malicious
  KB line or message can't hijack its role or leak the prompt).
- **Input caps** — questions truncated to 1000 chars.
- **Graceful degradation** — any LLM error/rate-limit falls back to extractive
  answers; the app never returns an error to the user for an LLM hiccup.
- **No secrets in the repo** — API keys live in `.env` (git-ignored) / host
  secrets, never committed.

---

## 8. Cost & latency design

Built for a $0 / low-RAM posture:

- **Embeddings are local & free** (`all-MiniLM-L6-v2`, CPU). This is essential —
  the clustering engine embeds *every* question; doing that via a paid API would
  be the biggest recurring cost.
- **Answer cache** skips the LLM for repeated questions — the single biggest
  lever for cost, latency, **and** the Gemini free-tier limit (20 requests/min).
- **Cheapest viable LLM** — Gemini `2.5-flash` free tier; bursts that exceed the
  RPM limit fall back to extractive answers instead of erroring.
- **SQLite + numpy** — no vector-DB service to pay for or run; brute-force cosine
  is fine at prototype scale.
- **Streaming** improves *perceived* latency (tokens appear immediately).

---

## 9. Tech stack & data model

**Stack:** Python · FastAPI · `sentence-transformers` (local) · scikit-learn
(agglomerative) · `rank-bm25` · SQLite · Gemini via the OpenAI-compatible API ·
APScheduler · static HTML/JS SPA · Docker → Hugging Face Spaces.

**SQLite tables:**

| Table | Holds |
|---|---|
| `kb_chunks` | KB content: `text`, `source`, `embedding[384]` |
| `query_logs` | every question: `text`, `embedding`, `ts`, `cluster_id`, `answer`, `score`, `cache_hit` |
| `clusters` | semantic-frequency state: `centroid`, `member_count`, `canonical_question/answer`, `representative_queries` |
| `feedback` | `log_id`, `vote` (±1), `ts` |
| `answer_cache` | `question`, `embedding`, `answer`, `citations`, `ts` |

---

## 10. Key design decisions & trade-offs

| Decision | Why | Trade-off |
|---|---|---|
| **Hybrid retrieval** (vector + BM25) | Vectors capture meaning; BM25 catches exact terms (course codes, names, "Wi-Fi") | Slightly more compute per query (negligible at this scale) |
| **Agglomerative, not HDBSCAN** | Same goal (no preset *k*, merge drift) with clean Windows wheels and no native build pain | Single global distance threshold; the nightly pass mitigates |
| **Local SQLite, not a vector DB** | Free, zero-setup, fits a prototype; `store.py` is the seam to swap in pgvector/OpenSearch later | Brute-force cosine won't scale to millions of vectors |
| **Cache only first-turn, confident answers** | Follow-ups depend on history; greetings shouldn't be cached | Multi-turn answers aren't cached |
| **Persist LLM titles only on success** | A rate-limited title falling back to raw user text would otherwise be cached forever | Title may show a representative phrasing until the next successful generation |
| **Pluggable LLM, free-tier first** | Gemini free tier reliable enough; OpenRouter/Anthropic/none available via one env var | Free tiers are rate-limited (mitigated by the cache + fallback) |
| **Deliberately NOT built** | No fine-tuning, no custom vector DB, no multi-agent | Scale doesn't justify them yet — add when a metric demands it |
