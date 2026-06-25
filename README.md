---
title: Dynamic FAQ Chatbot
emoji: 🎓
colorFrom: indigo
colorTo: green
sdk: docker
app_port: 8000
pinned: false
---

# Dynamic FAQ Chatbot — self-updating FAQ via semantic frequency

A GenAI FAQ chatbot for a college department (RAG + NLP + LLM) **plus** a FAQ
page that writes itself from what users actually ask the bot — ranked by how
often each question is asked, measured **semantically**.

The hard part, and the whole point: *"How do I register for courses?"*,
*"what's the enrollment process"*, and *"how to sign up for classes"* are counted
as **one question asked three times**, not three different questions. The system
embeds every question, groups them by meaning, counts the groups, and surfaces
the biggest groups as the FAQ — live.

## Why this runs at $0

| Piece | Choice | Cost |
|---|---|---|
| Embeddings (RAG + clustering) | local `sentence-transformers` (CPU) | **free** |
| Store + vector search | SQLite + numpy cosine (no account, no server) | **free** |
| Answer generation | **OpenRouter** open-source model (`:free` tier) — optional | **free** |

> Designed for a low-RAM laptop: nothing heavy runs locally except the small
> embedding model. Answer generation goes through **OpenRouter** (an
> OpenAI-compatible gateway to open-source models like Llama / Qwen / Mistral),
> using a `:free` model that costs $0 and runs on their servers.
>
> And it still works with **no API key at all**: it returns the best-matching
> knowledge-base content and uses the most representative real question as each
> FAQ entry. The **entire semantic-frequency engine** — the novel part — needs no
> LLM, no key, and no money. Add a free OpenRouter key only for polished answers.

## How it works

```
  source FAQ docs ──ingest──► kb_chunks (embedded, in SQLite)
                                   ▲ retrieve top-k
  user ─► POST /chat ─► embed ─────┘─► Claude Haiku (or extractive) ─► answer
                          │
                          ├─► log the question (text + embedding)
                          └─► REAL-TIME grouping: nearest cluster centroid?
                                 cosine ≥ threshold → count++  else → new cluster

  nightly BATCH job re-clusters ALL questions (agglomerative) → merges drift,
  regenerates canonical Q + A per top cluster

  user ─► GET /faq ─► top clusters by recency-weighted frequency
```

Hybrid by design: real-time grouping gives instant counts; the nightly batch
pass corrects the drift greedy assignment accumulates.

## Quick start

```bash
# 1. Install (first run downloads a small embedding model, ~80 MB)
python -m venv .venv
.venv\Scripts\activate            # Windows  (use: source .venv/bin/activate on macOS/Linux)
pip install -r requirements.txt

# 2. (optional) enable polished answers — get a FREE key at openrouter.ai/keys
copy .env.example .env            # then paste OPENROUTER_API_KEY  (optional)

# 3. Build the knowledge base from data/sources/
python -m scripts.ingest_kb

# 4. Run the app
uvicorn app.main:app --reload
```

Open **http://127.0.0.1:8000/** — a single page with two tabs:
- **Chat** — multi-turn conversation; history persists while you switch tabs and
  resets on a full page refresh (or the "＋ New chat" button).
- **Dynamic FAQ** — auto-generated, auto-refreshing. Only questions asked at
  least `FAQ_MIN_ASKS` times (default 3) appear, and each title is an
  LLM-generated *generic* phrasing of the intent (raw user wordings show under
  "people also asked"). Without an LLM key, titles fall back to a representative
  user phrasing.

## Prove it works (the key demo)

With the server running:

```bash
python -m scripts.seed_demo
```

It fires ~8 different phrasings of "how do I register", plus other intents, then
prints the FAQ. You should see each intent collapse into **one** cluster and rise
to the top of `/faq` — the semantic-frequency engine in action.

## Tuning

Everything is env-overridable — see [.env.example](.env.example):
`SIM_THRESHOLD` (grouping strictness), `FAQ_MIN_ASKS` (min asks before a question
shows on the FAQ), `FAQ_TOP_N`, `RECENCY_WINDOW_DAYS`, `RAG_TOP_K`,
`MAX_HISTORY_TURNS` (multi-turn context window).

## Features

- **RAG chat** — hybrid retrieval (vector + BM25, fused with RRF), grounded
  answers with citations, **streaming**, multi-turn with **follow-up query
  rewriting**, and a **friendly path for greetings/off-topic** messages.
- **Self-updating FAQ** — semantic-frequency clustering (real-time + nightly
  batch), shown above a frequency threshold with LLM-generated generic titles.
- **Feedback loop** — 👍/👎 per answer feeds the metrics.
- **Semantic answer cache** — near-duplicate questions skip the LLM (cuts cost
  and dodges free-tier rate limits).
- **Guardrails** — strict grounding, honest "I don't know," prompt-injection
  resistance, input length caps.
- **Insights tab** — usage/quality analytics + the **content-gap report**
  ("asked a lot, answered poorly") + add-knowledge form.
- **Eval harness** — golden set scored for retrieval recall, keyword coverage,
  and LLM-judge faithfulness (`python -m scripts.eval`).
- **Pluggable LLM** — Gemini (default) / OpenRouter / Anthropic / none.

## Evaluation

```bash
python -m scripts.ingest_kb && python -m scripts.eval
```
Prints retrieval recall, keyword coverage, and faithfulness — the numbers to
quote. Edit `data/eval/golden.json` to add cases.

## Multilingual (one-line switch)

Set `EMBED_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
(same 384-dim, runs locally) and re-ingest. Gemini already answers in any
language.

## Deploying

See **[DEPLOY.md](DEPLOY.md)** — ships a [`Dockerfile`](Dockerfile) and covers
the two things that shape deployment (≥1 GB RAM for the embedding model, and a
persistent volume for the SQLite FAQ history), with recommended hosts
(Hugging Face Spaces for a free demo; Render/Railway with a volume for durability).

## API

| Method | Path | Purpose |
|---|---|---|
| POST | `/chat` | `{question, history}` → grounded answer + `log_id`, `confidence`, `cache_hit`, cluster |
| POST | `/chat/stream` | same, streamed token-by-token (SSE) |
| POST | `/feedback` | `{log_id, vote}` — 👍 (1) / 👎 (-1) on an answer |
| GET | `/faq?top_n=8` | top clusters ranked by recency-weighted frequency |
| GET | `/analytics` | usage + quality metrics (helpful rate, cache hits, top/unanswered) |
| GET | `/admin/content-gaps` | "asked a lot, answered poorly" — what to document next |
| POST | `/admin/kb` | `{filename, text}` — add a KB doc + re-ingest |
| POST | `/ingest` | rebuild the knowledge base from `data/sources/` |
| POST | `/admin/recluster` | run the batch re-clustering pass now |
| GET | `/health` | status (llm enabled?, #clusters, #queries) |

The nightly batch job runs automatically at 03:00 via APScheduler; you can also
run it standalone for cron / Windows Task Scheduler:

```bash
python -m scripts.recluster
```

## Tests

```bash
pip install pytest
pytest
```

Tests verify the core guarantee (paraphrases collapse into one cluster, distinct
intents stay separate, FAQ ranks by frequency, retrieval orders by similarity)
using hand-made vectors — no model download or API key required.

## Project layout

```
app/        config, embeddings, store (SQLite+numpy), llm, rag,
            clustering (the engine), faq, ingest, main (FastAPI)
scripts/    ingest_kb, recluster, seed_demo
data/sources/  sample department FAQ docs (edit / add your own)
web/        chat UI + dynamic FAQ page
tests/      offline unit tests
```

## Notes & next steps

- **Cold start:** with little traffic the FAQ is thin — seed it with the demo or
  a few hand-written entries; it takes over as real questions accumulate.
- **Scaling past a prototype:** the functions in `app/store.py` are the seam
  where a MongoDB Atlas Vector Search backend (matching the resume stack) would
  slot in without touching the rest of the app.
- **Swapping the LLM:** `LLM_PROVIDER` selects `openrouter` (default, free
  open-source models), `anthropic` (Claude), or `none` (extractive). Change
  `OPENROUTER_MODEL` to any model id from openrouter.ai/models. To go fully
  offline later, point the OpenAI-compatible client in `app/llm.py` at a local
  Ollama server — nothing else changes.
