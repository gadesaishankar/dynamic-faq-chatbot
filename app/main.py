"""FastAPI app: chat (+streaming), dynamic FAQ, feedback, analytics, admin, UI.

Routes:
  POST /chat            -> RAG answer + log + real-time clustering + cache
  POST /chat/stream     -> same, streamed token-by-token (SSE)
  POST /feedback        -> 👍/👎 on an answer
  GET  /faq             -> top clusters by recency-weighted frequency
  GET  /analytics       -> usage + quality metrics
  GET  /admin/content-gaps -> "asked a lot, answered poorly"
  POST /admin/kb        -> add a knowledge-base doc + re-ingest
  POST /ingest          -> rebuild the KB from data/sources
  POST /admin/recluster -> run the batch re-clustering pass now
  GET  /                -> single-page UI
"""
from __future__ import annotations

import json
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import FileResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import analytics, clustering, faq, ingest, llm, rag, store
from .config import settings
from .schemas import (
    AddKbRequest,
    AnalyticsResponse,
    CategoriesResponse,
    ChatRequest,
    ChatResponse,
    ContentGapsResponse,
    FaqResponse,
    FeedbackRequest,
    IngestResponse,
    ReclusterResponse,
)

MAX_QUESTION_CHARS = 1000
scheduler = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    store.init_db()
    global scheduler
    try:
        from apscheduler.schedulers.background import BackgroundScheduler

        scheduler = BackgroundScheduler(daemon=True)
        scheduler.add_job(clustering.recluster_batch, "cron", hour=3, id="recluster")
        scheduler.start()
    except Exception:
        scheduler = None
    yield
    if scheduler is not None:
        scheduler.shutdown(wait=False)


app = FastAPI(title="Dynamic FAQ Chatbot", lifespan=lifespan)


def _clean_question(q: str) -> str:
    return (q or "").strip()[:MAX_QUESTION_CHARS]


def _history(req: ChatRequest) -> list[dict]:
    return [{"role": t.role, "content": t.content} for t in (req.history or [])]


def _record(question: str, result: dict) -> tuple[int, bool, int, str]:
    """Shared post-processing: real-time clustering + logging. Returns
    (cluster_id, new_cluster, log_id, confidence)."""
    cluster_id, is_new = clustering.assign_realtime(question, result["query_vec"])
    log_id = store.add_query_log(
        question, result["query_vec"], cluster_id, result["answer"],
        score=result["score"], cache_hit=result["cache_hit"],
    )
    confidence = "high" if result["score"] >= settings.RELEVANCE_THRESHOLD else "low"
    return cluster_id, is_new, log_id, confidence


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    question = _clean_question(req.question)
    result = rag.answer(question, history=_history(req))
    cluster_id, is_new, log_id, confidence = _record(question, result)
    return ChatResponse(
        answer=result["answer"], citations=result["citations"], cluster_id=cluster_id,
        new_cluster=is_new, llm_used=result["llm_used"], log_id=log_id,
        confidence=confidence, cache_hit=result["cache_hit"],
    )


@app.post("/chat/stream")
def chat_stream(req: ChatRequest) -> StreamingResponse:
    """Stream the answer token-by-token as SSE, then a final `meta` event with
    cluster/log/citations. Falls back to a full answer if the LLM can't stream."""
    question = _clean_question(req.question)
    history = _history(req)

    def event_gen():
        p = rag.prepare(question, history)
        cache = p["cached"]
        if cache is not None:
            answer_text = cache["answer"]
            citations = cache["citations"]
            llm_used, cache_hit = False, True
            yield _sse("token", {"text": answer_text})
        else:
            citations = p["contexts"]
            stream = llm.chat_stream(
                llm.ANSWER_SYSTEM,
                llm.build_answer_turns(question, citations, history),
                512,
            )
            parts: list[str] = []
            if stream is not None:
                try:
                    for delta in stream:
                        parts.append(delta)
                        yield _sse("token", {"text": delta})
                except Exception:
                    pass
            if parts:
                answer_text, llm_used = "".join(parts), True
            else:
                answer_text = llm.fallback_answer(question, citations)
                llm_used = False
                yield _sse("token", {"text": answer_text})
            cache_hit = False
            rag._maybe_cache(question, p, answer_text, history)

        result = {
            "query_vec": p["query_vec"], "answer": answer_text,
            "score": (citations[0]["score"] if (cache_hit and citations) else p["relevance"]),
            "cache_hit": cache_hit,
        }
        cluster_id, is_new, log_id, confidence = _record(question, result)
        yield _sse("meta", {
            "cluster_id": cluster_id, "new_cluster": is_new, "log_id": log_id,
            "confidence": confidence, "llm_used": llm_used, "cache_hit": cache_hit,
            "citations": citations,
        })

    return StreamingResponse(event_gen(), media_type="text/event-stream")


@app.post("/feedback")
def feedback(req: FeedbackRequest) -> dict:
    store.add_feedback(req.log_id, req.vote)
    return {"ok": True}


@app.get("/faq", response_model=FaqResponse)
def get_faq(top_n: int | None = None) -> FaqResponse:
    return FaqResponse(**faq.build_faq(top_n))


@app.get("/analytics", response_model=AnalyticsResponse)
def get_analytics() -> AnalyticsResponse:
    return AnalyticsResponse(**analytics.overview())


@app.get("/categories", response_model=CategoriesResponse)
def categories() -> CategoriesResponse:
    return CategoriesResponse(categories=analytics.by_category())


@app.get("/admin/content-gaps", response_model=ContentGapsResponse)
def content_gaps() -> ContentGapsResponse:
    return ContentGapsResponse(gaps=analytics.content_gaps())


@app.post("/admin/kb", response_model=IngestResponse)
def add_kb(req: AddKbRequest) -> IngestResponse:
    return IngestResponse(**ingest.add_document(req.filename, req.text))


@app.post("/ingest", response_model=IngestResponse)
def post_ingest() -> IngestResponse:
    return IngestResponse(**ingest.ingest_directory())


@app.post("/admin/recluster", response_model=ReclusterResponse)
def post_recluster() -> ReclusterResponse:
    return ReclusterResponse(**clustering.recluster_batch())


@app.get("/health")
def health() -> dict:
    return {
        "ok": True,
        "llm_enabled": settings.llm_enabled,
        "clusters": store.count_clusters(),
        "queries_logged": store.total_query_count(),
    }


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


# --- static UI --------------------------------------------------------------
app.mount("/static", StaticFiles(directory=settings.WEB_DIR), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(settings.WEB_DIR / "index.html")


@app.get("/faq-page")
def faq_page() -> RedirectResponse:
    return RedirectResponse("/#faq")
