"""FastAPI app: chat endpoint, dynamic FAQ endpoint, ingest/admin, static UI.

Wiring:
  POST /chat          -> RAG answer + log query + real-time cluster assignment
  GET  /faq           -> top clusters ranked by recency-weighted frequency
  POST /ingest        -> (re)build the knowledge base from data/sources
  POST /admin/recluster -> run the batch re-clustering pass now
  GET  /              -> chat UI ; GET /faq-page -> dynamic FAQ UI
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from . import clustering, faq, ingest, rag, store
from .config import settings
from .schemas import (
    ChatRequest,
    ChatResponse,
    FaqResponse,
    IngestResponse,
    ReclusterResponse,
)

scheduler = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    store.init_db()
    global scheduler
    try:
        from apscheduler.schedulers.background import BackgroundScheduler

        scheduler = BackgroundScheduler(daemon=True)
        # Nightly batch re-clustering at 03:00 server time.
        scheduler.add_job(clustering.recluster_batch, "cron", hour=3, id="recluster")
        scheduler.start()
    except Exception:
        scheduler = None  # APScheduler optional; manual /admin/recluster still works.
    yield
    if scheduler is not None:
        scheduler.shutdown(wait=False)


app = FastAPI(title="Dynamic FAQ Chatbot", lifespan=lifespan)


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    history = [{"role": t.role, "content": t.content} for t in (req.history or [])]
    result = rag.answer(req.question, history=history)
    # Real-time semantic grouping using the SAME vector we just embedded.
    cluster_id, is_new = clustering.assign_realtime(req.question, result["query_vec"])
    store.add_query_log(req.question, result["query_vec"], cluster_id, result["answer"])
    return ChatResponse(
        answer=result["answer"],
        citations=result["citations"],
        cluster_id=cluster_id,
        new_cluster=is_new,
        llm_used=result["llm_used"],
    )


@app.get("/faq", response_model=FaqResponse)
def get_faq(top_n: int | None = None) -> FaqResponse:
    return FaqResponse(**faq.build_faq(top_n))


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


# --- static UI --------------------------------------------------------------
# Mounted last so the API routes above take precedence.
app.mount("/static", StaticFiles(directory=settings.WEB_DIR), name="static")


@app.get("/")
def index() -> FileResponse:
    # Single-page app: Chat + FAQ are tabs, so navigating between them keeps
    # the conversation in memory; a browser refresh starts a fresh chat.
    return FileResponse(settings.WEB_DIR / "index.html")


@app.get("/faq-page")
def faq_page() -> RedirectResponse:
    # Kept for old bookmarks — open the SPA on its FAQ tab.
    return RedirectResponse("/#faq")
