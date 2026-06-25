"""Load department FAQ source documents into the knowledge base.

Reads .md / .txt files from the sources directory, splits them into chunks,
embeds them locally, and stores them for RAG retrieval.
"""
from __future__ import annotations

from pathlib import Path

from . import embeddings, store
from .config import settings


def _chunk(text: str, max_chars: int) -> list[str]:
    """Split on blank lines, then pack paragraphs up to ~max_chars per chunk."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    buf = ""
    for p in paragraphs:
        if not buf:
            buf = p
        elif len(buf) + len(p) + 2 <= max_chars:
            buf += "\n\n" + p
        else:
            chunks.append(buf)
            buf = p
    if buf:
        chunks.append(buf)
    return chunks


def ingest_directory(path: str | None = None, *, reset: bool = True) -> dict:
    src_dir = Path(path or settings.SOURCES_DIR)
    files = sorted(
        [p for p in src_dir.glob("**/*") if p.suffix.lower() in (".md", ".txt")]
    )

    items: list[tuple[str, str]] = []  # (chunk_text, source_name)
    for f in files:
        text = f.read_text(encoding="utf-8", errors="ignore")
        for chunk in _chunk(text, settings.CHUNK_CHARS):
            items.append((chunk, f.name))

    if reset:
        store.clear_kb()

    if items:
        vecs = embeddings.embed([t for t, _ in items])
        store.add_chunks(items, vecs)

    return {
        "chunks_indexed": len(items),
        "sources": [f.name for f in files],
    }


def add_document(filename: str, text: str) -> dict:
    """Write a new KB source file and re-ingest. Used by the admin UI to fill
    content gaps without redeploying."""
    src_dir = Path(settings.SOURCES_DIR)
    src_dir.mkdir(parents=True, exist_ok=True)
    # basename only — never let a caller write outside data/sources.
    name = Path(filename).name or "untitled.md"
    if not name.lower().endswith((".md", ".txt")):
        name += ".md"
    (src_dir / name).write_text(text, encoding="utf-8")
    return ingest_directory()
