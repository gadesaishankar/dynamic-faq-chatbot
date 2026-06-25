"""Shared test fixtures.

Tests operate directly on the store/clustering math with hand-made unit vectors,
so they need neither the embedding model (torch) nor an API key — they're fast
and fully offline.
"""
import numpy as np
import pytest

from app import store
from app.config import settings


@pytest.fixture()
def temp_db(tmp_path, monkeypatch):
    db_file = tmp_path / "test.db"
    monkeypatch.setattr(settings, "DB_PATH", str(db_file))
    # Tests must be fully offline — disable the LLM regardless of any real .env
    # (so canonical generation uses the deterministic fallback, no network).
    monkeypatch.setattr(settings, "LLM_PROVIDER", "none")
    # Drop any cached thread-local connection so we bind to the temp DB.
    if hasattr(store._local, "conn"):
        del store._local.conn
    store.init_db()
    yield store
    if hasattr(store._local, "conn"):
        store._local.conn.close()
        del store._local.conn


def unit(values) -> np.ndarray:
    v = np.asarray(values, dtype=np.float32)
    return v / np.linalg.norm(v)
