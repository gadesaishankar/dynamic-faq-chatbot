"""Shared test fixtures.

Tests operate directly on the store/clustering math with hand-made unit vectors,
so they need neither the embedding model (torch) nor an API key — they're fast
and fully offline.
"""
import numpy as np
import pytest

from app import store
from app.config import settings
from app.stores import sqlite_backend


@pytest.fixture()
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "STORE_BACKEND", "sqlite")
    monkeypatch.setattr(settings, "DB_PATH", str(tmp_path / "test.db"))
    # Tests must be fully offline — disable the LLM regardless of any real .env
    # (so canonical generation uses the deterministic fallback, no network).
    monkeypatch.setattr(settings, "LLM_PROVIDER", "none")
    store._reset()
    if hasattr(sqlite_backend._local, "conn"):
        del sqlite_backend._local.conn
    store.init_db()
    yield store
    if hasattr(sqlite_backend._local, "conn"):
        sqlite_backend._local.conn.close()
        del sqlite_backend._local.conn
    store._reset()


def unit(values) -> np.ndarray:
    v = np.asarray(values, dtype=np.float32)
    return v / np.linalg.norm(v)
