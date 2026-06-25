# Dynamic FAQ Chatbot — container image.
# Python 3.12 (stable wheels for torch/sentence-transformers on Linux).
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.hfcache

WORKDIR /app

# Install deps first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY . .

# Bake the embedding model into the image so the first request isn't slow and
# the container needs no model download at runtime.
RUN python -c "from sentence_transformers import SentenceTransformer; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

EXPOSE 8000

# Rebuild the KB from data/sources on boot (keeps it in sync; preserves the
# separate query_logs/clusters tables), then serve. $PORT is set by most PaaS.
CMD ["sh", "-c", "python -m scripts.ingest_kb && uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
