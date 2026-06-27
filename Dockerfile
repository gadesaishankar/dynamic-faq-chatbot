# Dynamic FAQ Chatbot — multi-stage build.
# Stage 1: build React frontend. Stage 2: Python + baked model.

# --- Stage 1: React build ---------------------------------------------------
FROM node:20-slim AS frontend
WORKDIR /build
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ .
RUN npm run build
# Output is in /build/../web → we COPY from /web in the next stage.
# Vite config sets outDir: '../web', so from /build that's /web.

# --- Stage 2: Python app ----------------------------------------------------
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.hfcache

WORKDIR /app

# Install Python deps first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code (excluding frontend source — we only need the built assets)
COPY app/ app/
COPY scripts/ scripts/
COPY data/ data/
COPY conftest.py .

# Copy the React build output from stage 1
COPY --from=frontend /web/ web/

# Bake the embedding model into the image
RUN python -c "from sentence_transformers import SentenceTransformer; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

EXPOSE 8000

CMD ["sh", "-c", "python -m scripts.ingest_kb && uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
