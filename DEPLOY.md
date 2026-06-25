# Deployment plan

This app has two deployment-shaping characteristics — plan around them:

1. **It loads a local embedding model (torch + all-MiniLM-L6-v2).** Budget
   **~1–1.5 GB RAM**. The common "free 512 MB" tiers will OOM. Pick a host with
   ≥ 1 GB, or one whose free tier is generous on RAM.
2. **State lives in SQLite (`data/faq.db`).** Query logs + clusters are what make
   the FAQ "self-updating". On hosts with **ephemeral disk**, that file resets on
   every redeploy/restart — so use a **persistent volume**, or migrate the store
   to a managed DB (the functions in `app/store.py` are the single seam for that).

The repo ships a [`Dockerfile`](Dockerfile) that bakes the model into the image
(no runtime download), rebuilds the knowledge base on boot, and serves on
`$PORT`. It runs as-is on any container host.

---

## Recommended paths

### A. Hugging Face Spaces (Docker) — recommended free path
Best free fit: Spaces give **~16 GB RAM** on the free CPU tier, so torch is no
problem. The Space config already lives in the `README.md` metadata block
(`sdk: docker`, `app_port: 8000`) and the `Dockerfile` is auto-detected.

1. Create a free account at huggingface.co, then **New → Space**:
   - Owner: you · Space name: e.g. `dynamic-faq-chatbot`
   - SDK: **Docker** → **Blank** · Hardware: **CPU basic (free)**
2. Push this repo to the Space's git remote (it gives you the URL):
   ```bash
   git init && git add -A && git commit -m "Dynamic FAQ chatbot"
   git remote add space https://huggingface.co/spaces/<you>/dynamic-faq-chatbot
   git push space main
   ```
   (When prompted, username = your HF username, password = a **write access
   token** from huggingface.co/settings/tokens.)
3. In the Space → **Settings → Variables and secrets**:
   - **Variable**: `LLM_PROVIDER = gemini`
   - **Secret**: `GEMINI_API_KEY = <your key>`
4. The Space builds the Docker image (a few minutes — it bakes the embedding
   model in) and starts automatically. Open the Space URL.

> Caveats:
> - First build is slow (installs torch, downloads the model) — subsequent boots
>   are fast.
> - Free Space storage is **ephemeral** — the FAQ history resets when the Space
>   restarts/rebuilds. Fine for a demo. To persist it: enable HF **persistent
>   storage** (paid) and set `DB_PATH=/data/faq.db`, **or** snapshot it for free
>   with `python -m scripts.backup_db` (set `HF_TOKEN` + `HF_BACKUP_DATASET`) and
>   restore `faq.db` on boot.
> - Never commit `.env` (it's already git-ignored) — set the key via Secrets.

#### Auto-deploy on every push (GitHub Actions)
The repo ships [`.github/workflows/sync-to-hf.yml`](.github/workflows/sync-to-hf.yml),
which mirrors GitHub → your Space on every push to `main`. To enable it, host the
repo on GitHub and add (under **Settings → Secrets and variables → Actions**):
- Secret: `HF_TOKEN` = your HF **write** token
- Variables: `HF_USERNAME` = your HF username, `HF_SPACE` = the Space name

After that you only ever `git push origin main`; the Space rebuilds itself. (You
still need to do the very first push to create the Space, or create it in the UI.)

### B. Render (Docker Web Service) — durable, low cost
1. New → **Web Service** → connect the repo → environment **Docker**.
2. Instance type: **≥ 1 GB RAM** (free 512 MB is too small for torch).
3. Add a **Persistent Disk** mounted at `/app/data` (so `faq.db` survives
   restarts).
4. **Environment** → add `LLM_PROVIDER=gemini`, `GEMINI_API_KEY=…` (as secrets).
5. Render sets `$PORT` automatically; the Dockerfile already honors it.

### C. Railway / Fly.io
Same shape: deploy the Docker image, attach a **volume** at `/app/data`, set the
env vars, pick an instance with ≥ 1 GB RAM.

### D. Google Cloud Run — scales to zero, generous free tier
Works, with two caveats: filesystem is **ephemeral** (SQLite won't persist —
use a managed DB or GCS, or accept resets), and the in-process nightly job won't
run while scaled to zero (see Scheduling below). Allocate ≥ 1 GB memory; expect a
cold-start delay while the model loads.

---

## Configuration (all hosts)

Set these as platform **environment variables / secrets** — never commit `.env`:

| Var | Value |
|---|---|
| `LLM_PROVIDER` | `gemini` (or `openrouter` / `anthropic` / `none`) |
| `GEMINI_API_KEY` | your key (rotate the one pasted in chat) |
| `GEMINI_MODEL` | `gemini-2.5-flash` (default) |
| `FAQ_MIN_ASKS` | `3` (min asks before a question shows on the FAQ) |
| `SIM_THRESHOLD` | `0.45` (semantic grouping strictness) |

Even with no key set, the app runs (extractive answers) — so a missing key
degrades gracefully rather than failing the deploy.

## Scheduling the nightly re-cluster

The batch re-cluster runs in-process via APScheduler at 03:00 — fine on
always-on hosts (A, B, C). On scale-to-zero hosts (Cloud Run), instead trigger
it externally with a scheduled `POST /admin/recluster` (Cloud Scheduler, GitHub
Actions cron, etc.) or run `python -m scripts.recluster` as a scheduled job.

## Pre-deploy checklist
- [ ] `docker build -t faqbot .` succeeds locally
- [ ] `docker run -p 8000:8000 -e LLM_PROVIDER=gemini -e GEMINI_API_KEY=… faqbot` → `/health` returns `llm_enabled: true`
- [ ] Persistent volume mounted at `/app/data` (if you want durable FAQ history)
- [ ] Secrets set in the platform dashboard; `.env` is **not** in the image (it's in `.dockerignore`)
- [ ] Instance has ≥ 1 GB RAM
