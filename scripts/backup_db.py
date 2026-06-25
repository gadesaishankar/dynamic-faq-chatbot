"""Optional persistence: back up the SQLite DB to a Hugging Face dataset.

Free Spaces have ephemeral storage, so the FAQ history resets on rebuild. Run
this (e.g. on a schedule, or from the nightly job) to snapshot it. To restore,
download faq.db from the dataset into data/ before startup.

Set HF_TOKEN (write) and HF_BACKUP_DATASET (e.g. "you/faqbot-data"). No-op if
either is unset, so it's safe to wire in unconditionally.

Usage:  python -m scripts.backup_db
"""
from __future__ import annotations

import os
from pathlib import Path

from app.config import settings


def main() -> None:
    token = os.getenv("HF_TOKEN")
    repo = os.getenv("HF_BACKUP_DATASET")
    if not token or not repo:
        print("Backups disabled — set HF_TOKEN and HF_BACKUP_DATASET to enable.")
        return
    if not Path(settings.DB_PATH).exists():
        print(f"No DB at {settings.DB_PATH} yet — nothing to back up.")
        return

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo, repo_type="dataset", exist_ok=True)
    api.upload_file(
        path_or_fileobj=settings.DB_PATH,
        path_in_repo="faq.db",
        repo_id=repo,
        repo_type="dataset",
    )
    print(f"Backed up {settings.DB_PATH} -> {repo}/faq.db")


if __name__ == "__main__":
    main()
