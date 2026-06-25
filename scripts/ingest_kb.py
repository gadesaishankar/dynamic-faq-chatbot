"""Build the knowledge base from data/sources/.

Usage:  python -m scripts.ingest_kb
"""
from app import store
from app.ingest import ingest_directory


def main() -> None:
    store.init_db()
    result = ingest_directory()
    print(f"Indexed {result['chunks_indexed']} chunks from {len(result['sources'])} file(s):")
    for s in result["sources"]:
        print(f"  - {s}")


if __name__ == "__main__":
    main()
