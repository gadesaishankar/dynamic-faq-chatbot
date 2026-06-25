"""Run the batch re-clustering pass standalone (e.g. from cron / Task Scheduler).

Usage:  python -m scripts.recluster
"""
from app import store
from app.clustering import recluster_batch


def main() -> None:
    store.init_db()
    result = recluster_batch()
    print("Re-clustering complete:")
    print(f"  clusters: {result['clusters_before']} -> {result['clusters_after']}")
    print(f"  queries processed: {result['queries_processed']}")
    print(f"  canonical Q/A generated: {result['canonical_generated']}")


if __name__ == "__main__":
    main()
