"""Prove the semantic frequency engine end to end.

Fires many PARAPHRASES of the same intents at the running API, then prints the
dynamic FAQ. The whole point: ~10 different ways of asking "how do I register"
should collapse into ONE cluster with count ~10 — not 10 clusters of 1.

Start the server first:  uvicorn app.main:app --reload
Then run:                python -m scripts.seed_demo
"""
from __future__ import annotations

import json
import urllib.request

BASE = "http://127.0.0.1:8000"

# Each inner list = the SAME question intent, phrased many different ways.
INTENTS = {
    "course registration": [
        "How do I register for courses?",
        "What's the enrollment process?",
        "how to sign up for classes",
        "Where can I enroll in subjects this semester?",
        "steps to register for classes",
        "I want to add courses, how?",
        "How do students sign up for courses?",
        "course registration procedure please",
    ],
    "library hours": [
        "What are the library hours?",
        "When does the library open?",
        "library timings?",
        "Till what time is the library open?",
        "is the library open on weekends",
    ],
    "fee payment": [
        "How do I pay my tuition fees?",
        "What are the ways to pay fees?",
        "fee payment methods",
        "Can I pay fees online?",
    ],
}


def post(path: str, payload: dict) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        BASE + path, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def get(path: str) -> dict:
    with urllib.request.urlopen(BASE + path) as resp:
        return json.loads(resp.read())


def main() -> None:
    print("Sending paraphrased questions...\n")
    cluster_of: dict[str, set[int]] = {}
    for intent, questions in INTENTS.items():
        ids = set()
        for q in questions:
            r = post("/chat", {"question": q})
            ids.add(r["cluster_id"])
        cluster_of[intent] = ids
        print(f"  {intent!r}: {len(questions)} phrasings -> cluster ids {sorted(ids)}")

    print("\nThe key result: each intent should map to (mostly) ONE cluster id.\n")

    print("Optional: running batch re-clustering to clean up...")
    post("/admin/recluster", {})

    print("\nTop dynamic FAQ (GET /faq):\n")
    faq = get("/faq")
    for i, item in enumerate(faq["items"], 1):
        print(f"{i}. [{item['ask_count']} asks] {item['question']}")
        if item["answer"]:
            print(f"     {item['answer'][:120]}")
        print(f"     examples: {item['examples']}")
    print(f"\n(generated from {faq['generated_from_queries']} logged questions)")


if __name__ == "__main__":
    main()
