"""Offline evaluation harness — gives you numbers to quote.

For each golden question it measures three things:
  - retrieval recall:   did the expected source make it into the top-k chunks?
  - keyword coverage:    does the answer contain the expected facts?
  - faithfulness (judge): is the answer fully supported by the retrieved context?
                          (LLM-as-judge — Gemini scores YES/NO)

Usage:  python -m scripts.eval
Run `python -m scripts.ingest_kb` first so the KB is populated. With no LLM key,
faithfulness is skipped and the rest still runs.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from app import llm, rag, store
from app.config import settings

GOLDEN = Path(settings.SOURCES_DIR).parent / "eval" / "golden.json"


def _judge(answer: str, contexts: list[dict]) -> bool | None:
    if not settings.llm_enabled:
        return None
    ctx = "\n\n".join(c["text"] for c in contexts)
    system = (
        "You are a strict evaluator. Reply with exactly YES or NO. "
        "YES if the ANSWER is fully supported by the CONTEXT, NO otherwise."
    )
    user = f"CONTEXT:\n{ctx}\n\nANSWER:\n{answer}\n\nIs the answer fully supported?"
    out = llm._chat(system, [{"role": "user", "content": user}], max_tokens=5)
    return out.strip().upper().startswith("YES") if out else None


def main() -> None:
    store.init_db()
    cases = json.loads(GOLDEN.read_text(encoding="utf-8"))
    n = len(cases)
    recall = cover = judged = faithful = 0

    print(f"Evaluating {n} golden questions...\n")
    for case in cases:
        res = rag.answer(case["question"])
        sources = {c["source"] for c in res["citations"]}
        hit = case["expected_source"] in sources
        kw = all(k.lower() in res["answer"].lower() for k in case["must_include"])
        verdict = _judge(res["answer"], res["citations"])
        recall += hit
        cover += kw
        if verdict is not None:
            judged += 1
            faithful += verdict
        flags = f"recall={'Y' if hit else 'N'} keywords={'Y' if kw else 'N'}"
        if verdict is not None:
            flags += f" faithful={'Y' if verdict else 'N'}"
        print(f"  [{flags}] {case['question']}")
        time.sleep(3)  # stay under the free-tier RPM limit

    print("\n=== metrics ===")
    print(f"  retrieval recall : {recall}/{n} = {recall / n:.0%}")
    print(f"  keyword coverage : {cover}/{n} = {cover / n:.0%}")
    if judged:
        print(f"  faithfulness     : {faithful}/{judged} = {faithful / judged:.0%}")
    else:
        print("  faithfulness     : skipped (no LLM key)")


if __name__ == "__main__":
    main()
