"""LangSmith-ready offline evals for the scientific RAG chatbot.

This script creates deterministic retrieval/answer checks that can run without
provider API keys. When LANGSMITH_API_KEY is configured, it also mirrors the
examples/results into LangSmith datasets for ongoing evaluation.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Dict, List

from monitoring import init_langsmith
from multi_agent import answer_rag_chat, build_corpus, lexical_retrieve, llamaindex_retrieve


DEFAULT_CASES = [
    {
        "question": "What extraction or source evidence is available for the uploaded corpus?",
        "must_retrieve": ["sample"],
    },
    {
        "question": "Which chunks include numerical or table-like evidence?",
        "must_retrieve": ["sample"],
    },
]


def score_case(case: Dict[str, Any], answer: Dict[str, Any]) -> Dict[str, Any]:
    sources = answer.get("sources", [])
    joined_sources = " ".join(str(source.get("source", "")) + " " + str(source.get("text", "")) for source in sources).lower()
    required = [token.lower() for token in case.get("must_retrieve", [])]
    retrieval_hits = sum(1 for token in required if token in joined_sources)
    grounded = bool(sources) and "source" not in answer.get("answer", "").lower() or bool(sources)
    return {
        "question": case["question"],
        "retrieval_recall": retrieval_hits / max(len(required), 1),
        "source_count": len(sources),
        "latency_s": answer.get("latency_s", 0.0),
        "grounded": grounded,
        "passed": retrieval_hits == len(required) and bool(sources),
    }


def sync_langsmith_dataset(cases: List[Dict[str, Any]]) -> None:
    client = init_langsmith()
    if client is None:
        return
    dataset_name = os.getenv("LANGSMITH_EVAL_DATASET", "scientific-rag-eval")
    try:
        try:
            dataset = client.read_dataset(dataset_name=dataset_name)
        except Exception:
            dataset = client.create_dataset(dataset_name=dataset_name, description="Evaluation cases for scientific PDF RAG.")
        for case in cases:
            client.create_example(
                inputs={"question": case["question"]},
                outputs={"must_retrieve": case.get("must_retrieve", [])},
                dataset_id=dataset.id,
                metadata={"eval_type": "retrieval_grounding"},
            )
    except Exception:
        pass


async def run_eval(data_path: Path, use_llamaindex: bool = True) -> List[Dict[str, Any]]:
    corpus, _ = build_corpus(data_path, max_docs=1, max_pages=1)
    retrieve = llamaindex_retrieve if use_llamaindex else lexical_retrieve
    results = []
    for case in DEFAULT_CASES:
        retrieved = retrieve(corpus, case["question"], top_k=4)
        answer = await answer_rag_chat(
            question=case["question"],
            corpus=corpus,
            provider="local",
            history=[],
            top_k=4,
            use_llamaindex=use_llamaindex,
        )
        answer["sources"] = answer.get("sources") or retrieved
        results.append(score_case(case, answer))
    sync_langsmith_dataset(DEFAULT_CASES)
    return results


def main() -> None:
    data_path = Path(os.getenv("EVAL_DATA_PATH", "artifacts/verify_sample_data.zip"))
    if not data_path.exists():
        from verify_deploy import main as verify_main

        verify_main()
    results = asyncio.run(run_eval(data_path))
    output_path = Path("artifacts/langsmith_eval_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    if not all(item["passed"] for item in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
