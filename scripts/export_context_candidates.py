#!/usr/bin/env python3
"""Export ranked candidate lists for the contexting-strategies rerank bridge."""

import argparse
import json
from pathlib import Path

from reranking_strategies.data import load_documents, load_queries, validate_dataset
from reranking_strategies.evaluation import evaluate
from reranking_strategies.rerankers import HeuristicReranker, MMRReranker
from reranking_strategies.retrieval import BM25Retriever

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Export rerank rankings for contexting-strategies")
    parser.add_argument("--output", default="context_candidates.json")
    parser.add_argument("--documents", default=str(ROOT / "data/documents.json"))
    parser.add_argument("--queries", default=str(ROOT / "data/golden.json"))
    parser.add_argument("--candidate-k", type=int, default=10)
    parser.add_argument("--metric-k", type=int, default=5)
    args = parser.parse_args()

    documents = load_documents(Path(args.documents))
    queries = load_queries(Path(args.queries))
    validate_dataset(documents, queries)
    bm25 = BM25Retriever(documents)
    report = evaluate(
        queries,
        lambda query, limit: bm25.search(query, limit),
        rerankers=[HeuristicReranker(), MMRReranker()],
        candidate_k=args.candidate_k,
        metric_k=args.metric_k,
    )
    payload = {
        "source": "reranking-strategies export_context_candidates.py",
        "results": report["results"],
    }
    output = Path(args.output)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"output": str(output.resolve()), "query_count": len(report["results"])}, indent=2))


if __name__ == "__main__":
    main()
