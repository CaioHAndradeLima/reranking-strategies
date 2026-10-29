import argparse
import json
from pathlib import Path

from .data import load_documents, load_queries, validate_dataset
from .evaluation import evaluate
from .rerankers import CrossEncoderReranker, HeuristicReranker, MMRReranker, OpenAIListwiseReranker
from .retrieval import BM25Retriever, rrf_fuse

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval and reranking strategies")
    parser.add_argument("command", choices=["evaluate"])
    parser.add_argument("--documents", default=str(ROOT / "data/documents.json"))
    parser.add_argument("--queries", default=str(ROOT / "data/golden.json"))
    parser.add_argument("--output", default="")
    parser.add_argument("--candidate-k", type=int, default=10)
    parser.add_argument("--metric-k", type=int, default=5)
    parser.add_argument("--dense", action="store_true", help="Index and compare dense/hybrid Qdrant retrieval")
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    parser.add_argument("--llm-reranker", action="store_true", help="Use optional OpenAI listwise reranking")
    parser.add_argument("--cross-encoder", action="store_true", help="Use optional local cross-encoder reranking")
    parser.add_argument("--cross-encoder-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    args = parser.parse_args()

    documents = load_documents(args.documents)
    queries = load_queries(args.queries)
    validate_dataset(documents, queries)
    bm25 = BM25Retriever(documents)
    rerankers = [HeuristicReranker(), MMRReranker()]
    if args.llm_reranker:
        rerankers.append(OpenAIListwiseReranker())
    if args.cross_encoder:
        rerankers.append(CrossEncoderReranker(args.cross_encoder_model))
    reports = {"bm25": evaluate(queries, bm25.search, rerankers, args.candidate_k, args.metric_k)}
    if args.dense:
        from .qdrant_store import QdrantDenseStore

        store = QdrantDenseStore(url=args.qdrant_url)
        store.index(documents)
        by_id = {doc.id: doc for doc in documents}
        dense = lambda query, limit: store.search(query, by_id, limit)
        hybrid = lambda query, limit: rrf_fuse(
            [bm25.search(query, limit), dense(query, limit)], limit=limit
        )
        reports["dense"] = evaluate(queries, dense, rerankers, args.candidate_k, args.metric_k)
        reports["hybrid_rrf"] = evaluate(queries, hybrid, rerankers, args.candidate_k, args.metric_k)
    report = {"retrieval_modes": reports}
    serialized = json.dumps(report, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized + "\n")
        print(json.dumps({"report": str(output), "summary": {
            name: data["summary"] for name, data in report["retrieval_modes"].items()
        }}, indent=2))
    else:
        print(serialized)


if __name__ == "__main__":
    main()
