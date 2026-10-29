import argparse
import json
from pathlib import Path

from .answer_judging import OpenAIAnswerJudge, evaluate_answers
from .data import load_documents, load_edges, load_queries, validate_dataset
from .evaluation import evaluate, recommend_best_strategy
from .judge_calibration import calibrate_judge
from .pricing import (
    CallBudget,
    OPENAI_STANDARD_RATES_PER_MILLION,
    PRICING_AS_OF,
    PRICING_SOURCE,
)
from .rerankers import (
    CrossEncoderReranker,
    EdgeStructureReranker,
    HeuristicReranker,
    MMRReranker,
    OpenAIPairwiseJudge,
    OpenAIListwiseReranker,
    PairwiseJudgeReranker,
    RetryingReranker,
)
from .retrieval import BM25Retriever, rrf_fuse

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval and reranking strategies")
    parser.add_argument(
        "command", choices=[
            "evaluate", "benchmark", "benchmark-edges", "judge-answers", "calibrate-judge",
        ]
    )
    parser.add_argument("--documents", default=str(ROOT / "data/documents.json"))
    parser.add_argument("--queries", default=str(ROOT / "data/golden.json"))
    parser.add_argument("--edges", default=str(ROOT / "data/edges.json"))
    parser.add_argument("--edge-queries", default=str(ROOT / "data/edge_golden.json"))
    parser.add_argument("--output", default="")
    parser.add_argument("--answers", default="", help="JSON answer samples for judge-answers")
    parser.add_argument("--ratings", default="", help="Paired judge and reviewer scores for calibrate-judge")
    parser.add_argument("--judge-model", default="gpt-6-luna", help="Model used by judge-answers")
    parser.add_argument("--candidate-k", type=int, default=10)
    parser.add_argument("--metric-k", type=int, default=5)
    parser.add_argument("--dense", action="store_true", help="Index and compare dense/hybrid Qdrant retrieval")
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    parser.add_argument("--llm-reranker", action="store_true", help="Use optional OpenAI listwise reranking")
    parser.add_argument("--llm-model", default="gpt-6-luna")
    parser.add_argument("--llm-pairwise", action="store_true", help="Use optional OpenAI pairwise judging")
    parser.add_argument("--llm-loop", action="store_true", help="Retry low-confidence listwise rankings once")
    parser.add_argument("--llm-fallback-model", default="gpt-6.1-terra")
    parser.add_argument("--retry-confidence-threshold", type=float, default=0.6)
    parser.add_argument("--live-llm", action="store_true", help="Add model comparisons to benchmark")
    parser.add_argument("--benchmark-models", default="gpt-6-luna,gpt-6.1-terra")
    parser.add_argument("--with-pairwise", action="store_true", help="Include pairwise methods in benchmark")
    parser.add_argument("--with-loop", action="store_true", help="Include one-retry pipelines in benchmark")
    parser.add_argument("--max-llm-calls", type=int, default=200, help="Hard request cap for one run")
    parser.add_argument("--pairwise-candidates", type=int, default=5,
                        help="Pairwise judge only compares the top candidate shortlist")
    parser.add_argument("--selection-metric", choices=["recall", "mrr", "ndcg"], default="ndcg")
    parser.add_argument("--cross-encoder", action="store_true", help="Use optional local cross-encoder reranking")
    parser.add_argument("--cross-encoder-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    args = parser.parse_args()

    if args.command == "calibrate-judge":
        if not args.ratings:
            parser.error("calibrate-judge requires --ratings")
        try:
            rating_payload = json.loads(Path(args.ratings).read_text())
        except json.JSONDecodeError as exc:
            parser.error(f"{args.ratings} contains invalid JSON: {exc.msg}")
        try:
            report = calibrate_judge(rating_payload)
        except ValueError as exc:
            parser.error(str(exc))
        serialized = json.dumps(report, indent=2)
        if args.output:
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(serialized + "\n")
            print(json.dumps({"report": str(output), "label_source": report["label_source"]}, indent=2))
        else:
            print(serialized)
        return

    evidence_documents = load_documents(args.documents)
    edge_mode = args.command == "benchmark-edges"
    if edge_mode:
        edges = load_edges(args.edges, evidence_documents)
        documents = [edge.as_document() for edge in edges]
        queries = load_queries(args.edge_queries)
    else:
        documents = evidence_documents
        queries = load_queries(args.queries)
    validate_dataset(documents, queries)
    if args.command == "judge-answers":
        if not args.answers:
            parser.error("judge-answers requires --answers")
        try:
            answer_samples = json.loads(Path(args.answers).read_text())
        except json.JSONDecodeError as exc:
            parser.error(f"{args.answers} contains invalid JSON: {exc.msg}")
        if not isinstance(answer_samples, list) or not answer_samples:
            parser.error("--answers must contain a non-empty JSON array")
        if len(answer_samples) > args.max_llm_calls:
            parser.error(
                f"{len(answer_samples)} answer judgments exceed --max-llm-calls "
                f"({args.max_llm_calls}); increase the cap or reduce samples"
            )
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install with: pip install -e '.[llm-judge]'") from exc
        judge = OpenAIAnswerJudge(
            args.judge_model, OpenAI(), CallBudget(args.max_llm_calls)
        )
        report = {
            "experiment": {
                "judge_model": args.judge_model,
                "max_llm_calls": args.max_llm_calls,
                "pricing_as_of": PRICING_AS_OF,
                "pricing_source": PRICING_SOURCE,
                "pricing_rates_usd_per_million_tokens": OPENAI_STANDARD_RATES_PER_MILLION,
            },
            **evaluate_answers(answer_samples, queries, documents, judge),
        }
        serialized = json.dumps(report, indent=2)
        if args.output:
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(serialized + "\n")
            print(json.dumps({"report": str(output), "summary": report["summary"]}, indent=2))
        else:
            print(serialized)
        return
    if args.pairwise_candidates <= 0:
        parser.error("--pairwise-candidates must be positive")
    bm25 = BM25Retriever(documents)
    rerankers = [HeuristicReranker(), MMRReranker()]
    if edge_mode:
        rerankers.append(EdgeStructureReranker())
    llm_enabled = args.llm_reranker or args.llm_pairwise or args.llm_loop or args.live_llm
    estimated_llm_calls = 0
    benchmark_mode = args.command in {"benchmark", "benchmark-edges"}
    if not benchmark_mode and (args.live_llm or args.with_pairwise or args.with_loop):
        parser.error("--live-llm, --with-pairwise, and --with-loop are benchmark-only options")
    if benchmark_mode:
        if (args.with_pairwise or args.with_loop) and not args.live_llm:
            parser.error("--with-pairwise and --with-loop require --live-llm in benchmark mode")
        if args.live_llm:
            benchmark_models = [model.strip() for model in args.benchmark_models.split(",") if model.strip()]
            if not benchmark_models or len(benchmark_models) != len(set(benchmark_models)):
                parser.error("--benchmark-models must contain unique, comma-separated model IDs")
            if args.with_loop and len(benchmark_models) < 2:
                parser.error("--with-loop requires at least two models in --benchmark-models")
            per_pairwise_model = min(args.pairwise_candidates, args.candidate_k)
            estimated_calls_per_query = len(benchmark_models)
            if args.with_pairwise:
                estimated_calls_per_query += len(benchmark_models) * (
                    per_pairwise_model * (per_pairwise_model - 1) // 2
                )
            if args.with_loop:
                estimated_calls_per_query += 2  # primary plus one possible fallback
            estimated_calls = estimated_calls_per_query * len(queries) * (3 if args.dense else 1)
            estimated_llm_calls = estimated_calls
            if estimated_calls > args.max_llm_calls:
                parser.error(
                    f"estimated worst-case {estimated_calls} LLM calls exceeds --max-llm-calls "
                    f"({args.max_llm_calls}); reduce pairwise candidates/models or raise the cap"
                )
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise RuntimeError("Install with: pip install -e '.[llm-judge]'") from exc
            client = OpenAI()
            budget = CallBudget(args.max_llm_calls)
            rerankers.extend(OpenAIListwiseReranker(model, client, budget) for model in benchmark_models)
            if args.with_pairwise:
                for model in benchmark_models:
                    judge = OpenAIPairwiseJudge(model, client, budget)
                    rerankers.append(PairwiseJudgeReranker(
                        judge, candidate_limit=args.pairwise_candidates
                    ))
            if args.with_loop:
                primary = OpenAIListwiseReranker(benchmark_models[0], client, budget)
                fallback = OpenAIListwiseReranker(benchmark_models[-1], client, budget)
                rerankers.append(RetryingReranker(
                    primary, fallback, confidence_threshold=args.retry_confidence_threshold
                ))
        elif args.llm_reranker or args.llm_pairwise or args.llm_loop:
            parser.error("Use --live-llm with the benchmark command to compare model-based methods")
    elif llm_enabled:
        pairwise_size = min(args.pairwise_candidates, args.candidate_k)
        estimated_calls_per_query = (1 if args.llm_reranker else 0)
        if args.llm_pairwise:
            estimated_calls_per_query += pairwise_size * (pairwise_size - 1) // 2
        if args.llm_loop:
            estimated_calls_per_query += 2
        estimated_calls = estimated_calls_per_query * len(queries) * (3 if args.dense else 1)
        estimated_llm_calls = estimated_calls
        if estimated_calls > args.max_llm_calls:
            parser.error(
                f"estimated worst-case {estimated_calls} LLM calls exceeds --max-llm-calls "
                f"({args.max_llm_calls}); reduce pairwise candidates or raise the cap"
            )
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install with: pip install -e '.[llm-judge]'") from exc
        client = OpenAI()
        budget = CallBudget(args.max_llm_calls)
        primary = OpenAIListwiseReranker(args.llm_model, client, budget)
        if args.llm_reranker:
            rerankers.append(primary)
        if args.llm_pairwise:
            judge = OpenAIPairwiseJudge(args.llm_model, client, budget)
            rerankers.append(PairwiseJudgeReranker(
                judge, candidate_limit=args.pairwise_candidates
            ))
        if args.llm_loop:
            fallback = OpenAIListwiseReranker(args.llm_fallback_model, client, budget)
            rerankers.append(RetryingReranker(
                primary, fallback, confidence_threshold=args.retry_confidence_threshold
            ))
    if args.cross_encoder:
        rerankers.append(CrossEncoderReranker(args.cross_encoder_model))
    reports = {"bm25": evaluate(queries, bm25.search, rerankers, args.candidate_k, args.metric_k)}
    if args.dense:
        from .qdrant_store import QdrantDenseStore

        store = QdrantDenseStore(
            url=args.qdrant_url,
            collection="reranking_edge_demo" if edge_mode else "reranking_demo",
        )
        if edge_mode:
            store.index(
                documents,
                indexed_payload_fields=("kind", "source_id", "relation", "target_id"),
            )
        else:
            store.index(documents)
        by_id = {doc.id: doc for doc in documents}
        dense = lambda query, limit: store.search(query, by_id, limit)
        hybrid = lambda query, limit: rrf_fuse(
            [bm25.search(query, limit), dense(query, limit)], limit=limit
        )
        reports["dense"] = evaluate(queries, dense, rerankers, args.candidate_k, args.metric_k)
        reports["hybrid_rrf"] = evaluate(queries, hybrid, rerankers, args.candidate_k, args.metric_k)
    report = {"retrieval_modes": reports}
    report["recommendations"] = {
        name: recommend_best_strategy(result["summary"], args.selection_metric)
        for name, result in reports.items()
    }
    if benchmark_mode:
        report["experiment"] = {
            "corpus_type": "graph_edges" if edge_mode else "passages",
            "edge_count": len(documents) if edge_mode else None,
            "evidence_document_count": len(evidence_documents) if edge_mode else None,
            "live_llm": args.live_llm,
            "models": benchmark_models if args.live_llm else [],
            "with_pairwise": args.with_pairwise,
            "with_retry_loop": args.with_loop,
            "retry_confidence_threshold": args.retry_confidence_threshold if args.with_loop else None,
            "pairwise_candidate_limit": args.pairwise_candidates if args.with_pairwise else None,
            "candidate_k": args.candidate_k,
            "metric_k": args.metric_k,
            "selection_metric": args.selection_metric,
            "max_llm_calls": args.max_llm_calls if args.live_llm else 0,
            "estimated_worst_case_llm_calls": estimated_llm_calls,
            "pricing_as_of": PRICING_AS_OF,
            "pricing_source": PRICING_SOURCE,
            "pricing_rates_usd_per_million_tokens": OPENAI_STANDARD_RATES_PER_MILLION,
        }
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
