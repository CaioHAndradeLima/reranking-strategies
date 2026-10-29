"""Fine-tune and evaluate a cross-encoder on the project's tiny labeled corpus.

This is a workflow demonstration, not a meaningful estimate of generalization.
Every query is held out once (leave-one-query-out cross-validation), and each
fold trains only on other query IDs. Hyperparameters are fixed in advance.
"""

import argparse
from importlib.metadata import version
import json
import platform
from statistics import median
import tempfile
from pathlib import Path
from statistics import mean
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_MODEL_REVISION = "233902d25c440f23af6f7d6e94d2946bac0bee0a"
LATENCY_REPEATS = 5


def percentile(values: list[float], fraction: float) -> float:
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * fraction
    low = int(position)
    high = min(low + 1, len(values) - 1)
    portion = position - low
    return values[low] * (1 - portion) + values[high] * portion


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--candidate-k", type=int, default=10)
    parser.add_argument("--metric-k", type=int, default=5)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--model-output", default="models/demo-cross-encoder")
    parser.add_argument("--output", default="reports/cross-encoder-finetuning-demo.json")
    args = parser.parse_args()
    if args.candidate_k <= 0 or args.metric_k <= 0 or args.epochs <= 0:
        parser.error("candidate-k, metric-k, and epochs must be positive")

    try:
        import torch
        from datasets import Dataset
        from sentence_transformers.cross_encoder import (
            CrossEncoder,
            CrossEncoderTrainer,
            CrossEncoderTrainingArguments,
        )
        from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss
    except ImportError as exc:
        raise SystemExit(
            "Install training dependencies with: pip install -e '.[cross-encoder-training]'"
        ) from exc

    from reranking_strategies.data import load_documents, load_queries
    from reranking_strategies.metrics import score_ranking
    from reranking_strategies.models import Hit
    from reranking_strategies.retrieval import BM25Retriever

    torch.manual_seed(42)
    documents = load_documents(ROOT / "data/documents.json")
    queries = load_queries(ROOT / "data/golden.json")
    retriever = BM25Retriever(documents)
    candidate_sets = {
        query.id: retriever.search(query.query, args.candidate_k)
        for query in queries
    }

    def model_metrics(model) -> dict:
        per_query = []
        latencies = []
        # Warm up tokenizer/model work so the first call does not dominate timing.
        first_query = queries[0]
        first_pair = (first_query.query,
                      candidate_sets[first_query.id][0].document.searchable_text)
        model.predict([first_pair], show_progress_bar=False)
        for query in queries:
            candidates = candidate_sets[query.id]
            pairs = [(query.query, hit.document.searchable_text) for hit in candidates]
            repeated_latencies = []
            scores = None
            for _ in range(LATENCY_REPEATS):
                started = perf_counter()
                scores = model.predict(pairs, show_progress_bar=False)
                repeated_latencies.append((perf_counter() - started) * 1000)
            latencies.append(median(repeated_latencies))
            ranked = sorted(
                (Hit(hit.document, float(score), "cross_encoder")
                 for hit, score in zip(candidates, scores)),
                key=lambda hit: (-hit.score, hit.document.id),
            )[:args.metric_k]
            per_query.append(score_ranking(ranked, query.relevance, args.metric_k))
        return {
            "query_count": len(per_query),
            "recall": mean(row["recall"] for row in per_query),
            "mrr": mean(row["mrr"] for row in per_query),
            "ndcg": mean(row["ndcg"] for row in per_query),
            "latency_ms_mean": mean(latencies),
            "latency_ms_p95": percentile(latencies, 0.95),
            "per_query": per_query,
        }

    # The pretrained model and each fold use the same BM25 candidate lists.
    base_model = CrossEncoder(args.model, revision=args.revision)
    baseline = model_metrics(base_model)
    fold_results = []
    training_seconds = []

    for held_out in queries:
        train_rows = []
        for query in queries:
            if query.id == held_out.id:
                continue
            for hit in candidate_sets[query.id]:
                grade = query.relevance.get(hit.document.id, 0)
                # Map the project's 0..3 graded label to the BCE 0..1 target.
                train_rows.append({
                    "query": query.query,
                    "passage": hit.document.searchable_text,
                    "label": grade / 3.0,
                })
        if not any(row["label"] > 0 for row in train_rows):
            raise RuntimeError(f"Training fold for {held_out.id} has no positive pairs")

        torch.manual_seed(42)
        model = CrossEncoder(args.model, num_labels=1, revision=args.revision)
        training_data = Dataset.from_list(train_rows)
        with tempfile.TemporaryDirectory(prefix="reranker-fold-") as fold_output:
            training_args = CrossEncoderTrainingArguments(
                output_dir=fold_output,
                num_train_epochs=args.epochs,
                per_device_train_batch_size=8,
                learning_rate=2e-5,
                warmup_steps=0,
                eval_strategy="no",
                save_strategy="no",
                logging_strategy="no",
                report_to="none",
                seed=42,
                data_seed=42,
            )
            trainer = CrossEncoderTrainer(
                model=model,
                args=training_args,
                train_dataset=training_data,
                loss=BinaryCrossEntropyLoss(model),
            )
            started = perf_counter()
            trainer.train()
            training_seconds.append(perf_counter() - started)
            held_out_metrics = model_metrics_for_one(model, held_out, candidate_sets,
                                                     args.metric_k)
            fold_results.append({
                "held_out_query_id": held_out.id,
                "train_query_ids": [query.id for query in queries if query.id != held_out.id],
                **held_out_metrics,
            })

    # After cross-validation, fit a convenience checkpoint on all labeled demo queries.
    # Its in-sample scores are not used in the reported cross-validation metrics.
    all_rows = [
        {
            "query": query.query,
            "passage": hit.document.searchable_text,
            "label": query.relevance.get(hit.document.id, 0) / 3.0,
        }
        for query in queries
        for hit in candidate_sets[query.id]
    ]
    final_model = CrossEncoder(args.model, num_labels=1, revision=args.revision)
    with tempfile.TemporaryDirectory(prefix="reranker-final-train-") as training_output:
        final_args = CrossEncoderTrainingArguments(
            output_dir=training_output,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=8,
            learning_rate=2e-5,
            warmup_steps=0,
            eval_strategy="no",
            save_strategy="no",
            logging_strategy="no",
            report_to="none",
            seed=42,
            data_seed=42,
        )
        final_trainer = CrossEncoderTrainer(
            model=final_model,
            args=final_args,
            train_dataset=Dataset.from_list(all_rows),
            loss=BinaryCrossEntropyLoss(final_model),
        )
        final_started = perf_counter()
        final_trainer.train()
        final_training_seconds = perf_counter() - final_started
    model_output = Path(args.model_output)
    model_output.mkdir(parents=True, exist_ok=True)
    final_model.save_pretrained(model_output)

    fine_tuned_per_query = [fold["metrics"] for fold in fold_results]
    fine_tuned_latencies = [fold["latency_ms"] for fold in fold_results]
    fine_tuned = {
        "query_count": len(fine_tuned_per_query),
        **{
            metric: mean(row[metric] for row in fine_tuned_per_query)
            for metric in ("recall", "mrr", "ndcg")
        },
        "latency_ms_mean": mean(fine_tuned_latencies),
        "latency_ms_p95": percentile(fine_tuned_latencies, 0.95),
    }

    report = {
        "experiment": "leave_one_query_out_cross_encoder_finetuning_demo",
        "data_source": "small synthetic corpus and six-query golden set in this repository",
        "candidate_source": "BM25 top candidate_k; train and test use the same retrieval stage",
        "label_mapping": "graded relevance 0..3 mapped linearly to BCE targets 0..1",
        "base_model": args.model,
        "base_model_revision": args.revision,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "sentence_transformers": version("sentence-transformers"),
            "transformers": version("transformers"),
            "device": "mps" if torch.backends.mps.is_available() else
                      "cuda" if torch.cuda.is_available() else "cpu",
        },
        "epochs": args.epochs,
        "candidate_k": args.candidate_k,
        "metric_k": args.metric_k,
        "latency_repeats_per_query": LATENCY_REPEATS,
        "fold_count": len(fold_results),
        "mean_training_seconds_per_fold": mean(training_seconds),
        "total_training_seconds": sum(training_seconds),
        "final_checkpoint_training_seconds": final_training_seconds,
        "final_checkpoint_path": str(model_output),
        "baseline_pretrained_cross_encoder": baseline,
        "fine_tuned_cross_encoder_leave_one_query_out": fine_tuned,
        "folds": fold_results,
        "limitations": [
            "Only six synthetic queries are available; the cross-validation estimate is highly unstable.",
            "No independent final test set exists in this example; do not claim production generalization.",
            "Training hyperparameters are fixed for demonstration and were not selected on a validation set.",
            "Treating unlisted documents as grade zero follows this toy golden set and needs human review in real data.",
        ],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "report": str(output),
        "baseline": baseline,
        "fine_tuned_cross_validation": fine_tuned,
        "total_training_seconds": report["total_training_seconds"],
        "final_checkpoint_training_seconds": final_training_seconds,
        "final_checkpoint_path": str(model_output),
    }, indent=2))


def model_metrics_for_one(model, query, candidate_sets, metric_k):
    from reranking_strategies.metrics import score_ranking
    from reranking_strategies.models import Hit

    candidates = candidate_sets[query.id]
    pairs = [(query.query, hit.document.searchable_text) for hit in candidates]
    model.predict([pairs[0]], show_progress_bar=False)  # Warm-up before timing.
    repeated_latencies = []
    scores = None
    for _ in range(LATENCY_REPEATS):
        started = perf_counter()
        scores = model.predict(pairs, show_progress_bar=False)
        repeated_latencies.append((perf_counter() - started) * 1000)
    latency_ms = median(repeated_latencies)
    ranked = sorted(
        (Hit(hit.document, float(score), "cross_encoder")
         for hit, score in zip(candidates, scores)),
        key=lambda hit: (-hit.score, hit.document.id),
    )[:metric_k]
    return {
        "latency_ms": latency_ms,
        "metrics": score_ranking(ranked, query.relevance, metric_k),
    }


if __name__ == "__main__":
    main()
