from collections import defaultdict
from collections.abc import Callable, Iterable
from statistics import mean
from time import perf_counter

from .metrics import score_ranking
from .models import Hit, QueryCase
from .rerankers import Reranker


def evaluate(
    queries: Iterable[QueryCase],
    retriever: Callable[[str, int], list[Hit]],
    rerankers: Iterable[Reranker] = (),
    candidate_k: int = 20,
    metric_k: int = 5,
) -> dict:
    if type(candidate_k) is not int or candidate_k <= 0:
        raise ValueError("candidate_k must be a positive integer")
    if type(metric_k) is not int or metric_k <= 0:
        raise ValueError("metric_k must be a positive integer")
    strategies: dict[str, list[dict[str, float]]] = defaultdict(list)
    rows = []
    rerankers = list(rerankers)
    latencies: dict[str, list[float]] = defaultdict(list)
    for case in queries:
        started = perf_counter()
        candidates = retriever(case.query, candidate_k)
        retrieval_ms = (perf_counter() - started) * 1000
        candidate_ids = [hit.document.id for hit in candidates]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError(f"Retriever returned duplicate document IDs for query '{case.id}'")
        timings = {"retrieval": retrieval_ms}
        scored = {"retrieval": candidates}
        for reranker in rerankers:
            started = perf_counter()
            ranked = reranker.rerank(case.query, candidates, metric_k)
            timings[reranker.name] = (perf_counter() - started) * 1000
            ranked_ids = [hit.document.id for hit in ranked]
            if len(ranked_ids) != len(set(ranked_ids)):
                raise ValueError(f"Reranker '{reranker.name}' returned duplicate document IDs")
            if not set(ranked_ids).issubset(candidate_ids):
                raise ValueError(f"Reranker '{reranker.name}' returned a document outside its candidates")
            scored[reranker.name] = ranked
        scored["retrieval"] = candidates[:metric_k]
        row = {"query_id": case.id, "query": case.query, "answerable": case.answerable,
               "candidate_ids": [hit.document.id for hit in candidates], "retrieval_ms": retrieval_ms,
               "strategies": {}}
        for name, hits in scored.items():
            metrics = score_ranking(hits, case.relevance, metric_k)
            strategies[name].append(metrics)
            latencies[name].append(timings[name])
            row["strategies"][name] = {
                **metrics, "ranked_ids": [hit.document.id for hit in hits],
                "scores": [hit.score for hit in hits], "latency_ms": timings[name],
            }
        rows.append(row)
    return {"query_count": len(rows), "candidate_k": candidate_k, "metric_k": metric_k,
            "summary": {name: {**{metric: mean(row[metric] for row in values)
                                 for metric in ("recall", "mrr", "ndcg")},
                                "latency_ms": mean(latencies[name])}
                        for name, values in strategies.items()},
            "results": rows}
