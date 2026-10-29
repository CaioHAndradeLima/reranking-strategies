import math
from collections.abc import Sequence

from .models import Hit


def recall_at_k(ranked_ids: Sequence[str], relevance: dict[str, int], k: int) -> float:
    relevant = {doc_id for doc_id, grade in relevance.items() if grade > 0}
    if not relevant:
        return 0.0
    return len(set(ranked_ids[:k]) & relevant) / len(relevant)


def mrr_at_k(ranked_ids: Sequence[str], relevance: dict[str, int], k: int) -> float:
    for rank, doc_id in enumerate(ranked_ids[:k], start=1):
        if relevance.get(doc_id, 0) > 0:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked_ids: Sequence[str], relevance: dict[str, int], k: int) -> float:
    def dcg(grades: Sequence[int]) -> float:
        return sum((2**grade - 1) / math.log2(rank + 1) for rank, grade in enumerate(grades, start=1))
    actual = [relevance.get(doc_id, 0) for doc_id in ranked_ids[:k]]
    ideal = sorted(relevance.values(), reverse=True)[:k]
    ideal_dcg = dcg(ideal)
    return dcg(actual) / ideal_dcg if ideal_dcg else 0.0


def score_ranking(hits: Sequence[Hit], relevance: dict[str, int], k: int = 5) -> dict[str, float]:
    ids = [hit.document.id for hit in hits]
    return {"recall": recall_at_k(ids, relevance, k),
            "mrr": mrr_at_k(ids, relevance, k),
            "ndcg": ndcg_at_k(ids, relevance, k)}
