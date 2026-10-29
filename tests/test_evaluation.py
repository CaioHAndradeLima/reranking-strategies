import unittest

from reranking_strategies.evaluation import evaluate, recommend_best_strategy
from reranking_strategies.models import Document, Hit, QueryCase


class DuplicateReranker:
    name = "duplicate"

    def rerank(self, query, candidates, limit=10):
        return [candidates[0], candidates[0]]


class ForeignReranker:
    name = "foreign"

    def rerank(self, query, candidates, limit=10):
        return [Hit(Document("foreign", "Foreign", "not a candidate"), 1.0, self.name)]


class MeasuredReranker:
    name = "measured"

    def __init__(self):
        self.last_usage = {
            "request_count": 1, "input_tokens": 100, "cached_input_tokens": 0,
            "output_tokens": 10, "estimated_cost_usd": 0.002,
        }

    def rerank(self, query, candidates, limit=10):
        return list(candidates[:limit])


class EvaluationValidationTests(unittest.TestCase):
    def test_candidate_and_metric_cutoffs_must_be_positive(self):
        for kwargs, message in [
            ({"candidate_k": 0}, "candidate_k"),
            ({"candidate_k": -1}, "candidate_k"),
            ({"candidate_k": 1.5}, "candidate_k"),
            ({"candidate_k": True}, "candidate_k"),
            ({"metric_k": 0}, "metric_k"),
            ({"metric_k": -1}, "metric_k"),
            ({"metric_k": 1.5}, "metric_k"),
            ({"metric_k": True}, "metric_k"),
        ]:
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, message):
                evaluate([], lambda query, limit: [], **kwargs)

    def test_evaluation_rejects_duplicate_retrieval_candidates(self):
        doc = Document("d", "Doc", "text")
        case = QueryCase("q", "query", {"d": 1})

        with self.assertRaisesRegex(ValueError, "Retriever returned duplicate"):
            evaluate([case], lambda query, limit: [Hit(doc, 1.0, "test")] * 2)

    def test_evaluation_rejects_duplicate_or_foreign_reranker_results(self):
        doc = Document("d", "Doc", "text")
        case = QueryCase("q", "query", {"d": 1})
        retriever = lambda query, limit: [Hit(doc, 1.0, "test")]

        with self.assertRaisesRegex(ValueError, "duplicate document IDs"):
            evaluate([case], retriever, [DuplicateReranker()])
        with self.assertRaisesRegex(ValueError, "outside its candidates"):
            evaluate([case], retriever, [ForeignReranker()])

    def test_evaluation_reports_latency_token_and_cost_metrics(self):
        doc = Document("d", "Doc", "text")
        case = QueryCase("q", "query", {"d": 1})
        report = evaluate([case, case], lambda query, limit: [Hit(doc, 1.0, "test")],
                          [MeasuredReranker()])

        summary = report["summary"]["measured"]
        self.assertEqual(summary["request_count"], 2)
        self.assertEqual(summary["input_tokens"], 200)
        self.assertEqual(summary["output_tokens"], 20)
        self.assertAlmostEqual(summary["estimated_cost_usd"], 0.004)
        self.assertGreaterEqual(summary["latency_ms_p95"], summary["latency_ms_p50"])
        self.assertEqual(report["results"][0]["strategies"]["measured"]["request_count"], 1)

    def test_recommendation_uses_quality_then_latency_and_cost(self):
        best = recommend_best_strategy({
            "slow": {"ndcg": 0.9, "latency_ms_p95": 80, "estimated_cost_usd": 0.01},
            "fast": {"ndcg": 0.9, "latency_ms_p95": 10, "estimated_cost_usd": 0.02},
            "low": {"ndcg": 0.8, "latency_ms_p95": 1, "estimated_cost_usd": 0.0},
        })
        self.assertEqual(best["strategy"], "fast")


if __name__ == "__main__":
    unittest.main()
