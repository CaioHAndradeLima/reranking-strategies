import unittest

from reranking_strategies.metrics import mrr_at_k, ndcg_at_k, recall_at_k


class MetricsTests(unittest.TestCase):
    def test_ranking_metrics_use_graded_relevance(self):
        relevance = {"relevant": 3, "partial": 1}
        self.assertEqual(recall_at_k(["irrelevant", "relevant"], relevance, 2), 0.5)
        self.assertEqual(mrr_at_k(["irrelevant", "relevant"], relevance, 2), 0.5)
        self.assertAlmostEqual(ndcg_at_k(["relevant", "partial"], relevance, 2), 1.0)

    def test_empty_relevance_returns_zero(self):
        self.assertEqual(recall_at_k(["a"], {}, 5), 0)
        self.assertEqual(mrr_at_k(["a"], {}, 5), 0)
        self.assertEqual(ndcg_at_k(["a"], {}, 5), 0)
