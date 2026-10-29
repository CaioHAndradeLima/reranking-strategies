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

    def test_duplicate_document_ids_do_not_inflate_ranking_metrics(self):
        relevance = {"relevant": 3, "partial": 1}
        ranked = ["relevant", "relevant", "partial"]
        self.assertEqual(recall_at_k(ranked, relevance, 2), 1.0)
        self.assertEqual(mrr_at_k(ranked, relevance, 2), 1.0)
        self.assertAlmostEqual(ndcg_at_k(ranked, relevance, 2), 1.0)

    def test_cutoff_must_be_positive_integer(self):
        for k in (0, -1, 1.5, True):
            with self.subTest(k=k), self.assertRaisesRegex(ValueError, "positive integer"):
                ndcg_at_k(["a"], {"a": 1}, k)
