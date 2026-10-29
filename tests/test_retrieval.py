import unittest

from reranking_strategies.models import Document, Hit
from reranking_strategies.retrieval import BM25Retriever, rrf_fuse


class RetrievalTests(unittest.TestCase):
    def test_bm25_finds_exact_identifier(self):
        docs = [Document("a", "Other", "general process"), Document("b", "Incident", "INC-4821 billing outage")]
        self.assertEqual(BM25Retriever(docs).search("INC-4821")[0].document.id, "b")

    def test_rrf_is_deterministic_and_rewards_consensus(self):
        a, b, c = [Document(name, name, name) for name in "abc"]
        left = [Hit(a, 0.2, "dense"), Hit(b, 0.1, "dense")]
        right = [Hit(b, 0.9, "bm25"), Hit(a, 0.1, "bm25"), Hit(c, 0.05, "bm25")]
        result = rrf_fuse([left, right], k=1)
        self.assertEqual(result[0].document.id, "a")
        self.assertEqual([hit.document.id for hit in result], ["a", "b", "c"])

    def test_rrf_counts_a_document_once_per_input_ranking(self):
        a = Document("a", "A", "alpha")
        b = Document("b", "B", "beta")
        hits = [Hit(a, 1.0, "source"), Hit(a, 0.9, "source"), Hit(b, 0.8, "source")]

        result = rrf_fuse([hits], k=1)

        self.assertEqual([hit.document.id for hit in result], ["a", "b"])
        self.assertAlmostEqual(result[0].score, 0.5)
        self.assertAlmostEqual(result[1].score, 1 / 3)

    def test_bm25_rejects_duplicate_document_ids_and_invalid_parameters(self):
        a = Document("same", "One", "one")
        b = Document("same", "Two", "two")
        with self.assertRaisesRegex(ValueError, "IDs must be unique"):
            BM25Retriever([a, b])
        with self.assertRaisesRegex(ValueError, "k1"):
            BM25Retriever([a], k1=0)
        with self.assertRaisesRegex(ValueError, "b"):
            BM25Retriever([a], b=1.1)

    def test_search_and_rrf_reject_invalid_cutoffs(self):
        retriever = BM25Retriever([Document("a", "A", "alpha")])
        with self.assertRaisesRegex(ValueError, "limit"):
            retriever.search("alpha", -1)
        with self.assertRaisesRegex(ValueError, "k must"):
            rrf_fuse([], k=0)
