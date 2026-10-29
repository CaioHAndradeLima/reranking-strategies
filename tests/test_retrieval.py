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
