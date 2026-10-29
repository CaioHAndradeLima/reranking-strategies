import unittest
from pathlib import Path

from reranking_strategies.data import load_documents, load_queries

ROOT = Path(__file__).resolve().parents[1]


class GoldenDataTests(unittest.TestCase):
    def test_queries_reference_known_documents_and_valid_grades(self):
        documents = load_documents(ROOT / "data/documents.json")
        queries = load_queries(ROOT / "data/golden.json")
        known_ids = {doc.id for doc in documents}
        self.assertTrue(queries)
        self.assertEqual(len({query.id for query in queries}), len(queries))
        for query in queries:
            self.assertTrue(set(query.relevance).issubset(known_ids))
            self.assertTrue(all(0 <= grade <= 3 for grade in query.relevance.values()))
            if not query.answerable:
                self.assertFalse(query.relevance)


if __name__ == "__main__":
    unittest.main()
