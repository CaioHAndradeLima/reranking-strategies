import unittest

from reranking_strategies.evaluation import evaluate
from reranking_strategies.models import Document, Hit, QueryCase


class DuplicateReranker:
    name = "duplicate"

    def rerank(self, query, candidates, limit=10):
        return [candidates[0], candidates[0]]


class ForeignReranker:
    name = "foreign"

    def rerank(self, query, candidates, limit=10):
        return [Hit(Document("foreign", "Foreign", "not a candidate"), 1.0, self.name)]


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


if __name__ == "__main__":
    unittest.main()
