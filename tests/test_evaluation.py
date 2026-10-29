import unittest

from reranking_strategies.evaluation import evaluate


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


if __name__ == "__main__":
    unittest.main()
