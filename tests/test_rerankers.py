import unittest

from reranking_strategies.models import Document, Hit
from reranking_strategies.rerankers import (
    CrossEncoderReranker, HeuristicReranker, MMRReranker,
    OpenAIPairwiseJudge, OpenAIListwiseReranker, PairwiseJudgeReranker,
    RetryingReranker,
)


def candidates():
    return [
        Hit(Document("b", "Retry policy", "Payment retries use backoff."), 0.9, "test"),
        Hit(Document("a", "Other", "Unrelated catalog information."), 0.8, "test"),
    ]


class RerankerTests(unittest.TestCase):
    def test_heuristic_reranker_prioritizes_title_match(self):
        result = HeuristicReranker().rerank("retry policy", candidates())
        self.assertEqual(result[0].document.id, "b")

    def test_mmr_handles_empty_and_limits_results(self):
        reranker = MMRReranker()
        self.assertEqual(reranker.rerank("retry", []), [])
        self.assertEqual(len(reranker.rerank("retry", candidates(), limit=1)), 1)

    def test_mmr_preserves_order_of_negative_relevance_scores(self):
        candidates_with_negative_scores = [
            Hit(Document("a", "Lower", "lower score"), -0.9, "dense"),
            Hit(Document("b", "Higher", "higher score"), -0.5, "dense"),
        ]

        result = MMRReranker(relevance_weight=1.0).rerank("query", candidates_with_negative_scores)

        self.assertEqual([hit.document.id for hit in result], ["b", "a"])

    def test_rerankers_reject_duplicate_candidates_and_negative_limits(self):
        repeated = candidates() + [candidates()[0]]
        with self.assertRaisesRegex(ValueError, "Candidate document IDs must be unique"):
            HeuristicReranker().rerank("retry", repeated)
        with self.assertRaisesRegex(ValueError, "limit must be a non-negative integer"):
            HeuristicReranker().rerank("retry", candidates(), limit=-1)

    def test_cross_encoder_rejects_incomplete_score_output(self):
        reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
        reranker.model = type("Model", (), {"predict": lambda self, pairs: [0.5]})()

        with self.assertRaisesRegex(ValueError, "one score for every candidate"):
            reranker.rerank("retry", candidates())

    def test_pairwise_adapter_uses_injected_judge(self):
        reranker = PairwiseJudgeReranker(lambda query, left, right: 1 if left.document.id == "a" else -1)
        self.assertEqual(reranker.rerank("q", candidates())[0].document.id, "a")

    def test_pairwise_candidate_limit_caps_comparator_calls(self):
        calls = []
        judge = lambda query, left, right: calls.append((left.document.id, right.document.id)) or 1
        result = PairwiseJudgeReranker(judge, candidate_limit=1).rerank("q", candidates())

        self.assertEqual(calls, [])
        self.assertEqual([hit.document.id for hit in result], ["b", "a"])

    def test_openai_pairwise_judge_reports_tokens_and_selects_candidate(self):
        class Responses:
            def create(self, **kwargs):
                self.request = kwargs
                usage = type("Usage", (), {"input_tokens": 100, "output_tokens": 5})()
                return type("Response", (), {"output_text": '{"winner_id":"a"}', "usage": usage})()

        client = type("Client", (), {"responses": Responses()})()
        judge = OpenAIPairwiseJudge(model="gpt-6-luna", client=client)
        by_id = {hit.document.id: hit for hit in candidates()}

        outcome = judge("retry", by_id["a"], by_id["b"])
        pairwise = PairwiseJudgeReranker(judge)
        ranked = pairwise.rerank("retry", list(by_id.values()))

        self.assertEqual(outcome, 1)
        self.assertFalse(client.responses.request["store"])
        self.assertEqual([hit.document.id for hit in ranked][0], "a")
        self.assertEqual(judge.usage_events[0]["request_count"], 1)
        self.assertEqual(pairwise.last_usage["request_count"], 1)
        self.assertGreater(pairwise.last_usage["estimated_cost_usd"], 0)

    def test_openai_listwise_adapter_validates_candidate_ids(self):
        class Responses:
            def create(self, **kwargs):
                self.request = kwargs
                usage = type("Usage", (), {"input_tokens": 100, "output_tokens": 5})()
                return type("Response", (), {
                    "output_text": '{"ranked_ids":["a","b"],"confidence":0.95}',
                    "usage": usage,
                })()

        class Client:
            responses = Responses()

        reranker = OpenAIListwiseReranker(client=Client())
        result = reranker.rerank("retry", candidates())
        self.assertEqual([hit.document.id for hit in result], ["a", "b"])
        self.assertFalse(Client.responses.request["store"])
        self.assertEqual(reranker.last_confidence, 0.95)
        self.assertEqual(reranker.last_usage["input_tokens"], 100)
        self.assertGreater(reranker.last_usage["estimated_cost_usd"], 0)

    def test_openai_listwise_rejects_missing_or_duplicate_ids(self):
        class Responses:
            def create(self, **kwargs):
                return type("Response", (), {
                    "output_text": '{"ranked_ids":["a","a"],"confidence":0.2}'
                })()

        client = type("Client", (), {"responses": Responses()})()
        with self.assertRaises(ValueError):
            OpenAIListwiseReranker(client=client).rerank("retry", candidates())

    def test_retrying_reranker_calls_fallback_only_below_confidence_threshold(self):
        class FakeReranker:
            def __init__(self, name, confidence, order):
                self.name = name
                self.last_confidence = confidence
                self.last_usage = {
                    "request_count": 1, "input_tokens": 10, "cached_input_tokens": 0,
                    "output_tokens": 2, "estimated_cost_usd": 0.001,
                }
                self.order = order
                self.calls = 0

            def rerank(self, query, candidates, limit=10):
                self.calls += 1
                by_id = {hit.document.id: hit for hit in candidates}
                return [by_id[doc_id] for doc_id in self.order][:limit]

        primary = FakeReranker("cheap", 0.3, ["b", "a"])
        fallback = FakeReranker("strong", 0.9, ["a", "b"])
        retry = RetryingReranker(primary, fallback, confidence_threshold=0.6)

        result = retry.rerank("retry", candidates())

        self.assertEqual([hit.document.id for hit in result], ["a", "b"])
        self.assertEqual((primary.calls, fallback.calls), (1, 1))
        self.assertEqual(retry.last_usage["request_count"], 2)
        self.assertEqual(retry.last_usage["retry_count"], 1)

    def test_retrying_reranker_does_not_call_fallback_above_threshold(self):
        primary = type("Primary", (), {
            "name": "primary", "last_confidence": 0.9,
            "last_usage": {"request_count": 1, "input_tokens": 10,
                           "cached_input_tokens": 0, "output_tokens": 2,
                           "estimated_cost_usd": 0.001},
            "rerank": lambda self, query, hits, limit=10: list(hits[:limit]),
        })()
        fallback = type("Fallback", (), {
            "name": "fallback", "calls": 0,
            "rerank": lambda self, query, hits, limit=10: setattr(self, "calls", self.calls + 1) or list(hits),
        })()
        retry = RetryingReranker(primary, fallback)

        retry.rerank("retry", candidates())

        self.assertEqual(fallback.calls, 0)
        self.assertEqual(retry.last_usage["retry_count"], 0)
