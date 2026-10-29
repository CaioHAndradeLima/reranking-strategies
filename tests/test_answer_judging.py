import json
import unittest

from reranking_strategies.answer_judging import OpenAIAnswerJudge, evaluate_answers
from reranking_strategies.models import Document, QueryCase


def judge_output(**overrides):
    result = {
        "correctness": 4,
        "groundedness": 4,
        "assertiveness_calibration": 4,
        "abstained": False,
        "unsupported_claims": [],
        "rationale": "The answer matches the evidence.",
    }
    result.update(overrides)
    return result


class AnswerJudgingTests(unittest.TestCase):
    def setUp(self):
        self.query = QueryCase(
            "q1", "How long are audit logs retained?", {"d1": 3},
            answerable=True, reference_answer="Security audit events are kept for one year.",
        )
        self.document = Document("d1", "Retention", "Security audit events are retained for one year.")

    def test_openai_answer_judge_uses_structured_schema_and_reports_usage(self):
        class Responses:
            def create(inner_self, **kwargs):
                inner_self.request = kwargs
                usage = type("Usage", (), {"input_tokens": 120, "output_tokens": 18})()
                return type("Response", (), {
                    "output_text": json.dumps(judge_output()), "usage": usage,
                })()

        responses = Responses()
        result = OpenAIAnswerJudge(
            model="gpt-6-luna", client=type("Client", (), {"responses": responses})()
        ).judge(self.query, "One year.", [self.document])

        self.assertFalse(responses.request["store"])
        self.assertEqual(responses.request["text"]["format"]["type"], "json_schema")
        self.assertIn("assertiveness_calibration", responses.request["instructions"])
        self.assertEqual(result["correctness"], 4)
        self.assertTrue(result["abstention_correct"])
        self.assertEqual(result["request_count"], 1)
        self.assertGreater(result["estimated_cost_usd"], 0)

    def test_unanswerable_query_rewards_abstention_as_deterministic_flag(self):
        query = QueryCase("q2", "What about recordings?", {}, answerable=False)

        class Responses:
            def create(self, **kwargs):
                return type("Response", (), {
                    "output_text": '{"correctness":4,"groundedness":4,'
                    '"assertiveness_calibration":4,"abstained":true,'
                    '"unsupported_claims":[],"rationale":"No evidence was supplied."}',
                })()

        judge = OpenAIAnswerJudge(client=type(
            "Client", (), {"responses": Responses()}
        )())

        self.assertTrue(judge.judge(query, "The supplied information does not say.", [])[
            "abstention_correct"
        ])

    def test_judge_rejects_missing_reference_and_invalid_scores(self):
        no_reference = QueryCase("q3", "Question?", {})
        with self.assertRaisesRegex(ValueError, "requires a reference_answer"):
            OpenAIAnswerJudge(client=object()).judge(no_reference, "answer", [])

        class Responses:
            def create(self, **kwargs):
                return type("Response", (), {"output_text": '{"correctness":5}'})()

        judge = OpenAIAnswerJudge(client=type(
            "Client", (), {"responses": Responses()}
        )())
        with self.assertRaisesRegex(ValueError, "correctness must be an integer"):
            judge.judge(self.query, "answer", [])

    def test_answer_evaluation_summarizes_scores_by_strategy(self):
        class FakeJudge:
            def judge(self, query, answer, evidence):
                score = 4 if "accurate" in answer else 1
                return {
                    **judge_output(correctness=score, groundedness=score,
                                   assertiveness_calibration=score),
                    "abstention_correct": True,
                    "request_count": 1,
                    "retry_count": 0,
                    "input_tokens": 10,
                    "cached_input_tokens": 0,
                    "output_tokens": 5,
                    "estimated_cost_usd": 0.001,
                }

        report = evaluate_answers([
            {"query_id": "q1", "strategy": "dense", "answer": "accurate", "evidence_ids": ["d1"]},
            {"query_id": "q1", "strategy": "bm25", "answer": "weak", "evidence_ids": []},
        ], [self.query], [self.document], FakeJudge())

        self.assertEqual(report["summary"]["dense"]["correctness"], 4)
        self.assertEqual(report["summary"]["bm25"]["groundedness"], 1)
        self.assertEqual(report["summary"]["dense"]["request_count"], 1)
        self.assertEqual(report["summary"]["dense"]["estimated_cost_usd"], 0.001)

    def test_answer_samples_validate_query_evidence_and_duplicate_strategy(self):
        class FakeJudge:
            def judge(self, query, answer, evidence):
                return judge_output() | {"request_count": 1}

        base = {"query_id": "q1", "strategy": "bm25", "answer": "answer", "evidence_ids": []}
        with self.assertRaisesRegex(ValueError, "unknown evidence IDs"):
            evaluate_answers([base | {"evidence_ids": ["missing"]}], [self.query], [self.document], FakeJudge())
        with self.assertRaisesRegex(ValueError, "Duplicate answer sample"):
            evaluate_answers([base, base], [self.query], [self.document], FakeJudge())


if __name__ == "__main__":
    unittest.main()
