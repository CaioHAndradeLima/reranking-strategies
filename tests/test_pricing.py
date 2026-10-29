import unittest

from reranking_strategies.pricing import CallBudget, estimate_openai_cost, response_usage


class PricingTests(unittest.TestCase):
    def test_cost_estimate_accounts_for_cached_input_tokens(self):
        cost = estimate_openai_cost("gpt-6-luna", input_tokens=10_000,
                                    output_tokens=1_000, cached_input_tokens=4_000)
        self.assertAlmostEqual(cost, 0.00114)

    def test_unknown_models_have_no_estimated_cost(self):
        self.assertIsNone(estimate_openai_cost("custom-model", 100, 10))

    def test_response_usage_records_requests_and_unknown_usage(self):
        usage = response_usage("gpt-6-luna", None)
        self.assertEqual(usage["request_count"], 1)
        self.assertIsNone(usage["estimated_cost_usd"])

    def test_call_budget_enforces_a_hard_request_cap(self):
        budget = CallBudget(1)
        budget.consume()
        with self.assertRaisesRegex(RuntimeError, "budget of 1"):
            budget.consume()


if __name__ == "__main__":
    unittest.main()
