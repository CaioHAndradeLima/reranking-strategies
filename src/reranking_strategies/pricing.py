"""OpenAI standard text-token rates used for approximate benchmark cost estimates.

Rates were checked on 2026-10-29. Update them from the pricing page before relying
on cost comparisons for a later billing period or a different service tier.
"""

from collections.abc import Iterable


PRICING_AS_OF = "2026-10-29"
PRICING_SOURCE = "https://developers.openai.com/api/docs/pricing"
OPENAI_STANDARD_RATES_PER_MILLION = {
    "gpt-6-luna": {"input": 0.10, "cached_input": 0.01, "output": 0.50},
    "gpt-6.1-terra": {"input": 1.00, "cached_input": 0.05, "output": 5.00},
}


def estimate_openai_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cached_input_tokens: int = 0,
) -> float | None:
    rates = OPENAI_STANDARD_RATES_PER_MILLION.get(model)
    if rates is None:
        return None
    cached = min(max(cached_input_tokens, 0), input_tokens)
    uncached = max(input_tokens - cached, 0)
    return (
        uncached * rates["input"]
        + cached * rates["cached_input"]
        + output_tokens * rates["output"]
    ) / 1_000_000


def response_usage(model: str, usage) -> dict:
    if usage is None:
        return {
            "request_count": 1,
            "input_tokens": None,
            "cached_input_tokens": None,
            "output_tokens": None,
            "estimated_cost_usd": None,
        }
    input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
    output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
    input_details = getattr(usage, "input_tokens_details", None)
    cached_input_tokens = int(getattr(input_details, "cached_tokens", 0) or 0)
    return {
        "request_count": 1,
        "input_tokens": input_tokens,
        "cached_input_tokens": cached_input_tokens,
        "output_tokens": output_tokens,
        "estimated_cost_usd": estimate_openai_cost(
            model, input_tokens, output_tokens, cached_input_tokens
        ),
    }


def combine_usage(usages: Iterable[dict | None], retries: int = 0) -> dict:
    values = [usage for usage in usages if usage]
    result = {"request_count": 0, "retry_count": retries}
    for key in ("input_tokens", "cached_input_tokens", "output_tokens", "estimated_cost_usd"):
        items = [usage.get(key) for usage in values]
        result[key] = None if any(value is None for value in items) else sum(items)
    result["request_count"] = sum(usage.get("request_count", 0) for usage in values)
    return result


class CallBudget:
    """Shared hard cap for API calls made during one benchmark run."""

    def __init__(self, maximum: int = 100):
        if type(maximum) is not int or maximum <= 0:
            raise ValueError("maximum LLM calls must be a positive integer")
        self.maximum = maximum
        self.used = 0

    def consume(self) -> None:
        if self.used >= self.maximum:
            raise RuntimeError(f"LLM call budget of {self.maximum} requests has been reached")
        self.used += 1
