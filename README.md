# Reranking Strategies

A reproducible lab for comparing hybrid retrieval and reranking methods on a synthetic RAG corpus. The first milestone focuses on transparent baselines and offline evaluation; it does not require an API key.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
python -m reranking_strategies.cli evaluate
python -m unittest discover -s tests -v
```

Start the optional local Qdrant service with `docker compose up -d qdrant`, install `pip install -e '.[qdrant]'`, then run `python -m reranking_strategies.cli evaluate --dense`. FastEmbed downloads its embedding model on first use. For a local cross-encoder, install `pip install -e '.[cross-encoder]'` and add `--cross-encoder` (optionally set `--cross-encoder-model`). To compare OpenAI listwise reranking, install `pip install -e '.[llm-judge]'`, set `OPENAI_API_KEY` (and optionally `OPENAI_MODEL`), then add `--llm-reranker`. The default evaluation harness and tests run offline.

## What's included

- Synthetic corpus and graded query relevance labels (`data/`).
- BM25 retrieval, reciprocal rank fusion (RRF), and a common candidate/ranking interface.
- Heuristic, cross-encoder, and optional LLM pairwise reranking strategies.
- Recall@k, MRR@k, and nDCG@k evaluation, per-query results, and JSON reports.
- Unit tests for BM25, RRF, reranking behavior, metrics, and golden-data integrity.
- Optional Docker Compose Qdrant adapter for local dense + BM25 hybrid evaluation.

The corpus is intentionally synthetic. It is useful for controlled regression tests, but results should be validated against manually reviewed queries before making quality claims. Keep held-out queries untouched during tuning. The pairwise comparator accepts an injected judge callback for deterministic tests; the optional OpenAI listwise strategy makes one structured ranking request per query.
