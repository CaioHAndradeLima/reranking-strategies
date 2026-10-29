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

Run the Qdrant integration check against the Compose service with:

```bash
RERANKING_QDRANT_URL=http://localhost:6333 python -m unittest discover -s tests -p 'test_qdrant_store.py' -v
```

Indexing replaces the configured Qdrant collection with a snapshot of the supplied corpus, so rerunning evaluation cannot leave stale documents behind. The integration check downloads the FastEmbed model on first use.

## What's included

- Synthetic corpus and graded query relevance labels (`data/`).
- BM25 retrieval, reciprocal rank fusion (RRF), and a common candidate/ranking interface.
- Heuristic, cross-encoder, and optional LLM pairwise reranking strategies.
- Recall@k, MRR@k, and nDCG@k evaluation, per-query results, and JSON reports.
- Unit tests for BM25, RRF, reranking behavior, metrics, and golden-data integrity.
- Optional Docker Compose Qdrant adapter for local dense + BM25 hybrid evaluation.

The corpus is intentionally synthetic. It is useful for controlled regression tests, but results should be validated against manually reviewed queries before making quality claims. Keep held-out queries untouched during tuning. The pairwise comparator accepts an injected judge callback for deterministic tests; the optional OpenAI listwise strategy makes one structured ranking request per query.

The JSON loaders validate row shapes, unique IDs, relevance grades, query answerability, and cross references before the CLI runs an evaluation. Both `candidate_k` and `metric_k` must be positive integers.
