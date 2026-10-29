# Reranking Strategies

A reproducible lab for comparing retrieval and reranking strategies on a small synthetic RAG corpus. It makes the ranking step visible: retrieve a broad candidate set, reorder it using query and passage relevance, then evaluate what should reach the context window.

## Why rerank?

Retrieval systems quickly find passages that may answer a query. Their top results can still contain related but non-answering material. A reranker examines the query and each candidate together and reorders that set for the specific query.

```text
Query
  → Retrieve a broad candidate set
  → Optionally fuse dense and keyword results
  → Rerank candidates against the query
  → Keep the final top K passages for context
```

Reranking cannot recover a relevant passage that retrieval left out. Candidate recall therefore sets an upper bound on the quality a reranker can achieve. Reranking also adds latency and, for hosted judges, may add API cost.

| Strategy | What it does | Trade-off |
| --- | --- | --- |
| Retrieval only | Uses the initial BM25 or dense ranking | Simple and fast; may favor related text over direct evidence |
| Retrieval plus reranking | Reorders one candidate list for the query | Can improve evidence order; adds compute and latency |
| Hybrid retrieval plus reranking | Fuses keyword and dense candidates, then reranks | Combines exact matches and semantic similarity; adds more components to tune |

Apply access and metadata filters during retrieval, before reranking. The full design guidance is in [Retrieval strategy: reranking](https://github.com/CaioHAndradeLima/AI-System-Architecturing-Engineering/blob/main/retrieval/search/retrieval-strategy-reranking.md).

## What this project compares

- **BM25** keyword retrieval as the offline baseline.
- **Dense retrieval** using Qdrant and FastEmbed, plus **hybrid retrieval** using reciprocal rank fusion (RRF).
- **Heuristic** lexical/title scoring and **MMR** relevance/diversity reranking.
- Optional **cross-encoder** reranking and **OpenAI listwise** reranking. A generic pairwise judge adapter is available for an injected comparator.
- Per-query and aggregate **Recall@K**, **MRR@K**, **nDCG@K**, mean/p50/p95 stage latency, LLM request/retry counts, token usage, and estimated API cost.

The golden set uses graded relevance: 0 is irrelevant, 1 is partial, 2 is useful, and 3 directly answers the query. The evaluation runs each reranker on the same retrieved candidates, so it can distinguish retrieval misses from ordering improvements.

For guidance on golden sets, candidate recall, ranking metrics, hard negatives, latency, and fine-tuning decisions, see [Reranker evaluation and fine-tuning](https://github.com/CaioHAndradeLima/AI-System-Architecturing-Engineering/blob/main/retrieval/search/retrieval-reranker-evaluation-and-fine-tuning.md).

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
python -m reranking_strategies.cli evaluate
```

The default run evaluates BM25, heuristic reranking, and MMR using the included data. The defaults retrieve 10 candidates and score the top 5. Save a detailed JSON report with:

```bash
python -m reranking_strategies.cli evaluate --candidate-k 20 --metric-k 5 --output reports/bm25.json
```

Both cutoffs must be positive integers. The input loaders check JSON structure, unique IDs, relevance grades, answerability, and references from query labels to documents before evaluation.

## Run dense and hybrid retrieval

Start Qdrant, install its optional client and embedding dependencies, and run the dense and hybrid comparisons:

```bash
docker compose up -d qdrant
pip install -e '.[qdrant]'
python -m reranking_strategies.cli evaluate --dense --output reports/qdrant.json
```

FastEmbed downloads the embedding model on first use. Each run replaces the configured Qdrant collection with a snapshot of the current corpus, so deleted documents do not remain searchable. Stop the service with `docker compose down`.

Run the Docker-backed reindex regression test with:

```bash
RERANKING_QDRANT_URL=http://localhost:6333 python -m unittest discover -s tests -p 'test_qdrant_store.py' -v
```

## Optional rerankers

Install a local cross-encoder and include it in the evaluation:

```bash
pip install -e '.[cross-encoder]'
python -m reranking_strategies.cli evaluate --cross-encoder
```

For OpenAI listwise ranking, install the client, set `OPENAI_API_KEY`, and add `--llm-reranker`. `--llm-model` selects the model; the default is `gpt-6-luna`.

### LLM as a judge: methods and test coverage

The project has three LLM judging patterns, each covered by deterministic unit tests:

| Method | Strategy and cost shape | Available as |
| --- | --- | --- |
| Listwise judge | One request ranks the candidate set. Low request count; prompt size grows with candidate count. | `--llm-reranker` in `evaluate`, or each model in `benchmark --live-llm`. |
| Pairwise judge | Compares pairs and sorts a shortlist. With `n` candidates the comparator can make up to `n(n-1)/2` calls, so the benchmark defaults to a top-five shortlist. | `--llm-pairwise`, or add `--with-pairwise` in live benchmark mode. |
| Bounded retry | Runs a primary listwise judge and calls one fallback model only when the primary's self-reported confidence is below the threshold. It never loops indefinitely. | `--llm-loop`, or add `--with-loop` in live benchmark mode. |

Pairwise and listwise prompts treat passage contents as untrusted, and requests set `store=False`. The retry confidence is a model-reported estimate, **not a calibrated probability**. Retrying may improve ordering, leave it unchanged, or make it worse; the evaluator measures the final ranking and counts the extra requests. Do not treat confidence as a quality label until it has been calibrated against human judgments.

The tests use fake Responses clients and fixed comparators; they verify schema validation, request accounting, cost calculation, the one-retry condition, and the call cap without making API calls. The benchmark can compare two or more models, all enabled strategies, and a no-retry versus one-retry result on the same query set:

```bash
pip install -e '.[llm-judge]'
export OPENAI_API_KEY="..."
# Baselines, both listwise models, pairwise shortlist, and primary→fallback retry.
# --max-llm-calls is a hard cap; the preflight rejects a larger worst case.
python -m reranking_strategies.cli benchmark --live-llm \
  --benchmark-models gpt-6-luna,gpt-6.1-terra \
  --with-pairwise --with-loop --max-llm-calls 200 \
  --output reports/live-llm.json

# Add dense and hybrid retrieval too (three retrieval modes, larger call budget).
python -m reranking_strategies.cli benchmark --live-llm --dense \
  --benchmark-models gpt-6-luna,gpt-6.1-terra \
  --with-pairwise --with-loop --max-llm-calls 500 \
  --output reports/live-llm-qdrant.json
```

`benchmark` always includes BM25, heuristic, and MMR. With `--live-llm`, it also includes one listwise run per model; the flags add pairwise and retry strategies. Its report includes a per-query trace and per-strategy quality, mean/p50/p95 reranker latency, request and retry counts, token counts, and estimated USD cost. It recommends the highest mean value of `--selection-metric` (nDCG by default), breaking ties by p95 latency and then estimated cost. Choose the candidate shortlist, model set, and retrieval mode before looking at held-out results; selecting and reporting on the same tiny golden set can overfit.

Cost estimates use the standard token rates recorded in [`pricing.py`](src/reranking_strategies/pricing.py), checked on 2026-10-29 against the [OpenAI API pricing page](https://developers.openai.com/api/docs/pricing). They account for cached input tokens when usage reports them. Billing can differ by service tier, model, account, and later price changes; unknown model rates are reported as unknown rather than zero.

### Measured offline baseline

The following is one local Docker-backed run on the included **six-query synthetic golden set**, `candidate_k=10`, `metric_k=5`. Ranking metrics are means across the six queries; latency is p95 for the named retrieval or reranking stage. LLM requests, tokens, and cost are zero for these non-LLM strategies. These small-corpus results are a pipeline smoke benchmark, not evidence that one method will win on another corpus.

| Retrieval mode | Strategy | Recall@5 | MRR@5 | nDCG@5 | Stage p95 (ms) | LLM requests / retries | Estimated cost |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 | Retrieval only | 0.778 | 0.667 | 0.710 | 0.018 | 0 / 0 | $0 |
| BM25 | Heuristic | 0.833 | 0.750 | **0.744** | 0.036 | 0 / 0 | $0 |
| BM25 | MMR | 0.778 | 0.667 | 0.702 | 0.509 | 0 / 0 | $0 |
| Dense | Retrieval only | 0.833 | 0.750 | 0.742 | 6.383 | 0 / 0 | $0 |
| Dense | Heuristic | 0.833 | 0.750 | **0.744** | 0.110 | 0 / 0 | $0 |
| Dense | MMR | 0.833 | 0.750 | 0.738 | 1.396 | 0 / 0 | $0 |
| Hybrid RRF | Retrieval only | 0.833 | 0.667 | 0.734 | 5.615 | 0 / 0 | $0 |
| Hybrid RRF | Heuristic | 0.833 | 0.750 | **0.744** | 0.107 | 0 / 0 | $0 |
| Hybrid RRF | MMR | 0.722 | 0.639 | 0.675 | 1.395 | 0 / 0 | $0 |

In this run, heuristic reranking had the top nDCG in all three retrieval modes. The initial candidate set capped recall, and MMR's diversity penalty hurt the hybrid result. Small timing values are machine/run dependent; repeat measurements on representative data before choosing a production strategy.

**Live LLM results (no retry vs one-retry pipeline): not measured in this checkout.** No `OPENAI_API_KEY` is configured, so there are no honest model quality, latency, token, or cost numbers to publish yet. Run the live benchmark above to populate those rows from provider usage; the report compares each model's listwise result and the retry pipeline on the same queries. In the retry result, request and cost totals include the primary and any fallback calls actually made.

## Tests

Run the offline unit suite with:

```bash
python -m unittest discover -s tests -v
```

The GitHub Actions workflow runs those tests and a Qdrant integration test against the Docker Compose service on each push and pull request. Optional cross-encoder and OpenAI calls are not required by the offline suite.

## Scope and limitations

The included corpus is synthetic and useful for controlled regression checks, not for making quality claims about a production corpus. For meaningful comparisons, build a manually reviewed golden set, keep held-out queries untouched during tuning, and measure quality and latency on representative traffic. This project evaluates ranking; it does not fine-tune or deploy a reranker.
