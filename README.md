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

### Retrieve and rerank graph edges

Qdrant's HNSW graph is an internal nearest-neighbor index; its links are not application relationships. This experiment stores each domain relationship as its own vector point and keeps the directed endpoints, relation label, and supporting document IDs in the point payload. Qdrant points support a vector plus optional payload ([Qdrant points](https://qdrant.tech/documentation/concepts/points/)); indexed payload fields can filter vector search ([Qdrant filtering](https://qdrant.tech/documentation/search/filtering/)).

The example edges in [`edges.json`](data/edges.json) are extracted from the sample passages. [`edge_golden.json`](data/edge_golden.json) labels relevant edges for six queries. `benchmark-edges` uses the existing BM25, dense Qdrant, and hybrid RRF retrievers, then evaluates the normal heuristic and MMR rerankers plus `EdgeStructureReranker`. That edge-aware reranker combines 75% of the candidate's normalized retrieval score with 25% lexical overlap across the query and the edge's source, relation, target, and description. This preserves most of the retriever's signal while testing whether edge fields can refine it.

Run the BM25 experiment without Docker:

```bash
python -m reranking_strategies.cli benchmark-edges \
  --output reports/edges-bm25.json
```

Run the dense and hybrid edge experiment with Qdrant:

```bash
docker compose up -d qdrant
pip install -e '.[qdrant]'
python -m reranking_strategies.cli benchmark-edges --dense \
  --output reports/edges-qdrant.json
```

Each edge retains its `evidence_ids`, so after retrieving an edge you can fetch the supporting passages. The Qdrant adapter can also filter an edge search by indexed payload fields such as its source, relation, or target:

```python
from reranking_strategies.data import load_documents, load_edges
from reranking_strategies.qdrant_store import QdrantDenseStore

source_documents = load_documents("data/documents.json")
edges = [edge.as_document() for edge in load_edges("data/edges.json", source_documents)]
edge_documents_by_id = {edge.id: edge for edge in edges}
store = QdrantDenseStore(collection="reranking_edge_demo")
store.index(edges, indexed_payload_fields=("kind", "source_id", "relation", "target_id"))

store.search(
    "Which service schedules invoice retries?",
    edge_documents_by_id,
    limit=10,
    filters={"kind": "edge", "source_id": "service:retry-worker"},
)
```

This prototype retrieves individual edges; it does not yet expand graph paths or infer entities from arbitrary questions. For multi-hop questions, the next step is to identify seed entities, expand a bounded number of adjacent edges, then rerank candidate paths for endpoint and relation fit, direction, evidence quality, and path continuity. Keep that path benchmark separate from single-edge retrieval because it measures a different task.

#### Measured edge-point baseline

One Docker-backed run, 19 edge points, six queries, `candidate_k=10`, `metric_k=5`. Quality values are means; p95 latency is for retrieval or reranking stage only. Costs are zero for these non-LLM methods. The small corpus is a smoke benchmark.

| Retrieval mode | Strategy | Recall@5 | MRR@5 | nDCG@5 | Stage p95 (ms) |
| --- | --- | ---: | ---: | ---: | ---: |
| BM25 | Retrieval only | 0.736 | 0.750 | 0.764 | 0.096 |
| BM25 | Heuristic | 0.750 | 0.667 | 0.711 | 0.050 |
| BM25 | MMR | 0.736 | 0.750 | 0.762 | 0.632 |
| BM25 | Edge structure | 0.736 | 0.750 | 0.762 | 0.047 |
| Dense | Retrieval only | 0.750 | 0.833 | 0.797 | 5.417 |
| Dense | Heuristic | 0.750 | 0.667 | 0.711 | 0.134 |
| Dense | MMR | 0.750 | 0.833 | 0.784 | 1.328 |
| Dense | Edge structure | 0.750 | 0.833 | 0.797 | 0.097 |
| Hybrid RRF | Retrieval only | 0.792 | 0.833 | **0.816** | 4.804 |
| Hybrid RRF | Heuristic | 0.750 | 0.667 | 0.711 | 0.125 |
| Hybrid RRF | MMR | 0.750 | 0.833 | 0.758 | 1.285 |
| Hybrid RRF | Edge structure | 0.792 | 0.750 | 0.770 | 0.100 |

On this tiny set, hybrid retrieval alone ranks the labeled edges best. The structural reranker tied dense retrieval and nearly tied BM25+MMR, but reduced hybrid nDCG; the ordinary heuristic also reduced nDCG. So edges are retrievable and can reuse the ranking evaluator, but these results do not support adding a reranker by default. We need more human-reviewed edge labels and multi-hop queries before drawing a broader conclusion. Live LLM edge reranking is available through `benchmark-edges --live-llm`, but has not been measured here because no API key is configured.

## Optional rerankers

Install a local cross-encoder and include it in the evaluation:

```bash
pip install -e '.[cross-encoder]'
python -m reranking_strategies.cli evaluate --cross-encoder
```

For OpenAI listwise ranking, install the client, set `OPENAI_API_KEY`, and add `--llm-reranker`. `--llm-model` selects the model; the default is `gpt-6-luna`.

### LLM as a judge: methods and test coverage

The project has three LLM judging patterns for passage ranking, covered by deterministic unit tests:

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

### Judge generated answers for correctness and assertiveness

Ranking metrics tell us whether useful passages appear near the top. They do not tell us whether a RAG system's **answer** is correct or overstates what those passages support. The separate `judge-answers` command evaluates supplied answer/evidence samples. It does not generate answers; run the same answer-generation model, prompt, and context budget for each retrieval strategy, then provide each output and its evidence IDs.

The golden query file now supports an optional `reference_answer`. The included answerable queries have initial references derived from the corpus; the unanswerable query has `reference_answer: null`. Have a human review references before treating scores as evaluation labels, and retain a held-out query set. A sample answer file looks like:

```json
[
  {
    "query_id": "q-003",
    "strategy": "dense+heuristic",
    "answer": "Security audit events are retained for one year.",
    "evidence_ids": ["doc-008"]
  },
  {
    "query_id": "q-006",
    "strategy": "dense+heuristic",
    "answer": "The supplied documents do not specify a retention period for chat recordings.",
    "evidence_ids": []
  }
]
```

Run the judge with the optional OpenAI dependency and API key:

```bash
python -m reranking_strategies.cli judge-answers \
  --answers reports/answers.json \
  --judge-model gpt-6-luna \
  --max-llm-calls 100 \
  --output reports/answer-judge.json
```

The judge returns per-sample and per-strategy mean scores on a **0–4 scale** for:

| Score | What it measures | Score of 4 |
| --- | --- | --- |
| Correctness | Agreement with the reference answer | Correct and complete for the query |
| Groundedness | Support for factual claims in the supplied evidence | All material claims are supported |
| Assertiveness calibration | Whether certainty is proportional to evidence; low scores penalize confident unsupported claims and also needless hedging when evidence is clear | Claims are appropriately qualified |
| Abstention correctness | Whether an answer abstains for an unanswerable query and answers an answerable one | Correct decision; reported as a rate from 0 to 1 |

The judge also returns unsupported claims, a short rationale, latency, request count, input/output tokens, and estimated cost. `assertiveness_calibration` measures evidence-relative overclaiming, not how confident or persuasive the writing sounds. Correctness and groundedness are separate: an answer may match the reference but lack support in the passages actually provided. The judge does not see the strategy label, reducing one source of scoring bias.

These are model judgments, not ground truth. Blindly sample outputs for human review, compare judge scores with human labels, and check disagreement before using the scores to select a production system. Keep the judge model fixed across strategies; where practical, use a judge different from the answer generator. **Live answer-judge scores have not been collected in this checkout** because no `OPENAI_API_KEY` is configured.

## Tests

Run the offline unit suite with:

```bash
python -m unittest discover -s tests -v
```

The GitHub Actions workflow runs those tests and a Qdrant integration test against the Docker Compose service on each push and pull request. Optional cross-encoder and OpenAI calls are not required by the offline suite.

## Scope and limitations

The included corpus is synthetic and useful for controlled regression checks, not for making quality claims about a production corpus. For meaningful comparisons, build a manually reviewed golden set, keep held-out queries untouched during tuning, and measure quality and latency on representative traffic. This project evaluates ranking; it does not fine-tune or deploy a reranker.
