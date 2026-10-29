# Reranking Strategies

An engineering lab for answering a practical RAG question: **which retrieval and reranking pipeline gives the best evidence for the least latency and cost?**

The project runs keyword, vector, and hybrid retrieval against the same small corpus, applies several reranking methods to the retrieved candidates, and reports ranking quality alongside latency and LLM usage. It is intended for learning and controlled experiments. It is not a hosted RAG service, a production capacity benchmark, or a fine-tuning framework.

## Start here

### Results achieved using different reranking strategies

These are the measured passage-ranking results from one local run over six synthetic queries. The evaluator retrieved 10 candidates and scored the top five. Each reranker received the same candidates within its retrieval mode.

| Strategy | Approach | nDCG@5: BM25 / dense / hybrid | Stage p95 ms: BM25 / dense / hybrid | Result in this run |
| --- | --- | ---: | ---: | --- |
| Retrieval only | Keep the order returned by BM25, dense search, or hybrid RRF. | 0.710 / 0.742 / 0.734 | 0.018 / 6.383 / 5.615 | Baseline. Dense retrieval led the individual modes; hybrid was close. |
| Heuristic | Reorder by lexical overlap and title match. | **0.744 / 0.744 / 0.744** | 0.036 / 0.110 / 0.107 | Highest measured nDCG in all three modes. |
| MMR | Balance relevance against similarity to already selected passages. | 0.702 / 0.738 / 0.675 | 0.509 / 1.396 / 1.395 | Did not improve on retrieval only; its diversity penalty lowered hybrid nDCG. |
| Edge structure | Mix retrieval score (75%) with lexical overlap across edge endpoints, relation, and description (25%). | See edge result below | See edge result below | On 19 edges, it did not beat hybrid retrieval without reranking. |

The separate graph-edge run used six queries and 19 edge points. Hybrid retrieval alone scored **0.816 nDCG@5**; the edge-structure reranker scored **0.770** on the same retrieval mode. This experiment does not support enabling that reranker by default.

Cross-encoder, listwise LLM, pairwise LLM, and bounded fallback methods are implemented, but their live/model-backed quality results have not been measured in this checkout. The LLM tests use fakes, so they verify request handling and accounting without making paid API calls. See the detailed [passage results](#passage-benchmark-result), [edge results](#edge-benchmark-result), and [LLM benchmark instructions](#live-llm-ranking-benchmark).

These results come from tiny synthetic datasets and one local run; they do not predict performance on another corpus. Reranking cannot recover evidence retrieval missed, so compare candidate recall as well as ranking metrics.

### Quick start

Requires Python 3.10 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
python -m reranking_strategies.cli evaluate
```

The default evaluation compares BM25 retrieval with retrieval-only ranking, heuristic reranking, and MMR. It uses the included corpus and labels, retrieves 10 candidates, and evaluates the top five. To save a detailed JSON report:

```bash
python -m reranking_strategies.cli evaluate \
  --candidate-k 20 --metric-k 5 \
  --output reports/bm25.json
```

## How the evaluation works

Each query has a set of relevant documents with graded labels: `0` irrelevant, `1` partially relevant, `2` useful, and `3` directly answers the query. For each retrieval mode, the evaluator gathers the candidate set once, then applies each reranker to those same candidates. This makes it possible to see whether a miss happened during retrieval or whether a relevant candidate was simply ranked too low.

The report includes per-query results as well as aggregate metrics:

| Metric | What it tells you |
| --- | --- |
| Recall@K | How much of the labeled relevant evidence appears in the top K. Low recall usually points to retrieval or candidate-depth problems. |
| MRR@K | How early the first relevant result appears. Higher values mean users or generators encounter useful evidence sooner. |
| nDCG@K | Whether the top K results are ordered well, accounting for the graded relevance labels. This is the default selection metric. |
| Mean, p50, p95 stage latency | How long retrieval or reranking took in the measured run. Timing depends on the machine, corpus, warm-up, and provider. |
| LLM requests, retries, tokens, estimated cost | The usage reported by live model calls. Offline strategies have zero LLM usage. |

Do not compare ranking scores without also checking candidate recall: a reranker can only reorder candidates it receives. Use a manually reviewed golden set, hold out queries during tuning, and repeat latency measurements on representative data. See the companion guides on [reranking strategy](https://github.com/CaioHAndradeLima/AI-System-Architecturing-Engineering/blob/main/retrieval/search/retrieval-strategy-reranking.md) and [reranker evaluation and fine-tuning](https://github.com/CaioHAndradeLima/AI-System-Architecturing-Engineering/blob/main/retrieval/search/retrieval-reranker-evaluation-and-fine-tuning.md).

## Retrieval options

### BM25 baseline

BM25 ranks passages by lexical match. It is a useful no-service baseline and often handles exact terms, names, and identifiers well.

```bash
python -m reranking_strategies.cli evaluate \
  --output reports/bm25.json
```

### Dense and hybrid retrieval

Dense retrieval embeds queries and passages, then searches Qdrant. Hybrid retrieval combines dense and BM25 rankings with reciprocal rank fusion (RRF), which can balance semantic matches with exact terms.

```bash
docker compose up -d qdrant
pip install -e '.[qdrant]'
python -m reranking_strategies.cli evaluate --dense \
  --output reports/qdrant.json
```

FastEmbed downloads its embedding model on first use. The evaluation adapter replaces the configured demo collection with the current corpus snapshot, including removing stale points. Stop Qdrant with `docker compose down`.

## Strategy guide: what to use and when

Use retrieval to find a candidate set; use reranking to improve that set's order. Start with a retrieval-only baseline, then add the least expensive strategy that addresses the measured failure.

| Strategy | What it does | Use it when | Trade-off and this project's evidence |
| --- | --- | --- | --- |
| **BM25 retrieval** | Finds candidates by matching query and passage terms. | Exact terms, names, IDs, and error codes matter. | Fast and easy to inspect; misses paraphrases. Included as the offline baseline. |
| **Dense retrieval** | Finds candidates by vector similarity between query and passage meaning. | The query and source use different wording. | Captures semantic similarity; may miss exact identifiers. This project uses Qdrant and FastEmbed. |
| **Hybrid retrieval (RRF)** | Combines BM25 and dense rankings with reciprocal rank fusion. | Both exact terms and semantic matches matter. | Adds a component to tune. It achieved the highest edge nDCG in the small edge run, before reranking. |
| **Retrieval only** | Passes the retriever's order through unchanged. | You need a baseline to determine whether reranking helps. | No additional model cost or reranking stage. It is not itself a reranker. |
| **Heuristic reranking** | Reorders by lexical overlap and title matches. | You need a cheap, transparent first reranker. | Limited semantic understanding. It had the best measured passage nDCG in the included six-query run. |
| **MMR** | Balances relevance with diversity among selected passages. | Top results repeat the same fact and broader coverage is useful. | Diversity can push down multiple relevant passages. It lowered nDCG in the included run. |
| **Cross-encoder** | Scores each query-passage pair jointly with a local model. | A general reranker is not accurate enough and the candidate set is small. | More compute per candidate than vector retrieval; evaluate on the actual shortlist. Implemented, but not measured in the published result table. |
| **Listwise LLM judge** | Gives a candidate set to an LLM and asks it to order the passages. | You want flexible reasoning over a small shortlist. | One request can rank several items, but prompt size grows with candidate text. Live quality results are not yet recorded. |
| **Pairwise LLM judge** | Asks an LLM to choose between candidates, then sorts from the comparisons. | Head-to-head judgments are valuable on a very short list. | Up to `n(n-1)/2` calls for `n` candidates; five candidates can mean ten comparisons. Live quality results are not yet recorded. |
| **Bounded fallback** | Calls a primary listwise model and retries once with a fallback under a condition. | You want to test whether a second attempt helps harder cases. | Adds latency and cost. Current trigger is model-reported confidence, not a calibrated probability; live quality results are not yet recorded. |
| **Edge-structure reranking** | Mixes retrieval score (75%) with lexical overlap across edge endpoints, relation, and description (25%). | The retrieved records are graph relationships and their structure matters. | Did not improve the small hybrid edge result: nDCG@5 was 0.770 versus 0.816 for hybrid retrieval alone. |

To add the optional cross-encoder:

```bash
pip install -e '.[cross-encoder]'
python -m reranking_strategies.cli evaluate --cross-encoder
```

### Should we fine-tune a reranker?

**Not yet for this sample corpus.** Fine-tuning is useful when a general cross-encoder still makes important, repeatable mistakes on a real target domain and you have enough reviewed examples to train and test against. This project's six synthetic queries are far too few, and the current benchmark has not established that a cross-encoder is the bottleneck.

Fine-tuning changes the model using examples of the relevance behavior you want. For a cross-encoder reranker, each example is a `(query, passage)` pair with a relevance label. A practical dataset should include:

- Positive passages that genuinely answer or support the query.
- **Hard negatives** returned by the same first-stage retriever that look relevant but do not answer the query.
- Human-reviewed relevance labels. Do not treat every unclicked or unlabeled passage as irrelevant.
- Query-disjoint train, validation, and test splits, so passages for one query do not leak across splits.

With binary labels, train a scalar cross-encoder score using a binary classification loss. With preference or graded labels, use a ranking objective that teaches the model which passage should be above another. Sentence Transformers documents [cross-encoder training](https://sbert.net/docs/cross_encoder/training_overview.html), [reranking evaluators](https://sbert.net/docs/package_reference/cross_encoder/evaluation.html), and [training examples for rerankers](https://sbert.net/examples/cross_encoder/training/rerankers/README.html). Training should use negatives drawn from the candidates that production retrieval actually returns; otherwise the training task and serving task differ.

For this project, the next implementation would be an optional `scripts/train_cross_encoder.py` plus a reviewed query/passage training dataset. It would save a versioned local checkpoint, then reuse the existing `--cross-encoder-model` option to compare that checkpoint with the original model using the same retrieval candidates and held-out queries. The final comparison should include nDCG/MRR, candidate recall, p95 latency, model size, and inference cost. **This repository does not currently include a training script or real training labels**; it can evaluate a cross-encoder, including a compatible local model path, but training is future work.

Fine-tune only if the held-out evaluation shows a meaningful quality gain that is worth the additional training, hosting, and maintenance cost. If retrieval is missing the evidence, fix retrieval or candidate depth first. If the general reranker already meets the quality and latency targets, fine-tuning adds complexity without a demonstrated benefit.

### Live LLM ranking benchmark

The `benchmark` command compares baselines, multiple listwise models, and optional pairwise and fallback strategies. Install the client and set an API key only when you want to make live, billable calls:

```bash
pip install -e '.[llm-judge]'
export OPENAI_API_KEY="..."
python -m reranking_strategies.cli benchmark --live-llm \
  --benchmark-models gpt-6-luna,gpt-6.1-terra \
  --with-pairwise --with-loop --max-llm-calls 200 \
  --output reports/live-llm.json
```

Add `--dense` to compare BM25, dense, and hybrid retrieval in the same benchmark; use a larger `--max-llm-calls` because the enabled strategies are evaluated for each retrieval mode. The call cap is checked before requests are sent. The report includes per-query traces, ranking metrics, mean/p50/p95 reranker latency, calls, retries, token usage, and estimated USD cost. The recommended strategy maximizes the selected metric (nDCG by default), then breaks ties by p95 latency and cost.

The pairwise option defaults to a top-five shortlist; comparing every pair among five items can take up to 10 comparator calls for a query. The fallback option retries at most once. Its trigger uses the model's self-reported confidence, which is **not a calibrated probability**. A retry can improve or worsen the ranking; compare its final metrics with the no-retry result.

LLM prompts treat passage text as untrusted input, and API requests set `store=False`. Cost calculations use the standard model rates in [`pricing.py`](src/reranking_strategies/pricing.py) and account for cached input when usage reports it. Model, service-tier, and regional rates can change; unknown rates are reported as unknown rather than zero.

## LLM judge for generated answers

Ranking metrics measure where relevant evidence appears. They do not tell you whether a generated answer is correct or whether it claims more than the evidence supports. The separate `judge-answers` command scores answers that you supply; it does **not** generate answers. To compare retrieval strategies fairly, generate answers with the same answer model, prompt, and context budget, then provide each answer and the evidence IDs it used.

The answer judge reports:

| Score | Meaning |
| --- | --- |
| Correctness | Agreement with the reference answer |
| Groundedness | Whether the supplied evidence supports the answer's material claims |
| Assertiveness calibration | Whether the answer's certainty matches the available evidence |
| Abstention correctness | Whether it answers answerable questions and abstains on unanswerable ones |

The first three scores use a 0–4 scale; abstention correctness is reported as a rate from 0 to 1. The report also includes unsupported claims, rationale, latency, requests, tokens, and estimated cost. These are model judgments, not ground truth. Review references and sample outputs with people, compare judge scores to human labels, and inspect disagreements before using the scores to choose a system.

Example answer input:

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

Run the judge with the `llm-judge` extra and `OPENAI_API_KEY` configured:

```bash
python -m reranking_strategies.cli judge-answers \
  --answers reports/answers.json \
  --judge-model gpt-6-luna \
  --max-llm-calls 100 \
  --output reports/answer-judge.json
```

The included answerable queries have initial reference answers derived from the corpus; the unanswerable query has no reference answer. Have a human review these labels before treating them as evaluation ground truth. No live answer-judge results are included in this repository yet.

### Calibrate judge scores against reviewer ratings

The `calibrate-judge` command tests whether numeric judge scores track ratings from reviewers. Give it paired scores for each answer and dimension. Keep some labeled answers in a **calibration** split to fit the mapping, and a separate **validation** split to measure it. The command fits a monotonic isotonic mapping independently for correctness, groundedness, and assertiveness, then compares raw and adjusted score error on held-out ratings. It also reports how much reviewers disagreed with one another. The mapping adjusts score scale; it does not rewrite answers or make an unreliable judge trustworthy.

A paired rating row looks like this:

```json
{
  "sample_id": "q-003::dense+heuristic",
  "dimension": "groundedness",
  "split": "calibration",
  "judge_score": 3,
  "human_scores": [2, 3, 2]
}
```

Store rows in a JSON object with `label_source: "human"` and a `records` array. Each row contains the LLM score and independent reviewer scores for the same answer and dimension. Use `calibration` and `validation` labels assigned before fitting. Each dimension needs at least five calibration rows, two validation rows, and two different judge-score values in the calibration split; a real study should use substantially more and cover the range of answer quality.

```bash
python -m reranking_strategies.cli calibrate-judge \
  --ratings data/human_judge_ratings.json \
  --output reports/judge-calibration.json
```

The report compares raw and calibrated mean absolute error, exact agreement, within-one-point agreement, and average judge bias on the held-out split. Review the reviewer-to-reviewer disagreement too: if people interpret the rubric differently, judge agreement has no single reliable target. Keep one final test set untouched while revising the rubric or judge prompt.

To try the pipeline before collecting reviewer data, this repository includes a **fictional** dataset. Generate it and run the calibration:

```bash
python scripts/generate_judge_calibration_demo.py
python -m reranking_strategies.cli calibrate-judge \
  --ratings data/judge_calibration_demo.json \
  --output reports/synthetic-judge-calibration.json
```

The demo invents latent scores, a judge with a known one-point bias, and three fictional reviewer-severity profiles. Its report is marked `synthetic_demo`; any improvement only shows the algorithm can recover this constructed bias. It is **not human data, not judge validation, and must not be presented as evidence** that a live model agrees with people. Replace it with independently collected ratings before making a quality claim.

## Graph-edge retrieval experiment

The project also tests retrieving application relationships as evidence. An edge is stored as its own vector point with directed source and target IDs, relation, description, and supporting passage IDs. This is separate from the HNSW links Qdrant uses internally to search vectors. See [Qdrant points](https://qdrant.tech/documentation/concepts/points/) and [payload filtering](https://qdrant.tech/documentation/search/filtering/).

The small example dataset is in [`edges.json`](data/edges.json), with six labeled queries in [`edge_golden.json`](data/edge_golden.json). Run the BM25 experiment locally or include dense/hybrid retrieval with Qdrant:

```bash
python -m reranking_strategies.cli benchmark-edges \
  --output reports/edges-bm25.json

docker compose up -d qdrant
pip install -e '.[qdrant]'
python -m reranking_strategies.cli benchmark-edges --dense \
  --output reports/edges-qdrant.json
```

After retrieving an edge, its `evidence_ids` identify passages to fetch as support. Searches can also filter on indexed fields such as source, relation, and target. The current implementation retrieves and reranks individual edges; it does not identify entities from arbitrary questions or expand multi-hop paths. A path-search experiment should have its own labels and metrics because it is a different task.

### Edge benchmark result

One local Docker run over 19 edge points and six queries (`candidate_k=10`, `metric_k=5`). Quality metrics are means; p95 latency is for the named stage. This is a smoke benchmark, not a performance claim.

| Retrieval | Ranking | Recall@5 | MRR@5 | nDCG@5 | Stage p95 (ms) |
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

On this dataset, hybrid retrieval alone ranks the labeled edges best. The structural reranker tied dense retrieval but lowered hybrid nDCG. There is not enough evidence to recommend an edge reranker by default; more reviewed edge labels and multi-hop queries are needed. Live LLM edge ranking is available with `benchmark-edges --live-llm`, but has not been measured here.

## Passage benchmark result

One local Docker-backed run on the included six-query synthetic set (`candidate_k=10`, `metric_k=5`). Quality metrics are means; p95 latency is for the retrieval or reranking stage. The LLM columns are zero because these strategies do not call an LLM.

| Retrieval | Ranking | Recall@5 | MRR@5 | nDCG@5 | Stage p95 (ms) | LLM calls / retries | Estimated cost |
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

Heuristic reranking had the highest measured nDCG in each retrieval mode in this run. MMR did not improve the result and reduced hybrid nDCG. The candidate set also limits the best possible reranker score. These small timing values depend on the machine and run; repeat the experiment on a representative corpus before making a design choice. Live LLM quality, latency, and cost results are not available because no API results have been recorded.

## Scalability and monthly cost estimate

One million requests in a 30-day month averages about **0.39 requests/second**; a 10x peak is about 3.9 requests/second. That average alone does not size a service. Corpus size, vector dimensions, filters, candidate depth, bursts, ingestion, latency targets, and additional LLM calls all affect capacity.

This checkout runs a single local Qdrant node and benchmarks only six queries and 19 edges. It has not been load-tested for production throughput or high availability. Its evaluation replaces the demo collection from a corpus snapshot; a production service should separate incremental ingestion from query serving. Validate any deployment using production-sized vectors and payloads, peak traffic, concurrent writes, and failure/recovery tests. Measure p95/p99 latency, errors, CPU, memory, disk I/O, and recall.

The following is a **compute-only planning estimate**, not a production architecture. It assumes one always-on Linux VM hosting self-managed Qdrant, 2 vCPU, 8 GiB RAM, 730 hours/month, on-demand rates, USD, and approximate list prices checked 2026-10-29. Rates vary by region and OS. AWS `m7i.large` specifications are listed by [AWS](https://aws.amazon.com/ec2/instance-types/m7i/) and its rate should be checked with the [AWS pricing page/calculator](https://aws.amazon.com/ec2/pricing/on-demand/). GCP price is from [Compute Engine pricing](https://cloud.google.com/products/compute/pricing/general-purpose); Azure East US price is from [Linux VM pricing](https://azure.microsoft.com/en-us/pricing/details/virtual-machines/linux/) and its [Retail Prices API](https://learn.microsoft.com/en-us/rest/api/cost-management/retail-prices/azure-retail-prices).

| Provider and example VM | VM for 730 hours | With 1M GPT-6 Luna listwise calls | With 1M GPT-6.1 Terra listwise calls |
| --- | ---: | ---: | ---: |
| AWS `m7i.large` | $73.58 | $273.58 | $2,073.58 |
| GCP `e2-standard-2` | $48.92 | $248.92 | $2,048.92 |
| Azure `Standard_D2s_v5` (East US) | $70.08 | $270.08 | $2,070.08 |

The model estimates assume one LLM call per user request with 1,500 uncached input tokens and 100 output tokens, no retries, and standard short-context rates from the [OpenAI API pricing page](https://developers.openai.com/api/docs/pricing), checked 2026-10-29. That is 1.5B input and 100M output tokens/month: about **$200 with GPT-6 Luna** or **$2,000 with GPT-6.1 Terra**. A Luna primary with a Terra fallback on 10% of requests adds about $200 in this token scenario (about $400 total model spend). A judge called in addition to ranking also adds tokens and requests. Actual prompts include candidate text, so use measured token counts for a real forecast.

Pairwise ranking over a five-item shortlist can make up to 10 comparisons per query. If every comparison used the same token allowance above, one million queries would cost about **$2,000 with Luna or $20,000 with Terra**, before infrastructure. Actual pairwise prompts may be smaller; call limits and early stopping may reduce usage.

The VM estimates exclude storage, snapshots, replicas or a multi-node cluster, managed vector database fees, API/worker compute, load balancers, logs and monitoring, egress, backups, support, taxes, and discounts. A single VM has no high availability. Use the linked provider calculators and real token/traffic measurements for a deployment-specific budget.

## Project layout

| Path | Contents |
| --- | --- |
| `src/reranking_strategies/retrieval.py` | BM25, dense search integration, and RRF fusion |
| `src/reranking_strategies/rerankers.py` | Heuristic, MMR, cross-encoder, listwise, pairwise, retry, and edge ranking |
| `src/reranking_strategies/evaluation.py`, `metrics.py` | Evaluation orchestration and ranking metrics |
| `src/reranking_strategies/answer_judging.py` | LLM-based evaluation of generated answers |
| `src/reranking_strategies/judge_calibration.py` | Isotonic score calibration and held-out comparison against reviewer ratings |
| `src/reranking_strategies/qdrant_store.py` | Qdrant indexing, filtering, and search adapter |
| `data/` | Example passages, graph edges, and labeled query sets |
| `scripts/generate_judge_calibration_demo.py` | Reproducible synthetic ratings for exercising the calibration pipeline |
| `tests/` | Offline unit tests and optional Qdrant integration tests |

## Tests

Run the offline suite:

```bash
python -m unittest discover -s tests -v
```

Run the Qdrant integration tests against the Docker service:

```bash
docker compose up -d qdrant
RERANKING_QDRANT_URL=http://localhost:6333 \
  python -m unittest discover -s tests -p 'test_qdrant_store.py' -v
docker compose down
```

GitHub Actions runs the unit and Qdrant integration suites on pushes and pull requests. Tests use fake model clients; live OpenAI calls are optional and are not part of the test suite.

## What this project does not claim

- The synthetic golden set is not representative of a production workload.
- The benchmark numbers do not prove that a reranker will improve another corpus.
- LLM judge scores are not human ground truth and need calibration and review.
- The single-node local Qdrant setup is not a highly available deployment.
- The project evaluates retrieval and ranking; it does not generate answers, operate a RAG application, or fine-tune models.
