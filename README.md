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
- Per-query and aggregate **Recall@K**, **MRR@K**, **nDCG@K**, and mean stage latency.

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

For OpenAI listwise ranking, install the client, set `OPENAI_API_KEY`, and add `--llm-reranker`. `OPENAI_MODEL` can select the model; the default is `gpt-4o-mini`.

### LLM as a judge: methods and test coverage

The project has two LLM judging patterns, each covered by a deterministic unit test:

| Method | Implementation | Existing test |
| --- | --- | --- |
| Pairwise comparison | [`PairwiseJudgeReranker`](src/reranking_strategies/rerankers.py) accepts an injected callback that compares two candidates for a query. Use it to plug in a custom comparator; this class does not make provider calls itself. | [`test_pairwise_adapter_uses_injected_judge`](tests/test_rerankers.py) supplies a fixed comparator and checks the ordering. |
| OpenAI listwise ranking | [`OpenAIListwiseReranker`](src/reranking_strategies/rerankers.py) sends the query and candidate IDs/titles/texts in one Responses API request, then requires each candidate ID exactly once. Use it for one structured ordering request per query; input size grows with the candidate set. | [`test_openai_listwise_adapter_validates_candidate_ids`](tests/test_rerankers.py) uses a fake Responses client; [`test_openai_listwise_rejects_missing_or_duplicate_ids`](tests/test_rerankers.py) checks invalid output. |

The CLI's `--llm-reranker` option selects the OpenAI listwise method in [`cli.py`](src/reranking_strategies/cli.py). It sets `store=False`. **There is no live-provider test in the unit suite:** OpenAI responses are replaced by a fake client, while the pairwise test injects a fixed comparator. These tests are deterministic, offline, and make no API calls. Running the CLI with `--llm-reranker` is the manual end-to-end check and requires an API key.

```bash
pip install -e '.[llm-judge]'
export OPENAI_API_KEY="..."
python -m reranking_strategies.cli evaluate --llm-reranker
```

## Tests

Run the offline unit suite with:

```bash
python -m unittest discover -s tests -v
```

The GitHub Actions workflow runs those tests and a Qdrant integration test against the Docker Compose service on each push and pull request. Optional cross-encoder and OpenAI calls are not required by the offline suite.

## Scope and limitations

The included corpus is synthetic and useful for controlled regression checks, not for making quality claims about a production corpus. For meaningful comparisons, build a manually reviewed golden set, keep held-out queries untouched during tuning, and measure quality and latency on representative traffic. This project evaluates ranking; it does not fine-tune or deploy a reranker.
