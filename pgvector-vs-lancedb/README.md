# pgvector vs LanceDB — reproducible vector search benchmark

Benchmark code and measured results for the article:

- EN: https://implicit-none.com/en/pgvector-vs-lancedb-benchmark/
- JA: https://implicit-none.com/ja/pgvector-vs-lancedb-benchmark/

Compares **pgvector 0.8.6** (PostgreSQL 17, HNSW) and **LanceDB 0.36.0**
(IVF_HNSW_SQ) on 100k DBpedia OpenAI embeddings (1536-dim, cosine, k=10)
across four axes: ingest + index build, recall-latency curves, concurrency
(1 vs 8 threads), and filtered search (1% / 10% selectivity).

Measured on an Apple M5 Pro (48 GB); pgvector runs in Docker capped at
8 CPUs / 6 GB. `results/` contains the JSONs behind every number in the
article.

## Reproduce

```bash
# 0. deps
python3 -m venv venv && venv/bin/pip install "lancedb==0.36.0" "psycopg[binary]" numpy pyarrow huggingface_hub tqdm matplotlib

# 1. pgvector server
docker compose up -d
docker exec pgvector-bench psql -U bench -c "CREATE EXTENSION IF NOT EXISTS vector"

# 2. data (downloads the needed HF parquet shards) + exact ground truth
venv/bin/python prepare_data.py 100000
venv/bin/python ground_truth.py 100000

# 3. run all four axes on both systems
for phase in ingest query concurrent filtered; do
  venv/bin/python bench_pgvector.py 100000 $phase
  venv/bin/python bench_lancedb.py 100000 $phase
done

# 4. charts
venv/bin/python plot.py
```

Notes:

- Vectors are L2-normalized so cosine top-k == inner-product top-k; ground
  truth is exact full-scan top-10, recomputed per filter condition.
- `bench_pgvector.py ingest` runs `ANALYZE` after the bulk load. Skipping it
  makes the planner pick HNSW + post-filter even at 1% selectivity and
  collapses filtered recall to ~0.09 — see the article.
- LanceDB's IVF_HNSW_SQ plateaus around recall 0.95 without `refine_factor`
  (scalar quantization); the sweep includes refined points to reach 0.99+.
