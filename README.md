# Hippo-qwen

This repository has been reset for a new memory engine design.

The previous Go vector/graph database runtime, HTTP server, web UI, and vector
index have been removed. The remaining Python code is research scaffolding for
synthetic data, selector experiments, training, and benchmarks while the next
storage/retrieval architecture is designed from scratch.

## Remaining Areas

- `python/synthetic`: synthetic memory and retrieval-case generation
- `python/selector`: selector, calibration, stress, and evolution experiments
- `python/librarian`: model and feature experiments
- `python/benchmarks`: benchmark harnesses and prior experiment references
- `requirements-train.txt`: Python training dependencies

## License

This project is released under the MIT License. Use, copying, modification,
distribution, sublicensing, and commercial use are allowed, provided the
copyright and permission notice for Cameron Badman is retained.

## Current Experiment

`python/benchmarks/rope_delta_grid.py` tests a deterministic rope delta grid:

- each memory is represented once per active 3D layer
- coordinates are quantized deltas from the first memory embedding
- high-dimensional runs can spread layers across the embedding instead of using
  consecutive triples
- node/query layer selection is deterministic and based on per-layer delta energy
- grid cells are stored as a binary arena and linked as per-layer ropes
- node records are binary arena entries with prev/next pointers inside each cell
- graph edges are stored in a separate binary arena and expanded after grid seed retrieval
- payload text is read lazily from binary payload files after ranking
- cell-local postings can be read with a deterministic scan cap for large-memory runs
- JSON is not used for hot-path node, edge, or payload reads; current JSON files are
  benchmark output/debug metadata

Latest local hard run:

```bash
python -m python.benchmarks.rope_delta_grid \
  --cases 3 \
  --pool-size 10000 \
  --growth-count 2000 \
  --growth-scenarios unrelated,semantic_decoy,conflict,repeated,combined \
  --determinism-repeats 2 \
  --cell-width 0.03125 \
  --radius 0 \
  --layers 12 \
  --min-layer-delta 0.02
```

Result summary: deterministic rebuild/search, context recall `1.0`, context
precision `1.0`, baseline p95 query latency about `32 ms`, and worst growth
p95 query latency about `70 ms` on the local hash-embedding benchmark.

Sparse 512-dimensional hash embedding check:

- `--dim-count 512 --layers 96 --layer-schedule spread --radius 0`
- `--min-node-layer-delta 0.01 --max-node-layers 24`
- `--min-layer-delta 0.01 --min-query-layers 8 --max-query-layers 24`
- 10k pool, 2k growth, one case across all hard growth scenarios
- context recall `1.0` and context precision `1.0`
- baseline p95 query latency about `45 ms`
- worst growth p95 query latency about `178 ms` in `repeated`
- worst growth node-record reads about `172k`

Sparse 1024-dimensional hash embedding check:

- `--dim-count 1024 --layers 128 --layer-schedule spread --radius 0`
- `--min-node-layer-delta 0.0075 --max-node-layers 24`
- `--min-layer-delta 0.0075 --min-query-layers 8 --max-query-layers 24`
- 10k pool, 2k growth, one case across all hard growth scenarios
- context recall `1.0` and context precision `1.0`
- baseline p95 query latency about `18 ms`
- worst growth p95 query latency about `132 ms` in `combined`
- worst growth node-record reads about `122k`

100k sparse 1024-dimensional scale check:

- `python/benchmarks/rope_delta_grid_scale.py`
- `--pool-size 100000 --growth-count 10000 --growth-scenarios combined`
- `--dim-count 1024 --layers 128 --layer-schedule spread`
- `--min-node-layer-delta 0.0075 --max-node-layers 24`
- `--min-layer-delta 0.0075 --min-query-layers 8 --max-query-layers 24`
- `--max-cell-scan 4096`
- baseline: context recall `1.0`, context precision `1.0`, p95 query latency
  about `41 ms`, and about `28.8k` node-record reads
- combined growth to 110k memories: context recall `1.0`, context precision
  `1.0`, p95 query latency about `81 ms`, and `65,536` node-record reads
- deterministic repeated search output with zero mismatches
- combined-growth binary index footprint about `138 MB`

75k vector-search comparison:

- `python/benchmarks/vector_db_compare.py`
- compares Python exact cosine scan, FAISS flat inner-product search, FAISS
  HNSW, hnswlib HNSW, and the bounded rope grid on the same synthetic
  agent-memory workload
- `--pool-size 75000 --growth-count 7500 --growth-scenarios combined`
- `--dim-count 1024 --max-cell-scan 4096 --repeat-searches 3`
- Python exact scan: baseline p95 about `2979 ms`, combined-growth p95 about
  `3275 ms`, and context recall/precision `0.0` in this workload
- FAISS flat: baseline p95 about `23 ms`, combined-growth p95 about `19 ms`,
  and context recall/precision `0.0` in this workload
- FAISS HNSW: baseline p95 about `0.52 ms`, combined-growth p95 about
  `0.31 ms`, and context recall/precision `0.0` in this workload
- hnswlib: baseline p95 about `0.27 ms`, combined-growth p95 about `0.34 ms`,
  and context recall/precision `0.0` in this workload
- bounded rope grid: baseline p95 about `42 ms`, combined-growth p95 about
  `59 ms`, context recall `1.0`, context precision `1.0`, and deterministic
  repeated output
- combined-growth rope index footprint about `95 MB`

Conclusion: sparse deterministic layer selection is the better high-dimensional
direction than a dense grid or ball-tree style index. It keeps growth-stable
ordering in these synthetic cases while staying under the 200 ms query target at
512, 1024, and the tested 100k-memory scale. The 100k test showed the important
failure mode was unbounded cell fan-out, not JSON serialization; bounded
deterministic posting scans fixed the query-time blow-up in the current
synthetic workload.

Public memory benchmark harness:

- `python/benchmarks/memorycraft_retrieval.py`
- loads MemoryCraft from Hugging Face or a local JSONL file
- supports `selected/sample.jsonl` and `full/longmemeval.jsonl`
- compares Python exact cosine scan, FAISS flat, FAISS HNSW, hnswlib HNSW, and
  the Hippo rope grid
- scores against explicit evidence IDs with recall@k, precision@k, MRR,
  context recall/precision under a token budget, latency, index size, and
  deterministic repeat mismatches
- `--unit auto` uses turn-level evidence when present; for LongMemEval rows
  where evidence labels name sessions, it uses answer-bearing turns when the
  dataset exposes `metadata.has_answer`

Example:

```bash
uv --cache-dir /tmp/uv-cache run \
  --with huggingface_hub \
  --with faiss-cpu \
  --with hnswlib \
  python -m python.benchmarks.memorycraft_retrieval \
  --hf-file full/longmemeval.jsonl \
  --limit-records 20 \
  --limit-questions 1 \
  --systems exact_vector,faiss_flat,faiss_hnsw,hnswlib,hippo_rope_grid \
  --output-json artifacts/memorycraft_retrieval/result.json \
  --output-md artifacts/memorycraft_retrieval/result.md
```

Hippo calibration transformer:

- `python/librarian/hippo_calibrator.py` defines a small transformer reranker
  over Hippo candidate neighborhoods
- `python/benchmarks/build_hippo_calibration_data.py` builds supervised
  MemoryCraft rows from raw Hippo retrieval, including base rank/score and
  evidence labels
- `python/librarian/train_hippo_calibrator.py` trains the reranker with BCE plus
  pairwise ranking loss
- `python/benchmarks/memorycraft_retrieval.py` supports `hippo_calibrated` when
  `--calibrator-checkpoint` is provided
- `python/benchmarks/memorycraft_ablation_suite.py` runs clean/adversarial
  MemoryCraft ablations across FAISS/HNSW, raw hybrid union, calibrated Hippo
  union, calibrator checkpoints, and candidate-pool sizes; see
  `MEMORYCRAFT_ABLATION_README.md` and `docs/memorycraft_ablation_suite.md`

Colab-oriented flow:

```bash
python -m python.benchmarks.build_hippo_calibration_data \
  --hf-file selected/sample.jsonl \
  --limit-records 20 \
  --limit-questions 80 \
  --max-candidates 128 \
  --final-fetch 128 \
  --inject-missing-relevant \
  --synthetic-hard-negatives 24 \
  --synthetic-hard-negative-weight 3.0 \
  --output artifacts/hippo_calibrator/memorycraft_train.jsonl

python -m python.librarian.train_hippo_calibrator \
  --dataset artifacts/hippo_calibrator/memorycraft_train.jsonl \
  --output artifacts/hippo_calibrator/calibrator.pt \
  --epochs 8 \
  --batch-size 64 \
  --max-candidates 128 \
  --embedding-dim 1024 \
  --feature-dim 24

python -m python.benchmarks.memorycraft_retrieval \
  --hf-file selected/sample.jsonl \
  --limit-records 3 \
  --limit-questions 20 \
  --systems hippo_rope_grid,hippo_calibrated \
  --calibrator-checkpoint artifacts/hippo_calibrator/calibrator.pt
```

## Reset Boundary

No production memory database runtime is currently present. The next runtime
should start with a fresh storage contract instead of extending the deleted
vector-index app.
