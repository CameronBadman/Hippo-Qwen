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

## Current Experiment

`python/benchmarks/rope_delta_grid.py` tests a deterministic rope delta grid:

- each memory is represented once per 3D layer
- coordinates are quantized deltas from the first memory embedding
- grid cells are stored as a binary arena and linked as per-layer ropes
- node records are binary arena entries with prev/next pointers inside each cell
- graph edges are stored in a separate binary arena and expanded after grid seed retrieval
- payload text is read lazily from binary payload files after ranking

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

512-dimensional hash embedding check:

- `--dim-count 512 --layers 96 --radius 0 --min-layer-delta 0.01`
- 10k pool, 2k growth, one case across all hard growth scenarios
- context recall `1.0` and context precision `1.0`
- baseline p95 query latency about `69 ms`
- worst growth p95 query latency about `204 ms` in `combined`
- worst growth node-record reads about `197k`

Conclusion: 512 dimensions can keep accuracy, but the current one-node-per-layer
arena layout needs a stronger layer sampler or posting cap before it is a good
default at high growth.

## Reset Boundary

No production memory database runtime is currently present. The next runtime
should start with a fresh storage contract instead of extending the deleted
vector-index app.
