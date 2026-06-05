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

## Reset Boundary

No production memory database runtime is currently present. The next runtime
should start with a fresh storage contract instead of extending the deleted
vector-index app.
