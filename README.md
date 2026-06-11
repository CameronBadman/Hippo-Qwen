# Hippo-Qwen

Deterministic memory infrastructure for AI agents.

Hippo-Qwen is an experimental memory layer built for agents that need to remember
across sessions without drifting, randomly changing answers, or blindly trusting
nearest-neighbor vector similarity. It combines a deterministic memory index with
a small transformer calibrator that learns which candidate memories should enter
the agent's context window.

This project targets the Qwen Cloud hackathon MemoryAgent track: persistent
memory, timely forgetting, limited-context recall, and increasingly accurate
cross-session decisions.

## Why This Exists

Most agent memory stacks are built on vector search. That is useful, but brittle:

- similar text can be the wrong memory
- stale preferences can beat newer corrections
- query-shaped decoys can rank above true evidence
- approximate indexes can return unstable results as memory grows
- agents often need a ranked context frame, not just nearest documents

Hippo-Qwen treats memory retrieval as a deterministic decision problem:

```text
same memory state + same query = same result
same mutation event + same state = same next state
```

That makes it easier to debug, benchmark, reproduce, and trust long-running
agents.

## What It Does

- Stores memory experiments and benchmarks for agent retrieval.
- Builds supervised calibration data from MemoryCraft-style memory tasks.
- Trains a compact transformer reranker over Hippo candidate neighborhoods.
- Tests against clean and adversarial memory profiles.
- Compares calibrated Hippo retrieval against FAISS/HNSW and hybrid baselines.
- Tracks recall, precision, MRR, context precision, hard-negative leakage,
  latency, and deterministic repeat mismatches.

The current best path is:

```text
query
  -> Hippo encoder / retrieval candidate set
  -> deterministic candidate ordering
  -> transformer calibrator
  -> ranked memory context
  -> Qwen agent
```

## Current Results

Latest A100 Colab run on the sanitized MemoryCraft holdout:

| scenario | system | p95 ms | recall@8 | precision@8 | context precision | hard neg@8 | MRR | determinism |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| clean | FAISS HNSW | 0.19 | 0.8801 | 0.2370 | 0.2871 | 0.0000 | 0.7338 | 0 mismatches |
| clean | Hippo calibrated | 25.25 | 0.9605 | 0.2620 | 0.4266 | 0.0000 | 1.0000 | 0 mismatches |
| query echo | FAISS HNSW | 0.22 | 0.0111 | 0.0028 | 0.1152 | 0.9907 | 0.1186 | 0 mismatches |
| query echo | Hippo calibrated | 27.89 | 0.9606 | 0.2565 | 0.4498 | 0.0167 | 1.0000 | 0 mismatches |
| mixed adversarial | FAISS HNSW | 0.21 | 0.0800 | 0.0213 | 0.1076 | 0.9500 | 0.1120 | 0 mismatches |
| mixed adversarial | Hippo calibrated | 29.96 | 0.9606 | 0.2565 | 0.4496 | 0.0167 | 1.0000 | 0 mismatches |

Candidate pool `64` is the current default. It reaches the same recall and
precision plateau as `128` and `256` while staying far below the `200 ms` query
target.

Full results: [docs/memorycraft_ablation_results_2026-06-06.md](docs/memorycraft_ablation_results_2026-06-06.md)

Hardened adversarial retest: [docs/memorycraft_hardened_retest_2026-06-11.md](docs/memorycraft_hardened_retest_2026-06-11.md)

Held-out family retest: [docs/memorycraft_heldout_family_retest_2026-06-11.md](docs/memorycraft_heldout_family_retest_2026-06-11.md)

Architecture diagram: [docs/hippo_qwen_architecture.md](docs/hippo_qwen_architecture.md)

## What It Is Good At

Hippo-Qwen is strongest when a memory system must avoid plausible but wrong
context:

- multi-session user preferences
- corrections and superseded facts
- same-entity wrong-time memories
- repeated memories and near duplicates
- answer-shaped decoys
- query-echo hard negatives
- deterministic replay of agent decisions

It is not trying to be the fastest ANN index. It is trying to be a reliable
agent memory layer where correctness, reproducibility, and context quality matter
more than sub-millisecond nearest-neighbor search.

## Repository Map

```text
python/benchmarks/
  memorycraft_retrieval.py          # memory benchmark harness
  memorycraft_ablation_suite.py     # clean/adversarial ablation runner
  build_hippo_calibration_data.py   # supervised calibration data builder
  vector_db_compare.py              # vector baseline comparison

python/librarian/
  hippo_calibrator.py               # compact transformer reranker
  train_hippo_calibrator.py         # calibrator training loop
  frame_builder.py                  # memory frame construction experiments

python/field_memory/
  token_field.py                    # token-field memory experiments
  token_encoder.py                  # prototype encoder work

docs/
  hippo_qwen_architecture.md
  memorycraft_heldout_family_retest_2026-06-11.md
  memorycraft_hardened_retest_2026-06-11.md
  memorycraft_ablation_results_2026-06-06.md
  memorycraft_ablation_suite.md
  retrieval_comparison.md
```

## Quick Start

Create an environment and install training/benchmark dependencies:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements-train.txt
```

Run a small MemoryCraft retrieval benchmark:

```bash
python -m python.benchmarks.memorycraft_retrieval \
  --hf-file selected/sample.jsonl \
  --limit-records 3 \
  --limit-questions 20 \
  --systems faiss_hnsw,hybrid_union_token
```

Build calibrator training data:

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
```

Train the compact calibrator:

```bash
python -m python.librarian.train_hippo_calibrator \
  --dataset artifacts/hippo_calibrator/memorycraft_train.jsonl \
  --output artifacts/hippo_calibrator/calibrator.pt \
  --epochs 8 \
  --batch-size 64 \
  --max-candidates 128 \
  --embedding-dim 1024 \
  --feature-dim 24
```

Run the ablation suite:

```bash
python -m python.benchmarks.memorycraft_ablation_suite \
  --hf-file selected/sample.jsonl \
  --limit-records 20 \
  --limit-questions 40 \
  --profiles clean,query_echo,mixed \
  --candidate-pools 32,64,128 \
  --calibrator relevance_only=artifacts/hippo_calibrator/calibrator.pt \
  --calibrated-systems hippo_calibrated_union \
  --work-dir artifacts/memorycraft_ablation_suite
```

## Qwen Cloud Plan

The hackathon-facing product is a Qwen-powered agent with Hippo memory behind it:

```text
Frontend / SDK / CLI
  -> Memory API
  -> Hippo deterministic memory runtime
  -> Qwen Cloud reasoning and memory summarization
  -> benchmark + trace viewer
```

Planned API surface:

- `POST /memories` - add a memory event
- `POST /query` - retrieve deterministic context for an agent turn
- `POST /feedback` - mark memories as useful, ignored, corrected, or stale
- `GET /trace/{query_id}` - inspect why a memory was selected
- `GET /health` - deployment health and model/index version

The engine should run with an append-only mutation log, snapshot-versioned reads,
stable tie-breaks, and pinned model/index versions. Qwen is used for the agent
workflow and teacher/evaluation loop; the memory engine remains deterministic by
design.

## Status

Working:

- benchmark harness
- adversarial profiles
- compact transformer calibrator
- Colab training/evaluation workflow
- checkpointed ablation outputs
- FAISS/HNSW baseline comparisons
- deterministic repeat checks

Next:

- hosted API
- Qwen Cloud integration
- CLI and Python SDK
- trace viewer
- deployment proof
- architecture diagram and demo video

## License

MIT License. Use, copying, modification, distribution, sublicensing, and
commercial use are allowed, provided the copyright and permission notice for
Cameron Badman is retained.
