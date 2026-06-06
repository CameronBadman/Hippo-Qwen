# Hippo MemoryCraft Ablation README

This is the main benchmark workflow for proving the Hippo memory-retrieval
claim:

> Hippo is a deterministic agent-memory retrieval layer that can preserve
> evidence recall and reduce hard-negative memory failures better than plain
> vector retrieval.

The benchmark is not trying to show that Hippo is faster than FAISS/HNSW.
FAISS/HNSW is expected to win raw nearest-neighbor latency. The point is to test
whether Hippo's candidate union plus calibration transformer improves retrieval
quality for agent memory, especially when the memory store contains plausible
but wrong decoys.

## What Gets Compared

The ablation suite compares these systems:

- `faiss_hnsw`: direct vector-search baseline.
- `hybrid_union_token`: FAISS/HNSW plus deterministic token-field candidate
  union, without the transformer.
- `hippo_calibrated_union`: the same union reranked by the Hippo calibration
  transformer.

It can also compare multiple calibrator checkpoints:

- `relevance_only`: transformer trained only on relevance ranking.
- `full_include_balanced`: relevance plus include-head objective and hard
  negative pressure.
- `no_synth_include`: include-head model trained without synthetic hard
  negatives.

## What The Adversarial Profiles Mean

The suite injects deterministic hard negatives into the held-out MemoryCraft
records. Each profile targets a common chatbot-memory failure mode:

- `clean`: no injected decoys.
- `query_echo`: false memories that repeat the query wording.
- `answer_shaped`: plausible answer-looking false memories.
- `stale_preference`: old user preferences that should no longer answer.
- `same_entity_wrong_time`: correct entities, wrong time window.
- `superseded_conflict`: old conflicting memories.
- `near_duplicate`: non-authoritative duplicate-looking memories.
- `evidence_adjacent`: context near the true evidence but not the evidence.
- `mixed`: deterministic mixture of all profiles.

## Metrics To Care About

The key metrics are:

- `recall@8`: evidence recovered in the first 8 returned memories.
- `precision@8`: how many of the first 8 returned memories are evidence.
- `context_recall`: evidence retained inside the token-budgeted context.
- `context_precision`: evidence density in the returned context.
- `hard_neg@8`: fraction of top-8 slots consumed by injected decoys.
- `hard_neg context`: fraction of context slots consumed by injected decoys.
- `MRR`: how early the first evidence hit appears.
- `det mismatches`: repeated-search mismatches. Target is always `0`.

Good outcome:

- `recall@8 >= 0.90`
- higher `precision@8` than FAISS/HNSW
- higher `context_precision` than FAISS/HNSW
- low `hard_neg@8` under adversarial profiles
- `det mismatches = 0`
- p95 latency below `200 ms`

## A100 Setup

A fresh Colab runtime needs:

```bash
pip install -q faiss-cpu huggingface_hub transformers sentence-transformers safetensors
git clone https://github.com/CameronBadman/Hippo-Qwen.git /content/Hippo-Qwen
git clone https://github.com/CameronBadman/Hippo-encoder.git /content/Hippo-encoder
```

Reassemble the Hippo encoder:

```bash
python /content/Hippo-encoder/scripts/reassemble_model_artifact.py \
  --manifest /content/Hippo-encoder/models/hippoencoder-bge-small-all-nli-pair-500k-c025-epoch3/chunks/chunks.json \
  --extract-to /content/hippoencoder-bge-small-all-nli-pair-500k-c025-epoch3
```

Expected encoder path:

```text
/content/hippoencoder-bge-small-all-nli-pair-500k-c025-epoch3/hippoencoder-bge-small-all-nli-pair-500k-c025-epoch3
```

## Data Preparation

Use MemoryCraft `selected/sample.jsonl`.

The raw dataset can contain malformed `\u` escapes, so sanitize the JSONL before
training/evaluation. Current standard split:

- train: records `0-119`
- holdout: records `120-259`

The ablation suite can sanitize a local slice with:

```bash
python -m python.benchmarks.memorycraft_ablation_suite \
  --dataset /path/to/sample.jsonl \
  --sanitize-dataset \
  --record-offset 120 \
  --record-limit 140 \
  --work-dir /content/hippo_big_ablation_v4 \
  --baseline-systems exact_vector \
  --profiles clean \
  --limit-questions 1
```

For serious runs, prebuild the train/holdout sanitized files once and reuse
them. That makes Colab restarts easier to recover from.

## Build Calibration Data

Full synthetic-hard-negative training set:

```bash
python -m python.benchmarks.build_hippo_calibration_data \
  --dataset /content/hippo_big_ablation_v4/memorycraft_train_0_120_sanitized.jsonl \
  --limit-records 120 \
  --limit-questions 220 \
  --candidate-source union \
  --max-candidates 128 \
  --final-fetch 128 \
  --inject-missing-relevant \
  --synthetic-hard-negatives 32 \
  --work-dir /content/hippo_big_ablation_v4/build_full \
  --output /content/hippo_big_ablation_v4/train_full.jsonl \
  --summary-json /content/hippo_big_ablation_v4/train_full_summary.json \
  --embedding-backend hippo \
  --hippo-checkpoint /content/hippoencoder-bge-small-all-nli-pair-500k-c025-epoch3/hippoencoder-bge-small-all-nli-pair-500k-c025-epoch3 \
  --hippo-encoder-src /content/Hippo-encoder/src \
  --hippo-batch-size 256 \
  --device cuda \
  --hybrid-candidate-fetch 1024 \
  --hybrid-token-candidate-fetch 1024
```

No-synthetic-negative control:

```bash
python -m python.benchmarks.build_hippo_calibration_data \
  --dataset /content/hippo_big_ablation_v4/memorycraft_train_0_120_sanitized.jsonl \
  --limit-records 120 \
  --limit-questions 220 \
  --candidate-source union \
  --max-candidates 128 \
  --final-fetch 128 \
  --inject-missing-relevant \
  --synthetic-hard-negatives 0 \
  --work-dir /content/hippo_big_ablation_v4/build_no_synth \
  --output /content/hippo_big_ablation_v4/train_no_synth.jsonl \
  --summary-json /content/hippo_big_ablation_v4/train_no_synth_summary.json \
  --embedding-backend hippo \
  --hippo-checkpoint /content/hippoencoder-bge-small-all-nli-pair-500k-c025-epoch3/hippoencoder-bge-small-all-nli-pair-500k-c025-epoch3 \
  --hippo-encoder-src /content/Hippo-encoder/src \
  --hippo-batch-size 256 \
  --device cuda \
  --hybrid-candidate-fetch 1024 \
  --hybrid-token-candidate-fetch 1024
```

## Train Calibrators

Relevance-only:

```bash
python -m python.librarian.train_hippo_calibrator \
  --dataset /content/hippo_big_ablation_v4/train_full.jsonl \
  --output /content/hippo_big_ablation_v4/relevance_only.pt \
  --epochs 6 \
  --batch-size 32 \
  --val-fraction 0.18 \
  --max-candidates 128 \
  --embedding-dim 1024 \
  --feature-dim 16 \
  --d-model 128 \
  --layers 3 \
  --heads 4 \
  --top-k 8 \
  --metric-head relevance \
  --selection-metric balanced \
  --include-loss-weight 0.0 \
  --include-rank-loss-weight 0.0 \
  --false-positive-loss-weight 0.0 \
  --rerank-relevance-weight 0.90 \
  --rerank-include-weight 0.0 \
  --rerank-base-weight 0.05 \
  --rerank-utility-weight 0.05
```

Include-head with synthetic negatives:

```bash
python -m python.librarian.train_hippo_calibrator \
  --dataset /content/hippo_big_ablation_v4/train_full.jsonl \
  --output /content/hippo_big_ablation_v4/full_include_balanced.pt \
  --epochs 8 \
  --batch-size 32 \
  --val-fraction 0.18 \
  --max-candidates 128 \
  --embedding-dim 1024 \
  --feature-dim 16 \
  --d-model 128 \
  --layers 3 \
  --heads 4 \
  --top-k 8 \
  --metric-head include \
  --selection-metric balanced \
  --include-loss-weight 1.0 \
  --include-rank-loss-weight 0.6 \
  --false-positive-loss-weight 0.25 \
  --false-positive-margin 0.25 \
  --rerank-relevance-weight 0.35 \
  --rerank-include-weight 0.60 \
  --rerank-base-weight 0.05 \
  --rerank-utility-weight 0.05
```

For the no-synthetic control, use the same include-head command with
`train_no_synth.jsonl` and output `no_synth_include.pt`.

## Run The Ablation Matrix

Recommended first serious run:

```bash
python -m python.benchmarks.memorycraft_ablation_suite \
  --dataset /content/hippo_big_ablation_v4/memorycraft_holdout_120_260_sanitized.jsonl \
  --work-dir /content/hippo_big_ablation_v4/ablation_matrix \
  --limit-records 0 \
  --limit-questions 40 \
  --profiles clean,query_echo,answer_shaped,stale_preference,same_entity_wrong_time,superseded_conflict,near_duplicate,evidence_adjacent,mixed \
  --adversarial-negatives 8 \
  --candidate-pools 64,128,256,512 \
  --calibrator relevance_only=/content/hippo_big_ablation_v4/relevance_only.pt \
  --calibrator full_include_balanced=/content/hippo_big_ablation_v4/full_include_balanced.pt \
  --calibrator no_synth_include=/content/hippo_big_ablation_v4/no_synth_include.pt \
  --repeat-searches 2 \
  --top-k 8 \
  --budget 900 \
  --final-fetch 128 \
  --hybrid-candidate-fetch 1024 \
  --hybrid-token-candidate-fetch 1024 \
  --embedding-backend hippo \
  --hippo-checkpoint /content/hippoencoder-bge-small-all-nli-pair-500k-c025-epoch3/hippoencoder-bge-small-all-nli-pair-500k-c025-epoch3 \
  --hippo-encoder-src /content/Hippo-encoder/src \
  --hippo-batch-size 256 \
  --device cuda
```

This emits:

- `/content/hippo_big_ablation_v4/ablation_matrix/summary.md`
- `/content/hippo_big_ablation_v4/ablation_matrix/summary.json`
- `/content/hippo_big_ablation_v4/ablation_matrix/experiment_log.md`
- `/content/hippo_big_ablation_v4/ablation_matrix/runs/*.json`
- `/content/hippo_big_ablation_v4/ablation_matrix/runs/*.md`

## How To Read The Result

Start with `summary.md`.

Best-case proof:

- clean Hippo calibrated beats FAISS/HNSW on `recall@8`, `precision@8`,
  `context_precision`, and `MRR`
- adversarial Hippo calibrated keeps high recall while FAISS/HNSW collapses
- `hard_neg@8` is much lower for Hippo than FAISS/HNSW
- `det mismatches` stays `0`

If `relevance_only` wins, the include head is probably not useful yet. If
`full_include_balanced` wins on hard-negative profiles, the include head is
starting to do useful suppression. If `no_synth_include` loses badly on
adversarial profiles, synthetic hard negatives are helping.

## Current Known Result

The prior 135-query holdout run showed:

| scenario | system | recall@8 | precision@8 | context precision | MRR | p95 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| clean | FAISS/HNSW | 0.880 | 0.237 | 0.287 | 0.734 | 0.18 ms |
| clean | Hippo calibrated | 0.960 | 0.260 | 0.414 | 1.000 | 25.17 ms |
| adversarial8 | FAISS/HNSW | 0.051 | 0.012 | 0.106 | 0.109 | 0.22 ms |
| adversarial8 | Hippo calibrated | 0.960 | 0.256 | 0.432 | 1.000 | 27.77 ms |

That is promising but not enough by itself. The ablation matrix exists to show
whether the same behavior holds across multiple hard-negative families and
candidate-pool sizes.

## Common Failure Modes

- Missing FAISS: install `faiss-cpu`.
- Missing previous Colab artifacts: rerun setup, data preparation, and training;
  `/content` is temporary.
- Malformed MemoryCraft JSON: sanitize `\u` escapes before slicing.
- High recall, low precision: train harder negatives or reduce candidate-pool
  noise.
- High `hard_neg@8`: the calibrator is falling for that adversarial family; add
  or overweight that profile in training.
- Nonzero determinism mismatches: treat as a blocker before using the result in
  a pitch.
