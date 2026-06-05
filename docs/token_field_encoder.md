# Token-Field Encoder Prototype

This prototype trains token-field heads on top of Hippo embeddings. The heads
emit action tokens consumed by the layered token-field index.

## Build Training Data

Use Hippo embeddings when running on the A100 runtime:

```bash
python -m python.field_memory.build_token_encoder_data \
  --dataset /path/to/longmemeval.jsonl \
  --embedding-backend hippo \
  --hippo-checkpoint /path/to/hippo_encoder.pt \
  --hippo-encoder-src /path/to/Hippo-encoder \
  --device cuda \
  --limit-records 0 \
  --hard-negatives 48 \
  --random-negatives 48 \
  --output artifacts/token_field_encoder/train.jsonl
```

## Train

```bash
python -m python.field_memory.train_token_encoder \
  --dataset artifacts/token_field_encoder/train.jsonl \
  --output artifacts/token_field_encoder/token_encoder.pt \
  --epochs 6 \
  --batch-size 32 \
  --embedding-dim 1024 \
  --action-count 512 \
  --bucket-count 65 \
  --query-token-count 48 \
  --node-token-count 48 \
  --candidate-cap 384
```

The first metric to optimize is `hard_candidate_recall`, then
`hard_recall@k`. The soft `val_candidate_recall` is still useful for tracking
the differentiable loss, but it can overstate quality because the index uses
exported hard tokens during retrieval.

## Build Hippo-Encoder Triplets

The token-field data can also be converted into Hippo-encoder triplets for a
larger from-scratch student run:

```bash
python -m python.field_memory.build_hippo_triplets \
  --input artifacts/token_field_encoder/train.jsonl \
  --output artifacts/hippo_encoder/memorycraft_triplets.jsonl \
  --max-positives 8 \
  --max-negatives 48 \
  --negatives-per-positive 2 \
  --include-random-negatives \
  --shuffle
```

## Benchmark

```bash
python -m python.benchmarks.token_field_retrieval \
  --dataset /path/to/longmemeval.jsonl \
  --embedding-backend hippo \
  --hippo-checkpoint /path/to/hippo_encoder.pt \
  --hippo-encoder-src /path/to/Hippo-encoder \
  --device cuda \
  --token-encoder-checkpoint artifacts/token_field_encoder/token_encoder.pt \
  --action-count 512 \
  --routing-layers 8 \
  --promotion-probability 0.45 \
  --routing-beam-width 96 \
  --query-token-count 48 \
  --node-token-count 48 \
  --bucket-radius 2 \
  --max-candidates 384 \
  --final-fetch 128
```

Target for the first useful checkpoint: improve 680-node candidate recall over
the random emitter while keeping p95 query latency below 200 ms.
