# MemoryCraft Calibration Ablation Suite

This suite is the current accuracy/precision proving ground for Hippo memory
retrieval. It repeatedly runs the MemoryCraft retrieval benchmark across clean
and adversarial conditions, then writes a single leaderboard and experiment log.

The goal is not to prove Hippo is a faster general vector database than FAISS.
The goal is narrower: show that a deterministic memory-specific retrieval layer
can preserve evidence recall and improve precision when normal vector retrieval
is polluted by hard memory decoys.

## What It Tests

The suite compares:

- `faiss_hnsw`: direct vector baseline.
- `hybrid_union_token`: FAISS/HNSW plus deterministic token-field candidate
  union, without the transformer calibrator.
- `hippo_calibrated_union`: the same candidate union reranked by a Hippo
  calibration transformer.

It can run the same systems across multiple adversarial profiles:

- `clean`: no injected decoys.
- `query_echo`: decoys that repeat the query wording.
- `answer_shaped`: plausible answer-looking false memories.
- `stale_preference`: old preferences that should be ignored.
- `same_entity_wrong_time`: right entities, wrong time window.
- `superseded_conflict`: older conflicting memories.
- `near_duplicate`: non-authoritative duplicates of evidence.
- `evidence_adjacent`: context next to the evidence but not the evidence.
- `mixed`: deterministic mixture of the above.

## Why These Metrics

The benchmark reports:

- `recall@8`: how much labelled evidence is recovered in the first 8 memories.
- `precision@8`: how dense the top 8 memories are with labelled evidence.
- `context_recall`: how much evidence survives the token budget.
- `context_precision`: how much of the returned context is useful evidence.
- `hard_neg@8`: how many top-8 slots are consumed by injected decoys.
- `MRR`: how soon the first evidence hit appears.
- `det mismatches`: repeated-search output mismatches. This should be `0`.

For Latitude 37 style evaluation, the strongest result is high recall and
precision with low `hard_neg@8` under adversarial profiles, not raw FAISS
latency.

## Colab Command

Example using the current Hippo encoder and one or more calibrator checkpoints:

```bash
python -m python.benchmarks.memorycraft_ablation_suite \
  --dataset /content/hippo_big_ablation_v3/memorycraft_holdout_120_260_sanitized.jsonl \
  --work-dir /content/hippo_big_ablation_v4 \
  --limit-records 0 \
  --limit-questions 40 \
  --profiles clean,query_echo,answer_shaped,stale_preference,same_entity_wrong_time,superseded_conflict,near_duplicate,evidence_adjacent,mixed \
  --adversarial-negatives 8 \
  --candidate-pools 64,128,256,512 \
  --calibrator relevance_only=/content/hippo_big_ablation_v3/relevance_only.pt \
  --calibrator full_include_balanced=/content/hippo_big_ablation_v3/full_include_balanced.pt \
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

For a larger holdout, create a sanitized JSONL slice first and point
`--dataset` at that file. The suite also has `--sanitize-dataset`,
`--record-offset`, and `--record-limit` for preparing a local slice, but using a
prebuilt sanitized file keeps Colab runs easier to resume.

## Outputs

The work directory contains:

- `summary.md`: leaderboard and scenario matrix.
- `summary.json`: machine-readable rows and run plan.
- `experiment_log.md`: explanation of the ablation axes and run plan.
- `runs/*.json`: raw benchmark output for each scenario/checkpoint/pool.
- `runs/*.md`: per-run MemoryCraft retrieval summaries.
- `prepared_dataset.jsonl`: only when dataset slicing/sanitization is requested.

## Reading The Results

Use `summary.md` first. The best rows should have:

- `recall@8 >= 0.90`
- rising `precision@8` and `context_precision`
- `hard_neg@8` close to `0.0` under adversarial profiles
- `det mismatches = 0`
- p95 latency below the target, currently `200 ms`

If recall is high but precision is low, the candidate union is finding evidence
but the calibrator is not suppressing enough near misses. If `hard_neg@8` is
high, the model is falling for that adversarial family and the next training set
should overweight that profile.

## Local Smoke Test

This does not require FAISS and is only meant to verify the runner:

```bash
python -m python.benchmarks.memorycraft_ablation_suite \
  --dataset python/benchmarks/fixtures/memorycraft_tiny.jsonl \
  --work-dir /tmp/hippo_ablation_smoke \
  --profiles clean,query_echo \
  --baseline-systems exact_vector \
  --candidate-pools 64 \
  --limit-records 0 \
  --limit-questions 2 \
  --repeat-searches 2 \
  --embedding-backend hash
```

Expected behavior: the command writes `summary.md` and `experiment_log.md`, and
the query-echo profile should expose hard-negative contamination in the tiny
fixture.
