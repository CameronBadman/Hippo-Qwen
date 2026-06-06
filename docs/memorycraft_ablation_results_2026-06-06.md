# MemoryCraft Ablation Results - 2026-06-06

These results were run in Google Colab on an A100 against the sanitized MemoryCraft holdout:

- dataset: `/content/hippo_big_ablation_v4/memorycraft_holdout_120_260_sanitized.jsonl`
- embedding backend: Hippo encoder checkpoint `hippoencoder-bge-small-all-nli-pair-500k-c025-epoch3`
- top-k: 8
- context budget: 900 tokens
- repeated deterministic searches: 2
- latency target: 200 ms p95

## Main Finding

The relevance-only calibrator is currently the strongest retrieval model. It dominates the full include-balanced calibrator and the no-synthetic-hard-negative calibrator across clean and adversarial profiles.

At candidate pool 128, relevance-only held about `0.960` recall@8, `0.256` precision@8, `0.44-0.45` context precision, `1.000` MRR, and `0` determinism mismatches across the adversarial profile set. P95 latency stayed around `25-31 ms`, comfortably under the `200 ms` target.

The no-synthetic-hard-negative ablation is the clearest training lesson: clean metrics remain decent, but adversarial hard-negative leakage jumps to roughly `0.70-0.77` hard neg@8. Synthetic decoys are not optional for this task.

## Pool Sweep

Focused sweep: profiles `clean`, `query_echo`, and `mixed`; calibrator `relevance_only`; pools `32,64,128,256`.

| scenario | pool | p95 ms | recall@8 | precision@8 | context precision | hard neg@8 | MRR | det mismatches |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| clean | 32 | 23.36 | 0.9575 | 0.2593 | 0.4247 | 0.0000 | 1.0000 | 0 |
| clean | 64 | 25.25 | 0.9605 | 0.2620 | 0.4266 | 0.0000 | 1.0000 | 0 |
| clean | 128 | 25.76 | 0.9605 | 0.2620 | 0.4266 | 0.0000 | 1.0000 | 0 |
| clean | 256 | 25.28 | 0.9605 | 0.2620 | 0.4266 | 0.0000 | 1.0000 | 0 |
| query_echo | 32 | 22.97 | 0.9540 | 0.2509 | 0.4456 | 0.0167 | 1.0000 | 0 |
| query_echo | 64 | 27.89 | 0.9606 | 0.2565 | 0.4498 | 0.0167 | 1.0000 | 0 |
| query_echo | 128 | 30.64 | 0.9606 | 0.2565 | 0.4498 | 0.0167 | 1.0000 | 0 |
| query_echo | 256 | 30.15 | 0.9606 | 0.2565 | 0.4498 | 0.0167 | 1.0000 | 0 |
| mixed | 32 | 23.25 | 0.9540 | 0.2509 | 0.4445 | 0.0167 | 1.0000 | 0 |
| mixed | 64 | 29.96 | 0.9606 | 0.2565 | 0.4496 | 0.0167 | 1.0000 | 0 |
| mixed | 128 | 28.32 | 0.9606 | 0.2565 | 0.4496 | 0.0167 | 1.0000 | 0 |
| mixed | 256 | 28.59 | 0.9606 | 0.2565 | 0.4496 | 0.0167 | 1.0000 | 0 |

Pool 64 is the current default recommendation. It reaches the same recall/precision plateau as 128 and 256 while avoiding the small recall drop seen at pool 32.

## Baseline Comparison

Clean retrieval is competitive for ordinary vector baselines, but adversarial profiles are where the calibrated system separates.

| scenario | system | p95 ms | recall@8 | precision@8 | context precision | hard neg@8 | MRR |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| clean | faiss_hnsw | 0.19 | 0.8801 | 0.2370 | 0.2871 | 0.0000 | 0.7338 |
| clean | hybrid_union_token | 6.69 | 0.8638 | 0.2287 | 0.3050 | 0.0000 | 0.7006 |
| clean | hippo_calibrated_union pool64 | 25.25 | 0.9605 | 0.2620 | 0.4266 | 0.0000 | 1.0000 |
| query_echo | faiss_hnsw | 0.22 | 0.0111 | 0.0028 | 0.1152 | 0.9907 | 0.1186 |
| query_echo | hybrid_union_token | 7.61 | 0.0185 | 0.0037 | 0.1150 | 0.9898 | 0.1245 |
| query_echo | hippo_calibrated_union pool64 | 27.89 | 0.9606 | 0.2565 | 0.4498 | 0.0167 | 1.0000 |
| mixed | faiss_hnsw | 0.21 | 0.0800 | 0.0213 | 0.1076 | 0.9500 | 0.1120 |
| mixed | hybrid_union_token | 7.73 | 0.0707 | 0.0185 | 0.0987 | 0.9556 | 0.1091 |
| mixed | hippo_calibrated_union pool64 | 29.96 | 0.9606 | 0.2565 | 0.4496 | 0.0167 | 1.0000 |

## Engineering Change

The ablation suite now checkpoints `summary.json` and `summary.md` after every completed case. Interrupted Colab cells should preserve all completed rows instead of losing the run.

## Next Work

The current bottleneck is precision density, not recall or latency. The next experiments should target:

- better evidence compaction inside the returned context
- stronger hard-negative curricula with same-entity, stale-time, and answer-shaped decoys
- thresholded abstention for low-confidence cases
- larger holdouts with more question styles and longer memory histories
