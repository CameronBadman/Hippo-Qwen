# Qwen Teacher Metadata Smoke Test - 2026-06-12

## Purpose

Test whether Qwen-generated derived metadata can recover retrieval signal when native memory metadata is unavailable.

This is a pipeline smoke test, not a final benchmark. The teacher cache was generated on a small balanced slice and the calibrator sanity run was trained and evaluated on the same 8 query cases.

## Setup

- Runtime: Google Colab A100
- Repo head: `b285a62`
- Teacher model: Qwen Cloud compatible chat completions API, `qwen-plus`
- Memory workload: `session_memory_stress`
- Memories: `1000`
- Query cases: `8`
- Native metadata availability: `0.0`
- Derived metadata: `qwen-cache`
- Metadata source: `derived`
- Embedding backend: deterministic hash embeddings
- Candidate pool: `128`
- Teacher-label batch:
  - `8` queries
  - `24` evidence memories
  - `96` hard negatives
  - `32` background memories

## Artifacts In Colab

- `/content/qwen_teacher_fields/balanced_teacher_items_1k_8q.jsonl`
- `/content/qwen_teacher_fields/qwen_teacher_cache_1k_8q.json`
- `/content/qwen_teacher_fields/qwen_teacher_audit_1k_8q.jsonl`
- `/content/qwen_teacher_fields/field_registry_after_qwen_1k_8q.json`
- `/content/qwen_teacher_fields/qwen_derived_calibrator.pt`
- `/content/qwen_teacher_fields/replay_1k_8q_qwen_calibrated.md`

## Retrieval-Only Result

With no calibrator checkpoint loaded, Qwen-derived metadata improved the hybrid candidate path over vector-only retrieval:

| system | recall@8 | recall@16 | recall@32 | precision@8 | context recall | context precision | hard neg ctx |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vector | 0.5000 | 0.6250 | 0.6667 | 0.1875 | 0.6667 | 0.0556 | 0.7431 |
| hybrid + Qwen fields | 0.6667 | 1.0000 | 1.0000 | 0.2500 | 1.0000 | 0.0834 | 0.7947 |

Interpretation: Qwen-derived fields recovered pool coverage even when original metadata was removed, but the uncalibrated context path still packed too much noise.

## Calibrated Smoke Result

A small 48-feature calibrator was trained from the same 8 query rows to verify the full derived-metadata-to-reranker path.

| system | recall@8 | precision@8 | context recall | context precision | hard neg ctx | included |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| hybrid + Qwen fields | 0.6667 | 0.2500 | 1.0000 | 0.0834 | 0.7947 | 36.00 |
| calibrated | 1.0000 | 0.3750 | 1.0000 | 0.0636 | 0.2860 | 47.38 |
| calibrated pack top3 | 1.0000 | 0.3750 | 1.0000 | 1.0000 | 0.0000 | 3.00 |
| calibrated pack top5 | 1.0000 | 0.3750 | 1.0000 | 0.6000 | 0.4000 | 5.00 |
| calibrated pack top8 | 1.0000 | 0.3750 | 1.0000 | 0.3750 | 0.6250 | 8.00 |
| calibrated pack p40/min3 | 1.0000 | 0.3750 | 1.0000 | 1.0000 | 0.0000 | 3.00 |

Interpretation: the full path works end to end. The top-3/threshold packing result is exactly what we want structurally: compact context, full recall, no hard negatives. But because this is trained and tested on the same small slice, treat it as a proof that the plumbing works, not proof that quality generalizes.

## Findings

1. Qwen teacher metadata is useful enough to keep. Even with native metadata removed, candidate coverage reached `1.0000` on the 8-query slice.
2. The derived metadata path is now a plausible answer to metadata degradation: Qwen can produce normalized fields that the deterministic retriever can index and the calibrator can consume.
3. Context packing remains mandatory. Without packing, high context recall still comes with unacceptable noise.
4. The next valid experiment must use train/test separation. The current calibrator run is intentionally overfit.

## Next Experiment

Generate a larger Qwen-labeled calibration set with a deterministic split:

- Train: 32-50 query cases
- Holdout: 16-25 query cases
- Keep balanced query/evidence/hard-negative/background sampling
- Report:
  - evidence-in-pool
  - recall@8/16/32
  - precision@8
  - context recall
  - context precision
  - hard-negative context rate
  - deterministic repeat mismatch rate

The most important validation is whether the Qwen-derived metadata keeps high pool coverage on holdout queries whose memories were not used to train the calibrator.
