# MemoryCraft Held-Out Family Retest - 2026-06-11

This retest addresses the remaining synthetic-decoy leakage risk in the hardened
MemoryCraft benchmark: the training and eval decoys still shared template
families. The control below trains without the two most obvious eval families,
then evaluates directly on those held-out families.

## Setup

- Runtime: Google Colab A100
- Repo commit: `20ff0e2`
- Dataset: MemoryCraft `selected/sample.jsonl`
- Embedding backend: deterministic hash backend
- Train split: first 80 records, 104 QA rows
- Eval split: records 80-119, 36 evaluated questions per scenario
- Candidate source: FAISS HNSW + token-field union
- Candidate pool: 64
- Top k: 8
- Repeat searches: 2
- Training synthetic negatives: 12 per row
- Training decoy style: `forensic`
- Training excluded decoy families: `query_echo,answer_shaped`
- Eval adversarial style: `forensic`
- Eval profiles: `query_echo,answer_shaped,mixed`

Artifacts in Colab:

```text
/content/hippo_heldout_retest/
  train_seen_only.jsonl
  seen_only_full.pt
  seen_only_no_shortcuts.pt
  eval_heldout_full/summary.md
  eval_heldout_no_shortcuts/summary.md
  compact_summary.json
```

## Result: Held-Out Families, Full Features

The full-feature calibrator was trained without `query_echo` and `answer_shaped`
synthetic hard negatives. It still generalized strongly to both held-out eval
families.

| scenario | system | p95 ms | recall@8 | precision@8 | context precision | hard neg@8 | MRR | determinism |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| query_echo | FAISS HNSW | 0.22 | 0.0598 | 0.0174 | 0.1006 | 0.9722 | 0.1059 | 0 |
| query_echo | Hippo calibrated | 27.62 | 0.9594 | 0.2188 | 0.2050 | 0.0556 | 1.0000 | 0 |
| answer_shaped | FAISS HNSW | 0.24 | 0.1154 | 0.0243 | 0.1025 | 0.9618 | 0.1095 | 0 |
| answer_shaped | Hippo calibrated | 27.17 | 0.9594 | 0.2188 | 0.2053 | 0.0764 | 1.0000 | 0 |
| mixed | FAISS HNSW | 0.20 | 0.3407 | 0.0729 | 0.1035 | 0.8507 | 0.1201 | 0 |
| mixed | Hippo calibrated | 28.17 | 0.9573 | 0.2153 | 0.1971 | 0.1215 | 1.0000 | 0 |

## Result: Train-Time No-Shortcut Ablation

This model was trained with shortcut/state/metadata features removed, then
evaluated with the same feature ablation. This is the stronger control than
zeroing those features only at eval time.

| scenario | system | p95 ms | recall@8 | precision@8 | context precision | hard neg@8 | MRR | determinism |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| query_echo | FAISS HNSW | 0.21 | 0.0598 | 0.0174 | 0.1006 | 0.9722 | 0.1059 | 0 |
| query_echo | Hippo calibrated no-shortcuts | 27.57 | 0.8261 | 0.1806 | 0.1853 | 0.0417 | 0.5557 | 0 |
| answer_shaped | FAISS HNSW | 0.24 | 0.1154 | 0.0243 | 0.1025 | 0.9618 | 0.1095 | 0 |
| answer_shaped | Hippo calibrated no-shortcuts | 28.54 | 0.7983 | 0.1771 | 0.1701 | 0.0972 | 0.5557 | 0 |
| mixed | FAISS HNSW | 0.24 | 0.3407 | 0.0729 | 0.1035 | 0.8507 | 0.1201 | 0 |
| mixed | Hippo calibrated no-shortcuts | 27.93 | 0.8122 | 0.1771 | 0.1696 | 0.1354 | 0.5524 | 0 |

## Interpretation

The original leakage concern was real, but the held-out control still supports
the core claim: the calibrator is not merely memorizing the `query_echo` and
`answer_shaped` template strings. With those families excluded from training,
the full model remains near `0.96 recall@8` on the held-out families and keeps
hard-negative leakage below `0.08` on those direct profiles.

The no-shortcut model is the more conservative lower bound. It drops MRR and
precision, but still beats FAISS HNSW recall by a wide margin on both held-out
families and suppresses most synthetic hard negatives. That suggests meaningful
signal remains in text, embedding, rank, and candidate-neighborhood structure
even after metadata/state shortcut removal.

This is still a fast validation run, not a final benchmark:

- hash embeddings, not the full Hippo encoder
- 36 evaluated questions per scenario
- synthetic decoys, not fully mutated real-turn adversaries
- no cross-encoder reranker baseline yet

The next defensible benchmark should use the Hippo encoder, a larger holdout,
mutated real-turn hard negatives, and a reranker baseline.
