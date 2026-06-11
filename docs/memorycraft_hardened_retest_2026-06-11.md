# MemoryCraft Hardened Adversarial Retest - 2026-06-11

This retest responds to a benchmark-validity issue in the original adversarial
suite: synthetic decoys had metadata/state/text fingerprints that made them too
easy for the calibrator to identify.

## What Changed

- Eval decoys now default to `--adversarial-style forensic`.
- Training hard negatives now default to `--synthetic-hard-negative-style forensic`.
- Forensic decoys clone visible metadata/state shape from real MemoryCraft cards.
- Decoy text no longer says things like "not evidence", "superseded", or
  "should be excluded".
- A new `--calibrator-feature-ablation no_shortcuts` mode zeros state,
  metadata-presence, and conflict-term shortcut features at evaluation time.

The old behavior remains available as `legacy` for comparison.

## Run Setup

- runtime: Google Colab A100
- code commit: `5d13ade`
- dataset: `daven3/MemoryCraft selected/sample.jsonl`
- train records: first 80 records
- eval records: offset 80, next 40 records
- evaluated questions per scenario: 36
- embedding backend: `hash`
- candidate source: union
- synthetic training negatives: 12 per row, forensic style
- eval decoys: 8 per QA, forensic style
- candidate pool: 64
- repeat searches: 2

This is a fast validation run, not the final Hippo-encoder result.

## Main Result

With forensic decoys and no feature ablation:

| scenario | system | p95 ms | recall@8 | precision@8 | context precision | hard neg@8 | MRR | det mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| clean | FAISS HNSW | 0.22 | 0.8444 | 0.1933 | 0.2290 | 0.0000 | 0.7136 | 0 |
| clean | Hippo calibrated | 25.79 | 0.9615 | 0.2245 | 0.2495 | 0.0000 | 1.0000 | 0 |
| query_echo | FAISS HNSW | 0.23 | 0.0598 | 0.0174 | 0.1006 | 0.9722 | 0.1059 | 0 |
| query_echo | Hippo calibrated | 27.67 | 0.9615 | 0.2222 | 0.2486 | 0.0382 | 0.9722 | 0 |
| answer_shaped | FAISS HNSW | 0.22 | 0.1154 | 0.0243 | 0.1025 | 0.9618 | 0.1095 | 0 |
| answer_shaped | Hippo calibrated | 28.22 | 0.9615 | 0.2222 | 0.2391 | 0.0590 | 0.9722 | 0 |
| mixed | FAISS HNSW | 0.21 | 0.3407 | 0.0729 | 0.1035 | 0.8507 | 0.1201 | 0 |
| mixed | Hippo calibrated | 27.93 | 0.9594 | 0.2188 | 0.2283 | 0.1493 | 0.9722 | 0 |

The calibrated system still beats FAISS/HNSW under harder decoys, but the
mixed-profile hard-negative leakage is no longer near-zero. That is a healthier
and more believable result.

## Shortcut Feature Ablation

With `--calibrator-feature-ablation no_shortcuts`, state, metadata-presence, and
conflict-term shortcut features are zeroed at evaluation time.

| scenario | Hippo recall@8 | Hippo precision@8 | Hippo hard neg@8 | Hippo MRR |
| --- | ---: | ---: | ---: | ---: |
| clean | 0.9106 | 0.2002 | 0.0000 | 0.6778 |
| query_echo | 0.8495 | 0.1875 | 0.0208 | 0.5875 |
| answer_shaped | 0.8217 | 0.1840 | 0.0521 | 0.5887 |
| mixed | 0.8175 | 0.1771 | 0.1562 | 0.5885 |

Interpretation:

- The previous shortcut concern was valid.
- Removing shortcut features hurts recall/MRR substantially.
- The model does not collapse to FAISS-level behavior, so the signal is not only
  metadata fingerprints.
- We need a larger Hippo-encoder retest before using the strongest claims in a
  pitch.

## Pitch Impact

Use this framing:

> The first adversarial benchmark had synthetic leakage. We hardened the decoys
> to match real memory metadata and added shortcut-feature ablations. The
> calibrated system still beats FAISS/HNSW under forensic decoys, but the result
> is more modest and we are treating the suite as an active benchmark, not a
> solved claim.

Avoid claiming:

> Hippo has proven 0.96 recall under all adversarial conditions at scale.

Defensible current claim:

> Hippo-Qwen is a deterministic memory retrieval research harness with a compact
> calibrator that improves clean retrieval and remains materially stronger than
> raw FAISS/HNSW under hardened synthetic decoys.

## Next Retest

The next run should use:

- Hippo encoder instead of hash embeddings
- larger record holdout
- held-out decoy families
- off-the-shelf cross-encoder reranker baseline
- 50k+ synthetic memory scale test with calibrated reranking
