# Session Metadata Degradation Results

Date: 2026-06-12

## Setup

- Runtime: Google Colab A100.
- Repo commit: `25f8efe`.
- Benchmark: `python.benchmarks.session_memory_stress`.
- Memory count: 50,000.
- Queries: 120.
- Seed: `72000`.
- Embedding backend: hash.
- Vector index: FAISS HNSW.
- Candidate generation:
  - vector fetch: 1024
  - token fetch: 1024
  - metadata fetch: 256
  - graph fetch: 256
  - final candidate pool: 128
- Calibrator: `/content/session_metadata_degradation_hash/stress_feature33_hash.pt`
- Packing policies:
  - `calibrated_pack_p40`: include probability `>= 0.40`
  - `calibrated_pack_top8`: fixed top-8 fallback
  - adaptive test: `p>=0.40` plus `--packing-threshold-min-items`

This is a hash-backed stress diagnostic. It is useful for policy selection and
failure attribution, but it does not replace the Hippo-encoder retest.

## Metadata Degradation Baseline

The first run used `p>=0.40` with no minimum fallback and compared it with a
fixed top-8 packer.

| case | system | pool | recall@8 | context recall | context precision | hard neg ctx | included | ctx tokens |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| clean | p>=0.40 | 1.0000 | 1.0000 | 0.9917 | 0.9167 | 0.0000 | 3.45 | 88.7 |
| clean | top-8 | 1.0000 | 1.0000 | 1.0000 | 0.3750 | 0.0229 | 8.00 | 200.8 |
| 70% metadata | p>=0.40 | 0.8278 | 0.8222 | 0.6917 | 0.8917 | 0.0000 | 2.40 | 61.8 |
| 70% metadata | top-8 | 0.8278 | 0.8222 | 0.8222 | 0.3083 | 0.0437 | 8.00 | 198.6 |
| 40% metadata | p>=0.40 | 0.6694 | 0.6556 | 0.3944 | 0.7917 | 0.0000 | 1.38 | 35.4 |
| 40% metadata | top-8 | 0.6694 | 0.6556 | 0.6556 | 0.2458 | 0.0083 | 8.00 | 197.2 |
| 0% metadata | p>=0.40 | 0.5972 | 0.5500 | 0.0000 | 0.0000 | 0.0000 | 0.00 | 0.0 |
| 0% metadata | top-8 | 0.5972 | 0.5500 | 0.5500 | 0.2062 | 0.0000 | 8.00 | 197.1 |
| 25% wrong metadata | p>=0.40 | 0.8694 | 0.8333 | 0.7139 | 0.9083 | 0.0000 | 2.48 | 64.0 |
| 25% wrong metadata | top-8 | 0.8694 | 0.8333 | 0.8333 | 0.3125 | 0.0354 | 8.00 | 198.3 |
| 50% wrong metadata | p>=0.40 | 0.7778 | 0.6889 | 0.4806 | 0.8167 | 0.0000 | 1.71 | 44.2 |
| 50% wrong metadata | top-8 | 0.7778 | 0.6889 | 0.6889 | 0.2583 | 0.0125 | 8.00 | 200.3 |

## Adaptive Minimum-Items Sweep

The no-metadata case exposed a packer failure: `p>=0.40` can abstain completely
when metadata is absent even though the top-8 ranking still contains evidence.
The next run tested `--packing-threshold-min-items` values. Colab became stale
while starting the `min_items=5` block, so the complete reliable data is for
minimums 1, 2, and 3.

| case | min items | recall@8 | context recall | context precision | hard neg ctx | included | ctx tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| clean | 1 | 1.0000 | 0.9722 | 0.9167 | 0.0000 | 3.39 | 87.1 |
| clean | 2 | 1.0000 | 0.9778 | 0.9167 | 0.0000 | 3.40 | 87.3 |
| clean | 3 | 1.0000 | 0.9889 | 0.9167 | 0.0000 | 3.43 | 88.2 |
| 70% metadata | 1 | 0.8500 | 0.6722 | 0.8917 | 0.0000 | 2.37 | 60.9 |
| 70% metadata | 2 | 0.8444 | 0.6833 | 0.8292 | 0.0000 | 2.53 | 65.0 |
| 70% metadata | 3 | 0.8361 | 0.7111 | 0.6833 | 0.0000 | 3.17 | 80.8 |
| 40% metadata | 1 | 0.7028 | 0.3972 | 0.8083 | 0.0000 | 1.49 | 38.2 |
| 40% metadata | 2 | 0.6972 | 0.4444 | 0.6333 | 0.0000 | 2.11 | 53.2 |
| 40% metadata | 3 | 0.7111 | 0.5111 | 0.5083 | 0.0000 | 3.02 | 75.5 |
| 0% metadata | 1 | 0.5639 | 0.2333 | 0.7000 | 0.0000 | 1.00 | 25.2 |
| 0% metadata | 2 | 0.5583 | 0.4028 | 0.6042 | 0.0000 | 2.00 | 48.5 |
| 0% metadata | 3 | 0.5556 | 0.4694 | 0.4694 | 0.0000 | 3.00 | 72.7 |
| 25% wrong metadata | 1 | 0.8278 | 0.6972 | 0.9083 | 0.0000 | 2.41 | 62.2 |
| 25% wrong metadata | 2 | 0.8222 | 0.7028 | 0.8417 | 0.0000 | 2.57 | 66.2 |
| 25% wrong metadata | 3 | 0.8306 | 0.7167 | 0.6972 | 0.0028 | 3.12 | 80.0 |
| 50% wrong metadata | 1 | 0.7028 | 0.4694 | 0.8167 | 0.0000 | 1.77 | 45.6 |
| 50% wrong metadata | 2 | 0.7083 | 0.4861 | 0.6292 | 0.0000 | 2.30 | 59.0 |
| 50% wrong metadata | 3 | 0.6917 | 0.5250 | 0.5083 | 0.0028 | 3.10 | 78.7 |

## Interpretation

- The clean `p>=0.40` result was real but metadata-rich.
- Missing or wrong metadata primarily hurts candidate generation:
  evidence-in-pool falls from 1.0000 clean to about 0.60 with no metadata.
- The calibrator still suppresses hard negatives well under degraded metadata.
- A pure threshold packer is too brittle because it can return no context when
  confidence drops.
- A minimum-3 fallback is the best observed tradeoff:
  it restores no-metadata context recall from 0.0000 to 0.4694 while keeping
  hard-negative context near zero.
- Top-8 is the high-recall fallback, but it roughly halves or thirds context
  precision compared with the calibrated threshold packer.

## Recommendation

Use this policy for the next implementation pass:

```text
score candidates with the calibrator
include candidates with include_probability >= 0.40
if fewer than 3 memories are included, add the top-ranked candidates until 3
never exceed the context budget
```

Treat this as a stopgap, not the final answer. The stronger fix is to train a
packing/abstention head on degraded-metadata examples and to improve candidate
generation so evidence-in-pool stays high without perfect metadata.

## Next Runs

1. Repeat the same sweep with the Hippo-encoder backend.
2. Complete the `min_items=5` block after a Colab runtime reset.
3. Add combined degradation cases:
   70% metadata plus 25% wrong, and 40% metadata plus 25% wrong.
4. Train the calibrator with metadata dropout/noise so it does not over-rely on
   clean structured fields.
5. Test graph/entity expansion with imperfect write-path entity extraction.
