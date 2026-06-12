# Hippo-Qwen Next Technical Plan

This is the current handoff plan after the 10k/50k session-memory stress test.

## Current State

Hippo-Qwen now has:

- A deterministic memory retrieval engine.
- A Hippo-encoder-backed candidate generator.
- A learned calibrator/reranker.
- A session-style 10k/50k stress benchmark:
  `python/benchmarks/session_memory_stress.py`
- Documented benchmark findings in `HIPPO_QWEN_NEXT_IDEAS.txt`.

The latest stress run used:

- Colab A100.
- Hippo encoder checkpoint:
  `/content/hippoencoder-model/hippoencoder-bge-small-all-nli-pair-500k-c025-epoch3`
- 120 queries.
- 10k and 50k generated memories.
- FAISS HNSW vector search.
- Token search.
- Candidate pool size 128.
- Vector fetch 1024.
- Token fetch 1024.
- Fresh 768-d set calibrator.

## Original Benchmark Result

The 50k result is not market-ready.

| memories | system | evidence in pool | recall@8 | precision@8 | hard neg@8 | context precision |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 10k | vector | 0.1167 | 0.0083 | 0.0031 | 0.8771 | 0.0055 |
| 10k | hybrid | 0.2944 | 0.2389 | 0.0896 | 0.5698 | 0.0227 |
| 10k | calibrated | 0.2944 | 0.2306 | 0.0865 | 0.0260 | 0.0217 |
| 50k | vector | 0.1028 | 0.0083 | 0.0031 | 0.7542 | 0.0052 |
| 50k | hybrid | 0.2444 | 0.1944 | 0.0729 | 0.5000 | 0.0175 |
| 50k | calibrated | 0.2444 | 0.1722 | 0.0646 | 0.0490 | 0.0177 |

Interpretation:

- The current 50k `recall@8` is too low for go-to-market.
- The main failure is not only reranking.
- The candidate pool only contains about 24% of relevant evidence at 50k.
- The calibrator cannot recover memories that never enter the pool.
- The calibrator is still useful because it cuts hard-negative exposure from
  0.5000 to 0.0490 at 50k.

## Updated Benchmark Result

After adding deterministic metadata/entity expansion and typed graph candidate
expansion, the 50k result changed materially.

| memories | mode | system | evidence in pool | recall@8 | recall@16 | recall@32 | hard neg@8 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 50k | baseline | calibrated | 0.2444 | 0.1722 | 0.1917 | 0.1972 | 0.0479 |
| 50k | expanded | calibrated | 1.0000 | 0.7194 | 0.8722 | 0.9972 | 0.2750 |
| 50k | expanded | hybrid | 1.0000 | 0.7056 | 0.9500 | 1.0000 | 0.5896 |
| 50k | expanded + stress-trained calibrator | calibrated | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0656 |
| 50k | expanded + stress calibrator + p>=0.40 packing | calibrated | 1.0000 | 1.0000 | n/a | n/a | 0.0281 |

Interpretation:

- The graph/entity candidate-generation hypothesis worked on the synthetic
  stress benchmark.
- Candidate generation is no longer the main blocker when structured metadata
  and typed graph expansion are available.
- 50k recall@8 moved past the rough 0.50 go-to-market threshold for this
  benchmark.
- After training on the expanded stress candidate pools, the calibrator reached
  recall@8 1.0000 with hard-negative top-k 0.0656.
- A fresh hash-backed packing sweep showed that the default budget packer was
  the context-collapse point: it returned about 40 memories. A simple include
  probability threshold of p>=0.40 returned about 3.5 memories with context
  recall 0.9917, context precision 0.9167, and hard-negative context rate 0.0
  on the 50k synthetic stress run.
- This is promising, but it is still a synthetic metadata-rich benchmark.
- The next risk is benchmark generosity: the graph layer currently gets clean
  user/project/brand metadata.

## Metadata Degradation Result

The partial/noisy metadata ablation has now been run on Colab A100 with the
hash-backed 50k stress setup. Detailed numbers are in
`docs/session_metadata_degradation_2026-06-12.md`.

Key findings:

- Clean metadata remains strong: `p>=0.40` packing reached recall@8 1.0000,
  context recall 0.9917, context precision 0.9167, and hard-negative context
  0.0000.
- With 70% metadata availability, pool recall dropped to 0.8278 and `p>=0.40`
  context recall dropped to 0.6917 while context precision stayed high at
  0.8917.
- With 40% metadata availability, pool recall dropped to 0.6694 and `p>=0.40`
  context recall dropped to 0.3944.
- With 0% metadata availability, pool recall was 0.5972 and pure `p>=0.40`
  returned no context at all.
- With 50% wrong metadata, pool recall was 0.7778 and `p>=0.40` context recall
  was 0.4806.

Adaptive packing was then tested with `--packing-threshold-min-items`.
The reliable completed sweep covers minimums 1, 2, and 3. Colab became stale
while starting the minimum-5 block, so minimum-5 still needs a clean rerun.

Best current policy:

```text
include memories with include_probability >= 0.40
if fewer than 3 memories are included, backfill from top-ranked candidates to 3
respect the token budget
```

That minimum-3 fallback fixed the no-context collapse:

| case | recall@8 | context recall | context precision | hard neg ctx | included |
| --- | ---: | ---: | ---: | ---: | ---: |
| clean | 1.0000 | 0.9889 | 0.9167 | 0.0000 | 3.43 |
| 70% metadata | 0.8361 | 0.7111 | 0.6833 | 0.0000 | 3.17 |
| 40% metadata | 0.7111 | 0.5111 | 0.5083 | 0.0000 | 3.02 |
| 0% metadata | 0.5556 | 0.4694 | 0.4694 | 0.0000 | 3.00 |
| 25% wrong metadata | 0.8306 | 0.7167 | 0.6972 | 0.0028 | 3.12 |
| 50% wrong metadata | 0.6917 | 0.5250 | 0.5083 | 0.0028 | 3.10 |

Interpretation:

- The calibrator is still doing useful suppression work under degraded
  metadata.
- Perfect structured metadata was inflating the prior clean result.
- The remaining bottleneck is a mix of candidate generation and confidence
  calibration under degraded metadata.
- The packer should not be a pure threshold. It needs an adaptive minimum or a
  learned stop/abstention head trained on metadata dropout/noise.
- The next credible quality run should use the Hippo-encoder backend and
  metadata dropout/noise during training.

## Core Diagnosis

Current retrieval behaves too much like global semantic search:

```text
query -> vector/token candidates -> calibrator -> packed context
```

For agent memory, relevant evidence is often connected by relationship rather
than by direct semantic similarity to the query.

Examples:

- Same user.
- Same project.
- Same brand.
- Same session.
- Same entity.
- Correction/supersession chain.
- Same decision thread.
- Same source document.
- Recent related preference.

The next architecture should make the graph a candidate generator, not just a
storage or visualization layer. The first implementation confirmed this on the
session stress benchmark.

## Next Architecture

Target retrieval shape:

```text
query
  -> vector/token/profile seed retrieval
  -> deterministic typed graph expansion
  -> source-quota candidate pool
  -> learned calibrator
  -> compact evidence frame
```

The graph should be deterministic, typed, bounded, and explainable.

Important edge types:

- `same_user`
- `same_project`
- `same_brand`
- `same_entity`
- `same_session`
- `temporal_next`
- `temporal_previous`
- `same_context`
- `corrects`
- `supersedes`
- `superseded_by`
- `anti_memory_for`
- `derived_from`

Expansion order should favor high-signal edges first:

1. `corrects`, `supersedes`, `superseded_by`, `anti_memory_for`
2. `same_entity`, `same_project`, `same_brand`
3. `same_user`, `same_session`, `same_context`
4. `temporal_next`, `temporal_previous`

Expansion must be stable:

- Same memory state plus same query gives same candidates.
- Tie-break by deterministic score, edge type priority, timestamp, and memory id.
- No random walk.
- No nondeterministic insertion order.

## Implementation Plan

### 1. Add Candidate Source Attribution

Before changing retrieval quality, make the benchmark explain misses.

Add per-relevant-memory attribution:

- Found by vector.
- Found by token.
- Found by metadata/entity lookup.
- Found by graph expansion.
- Missed.

Metrics to report:

- `evidence_in_vector_fetch`
- `evidence_in_token_fetch`
- `evidence_in_metadata_fetch`
- `evidence_in_graph_fetch`
- `evidence_in_final_pool`
- `hit_any_relevant`
- `all_relevant_found`
- `recall@8`
- `recall@16`
- `recall@32`

Expected value:

- Shows whether the next bottleneck is vector search, lexical search, metadata
  lookup, graph expansion, or reranking.

### 2. Build Deterministic Metadata Expansion

Add a candidate source that pulls memories by structured fields:

- `user_id`
- `project_id`
- `brand`
- `session_id`
- entity names
- topic tags

For the session stress benchmark, this should be the first recall lift because
the synthetic workload contains structured session/use data.

Candidate quota example:

```text
128 vector/token candidates
128 metadata/entity candidates
128 graph candidates
128 correction/profile candidates
```

Do not let one source dominate the pool.

### 3. Build Typed Graph Candidate Expansion

After initial seeds are found, expand one or two deterministic hops.

Inputs:

- Top vector/token seeds.
- Top metadata/entity seeds.
- Optional profile memories.

Expansion rules:

- Expand corrections/supersessions first.
- Expand same-entity/project/brand second.
- Expand temporal/session/context third.
- Cap per seed.
- Cap per edge type.
- Deduplicate by memory id.
- Track source attribution.

The output should be a candidate set, not final context.

The calibrator still decides final ranking.

### 4. Run Pool And Quota Sweeps

Test whether recall is limited by pool size or retrieval source quality.

Suggested sweep:

| vector fetch | token fetch | metadata quota | graph quota | final pool |
| ---: | ---: | ---: | ---: | ---: |
| 1024 | 1024 | 0 | 0 | 128 |
| 2048 | 2048 | 0 | 0 | 256 |
| 4096 | 4096 | 0 | 0 | 512 |
| 1024 | 1024 | 128 | 128 | 256 |
| 2048 | 2048 | 256 | 256 | 512 |
| 4096 | 4096 | 256 | 256 | 512 |

Target:

- 50k `evidence_in_pool` above 0.65.
- 50k `recall@8` above 0.50.
- Hard-negative top-k rate below 0.10.

### 5. Train On The Stress Distribution

The current calibrator was not trained for the 10k/50k session workload.

Generate training rows from `session_memory_stress.py`:

- Same generated distribution.
- Larger memory stores.
- Mined hard negatives from actual retrieval pools.
- Query-echo decoys.
- Wrong-project same-user decoys.
- Stale preference decoys.
- Correction/supersession cases.

Train objective should favor recall first:

- Missing true evidence is worse than including some extra context.
- Precision can be recovered by the packer.
- Candidate generation and reranking should optimize evidence survival.

### 6. Add Learned Include/Stop Packing

Only after recall improves, reduce context noise.

Current context precision is poor because the packer fills too much context.

Add:

- Include threshold.
- Global stop/no-memory threshold.
- Token-length penalty.
- Evidence-density utility head.

Goal:

- Return fewer memories when fewer are enough.
- Return zero memories when no evidence exists.
- Keep recall high while improving context precision.

## What Not To Claim Yet

Do not claim:

- Production-ready 50k retrieval quality.
- High recall at scale.
- That reranking alone solves memory retrieval.
- That vector search has been beaten generally.

Safe claim:

```text
Hippo-Qwen has a deterministic stress harness that exposes where agent memory
retrieval fails at 10k-50k memories. Current calibration strongly suppresses
hard negatives, but scale-quality now depends on graph/entity candidate
generation and recall-oriented training.
```

## Immediate Next Commit Target

The first implementation loop is complete:

1. Added attribution metrics to `session_memory_stress.py`.
2. Added deterministic metadata/entity candidate expansion.
3. Added typed graph expansion as an optional candidate source.
4. Ran 10k and 50k baseline vs expanded comparisons.
5. Updated `HIPPO_QWEN_NEXT_IDEAS.txt`.

Next implementation loop:

1. Repeat the metadata-degradation sweep with the Hippo-encoder backend.
2. Complete the minimum-5 adaptive packing block after a clean Colab reset.
3. Train with metadata dropout/noise so the calibrator does not depend on clean
   structured fields.
4. Add quota sweeps to find the smallest metadata/graph pool that keeps 50k
   recall@8 above 0.50.
5. Add learned include/stop packing if the minimum-3 rule is still too blunt.
6. Add a real-ish write-path entity extractor or Qwen teacher path to generate
   metadata instead of giving the benchmark perfect metadata.

The current 50k synthetic result is strong enough to continue. It should not be
marketed as production quality until it survives missing/noisy metadata and a
less template-driven holdout.
