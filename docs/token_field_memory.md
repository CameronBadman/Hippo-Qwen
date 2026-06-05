# Token Field Memory

This is the v2 retrieval experiment. It stops treating a query as a ball in
embedding space and instead treats memory retrieval as a deterministic token
field problem.

## Model

Each memory emits a small set of field tokens:

- `action_id`: a deterministic random projection/action slot.
- `bucket`: the quantized value produced by that action.
- `weight`: the strength of that action for the memory.

A query emits the same kind of field tokens. Retrieval starts from token
collisions, applies a deterministic collision cap, then reranks candidates with
field overlap, semantic similarity, and activation-mask overlap.

This is intentionally closer to a token/ViT-style action space than a geometric
radius search. The current implementation uses deterministic sparse random
projections as a non-neural baseline. A trained encoder can replace token
emission later without changing the index contract.

## Layered Index

The index has a dense layer 0 and sparse routing layers above it.

- Layer 0 contains every memory node.
- Higher layers contain promoted routing nodes.
- Promotion is deterministic pseudo-random: stable node hashes are compared
  against a configured promotion probability.
- Promoted nodes appear in every layer from 0 through their maximum level.
- Each layer has its own collision table keyed by
  `(layer, action_id, bucket)`.

Search runs high-to-low:

1. Emit query field tokens.
2. Search the highest sparse collision layers and keep a deterministic beam.
3. Descend through routing layers, carrying the beam as routing pressure.
4. Search layer 0.
5. Include or reject candidates using collision and optional overlap gates.
6. Rerank accepted candidates.

This is the missing structure behind the token-field idea: the graph/index is a
layered collision system, not just a flat inverted list.

## Determinism

The prototype is deterministic by design:

- Projection plans are derived from stable FNV-1a hashes.
- Lexical boosts map text tokens to stable sparse action ids.
- Promotion levels are derived from stable node hashes.
- Candidate pruning is sorted by collision strength and node index.
- Ranked results are tie-broken by node id.

Same memory state plus same query should produce the same result.

## Current Role

This is not yet the final memory system. It is a test harness for the next
encoder direction:

1. Train an encoder to emit selective action-token fields.
2. Use hard negatives to reduce raw collision saturation.
3. Keep query latency under 200 ms while pushing recall and precision upward.
4. Compare against exact vector, HNSW/FAISS-style baselines, and the prior Hippo
   rope-grid experiments.

During this stage, candidate-pool recall is more important than final top-k
precision. The index should first avoid dropping relevant memories. Precision can
then be improved by training the encoder, adding better include gates, and using
a stronger reranker.

The benchmark entrypoint is:

```bash
python -m python.benchmarks.token_field_retrieval
```
