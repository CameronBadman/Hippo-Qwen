# Retrieval Comparison Notes

These numbers are intended as a working baseline, not a win claim. The current
token-field system is deterministic and can bound memory reads, but FAISS and
hnswlib still win pure nearest-neighbor latency and quality on the small
MemoryCraft-style runs.

## MemoryCraft Sample

Run shape:

- Embeddings: Hippo encoder checkpoint trained from `BAAI/bge-small-en-v1.5`.
- Token field: `a100_hippo500_hardselect.pt`.
- Records/questions: 20.
- Average memories per record: 304.5.
- `top_k`: 8.
- `final_fetch`: 96.

| system | p95 ms | cand recall | cand precision | recall@8 | precision@8 | context recall | read/candidate avg | deterministic |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| exact_vector | 32.82 | 0.9500 | 0.0697 | 0.8252 | 0.2125 | 0.8281 | 304.5 | 1.0000 |
| faiss_flat | 0.35 | 0.9500 | 0.0697 | 0.8252 | 0.2125 | 0.8281 | 96 returned | 1.0000 |
| faiss_hnsw | 0.46 | 0.9500 | 0.0697 | 0.8252 | 0.2125 | 0.8281 | 96 returned | 1.0000 |
| hnswlib | 0.40 | 0.9500 | 0.0697 | 0.8252 | 0.2125 | 0.8281 | 96 returned | 1.0000 |
| token_field | 25.13 | 0.7900 | 0.0669 | 0.6314 | 0.1625 | 0.6914 | 66.9 scored | 1.0000 |

Result: FAISS wins this run. Token-field reads fewer candidate records than the
full vector baseline, but quality is lower and the Python token path is slower
than FAISS.

## LongMemEval Small Slice

Run shape:

- Records/questions: 20.
- Average memories per record: 28.1.
- Same Hippo encoder and token-field checkpoint.

| system | p95 ms | cand recall | cand precision | recall@8 | precision@8 | context recall | read/candidate avg | deterministic |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| exact_vector | 2.85 | 1.0000 | 0.1412 | 0.9228 | 0.3375 | 0.9196 | 28.1 | 1.0000 |
| faiss_flat | 0.17 | 1.0000 | 0.1412 | 0.9228 | 0.3375 | 0.9196 | 28 returned | 1.0000 |
| faiss_hnsw | 0.22 | 1.0000 | 0.1412 | 0.9228 | 0.3375 | 0.9196 | 28 returned | 1.0000 |
| hnswlib | 0.18 | 1.0000 | 0.1412 | 0.9228 | 0.3375 | 0.9196 | 28 returned | 1.0000 |
| token_field | 9.28 | 0.9362 | 0.1454 | 0.8764 | 0.3187 | 0.8806 | 24.1 scored | 1.0000 |

Result: FAISS also wins this small-memory run. Token-field is close on recall,
but not ahead.

## 10k Synthetic Agent-Memory Run

Run shape:

- Embeddings: deterministic hash backend, 768 dimensions.
- Pool size: 10,000.
- Growth: 0.
- This task currently favors graph/multihop bridge retrieval, so vector systems
  scoring 0.0 should not be treated as a universal FAISS failure.

| system | p95 ms | recall | precision | raw candidates p95 | reads p95 | deterministic |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| exact_vector | 537.91 | 0.0000 | 0.0000 | 10000 | 10000 | 1.0000 |
| faiss_flat | 2.91 | 0.0000 | 0.0000 | 96 | 96 returned | 1.0000 |
| faiss_hnsw | 0.60 | 0.0000 | 0.0000 | 96 | 96 returned | 1.0000 |
| hnswlib | 0.40 | 0.0000 | 0.0000 | 96 | 96 returned | 1.0000 |
| token_field | 167.23 | 0.3333 | 0.3333 | 788 | 512 scored | 1.0000 |

Result: token-field shows a possible advantage on graph-shaped retrieval, while
remaining under the 200 ms target at 10k nodes. This is promising, but not a fair
general vector DB benchmark yet because the synthetic labels are not pure nearest
neighbors.

## Current Verdict

Hippo/token-field is not beating FAISS as a general vector database yet. The path
worth pursuing is narrower and more interesting: beat vector-only retrieval on
agent-memory workloads where the answer depends on learned shapes, stable graph
growth, bridge memories, and bounded candidate reads.

The next fair target is a benchmark with:

- Real Hippo embeddings.
- Hard negatives.
- Labels for direct facts, preference drift, temporal facts, and multihop/bridge
  recall.
- Equal `top_k`, `final_fetch`, and context budget across systems.
- Separate reporting for returned payloads, known vector scans, and Hippo
  candidate records scored.
