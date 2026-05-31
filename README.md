# Hippo-Qwen

HippoGraph is a local-first prototype for a Qwen MemoryAgent: memory is stored
as a graph, and a librarian policy decides where new memories should connect.

This phase does not require Qwen Cloud credentials. It uses deterministic local
embeddings and a heuristic librarian behind the same contracts that the later
neighborhood transformer and Qwen teacher will use.

The vector layer is deliberately abstracted. The current runtime uses an exact
`LinearIndex`; the next acceleration step is an HNSW implementation that keeps
the same `VectorIndex` contract and only replaces seed/candidate discovery.

Edges intentionally carry compact routing features rather than prose: weight,
confidence, activation bitmask, last outcome, usage counts, and decay policy.
Traversal uses the activation mask to prefer edges that match the current query
without turning every edge into a text document.

## Run

```bash
go run ./cmd/hippograph -addr :8080
```

Open `http://localhost:8080`.

The server writes local graph state to `data/hippograph/`:

- `events.jsonl`: append-only event log
- `snapshot.json`: compact reload snapshot

## API

- `POST /memories`: insert a memory and place it in the graph
- `POST /search`: compare vector-only search with graph traversal search
- `POST /feedback`: strengthen or weaken returned nodes/edges
- `POST /maintenance/decay`: decay weak stale edges and garbage collect them
- `GET /graph`: return the current graph for visualization
- `GET /tools/list` and `POST /tools/call`: MCP-style tool wrapper

The web UI calls the same endpoints under `/api/*` to avoid static-route
collisions. The non-`/api` routes remain available for curl and compatibility.

## Python Librarian Placeholder

The Python service mirrors the PyTorch model contract:

```bash
python3 python/librarian/service.py --addr 127.0.0.1 --port 8090
```

Without a checkpoint it uses the heuristic librarian. With a checkpoint it
loads the trained neighborhood transformer:

```bash
python3 python/librarian/service.py \
  --checkpoint artifacts/librarian/neighborhood_transformer.pt
```

Generate heuristic-labeled synthetic cases:

```bash
python3 python/synthetic/generate.py \
  --output data/synthetic/librarian_cases.jsonl \
  --count 5000 \
  --candidates 32
```

Train in Colab or another PyTorch environment:

```bash
python3 -m python.training.train_librarian \
  --dataset data/synthetic/librarian_cases.jsonl \
  --output artifacts/librarian/neighborhood_transformer.pt \
  --epochs 8
```

The first model target is imitation of the heuristic librarian. Qwen teacher
labels can replace the synthetic labels later without changing the model or
service contract.

Generate harder local-first cases with retrieval labels:

```bash
python3 python/synthetic/generate.py \
  --output data/synthetic/librarian_hard_cases.jsonl \
  --count 12000 \
  --candidates 32
```

The current synthetic schema includes hard negatives, cross-project positives,
weak near-duplicates, compact memory-state features, and a `retrieval_task`
section for benchmarking retrieval under a context budget.

Run the benchmark without a model:

```bash
python3 -m python.benchmarks.benchmark_librarian \
  --dataset data/synthetic/librarian_hard_cases.jsonl \
  --limit 1000 \
  --output-md artifacts/librarian/benchmark.md
```

Run it with a trained checkpoint:

```bash
python3 -m python.benchmarks.benchmark_librarian \
  --dataset data/synthetic/librarian_hard_cases.jsonl \
  --checkpoint artifacts/librarian/neighborhood_transformer.pt \
  --limit 1000 \
  --output-json artifacts/librarian/benchmark.json \
  --output-md artifacts/librarian/benchmark.md
```
