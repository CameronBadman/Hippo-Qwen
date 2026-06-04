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

## Hippocampus Retrieval Scorecard

The hippocampus path is currently benchmark-first. It tests sparse memory basins
and associative graph recall without changing the Go runtime search behavior.
HNSW remains a future acceleration layer for candidate discovery; it is not the
memory algorithm.

Generate multi-hop associative cases:

```bash
python3 -m python.synthetic.generate \
  --scenario associative_multihop \
  --output data/synthetic/associative_multihop.jsonl \
  --count 5000 \
  --candidates 32
```

Run the scorecard:

```bash
python3 -m python.benchmarks.hippocampus_retrieval \
  --dataset data/synthetic/associative_multihop.jsonl \
  --limit 1000 \
  --output-json artifacts/hippocampus/scorecard.json \
  --output-md artifacts/hippocampus/scorecard.md
```

Run the same scorecard with a Hippo-encoder checkpoint:

```bash
python3 -m python.benchmarks.hippocampus_retrieval \
  --dataset data/synthetic/associative_multihop.jsonl \
  --limit 1000 \
  --embedding-backend hippo \
  --hippo-encoder-src /path/to/Hippo-encoder/src \
  --hippo-checkpoint /path/to/extracted/hippoencoder-bge-small-all-nli-pair-500k-c025-epoch3 \
  --output-json artifacts/hippocampus/scorecard_hippo_encoder.json \
  --output-md artifacts/hippocampus/scorecard_hippo_encoder.md
```

The report compares vector-only retrieval, sparse basin routing, and
associative recall. In addition to normal context metrics, it reports bridge
recall, target recall, path success, stale exposure, and wrong-context exposure.

Run the large-pool scaling benchmark when testing whether an encoder keeps
candidate retrieval from exploding as memory grows:

```bash
python3 -m python.benchmarks.large_pool_retrieval \
  --cases 20 \
  --pool-size 5000 \
  --calibration-strategies top16,top32,top64,pct99,z1.5,margin0.05 \
  --embedding-backend hippo \
  --hippo-encoder-src /path/to/Hippo-encoder/src \
  --hippo-checkpoint /path/to/extracted/hippoencoder-bge-small-all-nli-pair-500k-c025-epoch3 \
  --output-json artifacts/hippocampus/large_pool_hippo.json \
  --output-md artifacts/hippocampus/large_pool_hippo.md
```

This benchmark reports final context quality, pre-graph candidate-set
size/precision at score thresholds and top-N cutoffs, and calibrated basin
strategies with per-query graph-expansion latency.

Run the file-backed hierarchical ANN prototype when testing a hippocampus-style
coarse-to-fine traversal. It writes a binary node heap (`nodes.hgb`) plus a
small offset index, lazily seeks/decompresses only visited nodes, and exposes a
promotion heuristic knob for making important/helpful memories appear at higher
levels:

```bash
python3 -m python.benchmarks.hierarchical_file_ann \
  --cases 20 \
  --pool-size 10000 \
  --stable-growth \
  --growth-noise-count 8 \
  --promotion-bias 0.0 \
  --compact-limit 3 \
  --determinism-repeats 3 \
  --output-json artifacts/hippocampus/hierarchical_file_ann.json \
  --output-md artifacts/hippocampus/hierarchical_file_ann.md
```

Use `--promotion-bias` to increase or decrease the deterministic chance that a
memory is promoted into higher-level basins. The report includes precision,
context recall, path success, binary file reads, unique nodes read, cache hits,
lazy edge expansions, latency, promotion rate, and repeated-query determinism.
With stable growth enabled, basin routing uses fixed keys plus monotonic
activation masks and absolute inclusion gates, so new unrelated nodes should add
candidates rather than evict previously reachable relevant memories. Use
`--growth-noise-count` to test that property directly.

Run the hard deterministic memory regression when testing growth pressure and
context saturation. It rebuilds each index twice, repeats searches against the
same mmap store, and evaluates unrelated growth, semantic decoys, conflicts,
repeated inserts, and combined pressure:

```bash
python3 -m python.benchmarks.hard_memory_regression \
  --cases 10 \
  --pool-size 5000 \
  --growth-count 1000 \
  --stable-max-basins 8 \
  --determinism-repeats 3 \
  --fail-on-regression \
  --output-json artifacts/hippocampus/hard_memory_regression.json \
  --output-md artifacts/hippocampus/hard_memory_regression.md
```

The hard regression reports raw candidate volume, compacted context precision
and recall, growth retention, top-N retention, repeated-query determinism, and
p95 latency. Use the same `--embedding-backend hippo`,
`--hippo-encoder-src`, and `--hippo-checkpoint` flags as the other benchmarks
to run it with Hippo-encoder routing vectors. `--stable-max-basins 8` is the
current quality-preserving pressure setting; lower basin or leaf caps are
faster but can drop conflict-route recall under adversarial growth.

Run the full local-first evaluation suite:

```bash
python3 -m python.benchmarks.evaluation_suite \
  --work-dir artifacts/librarian/eval_suite \
  --cases 5000 \
  --eval-limit 1000 \
  --epochs 6
```

The suite generates hard cases, trains the full transformer plus ablations
without state features and without ranking loss, runs retrieval benchmarks, and
writes threshold-sweep metrics to `summary.json` and `summary.md`.

## Multi-Seed Context Selector

The context selector is a separate experiment for less greedy retrieval. It
trains on the whole candidate set for a retrieval task and learns which memories
belong in the final context, rather than scoring every edge independently.
It uses only observable query, anchor, candidate, and memory-state features;
the synthetic role labels are withheld from the model. The training objective
also includes optional explanation heads: a single reason class and a multi-label
auxiliary head for relevance, context match, preference match/conflict, stale
duplicates, and wrong-context decisions. Auxiliary training uses capped
per-label positive weighting by default, and benchmark reports include both
stored validation-calibrated thresholds and tuned-threshold diagnostic F1.

```bash
python3 -m python.selector.train_selector \
  --dataset data/synthetic/librarian_hard_cases.jsonl \
  --output artifacts/librarian/context_selector.pt \
  --epochs 6
```

Benchmark it:

```bash
python3 -m python.selector.benchmark_selector \
  --dataset data/synthetic/librarian_hard_cases.jsonl \
  --checkpoint artifacts/librarian/context_selector.pt \
  --limit 1000
```

Run selector ablations for query-only, multi-seed, no-state, and no-ranking-loss
variants. The generated summaries include retrieval metrics, reason-head macro
recall, auxiliary-head macro F1, and per-role exposure rates showing which
synthetic decoys enter top-k or budgeted context:

```bash
python3 -m python.selector.ablation_suite \
  --work-dir artifacts/librarian/selector_ablation \
  --cases 5000 \
  --eval-limit 1000 \
  --epochs 6
```

By default this uses the `longitudinal` synthetic scenario, which is designed to
stress the non-greedy selector: generic queries, stale same-context negatives,
popular wrong-context negatives, and memory-state features such as use count,
evidence count, and last outcome.

Use `--scenario adversarial` for a harder stress test with contradictory
preferences, high-similarity stale memories, protected old positives, lexical
decoys, popular wrong-project memories, and near duplicates:

```bash
python3 -m python.selector.temporal_evaluation \
  --work-dir artifacts/librarian/adversarial_eval \
  --scenario adversarial \
  --cases 8000 \
  --train-fraction 0.75 \
  --eval-limit 1000 \
  --epochs 6
```

Run the online memory-evolution benchmark to compare static retrieval against
retrieval after simulated helpful/ignored/corrected feedback updates. The
evolved path mutates memory state fields and learns graph-bias signatures from
observable relationships such as project match, preference relation, stale
status, and duplicate-like text:

```bash
python3 -m python.selector.evolution_benchmark \
  --scenario adversarial \
  --cases 2000 \
  --evolution-policies off,always,uncertainty_gated,low_confidence_only,risk_aware,risk_rescue \
  --evolution-bias-scales 0,0.1,0.25,0.5,1.0 \
  --output-json artifacts/librarian/evolution/summary.json \
  --output-md artifacts/librarian/evolution/summary.md
```

Pass `--checkpoint artifacts/librarian/temporal_eval/multi_seed_full.pt` to
include the transformer context selector in the same static-vs-evolved loop.
For selector checkpoints, post-rank graph bias is disabled by default so the
benchmark measures evolved memory state as model input; pass
`--selector-post-rank-bias` only when explicitly testing external reranking.
`always` applies and updates online memory state for every row. The
`uncertainty_gated` and `low_confidence_only` policies only apply/update that
state when the current ranking looks weak, and reports split `state_applied_rate`
from `bias_applied_rate` so drift-control runs can be inspected directly.
`risk_aware` adds a budget-level guard for preference-change queries: it blocks
state when the selected context already has current-preference evidence, and
only applies state on those rows when selector confidence is genuinely weak.
This is intended to prevent clean preference-shift runs from collapsing into
always-on memory mutation while still leaving a path for corrupted-state runs.
`risk_rescue` is a less conservative diagnostic policy: preference-change rows
can still apply state when current-preference evidence is thin and the selector
confidence looks damaged, so strong-corruption experiments can test limited
online repair without returning to always-on mutation.

To train the selector on the same evolved state it sees online, expand a case
file with simulated feedback first:

```bash
python3 -m python.selector.evolve_dataset \
  --input artifacts/librarian/adversarial_eval/train_cases.jsonl \
  --output artifacts/librarian/adversarial_eval/train_cases_evolved.jsonl \
  --passes 2 \
  --feedback-scorer heuristic_graph

python3 -m python.selector.train_selector \
  --dataset artifacts/librarian/adversarial_eval/train_cases_evolved.jsonl \
  --output artifacts/librarian/adversarial_eval/multi_seed_full_evolved.pt \
  --feature-dim 31 \
  --epochs 6
```

Run a cross-seed regression to compare raw selector training against
evolved-state-augmented selector training:

```bash
python3 -m python.selector.evolved_state_regression \
  --work-dir artifacts/librarian/evolved_state_regression \
  --seeds 51,53,55 \
  --cases 5000 \
  --epochs 4
```

The regression writes per-seed checkpoints and aggregate `summary.json` /
`summary.md` reports. By default it evaluates `always@0`, which means online
memory-state mutation is enabled but selector post-rank graph bias remains off.
To test selective memory growth, use
`--evolution-policies off,uncertainty_gated --evolved-variant uncertainty_gated@0`
or the corresponding `low_confidence_only@0`, `risk_aware@0`, or
`risk_rescue@0` variant.
Use `--scenario preference_shift` to stress changing user preferences, where
old high-use memories conflict with newer corrections. Use
`--eval-state-corruption mild` or `--eval-state-corruption strong` to perturb
`use_count`, `evidence_count`, `importance`, `last_outcome`, and sometimes age
at evaluation time:

```bash
python3 -m python.selector.evolved_state_regression \
  --work-dir artifacts/librarian/preference_shift_corrupt \
  --scenario preference_shift \
  --eval-state-corruption mild \
  --seeds 61,63,65 \
  --cases 5000 \
  --epochs 4
```

Run a temporal evaluation that trains on earlier generated cases and tests on
later cases:

```bash
python3 -m python.selector.temporal_evaluation \
  --work-dir artifacts/librarian/temporal_eval \
  --cases 8000 \
  --train-fraction 0.75 \
  --eval-limit 1000 \
  --epochs 6
```

Run a stress matrix when you want one report across multiple hard scenarios and
state-corruption settings. This is the best Colab entry point for deciding
whether the evolved-state selector is robust enough to keep:

```bash
python3 -m python.selector.stress_matrix \
  --work-dir artifacts/librarian/stress_matrix \
  --scenarios adversarial,preference_shift \
  --eval-state-corruptions none,mild \
  --seeds 51,53,55 \
  --cases 5000 \
  --epochs 4 \
  --condition-timeout-seconds 1800 \
  --progress-file artifacts/librarian/stress_matrix/progress.jsonl
```

The matrix reuses `evolved_state_regression`, writes one subdirectory per
scenario/corruption pair, and produces `stress_matrix_summary.json` plus
`stress_matrix_summary.md` with recall, precision, noise, second-half drift, and
quality-gate readouts. The gates turn stress results into promotion criteria:
adversarial cases must improve recall without adding noise, and preference-shift
cases must avoid regressing clean or mildly corrupted baselines. `progress.jsonl`
records condition start/finish events so a Colab run can be inspected even if
notebook stdout stalls.

Inspect selector failures by synthetic role:

```bash
python3 -m python.selector.error_analysis \
  --dataset artifacts/librarian/temporal_eval/eval_cases.jsonl \
  --checkpoint artifacts/librarian/temporal_eval/multi_seed_full.pt \
  --output-md artifacts/librarian/temporal_eval/multi_seed_full_errors.md
```
