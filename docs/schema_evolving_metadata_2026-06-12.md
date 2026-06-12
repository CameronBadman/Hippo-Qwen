# Schema-Evolving Metadata Classifier

Date: 2026-06-12

## What Landed

Hippo-Qwen now has a deterministic derived-metadata path for the session memory
stress benchmark.

The core idea:

```text
memory/query text
  -> field classifier
  -> derived_metadata fields with confidence/provenance
  -> metadata/graph candidate generation
  -> calibrator features
```

Original metadata is never overwritten. Derived fields live under
`derived_metadata`, so retrieval can compare:

- original metadata only
- derived metadata only
- original plus derived metadata

## New Components

- `python/librarian/field_schema.py`
  - seeded field registry
  - field predictions
  - deterministic JSON persistence
  - auto-promotion gates for proposed fields

- `python/librarian/field_classifier.py`
  - deterministic rule classifier for local stress tests
  - Qwen teacher cache reader
  - stable cache key based on prompt version, registry version, and text
  - derived metadata application helper

- `python/benchmarks/session_memory_stress.py`
  - `--derived-metadata none|rules|qwen-cache`
  - `--metadata-source original|derived|both`
  - `--field-registry`
  - `--field-cache`
  - `--output-field-registry`
  - schema auto-promotion gate flags

- `python/librarian/hippo_calibrator.py`
  - 48-feature mode can now include derived metadata match/confidence features
  - existing 16/33-feature checkpoints still work because features are truncated
    to checkpoint config

## How To Run A Local Smoke Test

Original plus derived fields:

```bash
python -m python.benchmarks.session_memory_stress \
  --memory-count 500 \
  --queries 8 \
  --vector-index numpy \
  --vector-fetch 64 \
  --token-fetch 64 \
  --metadata-fetch 64 \
  --graph-fetch 64 \
  --candidate-pool 64 \
  --embedding-backend hash \
  --derived-metadata rules \
  --metadata-source both \
  --packing-threshold 0.40 \
  --packing-threshold-min-items 3
```

Derived-only recovery after deleting original metadata:

```bash
python -m python.benchmarks.session_memory_stress \
  --memory-count 500 \
  --queries 8 \
  --vector-index numpy \
  --vector-fetch 64 \
  --token-fetch 64 \
  --metadata-fetch 64 \
  --graph-fetch 64 \
  --candidate-pool 64 \
  --embedding-backend hash \
  --metadata-availability 0 \
  --derived-metadata rules \
  --metadata-source derived \
  --packing-threshold 0.40 \
  --packing-threshold-min-items 3
```

## Smoke Result

The derived-only degraded smoke test completed with:

- `evidence_in_metadata_fetch`: 1.0000
- `evidence_in_graph_fetch`: 1.0000
- `evidence_in_pool`: 1.0000
- hybrid `recall@8`: 0.3333
- deterministic mismatches: 0

Interpretation:

- Derived metadata can recover the candidate pool when original metadata is
  removed in the small synthetic stress setup.
- Raw hybrid ranking remains noisy without the learned calibrator.
- The next real test is a 50k run with a 48-feature stress calibrator trained on
  metadata dropout/noise.

## Qwen Teacher Cache Shape

`--derived-metadata qwen-cache` reads cached teacher outputs. The cache key is:

```text
sha256(prompt_version + registry_version + field_names + text)
```

Each cache value can be either a list or an object with `fields`:

```json
{
  "fields": [
    {
      "field_name": "project",
      "value": "project_042",
      "confidence": 0.94,
      "source_span": "project_042"
    }
  ]
}
```

For offline experiments, `--qwen-cache-rule-fallback` can fill missing cache
entries with deterministic rule predictions and write them back to the cache.

## Building A Qwen Teacher Cache

The cache builder is:

```bash
python -m python.librarian.qwen_teacher_fields
```

It uses the OpenAI-compatible Qwen endpoint through standard-library HTTP. No
Alibaba CLI is required. Provide credentials through environment variables, not
through files:

```bash
export DASHSCOPE_API_KEY="..."
export QWEN_BASE_URL="https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
export QWEN_MODEL="qwen-plus"
```

Generate labels for a small generated stress sample:

```bash
python -m python.librarian.qwen_teacher_fields \
  --from-session-stress \
  --memory-count 1000 \
  --queries 20 \
  --limit 200 \
  --output-cache artifacts/field_classifier/qwen_teacher_cache.json \
  --audit-jsonl artifacts/field_classifier/qwen_teacher_audit.jsonl
```

Run a deterministic replay benchmark from that cache:

```bash
python -m python.benchmarks.session_memory_stress \
  --memory-count 1000 \
  --queries 20 \
  --vector-index numpy \
  --vector-fetch 128 \
  --token-fetch 128 \
  --metadata-fetch 128 \
  --graph-fetch 128 \
  --candidate-pool 128 \
  --embedding-backend hash \
  --metadata-availability 0 \
  --derived-metadata qwen-cache \
  --metadata-source derived \
  --field-cache artifacts/field_classifier/qwen_teacher_cache.json
```

Dry-run without network access:

```bash
python -m python.librarian.qwen_teacher_fields \
  --from-session-stress \
  --memory-count 120 \
  --queries 2 \
  --limit 3 \
  --output-cache /tmp/qwen_teacher_dry_cache.json \
  --audit-jsonl /tmp/qwen_teacher_dry_audit.jsonl \
  --dry-run
```

## Auto-Promotion

New field names are first recorded as `proposed`. They become routing fields
only if `--schema-auto-promote` is enabled and all deterministic gates pass:

- minimum observation count
- minimum mean confidence
- minimum distinct values
- minimum validation lift
- maximum hard-negative delta

Validation lift and hard-negative deltas are supplied as JSON files so promotion
is replayable and auditable. This avoids letting live model output mutate the
retrieval schema directly.

## Next Experiment

Run on Colab:

1. Build 50k metadata-degradation datasets.
2. Export calibration rows with `--derived-metadata rules --metadata-source both`.
3. Train a 48-feature calibrator with metadata dropout/noise.
4. Evaluate:
   - original only
   - derived only
   - both
   - degraded original plus derived recovery
5. Replace `rules` with `qwen-cache` once Qwen teacher labels are available.
