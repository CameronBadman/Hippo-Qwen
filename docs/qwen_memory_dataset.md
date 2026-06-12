# Qwen Memory Dataset

Hippo-Qwen can turn Qwen teacher labels into a reusable dataset instead of calling Qwen for every benchmark run.

The dataset has two purposes:

- Train or distill a smaller field classifier from Qwen-generated metadata labels.
- Train and evaluate Hippo's retrieval calibrator on query/candidate pairs with Qwen-derived fields attached.

## Export

```bash
python -m python.librarian.export_qwen_memory_dataset \
  --field-cache artifacts/field_classifier/qwen_teacher_cache.json \
  --output-dir artifacts/qwen_memory_dataset/v1 \
  --memory-count 10000 \
  --queries 48 \
  --train-query-limit 32 \
  --holdout-query-start 32 \
  --holdout-query-limit 16
```

If the cache was generated with a custom registry, pass the same registry used for labeling:

```bash
  --field-registry artifacts/field_classifier/field_registry.json
```

The cache key depends on text, prompt version, registry version, and registry field names, so using a different registry can intentionally produce cache misses.

## Files

The exporter writes:

- `manifest.json`: dataset version, split config, file paths, and row counts.
- `field_labels_all.jsonl`: normalized labeled items.
- `field_labels_train.jsonl`: train split for field extraction.
- `field_labels_holdout.jsonl`: holdout split for field extraction.
- `qwen_field_sft_train.jsonl`: Qwen-style chat/SFT rows.
- `qwen_field_sft_holdout.jsonl`: held-out chat/SFT rows.
- `retrieval_pairs_train.jsonl`: query/candidate retrieval labels for calibrator training.
- `retrieval_pairs_holdout.jsonl`: held-out retrieval labels for validation.

## Row Shapes

`field_labels_*.jsonl` rows contain:

```json
{
  "dataset_version": "hippo-qwen-memory-v1",
  "split": "train",
  "kind": "evidence",
  "item_id": "evidence::0::preference",
  "query_index": 0,
  "text": "...",
  "qwen_fields": [
    {
      "field_name": "project",
      "value": "project_091",
      "confidence": 1.0,
      "source_span": "project_091",
      "source_type": "qwen_cache",
      "teacher_version": "field-classifier-v1"
    }
  ],
  "teacher": {
    "model": "qwen-plus",
    "prompt_version": "field-classifier-v1",
    "registry_version": 1
  }
}
```

`qwen_field_sft_*.jsonl` rows contain OpenAI-compatible chat messages:

```json
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "{\"fields\":[...]}"}
  ],
  "metadata": {
    "split": "train",
    "kind": "evidence",
    "item_id": "evidence::0::preference"
  }
}
```

`retrieval_pairs_*.jsonl` rows contain:

```json
{
  "split": "holdout",
  "query_id": "query::32",
  "query_text": "...",
  "query_fields": [],
  "candidate_id": "hard::32::0",
  "candidate_kind": "hard_negative",
  "candidate_text": "...",
  "candidate_fields": [],
  "is_evidence": false,
  "include_label": 0.0
}
```

## Qwen Cloud Use

Treat `qwen_field_sft_train.jsonl` as the upload candidate for a Qwen/Alibaba fine-tuning or dataset-import workflow. The exact upload command depends on the Qwen Cloud surface available to the account, but the format is intentionally simple chat JSONL.

For retrieval experiments, use `retrieval_pairs_train.jsonl` and `retrieval_pairs_holdout.jsonl` locally. These are the rows that prove whether Qwen-derived metadata improves Hippo's candidate generation and calibration.

## Split Rule

The default split is deterministic:

- Train queries: `query::0` through `query::31`
- Holdout queries: `query::32` through `query::47`
- Train background: `background::0` through `background::95`
- Holdout background: `background::96` through `background::127`

This avoids the smoke-test mistake where the calibrator was trained and evaluated on the same query cases.
