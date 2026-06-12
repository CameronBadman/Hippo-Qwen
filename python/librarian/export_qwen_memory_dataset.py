from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from python.benchmarks.session_memory_stress import build_store
from python.librarian.field_classifier import (
    PROMPT_VERSION,
    FieldPredictionCache,
    parse_prediction_json,
    prediction_cache_key,
    stable_prediction_cache_key,
)
from python.librarian.field_schema import FieldPrediction, FieldRegistry, default_field_registry
from python.librarian.qwen_teacher_fields import TEACHER_SYSTEM_PROMPT, build_user_prompt


QUERY_ITEM_RE = re.compile(r"^(query|evidence|hard)::(\d+)(?:::.*)?$")
BACKGROUND_RE = re.compile(r"^background::(\d+)$")


def load_registry(path: str) -> FieldRegistry:
    if path:
        registry_path = Path(path)
        if registry_path.exists():
            return FieldRegistry.load(registry_path)
    return default_field_registry()


def item_text(item: dict[str, Any]) -> str:
    return str(item.get("text") or item.get("query") or "")


def item_id(item: dict[str, Any]) -> str:
    return str(item.get("id") or item.get("query_id") or "")


def query_case_index(identifier: str) -> int | None:
    match = QUERY_ITEM_RE.match(identifier)
    if not match:
        return None
    return int(match.group(2))


def split_for_item(
    identifier: str,
    *,
    train_query_limit: int,
    holdout_query_start: int,
    holdout_query_limit: int,
    background_train_limit: int,
    background_holdout_limit: int,
) -> str:
    query_index = query_case_index(identifier)
    if query_index is not None:
        if query_index < train_query_limit:
            return "train"
        if holdout_query_start <= query_index < holdout_query_start + holdout_query_limit:
            return "holdout"
        return "unused"
    background = BACKGROUND_RE.match(identifier)
    if background:
        index = int(background.group(1))
        if index < background_train_limit:
            return "train"
        if index < background_train_limit + background_holdout_limit:
            return "holdout"
    return "unused"


def kind_for_item(identifier: str) -> str:
    if identifier.startswith("query::"):
        return "query"
    if identifier.startswith("evidence::"):
        return "evidence"
    if identifier.startswith("hard::"):
        return "hard_negative"
    if identifier.startswith("background::"):
        return "background"
    return "unknown"


def predictions_for_text(cache: FieldPredictionCache, registry: FieldRegistry, text: str) -> list[FieldPrediction]:
    key = prediction_cache_key(text, registry, PROMPT_VERSION)
    cached = cache.get(key)
    if cached is None:
        cached = cache.get(stable_prediction_cache_key(text, PROMPT_VERSION))
    if cached is None:
        return []
    return parse_prediction_json(cached, source_type="qwen_cache", teacher_version=PROMPT_VERSION)


def prediction_dicts(predictions: list[FieldPrediction]) -> list[dict[str, Any]]:
    return [
        {
            "field_name": prediction.field_name,
            "value": prediction.value,
            "confidence": round(float(prediction.confidence), 8),
            "source_span": prediction.source_span,
            "source_type": prediction.source_type,
            "teacher_version": prediction.teacher_version,
        }
        for prediction in predictions
    ]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def build_items(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    memories, queries = build_store(args.memory_count, args.queries, args.seed)
    query_items = [{"id": query["id"], "text": query["text"], "source": "query", **query} for query in queries]
    return memories, query_items


def export_field_rows(
    args: argparse.Namespace,
    registry: FieldRegistry,
    cache: FieldPredictionCache,
    memories: list[dict[str, Any]],
    queries: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    all_items = queries + memories
    normalized_rows: list[dict[str, Any]] = []
    sft_rows: list[dict[str, Any]] = []
    stats = {
        "items_seen": 0,
        "items_exported": 0,
        "items_missing_cache": 0,
        "train_items": 0,
        "holdout_items": 0,
    }
    for item in all_items:
        identifier = item_id(item)
        text = item_text(item)
        split = split_for_item(
            identifier,
            train_query_limit=args.train_query_limit,
            holdout_query_start=args.holdout_query_start,
            holdout_query_limit=args.holdout_query_limit,
            background_train_limit=args.background_train_limit,
            background_holdout_limit=args.background_holdout_limit,
        )
        stats["items_seen"] += 1
        if split == "unused" or not text:
            continue
        predictions = predictions_for_text(cache, registry, text)
        if not predictions and not args.include_missing_cache:
            stats["items_missing_cache"] += 1
            continue
        fields = prediction_dicts(predictions)
        kind = kind_for_item(identifier)
        query_index = query_case_index(identifier)
        row = {
            "dataset_version": args.dataset_version,
            "split": split,
            "kind": kind,
            "item_id": identifier,
            "query_index": query_index,
            "text": text,
            "qwen_fields": fields,
            "teacher": {
                "model": args.teacher_model,
                "prompt_version": PROMPT_VERSION,
                "registry_version": registry.version,
            },
        }
        normalized_rows.append(row)
        sft_rows.append(
            {
                "messages": [
                    {"role": "system", "content": TEACHER_SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_prompt(text, registry)},
                    {"role": "assistant", "content": json.dumps({"fields": fields}, sort_keys=True, separators=(",", ":"))},
                ],
                "metadata": {
                    "dataset_version": args.dataset_version,
                    "split": split,
                    "kind": kind,
                    "item_id": identifier,
                    "query_index": query_index,
                    "teacher_model": args.teacher_model,
                    "prompt_version": PROMPT_VERSION,
                },
            }
        )
        stats["items_exported"] += 1
        stats[f"{split}_items"] += 1
    return normalized_rows, sft_rows, stats


def memory_by_id(memories: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(memory.get("id") or ""): memory for memory in memories}


def query_index_from_query(query: dict[str, Any]) -> int:
    value = str(query.get("id") or "")
    index = query_case_index(value)
    if index is None:
        raise ValueError(f"query id does not contain a query index: {value}")
    return index


def export_retrieval_rows(
    args: argparse.Namespace,
    registry: FieldRegistry,
    cache: FieldPredictionCache,
    memories: list[dict[str, Any]],
    queries: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    by_id = memory_by_id(memories)
    rows: list[dict[str, Any]] = []
    stats = {
        "retrieval_rows": 0,
        "train_retrieval_rows": 0,
        "holdout_retrieval_rows": 0,
        "retrieval_missing_cache": 0,
    }
    for query in queries:
        index = query_index_from_query(query)
        query_split = split_for_item(
            str(query.get("id") or ""),
            train_query_limit=args.train_query_limit,
            holdout_query_start=args.holdout_query_start,
            holdout_query_limit=args.holdout_query_limit,
            background_train_limit=args.background_train_limit,
            background_holdout_limit=args.background_holdout_limit,
        )
        if query_split == "unused":
            continue
        query_text = item_text(query)
        query_fields = prediction_dicts(predictions_for_text(cache, registry, query_text))
        candidate_ids = [str(item) for item in query.get("relevant_ids") or []]
        candidate_ids.extend(f"hard::{index}::{slot}" for slot in range(12))
        for candidate_id in candidate_ids:
            candidate = by_id.get(candidate_id)
            if candidate is None:
                continue
            candidate_text = item_text(candidate)
            candidate_fields = prediction_dicts(predictions_for_text(cache, registry, candidate_text))
            if not candidate_fields and not args.include_missing_cache:
                stats["retrieval_missing_cache"] += 1
                continue
            is_evidence = candidate_id in {str(item) for item in query.get("relevant_ids") or []}
            rows.append(
                {
                    "dataset_version": args.dataset_version,
                    "split": query_split,
                    "query_id": str(query.get("id") or ""),
                    "query_index": index,
                    "query_text": query_text,
                    "query_fields": query_fields,
                    "candidate_id": candidate_id,
                    "candidate_kind": "evidence" if is_evidence else "hard_negative",
                    "candidate_text": candidate_text,
                    "candidate_fields": candidate_fields,
                    "is_evidence": bool(is_evidence),
                    "include_label": 1.0 if is_evidence else 0.0,
                    "teacher": {
                        "model": args.teacher_model,
                        "prompt_version": PROMPT_VERSION,
                        "registry_version": registry.version,
                    },
                }
            )
            stats["retrieval_rows"] += 1
            stats[f"{query_split}_retrieval_rows"] += 1
    return rows, stats


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    registry = load_registry(args.field_registry)
    cache = FieldPredictionCache(args.field_cache)
    memories, queries = build_items(args)
    field_rows, sft_rows, field_stats = export_field_rows(args, registry, cache, memories, queries)
    retrieval_rows, retrieval_stats = export_retrieval_rows(args, registry, cache, memories, queries)

    train_fields = [row for row in field_rows if row["split"] == "train"]
    holdout_fields = [row for row in field_rows if row["split"] == "holdout"]
    train_sft = [row for row in sft_rows if row["metadata"]["split"] == "train"]
    holdout_sft = [row for row in sft_rows if row["metadata"]["split"] == "holdout"]
    train_retrieval = [row for row in retrieval_rows if row["split"] == "train"]
    holdout_retrieval = [row for row in retrieval_rows if row["split"] == "holdout"]

    paths = {
        "field_labels_all": output_dir / "field_labels_all.jsonl",
        "field_labels_train": output_dir / "field_labels_train.jsonl",
        "field_labels_holdout": output_dir / "field_labels_holdout.jsonl",
        "qwen_sft_train": output_dir / "qwen_field_sft_train.jsonl",
        "qwen_sft_holdout": output_dir / "qwen_field_sft_holdout.jsonl",
        "retrieval_train": output_dir / "retrieval_pairs_train.jsonl",
        "retrieval_holdout": output_dir / "retrieval_pairs_holdout.jsonl",
    }
    write_jsonl(paths["field_labels_all"], field_rows)
    write_jsonl(paths["field_labels_train"], train_fields)
    write_jsonl(paths["field_labels_holdout"], holdout_fields)
    write_jsonl(paths["qwen_sft_train"], train_sft)
    write_jsonl(paths["qwen_sft_holdout"], holdout_sft)
    write_jsonl(paths["retrieval_train"], train_retrieval)
    write_jsonl(paths["retrieval_holdout"], holdout_retrieval)

    manifest = {
        "dataset": "hippo_qwen_memory_teacher",
        "dataset_version": args.dataset_version,
        "teacher_model": args.teacher_model,
        "prompt_version": PROMPT_VERSION,
        "registry_version": registry.version,
        "memory_count": args.memory_count,
        "queries": args.queries,
        "seed": args.seed,
        "split": {
            "train_query_limit": args.train_query_limit,
            "holdout_query_start": args.holdout_query_start,
            "holdout_query_limit": args.holdout_query_limit,
            "background_train_limit": args.background_train_limit,
            "background_holdout_limit": args.background_holdout_limit,
        },
        "stats": {**field_stats, **retrieval_stats},
        "files": {name: str(path) for name, path in paths.items()},
    }
    manifest_path = output_dir / "manifest.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest["manifest"] = str(manifest_path)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--field-cache", required=True)
    parser.add_argument("--field-registry", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--memory-count", type=int, default=10000)
    parser.add_argument("--queries", type=int, default=48)
    parser.add_argument("--seed", type=int, default=72000)
    parser.add_argument("--train-query-limit", type=int, default=32)
    parser.add_argument("--holdout-query-start", type=int, default=32)
    parser.add_argument("--holdout-query-limit", type=int, default=16)
    parser.add_argument("--background-train-limit", type=int, default=96)
    parser.add_argument("--background-holdout-limit", type=int, default=32)
    parser.add_argument("--teacher-model", default="qwen-plus")
    parser.add_argument("--dataset-version", default="hippo-qwen-memory-v1")
    parser.add_argument("--include-missing-cache", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
