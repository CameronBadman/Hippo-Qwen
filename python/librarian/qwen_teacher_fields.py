from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from python.librarian.field_classifier import (
    PROMPT_VERSION,
    FieldPredictionCache,
    parse_prediction_json,
    prediction_cache_key,
    stable_predictions,
    stable_prediction_cache_key,
)
from python.librarian.field_schema import FieldPrediction, FieldRegistry, default_field_registry


DEFAULT_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen-plus"
TEACHER_SYSTEM_PROMPT = """You label agent-memory text for deterministic retrieval.
Return only valid JSON. Do not include markdown.
Extract compact fields useful for memory retrieval.
Use existing field names when possible. Propose a new field name only when the value is important and no existing field fits.
Every field item must include field_name, value, confidence, and source_span.
Confidence must be a number between 0 and 1.
Prefer stable canonical values over long prose."""


def registry_prompt(registry: FieldRegistry) -> str:
    fields = []
    for name, definition in sorted(registry.fields.items()):
        fields.append(
            {
                "name": name,
                "aliases": definition.aliases,
                "description": definition.description,
                "status": definition.status,
            }
        )
    return json.dumps(fields, sort_keys=True, separators=(",", ":"))


def build_user_prompt(text: str, registry: FieldRegistry) -> str:
    return "\n".join(
        [
            "Known field registry:",
            registry_prompt(registry),
            "",
            "Text to label:",
            str(text or ""),
            "",
            "Return JSON in this exact shape:",
            '{"fields":[{"field_name":"project","value":"example","confidence":0.9,"source_span":"example"}]}',
        ]
    )


def call_qwen_chat(
    *,
    api_key: str,
    base_url: str,
    model: str,
    text: str,
    registry: FieldRegistry,
    timeout: float,
    max_retries: int,
) -> list[FieldPrediction]:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": TEACHER_SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(text, registry)},
        ],
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    last_error: Exception | None = None
    for attempt in range(max(1, max_retries)):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
            content = data["choices"][0]["message"]["content"]
            predictions = parse_prediction_json(content, source_type="qwen_teacher", teacher_version=PROMPT_VERSION)
            return stable_predictions(predictions)
        except (urllib.error.HTTPError, urllib.error.URLError, KeyError, json.JSONDecodeError, ValueError) as exc:
            last_error = exc
            if attempt + 1 >= max_retries:
                break
            time.sleep(min(8.0, 1.5 * (attempt + 1)))
    raise RuntimeError(f"Qwen teacher call failed after {max_retries} attempts: {last_error}")


def load_registry(path: str) -> FieldRegistry:
    if path:
        registry_path = Path(path)
        if registry_path.exists():
            return FieldRegistry.load(registry_path)
    return default_field_registry()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_items(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.input_jsonl:
        return read_jsonl(Path(args.input_jsonl))
    if args.input_json:
        data = json.loads(Path(args.input_json).read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [dict(item) for item in data]
        if isinstance(data, dict):
            if isinstance(data.get("memories"), list):
                memories = [dict(item) for item in data.get("memories") or []]
                queries = [dict(item) for item in data.get("queries") or []]
                return memories + queries
            if isinstance(data.get("items"), list):
                return [dict(item) for item in data.get("items") or []]
        raise ValueError("--input-json must contain a list, items, or memories/queries")
    if args.from_session_stress:
        from python.benchmarks.session_memory_stress import build_store

        memories, queries = build_store(args.memory_count, args.queries, args.seed)
        return memories + queries
    raise ValueError("provide --input-jsonl, --input-json, or --from-session-stress")


def item_text(item: dict[str, Any], text_key: str) -> str:
    text = str(item.get(text_key) or item.get("text") or item.get("content") or item.get("question") or "")
    if text:
        return text
    if item.get("query"):
        return str(item.get("query") or "")
    return json.dumps(item, sort_keys=True)


def write_audit_row(path: Path | None, row: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def checkpoint_outputs(cache: FieldPredictionCache, registry: FieldRegistry, output_registry: str) -> None:
    cache.save()
    if output_registry:
        registry.save(output_registry)


def cache_key_for_text(text: str, registry: FieldRegistry, cache_key_mode: str) -> str:
    if cache_key_mode == "stable":
        return stable_prediction_cache_key(text, PROMPT_VERSION)
    if cache_key_mode == "registry":
        return prediction_cache_key(text, registry, PROMPT_VERSION)
    raise ValueError(f"unknown cache key mode: {cache_key_mode}")


def run(args: argparse.Namespace) -> dict[str, Any]:
    api_key = os.getenv(args.api_key_env)
    if not api_key and not args.dry_run:
        raise RuntimeError(f"missing API key env var: {args.api_key_env}")
    registry = load_registry(args.field_registry)
    cache = FieldPredictionCache(args.output_cache)
    items = load_items(args)
    if args.limit > 0:
        items = items[: args.limit]
    audit_path = Path(args.audit_jsonl) if args.audit_jsonl else None
    labelled = 0
    skipped = 0
    failures = 0
    for index, item in enumerate(items, start=1):
        text = item_text(item, args.text_key)
        key = cache_key_for_text(text, registry, args.cache_key_mode)
        cached = cache.get(key)
        if cached is not None and not args.refresh:
            for prediction in parse_prediction_json(
                cached,
                source_type="qwen_cache",
                teacher_version=PROMPT_VERSION,
            ):
                registry.record_predictions([prediction])
            skipped += 1
            continue
        if args.dry_run:
            predictions: list[FieldPrediction] = []
        else:
            predictions = call_qwen_chat(
                api_key=str(api_key),
                base_url=args.base_url,
                model=args.model,
                text=text,
                registry=registry,
                timeout=args.timeout,
                max_retries=args.max_retries,
            )
        cache.put(key, predictions)
        registry.record_predictions(predictions)
        labelled += 1
        write_audit_row(
            audit_path,
            {
                "index": index,
                "cache_key": key,
                "item_id": str(item.get("id") or item.get("qa_id") or index),
                "prediction_count": len(predictions),
                "predictions": [
                    {
                        "field_name": prediction.field_name,
                        "value": prediction.value,
                        "confidence": prediction.confidence,
                        "source_span": prediction.source_span,
                    }
                    for prediction in predictions
                ],
            },
        )
        if args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)
        if args.save_every > 0 and labelled % args.save_every == 0:
            checkpoint_outputs(cache, registry, args.output_registry)
    checkpoint_outputs(cache, registry, args.output_registry)
    return {
        "items": len(items),
        "labelled": labelled,
        "skipped_cached": skipped,
        "failures": failures,
        "cache": str(Path(args.output_cache)),
        "registry_version": registry.version,
        "model": args.model,
        "base_url": args.base_url,
        "cache_key_mode": args.cache_key_mode,
        "dry_run": bool(args.dry_run),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", default="")
    parser.add_argument("--input-json", default="")
    parser.add_argument("--from-session-stress", action="store_true")
    parser.add_argument("--memory-count", type=int, default=1000)
    parser.add_argument("--queries", type=int, default=20)
    parser.add_argument("--seed", type=int, default=72000)
    parser.add_argument("--text-key", default="text")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--field-registry", default="")
    parser.add_argument("--output-registry", default="")
    parser.add_argument("--output-cache", default="artifacts/field_classifier/qwen_teacher_cache.json")
    parser.add_argument("--audit-jsonl", default="")
    parser.add_argument("--api-key-env", default="DASHSCOPE_API_KEY")
    parser.add_argument("--base-url", default=os.getenv("QWEN_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--model", default=os.getenv("QWEN_MODEL", DEFAULT_MODEL))
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--cache-key-mode", choices=("registry", "stable"), default="registry")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
