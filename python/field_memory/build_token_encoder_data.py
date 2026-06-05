from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from python.benchmarks.hippocampus_retrieval import build_embedding_backend, ensure_backend_embeddings
from python.benchmarks.memorycraft_retrieval import (
    DEFAULT_HF_FILE,
    DEFAULT_HF_REPO,
    load_dataset_path,
    load_records,
    normalize_evidence,
    query_row,
    record_index_row,
    unit_mode,
)
from python.librarian.features import cosine


def compact_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(candidate.get("id") or ""),
        "text": str(candidate.get("text") or ""),
        "embedding": [float(value) for value in candidate.get("embedding") or []],
        "metadata": dict(candidate.get("metadata") or {}),
    }


def build_rows(record: dict[str, Any], backend: Any, args: argparse.Namespace, rng: random.Random) -> list[dict[str, Any]]:
    mode = unit_mode(record, args.unit)
    base = record_index_row(record, mode)
    if not base.get("candidates"):
        return []
    rows = []
    for qa_index, qa in enumerate(record.get("qa") or []):
        if bool(qa.get("abstention")) and not args.include_abstention:
            continue
        relevant = normalize_evidence(record, qa, mode)
        if not relevant:
            continue
        retrieval = query_row(base, qa, relevant, args.budget)
        embedded = ensure_backend_embeddings(retrieval, backend)
        task = embedded.get("retrieval_task") or {}
        query_embedding = [float(value) for value in task.get("query_embedding") or backend.embed_one(str(task.get("query") or ""))]
        positives = []
        negatives = []
        for candidate in embedded.get("candidates") or []:
            candidate_id = str(candidate.get("id") or "")
            if candidate_id in relevant:
                positives.append(compact_candidate(candidate))
            else:
                negatives.append(compact_candidate(candidate))
        if not positives or not negatives:
            continue
        scored_negatives = sorted(
            negatives,
            key=lambda item: (-cosine(query_embedding, item.get("embedding") or []), item["id"]),
        )
        hard = scored_negatives[: max(1, int(args.hard_negatives))]
        remaining = scored_negatives[max(1, int(args.hard_negatives)) :]
        rng.shuffle(remaining)
        random_negatives = remaining[: max(0, int(args.random_negatives))]
        rows.append(
            {
                "uid": str(record.get("uid") or ""),
                "source": str(record.get("source") or ""),
                "qa_index": qa_index,
                "unit": mode,
                "query": str(task.get("query") or ""),
                "query_embedding": query_embedding,
                "positive_ids": sorted(relevant),
                "positives": positives[: max(1, int(args.max_positives))],
                "hard_negatives": hard,
                "random_negatives": random_negatives,
                "budget": int(args.budget),
            }
        )
    return rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    rng = random.Random(args.seed)
    dataset_path = load_dataset_path(args)
    records = load_records(dataset_path, args.limit_records)
    backend = build_embedding_backend(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    row_count = 0
    positive_count = 0
    negative_count = 0
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            for row in build_rows(record, backend, args, rng):
                handle.write(json.dumps(row, sort_keys=True) + "\n")
                row_count += 1
                positive_count += len(row["positives"])
                negative_count += len(row["hard_negatives"]) + len(row["random_negatives"])
    result = {
        "dataset": str(dataset_path),
        "output": str(output),
        "rows": row_count,
        "positives": positive_count,
        "negatives": negative_count,
        "embedding_backend": backend.name,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="")
    parser.add_argument("--hf-repo", default=DEFAULT_HF_REPO)
    parser.add_argument("--hf-file", default=DEFAULT_HF_FILE)
    parser.add_argument("--output", default="artifacts/token_field_encoder/train.jsonl")
    parser.add_argument("--limit-records", type=int, default=0)
    parser.add_argument("--include-abstention", action="store_true")
    parser.add_argument("--unit", choices=["auto", "turn", "session"], default="auto")
    parser.add_argument("--budget", type=int, default=900)
    parser.add_argument("--max-positives", type=int, default=16)
    parser.add_argument("--hard-negatives", type=int, default=48)
    parser.add_argument("--random-negatives", type=int, default=48)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--embedding-backend", choices=["hash", "hippo"], default="hash")
    parser.add_argument("--hippo-checkpoint", default="")
    parser.add_argument("--hippo-encoder-src", default="")
    parser.add_argument("--hippo-max-length", type=int, default=128)
    parser.add_argument("--hippo-batch-size", type=int, default=128)
    parser.add_argument("--device", default="")
    parser.add_argument("--dim-count", type=int, default=1024)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()

