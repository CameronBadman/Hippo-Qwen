from __future__ import annotations

import argparse
import json
import sys
import time
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
from python.benchmarks.rope_delta_grid import build_rope_delta_grid, query_embedding_for, search_rope_delta_grid


def payload_for_ranked(
    row: dict[str, Any],
    ranked: list[tuple[str, float, str]],
    id_to_candidate: dict[str, dict[str, Any]],
    relevant: set[str],
    query_embedding: list[float],
    args: argparse.Namespace,
) -> dict[str, Any]:
    seen = set()
    candidates = []
    for rank, (candidate_id, score, _) in enumerate(ranked[: args.max_candidates], start=1):
        candidate = dict(id_to_candidate.get(candidate_id) or {})
        if not candidate:
            continue
        candidate["base_rank"] = rank
        candidate["base_score"] = float(score)
        candidates.append(candidate)
        seen.add(candidate_id)
    if args.inject_missing_relevant:
        for candidate_id in sorted(relevant - seen):
            candidate = dict(id_to_candidate.get(candidate_id) or {})
            if not candidate:
                continue
            candidate["base_rank"] = args.max_candidates + 1
            candidate["base_score"] = -1.0
            candidates.append(candidate)
            if len(candidates) >= args.max_candidates:
                break
    task = row.get("retrieval_task") or {}
    return {
        "query": str(task.get("query") or ""),
        "answer": str(task.get("answer") or ""),
        "qa_id": str(task.get("qa_id") or ""),
        "question_type": str(task.get("question_type") or ""),
        "query_embedding": query_embedding,
        "budget": int(task.get("budget") or args.budget),
        "relevant_ids": sorted(relevant),
        "candidates": candidates[: args.max_candidates],
    }


def build_rows(records: list[dict[str, Any]], backend: Any, args: argparse.Namespace) -> list[dict[str, Any]]:
    output_rows = []
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    for record_index, record in enumerate(records):
        mode = unit_mode(record, args.unit)
        base = record_index_row(record, mode)
        if not base.get("candidates"):
            continue
        first = query_row(base, {"question": base["anchor"]["text"], "qa_id": "index"}, set(), args.budget)
        embedded_base = ensure_backend_embeddings(first, backend)
        id_to_candidate = {str(candidate.get("id") or ""): dict(candidate) for candidate in embedded_base.get("candidates", [])}
        meta = build_rope_delta_grid(embedded_base, backend, work_dir / f"record_{record_index:05d}" / "hippo_rope", args)
        qa_count = 0
        for qa in record.get("qa") or []:
            if bool(qa.get("abstention")) and not args.include_abstention:
                continue
            relevant = normalize_evidence(record, qa, mode)
            if not relevant:
                continue
            row = query_row(embedded_base, qa, relevant, args.budget)
            row = ensure_backend_embeddings(row, backend)
            ranked, _, _ = search_rope_delta_grid(row, backend, meta, args)
            output_rows.append(payload_for_ranked(row, ranked, id_to_candidate, relevant, query_embedding_for(row, backend), args))
            qa_count += 1
            if args.limit_questions > 0 and qa_count >= args.limit_questions:
                break
    return output_rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    dataset_path = load_dataset_path(args)
    records = load_records(dataset_path, args.limit_records)
    backend = build_embedding_backend(args)
    rows = build_rows(records, backend, args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n")
    result = {
        "dataset": str(dataset_path),
        "output": str(output),
        "rows": len(rows),
        "records": len(records),
        "embedding_backend": backend.name,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    if args.summary_json:
        summary = Path(args.summary_json)
        summary.parent.mkdir(parents=True, exist_ok=True)
        summary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="")
    parser.add_argument("--hf-repo", default=DEFAULT_HF_REPO)
    parser.add_argument("--hf-file", default=DEFAULT_HF_FILE)
    parser.add_argument("--output", default="artifacts/hippo_calibrator/memorycraft_train.jsonl")
    parser.add_argument("--summary-json", default="")
    parser.add_argument("--limit-records", type=int, default=20)
    parser.add_argument("--limit-questions", type=int, default=20)
    parser.add_argument("--include-abstention", action="store_true")
    parser.add_argument("--unit", choices=["auto", "turn", "session"], default="auto")
    parser.add_argument("--inject-missing-relevant", action="store_true")
    parser.add_argument("--max-candidates", type=int, default=128)
    parser.add_argument("--budget", type=int, default=900)
    parser.add_argument("--work-dir", default="artifacts/hippo_calibrator/build")
    parser.add_argument("--layers", type=int, default=128)
    parser.add_argument("--layer-schedule", choices=["auto", "consecutive", "spread"], default="spread")
    parser.add_argument("--dim-count", type=int, default=1024)
    parser.add_argument("--cell-width", type=float, default=0.03125)
    parser.add_argument("--radius", type=int, default=0)
    parser.add_argument("--max-cell-scan", type=int, default=4096)
    parser.add_argument("--min-layer-delta", type=float, default=0.0075)
    parser.add_argument("--min-query-layers", type=int, default=8)
    parser.add_argument("--max-query-layers", type=int, default=24)
    parser.add_argument("--min-node-layer-delta", type=float, default=0.0075)
    parser.add_argument("--min-node-layers", type=int, default=1)
    parser.add_argument("--max-node-layers", type=int, default=24)
    parser.add_argument("--edge-seed-count", type=int, default=48)
    parser.add_argument("--graph-depth", type=int, default=2)
    parser.add_argument("--final-fetch", type=int, default=128)
    parser.add_argument("--embedding-backend", choices=["hash", "hippo"], default="hash")
    parser.add_argument("--hippo-checkpoint", default="")
    parser.add_argument("--hippo-encoder-src", default="")
    parser.add_argument("--hippo-max-length", type=int, default=128)
    parser.add_argument("--hippo-batch-size", type=int, default=128)
    parser.add_argument("--device", default="")
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
