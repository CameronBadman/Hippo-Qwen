from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from python.benchmarks.hard_memory_regression import aggregate
from python.benchmarks.hierarchical_file_ann import Ranked, ranked_signature
from python.benchmarks.hippocampus_retrieval import build_embedding_backend, ensure_backend_embeddings
from python.benchmarks.memorycraft_retrieval import (
    DEFAULT_HF_FILE,
    DEFAULT_HF_REPO,
    evidence_metrics,
    index_size,
    load_dataset_path,
    load_records,
    normalize_evidence,
    query_row,
    record_index_row,
    run_queries,
    unit_mode,
)
from python.benchmarks.rope_delta_grid import build_rope_delta_grid, search_rope_delta_grid
from python.benchmarks.vector_db_compare import exact_vector_search
from python.field_memory.token_field import build_token_field_index, search_token_field


def parse_systems(value: str) -> list[str]:
    systems = [item.strip() for item in value.split(",") if item.strip()]
    valid = {"exact_vector", "hippo_rope_grid", "token_field"}
    unknown = sorted(set(systems) - valid)
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown systems: {','.join(unknown)}")
    return systems


def candidate_pool_metrics(row: dict[str, Any], ranked: Ranked, candidate_ids: set[str]) -> dict[str, float]:
    relevant = {str(item) for item in (row.get("retrieval_task") or {}).get("relevant_ids") or []}
    candidates = {str(item) for item in candidate_ids}
    if not candidates:
        candidates = {str(item[0]) for item in ranked}
    if not relevant:
        return {
            "candidate_recall": 0.0,
            "candidate_precision": 0.0,
            "candidate_count": float(len(candidates)),
            "candidate_hits": 0.0,
        }
    hits = len(relevant & candidates)
    return {
        "candidate_recall": hits / len(relevant),
        "candidate_precision": hits / max(1, len(candidates)),
        "candidate_count": float(len(candidates)),
        "candidate_hits": float(hits),
    }


def evaluate_search(row: dict[str, Any], ranked: Ranked, stats: dict[str, float], candidate_ids: set[str], args: argparse.Namespace) -> dict[str, float]:
    out = {
        "latency_ms": float(stats.get("latency_ms") or 0.0),
        "unique_nodes_read": float(stats.get("unique_nodes_read") or 0.0),
        "payload_reads": float(stats.get("payload_reads") or 0.0),
        "node_records_read": float(stats.get("node_records_read") or 0.0),
        "edge_reads": float(stats.get("edge_reads") or 0.0),
        "edge_expansions": float(stats.get("edge_expansions") or 0.0),
        "routing_layer_reads": float(stats.get("routing_layer_reads") or 0.0),
        "routing_candidate_count": float(stats.get("routing_candidate_count") or 0.0),
        "raw_final_candidate_count": float(stats.get("raw_final_candidate_count") or 0.0),
        "final_candidate_count": float(stats.get("final_candidate_count") or len(ranked)),
        "calibrator_latency_ms": float(stats.get("calibrator_latency_ms") or 0.0),
    }
    out.update(evidence_metrics(row, ranked, args.top_k, args.budget))
    out.update(candidate_pool_metrics(row, ranked, candidate_ids))
    return out


def repeated(
    query_rows: list[dict[str, Any]],
    search: Any,
    args: argparse.Namespace,
) -> tuple[list[dict[str, float]], int]:
    rows = []
    mismatches = 0
    for row in query_rows:
        signatures = []
        metrics = []
        for _ in range(max(1, int(args.repeat_searches))):
            ranked, stats, candidate_ids = search(row)
            signatures.append(ranked_signature(ranked))
            metrics.append(evaluate_search(row, ranked, stats, candidate_ids, args))
        if any(signature != signatures[0] for signature in signatures[1:]):
            mismatches += 1
        rows.append(metrics[-1])
    return rows, mismatches


def run_record(record: dict[str, Any], record_number: int, backend: Any, work_dir: Path, args: argparse.Namespace) -> dict[str, Any] | None:
    mode = unit_mode(record, args.unit)
    base = record_index_row(record, mode)
    if not base.get("candidates"):
        return None
    qa_rows = []
    for qa in record.get("qa") or []:
        if bool(qa.get("abstention")) and not args.include_abstention:
            continue
        relevant = normalize_evidence(record, qa, mode)
        if not relevant:
            continue
        qa_rows.append(query_row(base, qa, relevant, args.budget))
        if args.limit_questions > 0 and len(qa_rows) >= args.limit_questions:
            break
    if not qa_rows:
        return None
    embedded_base = ensure_backend_embeddings(qa_rows[0], backend)
    for row in qa_rows:
        row["anchor"] = embedded_base["anchor"]
        row["candidates"] = embedded_base["candidates"]

    systems = []
    if "exact_vector" in args.systems:
        rows, mismatches = repeated(qa_rows, lambda row: exact_vector_search(row, backend, args), args)
        systems.append(
            {
                "name": "exact_vector",
                "metrics": aggregate(rows),
                "_metric_rows": rows,
                "determinism_mismatches": mismatches,
                "build_latency_ms": 0.0,
                "memory_count": len(embedded_base["candidates"]),
                "total_index_bytes": 0,
            }
        )
    if "hippo_rope_grid" in args.systems:
        meta = build_rope_delta_grid(embedded_base, backend, work_dir / f"record_{record_number:05d}" / "hippo_rope", args)
        rows, mismatches = repeated(qa_rows, lambda row: search_rope_delta_grid(row, backend, meta, args), args)
        systems.append(
            {
                "name": "hippo_rope_grid",
                "metrics": aggregate(rows),
                "_metric_rows": rows,
                "determinism_mismatches": mismatches,
                "build_latency_ms": float(meta.get("build_latency_ms") or 0.0),
                "memory_count": int(meta["memory_count"]),
                "total_index_bytes": index_size(meta),
            }
        )
    if "token_field" in args.systems:
        index = build_token_field_index(embedded_base, args)
        rows, mismatches = repeated(qa_rows, lambda row: search_token_field(row, backend, index, args), args)
        systems.append(
            {
                "name": "token_field",
                "metrics": aggregate(rows),
                "_metric_rows": rows,
                "determinism_mismatches": mismatches,
                "build_latency_ms": float(index.build_latency_ms),
                "memory_count": len(index.nodes),
                "total_index_bytes": int(index.index_bytes),
            }
        )

    return {
        "uid": str(record.get("uid") or ""),
        "source": str(record.get("source") or ""),
        "unit": mode,
        "question_count": len(qa_rows),
        "memory_count": len(embedded_base["candidates"]),
        "systems": systems,
    }


def system_rollup(records: list[dict[str, Any]], systems: list[str]) -> dict[str, Any]:
    out = {}
    for name in systems:
        metric_rows = []
        build_latency = []
        index_bytes = []
        mismatches = 0
        memories = []
        for record in records:
            for system in record.get("systems") or []:
                if system.get("name") != name:
                    continue
                metric_rows.extend(system.get("_metric_rows") or [])
                build_latency.append(float(system.get("build_latency_ms") or 0.0))
                index_bytes.append(float(system.get("total_index_bytes") or 0.0))
                mismatches += int(system.get("determinism_mismatches") or 0)
                memories.append(float(system.get("memory_count") or 0.0))
        out[name] = {
            "record_count": len(build_latency),
            "query_count": len(metric_rows),
            "metrics": aggregate(metric_rows) if metric_rows else {},
            "build_latency_ms_avg": sum(build_latency) / max(1, len(build_latency)),
            "index_mb_avg": (sum(index_bytes) / max(1, len(index_bytes))) / (1024 * 1024),
            "memory_count_avg": sum(memories) / max(1, len(memories)),
            "determinism_mismatches": mismatches,
        }
    return out


def public_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for record in records:
        item = dict(record)
        systems = []
        for system in item.get("systems") or []:
            public = dict(system)
            public.pop("_metric_rows", None)
            systems.append(public)
        item["systems"] = systems
        out.append(item)
    return out


def write_markdown(result: dict[str, Any], path: Path) -> None:
    lines = [
        "# Token Field Retrieval",
        "",
        f"- dataset: `{result['dataset']}`",
        f"- records: `{result['record_count']}`",
        f"- questions: `{result['question_count']}`",
        f"- systems: `{','.join(result['systems'])}`",
        f"- action_count: `{result['action_count']}`",
        f"- query_token_count: `{result['query_token_count']}`",
        f"- node_token_count: `{result['node_token_count']}`",
        f"- routing_layers: `{result['routing_layers']}`",
        "",
        "| system | records | avg memories | index MB | build ms | p95 ms | cand recall | cand precision | recall@k | precision@k | context recall | routed | read | mrr | deterministic |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, rollup in result["rollup"].items():
        metrics = rollup["metrics"]
        deterministic = 1.0 if int(rollup["determinism_mismatches"]) == 0 else 0.0
        lines.append(
            f"| {name} | {rollup['record_count']} | {rollup['memory_count_avg']:.1f} | "
            f"{rollup['index_mb_avg']:.2f} | {rollup['build_latency_ms_avg']:.2f} | "
            f"{metrics.get('latency_ms', {}).get('p95', 0.0):.2f} | "
            f"{metrics.get('candidate_recall', {}).get('avg', 0.0):.4f} | "
            f"{metrics.get('candidate_precision', {}).get('avg', 0.0):.4f} | "
            f"{metrics.get('recall_at_k', {}).get('avg', 0.0):.4f} | "
            f"{metrics.get('precision_at_k', {}).get('avg', 0.0):.4f} | "
            f"{metrics.get('context_recall', {}).get('avg', 0.0):.4f} | "
            f"{metrics.get('routing_candidate_count', {}).get('avg', 0.0):.1f} | "
            f"{metrics.get('unique_nodes_read', {}).get('avg', 0.0):.1f} | "
            f"{metrics.get('mrr', {}).get('avg', 0.0):.4f} | {deterministic:.4f} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    dataset_path = load_dataset_path(args)
    records = load_records(dataset_path, args.limit_records)
    backend = build_embedding_backend(args)
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    evaluated = []
    for index, record in enumerate(records):
        result = run_record(record, index, backend, work_dir, args)
        if result is not None:
            evaluated.append(result)
    result = {
        "benchmark": "token_field_retrieval",
        "dataset": str(dataset_path),
        "hf_repo": args.hf_repo,
        "hf_file": args.hf_file,
        "record_count": len(evaluated),
        "question_count": sum(int(record["question_count"]) for record in evaluated),
        "unit": args.unit,
        "systems": args.systems,
        "embedding_backend": backend.name,
        "token_encoder_checkpoint": args.token_encoder_checkpoint,
        "dim_count": args.dim_count,
        "action_count": args.action_count,
        "query_token_count": args.query_token_count,
        "node_token_count": args.node_token_count,
        "routing_layers": args.routing_layers,
        "promotion_probability": args.promotion_probability,
        "include_min_collision": args.include_min_collision,
        "include_min_overlap": args.include_min_overlap,
        "bucket_width": args.bucket_width,
        "bucket_radius": args.bucket_radius,
        "top_k": args.top_k,
        "budget": args.budget,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "rollup": system_rollup(evaluated, args.systems),
        "records": public_records(evaluated),
    }
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.output_md:
        write_markdown(result, Path(args.output_md))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="")
    parser.add_argument("--hf-repo", default=DEFAULT_HF_REPO)
    parser.add_argument("--hf-file", default=DEFAULT_HF_FILE)
    parser.add_argument("--limit-records", type=int, default=20)
    parser.add_argument("--limit-questions", type=int, default=1)
    parser.add_argument("--include-abstention", action="store_true")
    parser.add_argument("--unit", choices=["auto", "turn", "session"], default="auto")
    parser.add_argument("--systems", type=parse_systems, default=["exact_vector", "hippo_rope_grid", "token_field"])
    parser.add_argument("--repeat-searches", type=int, default=1)
    parser.add_argument("--work-dir", default="artifacts/token_field_retrieval")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--budget", type=int, default=900)
    parser.add_argument("--final-fetch", type=int, default=96)
    parser.add_argument("--dim-count", type=int, default=1024)
    parser.add_argument("--action-count", type=int, default=256)
    parser.add_argument("--query-token-count", type=int, default=32)
    parser.add_argument("--node-token-count", type=int, default=32)
    parser.add_argument("--projection-width", type=int, default=16)
    parser.add_argument("--bucket-width", type=float, default=0.055)
    parser.add_argument("--bucket-radius", type=int, default=1)
    parser.add_argument("--min-candidates", type=int, default=16)
    parser.add_argument("--max-candidates", type=int, default=192)
    parser.add_argument("--routing-layers", type=int, default=6)
    parser.add_argument("--promotion-probability", type=float, default=0.35)
    parser.add_argument("--promotion-bias", type=float, default=0.0)
    parser.add_argument("--routing-beam-width", type=int, default=48)
    parser.add_argument("--include-min-collision", type=float, default=0.0)
    parser.add_argument("--include-min-overlap", type=float, default=0.0)
    parser.add_argument("--layers", type=int, default=128)
    parser.add_argument("--layer-schedule", choices=["auto", "consecutive", "spread"], default="spread")
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
    parser.add_argument("--embedding-backend", choices=["hash", "hippo"], default="hash")
    parser.add_argument("--hippo-checkpoint", default="")
    parser.add_argument("--hippo-encoder-src", default="")
    parser.add_argument("--hippo-max-length", type=int, default=128)
    parser.add_argument("--hippo-batch-size", type=int, default=128)
    parser.add_argument("--device", default="")
    parser.add_argument("--token-encoder-checkpoint", default="")
    parser.add_argument("--token-encoder-device", default="")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-md", default="")
    parser.add_argument("--print-records", action="store_true")
    args = parser.parse_args()
    result = run(args)
    printed = result if args.print_records else {key: value for key, value in result.items() if key != "records"}
    print(json.dumps(printed, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
