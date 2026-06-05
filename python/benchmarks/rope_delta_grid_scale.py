from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from python.benchmarks.hard_memory_regression import aggregate, grow_row
from python.benchmarks.hierarchical_file_ann import ranked_signature
from python.benchmarks.hippocampus_retrieval import build_embedding_backend, ensure_backend_embeddings
from python.benchmarks.large_pool_retrieval import build_large_pool_case
from python.benchmarks.rope_delta_grid import build_rope_delta_grid, evaluate_one, search_rope_delta_grid
from python.benchmarks.skeleton_memory_index import compact_output, parse_scenarios


def run_dataset(
    name: str,
    row: dict[str, Any],
    backend: Any,
    case_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    build_started = time.perf_counter()
    meta = build_rope_delta_grid(row, backend, case_dir, args)
    build_ms = (time.perf_counter() - build_started) * 1000.0
    rows = []
    signatures = []
    for _ in range(max(1, int(args.repeat_searches))):
        raw_ranked, stats, protected_ids = search_rope_delta_grid(row, backend, meta, args)
        ranked, stats = compact_output(row, raw_ranked, protected_ids, stats, args)
        rows.append(evaluate_one(row, ranked, stats, args))
        signatures.append(ranked_signature(raw_ranked))
    mismatches = sum(1 for signature in signatures[1:] if signature != signatures[0])
    metrics = aggregate(rows)
    metrics["determinism_mismatches"] = {
        "avg": float(mismatches),
        "min": float(mismatches),
        "max": float(mismatches),
        "p50": float(mismatches),
        "p95": float(mismatches),
    }
    return {
        "name": name,
        "memory_count": meta["memory_count"],
        "node_record_count": meta["node_record_count"],
        "cell_count": meta["cell_count"],
        "edge_count": meta["edge_count"],
        "build_latency_ms": build_ms,
        "grid_bytes": meta["grid_bytes"],
        "payload_bytes": meta["payload_bytes"],
        "records_bytes": meta["records_bytes"],
        "edges_bytes": meta["edges_bytes"],
        "total_index_bytes": meta["grid_bytes"] + meta["payload_bytes"] + meta["records_bytes"] + meta["edges_bytes"],
        "metrics": metrics,
    }


def write_markdown(result: dict[str, Any], path: Path) -> None:
    lines = [
        "# Rope Delta Grid Scale",
        "",
        f"- backend: `{result['embedding_backend']}`",
        f"- pool_size: `{result['pool_size']}`",
        f"- growth_count: `{result['growth_count']}`",
        f"- repeat_searches: `{result['repeat_searches']}`",
        f"- dim_count: `{result['dim_count']}`",
        f"- layers: `{result['layers']}`",
        f"- layer_schedule: `{result['layer_schedule']}`",
        f"- max_node_layers: `{result['max_node_layers']}`",
        f"- max_query_layers: `{result['max_query_layers']}`",
        f"- max_cell_scan: `{result['max_cell_scan']}`",
        "",
        "| dataset | memories | nodes | cells | index MB | build ms | p50 ms | p95 ms | recall | precision | raw candidates p95 | node reads p95 | deterministic |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for dataset in result["datasets"]:
        metrics = dataset["metrics"]
        deterministic = 1.0 if metrics["determinism_mismatches"]["avg"] == 0.0 else 0.0
        lines.append(
            f"| {dataset['name']} | "
            f"{dataset['memory_count']} | "
            f"{dataset['node_record_count']} | "
            f"{dataset['cell_count']} | "
            f"{dataset['total_index_bytes'] / (1024 * 1024):.2f} | "
            f"{dataset['build_latency_ms']:.2f} | "
            f"{metrics.get('latency_ms', {}).get('p50', 0.0):.2f} | "
            f"{metrics.get('latency_ms', {}).get('p95', 0.0):.2f} | "
            f"{metrics.get('retrieval_context_recall', {}).get('avg', 0.0):.4f} | "
            f"{metrics.get('retrieval_context_precision', {}).get('avg', 0.0):.4f} | "
            f"{metrics.get('raw_final_candidate_count', {}).get('p95', 0.0):.2f} | "
            f"{metrics.get('node_records_read', {}).get('p95', 0.0):.2f} | "
            f"{deterministic:.4f} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    backend = build_embedding_backend(args)
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    base = ensure_backend_embeddings(build_large_pool_case(args.seed, args.pool_size), backend)
    datasets = [run_dataset("baseline", base, backend, work_dir / "baseline", args)]
    for scenario in args.growth_scenarios:
        grown = ensure_backend_embeddings(grow_row(base, scenario, args.growth_count, args.seed), backend)
        datasets.append(run_dataset(scenario, grown, backend, work_dir / scenario, args))
    result = {
        "benchmark": "rope_delta_grid_scale",
        "embedding_backend": backend.name,
        "pool_size": args.pool_size,
        "growth_count": args.growth_count,
        "growth_scenarios": args.growth_scenarios,
        "repeat_searches": args.repeat_searches,
        "seed": args.seed,
        "dim_count": args.dim_count,
        "layers": args.layers,
        "layer_schedule": args.layer_schedule,
        "cell_width": args.cell_width,
        "radius": args.radius,
        "max_cell_scan": args.max_cell_scan,
        "min_layer_delta": args.min_layer_delta,
        "min_query_layers": args.min_query_layers,
        "max_query_layers": args.max_query_layers,
        "min_node_layer_delta": args.min_node_layer_delta,
        "min_node_layers": args.min_node_layers,
        "max_node_layers": args.max_node_layers,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "datasets": datasets,
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
    parser.add_argument("--pool-size", type=int, default=100000)
    parser.add_argument("--growth-count", type=int, default=10000)
    parser.add_argument("--growth-scenarios", type=parse_scenarios, default=["combined"])
    parser.add_argument("--repeat-searches", type=int, default=5)
    parser.add_argument("--seed", type=int, default=62000)
    parser.add_argument("--work-dir", default="artifacts/rope_delta_grid_scale")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--budget", type=int, default=90)
    parser.add_argument("--compact-limit", type=int, default=3)
    parser.add_argument("--layers", type=int, default=128)
    parser.add_argument("--layer-schedule", choices=["auto", "consecutive", "spread"], default="spread")
    parser.add_argument("--dim-count", type=int, default=1024)
    parser.add_argument("--cell-width", type=float, default=0.03125)
    parser.add_argument("--radius", type=int, default=0)
    parser.add_argument("--max-cell-scan", type=int, default=0)
    parser.add_argument("--min-layer-delta", type=float, default=0.0075)
    parser.add_argument("--min-query-layers", type=int, default=8)
    parser.add_argument("--max-query-layers", type=int, default=24)
    parser.add_argument("--min-node-layer-delta", type=float, default=0.0075)
    parser.add_argument("--min-node-layers", type=int, default=1)
    parser.add_argument("--max-node-layers", type=int, default=24)
    parser.add_argument("--edge-seed-count", type=int, default=48)
    parser.add_argument("--graph-depth", type=int, default=2)
    parser.add_argument("--final-fetch", type=int, default=96)
    parser.add_argument("--embedding-backend", choices=["hash", "hippo"], default="hash")
    parser.add_argument("--hippo-checkpoint", default="")
    parser.add_argument("--hippo-encoder-src", default="")
    parser.add_argument("--hippo-max-length", type=int, default=128)
    parser.add_argument("--hippo-batch-size", type=int, default=128)
    parser.add_argument("--device", default="")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-md", default="")
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
