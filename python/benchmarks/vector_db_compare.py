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
from python.benchmarks.hierarchical_file_ann import Ranked, ranked_signature
from python.benchmarks.hippocampus_retrieval import build_embedding_backend, ensure_backend_embeddings
from python.benchmarks.large_pool_retrieval import build_large_pool_case
from python.benchmarks.rope_delta_grid import build_rope_delta_grid, evaluate_one, query_embedding_for, search_rope_delta_grid
from python.benchmarks.skeleton_memory_index import compact_output, parse_scenarios
from python.librarian.features import cosine


def exact_vector_search(row: dict[str, Any], backend: Any, args: argparse.Namespace) -> tuple[Ranked, dict[str, float], set[str]]:
    started = time.perf_counter()
    query_embedding = query_embedding_for(row, backend)
    scored = []
    scanned = 0
    for candidate in row.get("candidates", []):
        embedding = candidate.get("embedding") or []
        scored.append((str(candidate.get("id") or ""), cosine(query_embedding, embedding), str(candidate.get("text") or "")))
        scanned += 1
    scored.sort(key=lambda item: (-item[1], item[0]))
    fetch = scored[: max(1, int(args.final_fetch))]
    stats = {
        "latency_ms": (time.perf_counter() - started) * 1000.0,
        "unique_nodes_read": float(scanned),
        "payload_reads": float(len(fetch)),
        "node_records_read": float(scanned),
        "edge_reads": 0.0,
        "edge_expansions": 0.0,
        "cells_touched": 0.0,
        "active_query_layers": 0.0,
        "skipped_layers": 0.0,
        "raw_final_candidate_count": float(scanned),
        "final_candidate_count": float(len(fetch)),
    }
    return fetch, stats, set()


def run_repeated(
    name: str,
    row: dict[str, Any],
    search: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    rows = []
    signatures = []
    for _ in range(max(1, int(args.repeat_searches))):
        raw_ranked, stats, protected_ids = search(row)
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
    return {"name": name, "metrics": metrics}


def run_dataset(
    name: str,
    row: dict[str, Any],
    backend: Any,
    work_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    exact = run_repeated(
        "exact_vector",
        row,
        lambda item: exact_vector_search(item, backend, args),
        args,
    )
    build_started = time.perf_counter()
    meta = build_rope_delta_grid(row, backend, work_dir / name / "hippo_rope", args)
    build_ms = (time.perf_counter() - build_started) * 1000.0
    hippo = run_repeated(
        "hippo_rope_grid",
        row,
        lambda item: search_rope_delta_grid(item, backend, meta, args),
        args,
    )
    hippo.update(
        {
            "build_latency_ms": build_ms,
            "memory_count": meta["memory_count"],
            "node_record_count": meta["node_record_count"],
            "cell_count": meta["cell_count"],
            "edge_count": meta["edge_count"],
            "total_index_bytes": meta["grid_bytes"] + meta["payload_bytes"] + meta["records_bytes"] + meta["edges_bytes"],
        }
    )
    exact.update(
        {
            "build_latency_ms": 0.0,
            "memory_count": len(row.get("candidates", [])),
            "node_record_count": len(row.get("candidates", [])),
            "cell_count": 0,
            "edge_count": 0,
            "total_index_bytes": 0,
        }
    )
    return {"name": name, "systems": [exact, hippo]}


def write_markdown(result: dict[str, Any], path: Path) -> None:
    lines = [
        "# Vector DB Compare",
        "",
        f"- backend: `{result['embedding_backend']}`",
        f"- pool_size: `{result['pool_size']}`",
        f"- growth_count: `{result['growth_count']}`",
        f"- growth_scenarios: `{','.join(result['growth_scenarios'])}`",
        f"- repeat_searches: `{result['repeat_searches']}`",
        f"- dim_count: `{result['dim_count']}`",
        f"- max_cell_scan: `{result['max_cell_scan']}`",
        "",
        "| dataset | system | memories | index MB | build ms | p50 ms | p95 ms | recall | precision | raw candidates p95 | node reads p95 | deterministic |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for dataset in result["datasets"]:
        for system in dataset["systems"]:
            metrics = system["metrics"]
            deterministic = 1.0 if metrics["determinism_mismatches"]["avg"] == 0.0 else 0.0
            lines.append(
                f"| {dataset['name']} | "
                f"{system['name']} | "
                f"{system['memory_count']} | "
                f"{system['total_index_bytes'] / (1024 * 1024):.2f} | "
                f"{system['build_latency_ms']:.2f} | "
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
    datasets = [run_dataset("baseline", base, backend, work_dir, args)]
    for scenario in args.growth_scenarios:
        grown = ensure_backend_embeddings(grow_row(base, scenario, args.growth_count, args.seed), backend)
        datasets.append(run_dataset(scenario, grown, backend, work_dir, args))
    result = {
        "benchmark": "vector_db_compare",
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
    parser.add_argument("--pool-size", type=int, default=75000)
    parser.add_argument("--growth-count", type=int, default=7500)
    parser.add_argument("--growth-scenarios", type=parse_scenarios, default=["combined"])
    parser.add_argument("--repeat-searches", type=int, default=3)
    parser.add_argument("--seed", type=int, default=62000)
    parser.add_argument("--work-dir", default="artifacts/vector_db_compare")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--budget", type=int, default=90)
    parser.add_argument("--compact-limit", type=int, default=3)
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
