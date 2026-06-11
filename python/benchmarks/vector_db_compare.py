from __future__ import annotations

import argparse
import json
import math
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
from python.field_memory.token_field import (
    TokenFieldIndex,
    build_token_field_index,
    search_token_field,
    search_token_field_candidates,
    token_field_candidate_scores,
)
from python.librarian.features import cosine


def parse_systems(value: str) -> list[str]:
    systems = [item.strip() for item in value.split(",") if item.strip()]
    valid = {
        "exact_vector",
        "faiss_flat",
        "faiss_hnsw",
        "hnswlib",
        "hippo_rope_grid",
        "token_field",
        "hybrid_faiss_hnsw_token",
        "hybrid_union_token",
        "hippo_calibrated_union",
    }
    unknown = sorted(set(systems) - valid)
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown systems: {','.join(unknown)}")
    return systems


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
        "vector_index_scan_count": float(scanned),
    }
    return fetch, stats, set()


def normalized_float32(values: list[float]) -> Any:
    import numpy as np

    vector = np.asarray(values, dtype="float32")
    norm = float(np.linalg.norm(vector))
    if norm > 0.0 and math.isfinite(norm):
        vector = vector / norm
    return vector


def build_faiss(row: dict[str, Any], args: argparse.Namespace, kind: str) -> dict[str, Any]:
    try:
        import faiss
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("faiss systems require faiss-cpu or faiss-gpu to be installed") from exc

    candidates = list(row.get("candidates", []))
    if not candidates:
        raise ValueError("cannot build faiss index with no candidates")
    vectors = [normalized_float32([float(value) for value in candidate.get("embedding") or []]) for candidate in candidates]
    dim = int(vectors[0].shape[0])
    matrix = np.vstack(vectors).astype("float32", copy=False)
    if kind == "flat":
        index = faiss.IndexFlatIP(dim)
    elif kind == "hnsw":
        index = faiss.IndexHNSWFlat(dim, int(args.faiss_hnsw_m), faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = int(args.faiss_ef_construction)
        index.hnsw.efSearch = int(args.faiss_ef_search)
    else:
        raise ValueError(f"unknown faiss index kind: {kind}")
    index.add(matrix)
    return {
        "index": index,
        "ids": [str(candidate.get("id") or "") for candidate in candidates],
        "texts": [str(candidate.get("text") or "") for candidate in candidates],
        "memory_count": len(candidates),
        "index_bytes": int(matrix.nbytes),
        "kind": kind,
    }


def faiss_search(row: dict[str, Any], backend: Any, built: dict[str, Any], args: argparse.Namespace, fetch_count: int | None = None) -> tuple[Ranked, dict[str, float], set[str]]:
    started = time.perf_counter()
    query = normalized_float32(query_embedding_for(row, backend)).reshape(1, -1)
    limit = int(args.final_fetch) if fetch_count is None else int(fetch_count)
    limit = max(1, min(limit, int(built["memory_count"])))
    scores, indices = built["index"].search(query.astype("float32", copy=False), limit)
    ranked = []
    for score, index in zip(scores[0].tolist(), indices[0].tolist()):
        if index < 0:
            continue
        ranked.append((built["ids"][index], float(score), built["texts"][index]))
    stats = {
        "latency_ms": (time.perf_counter() - started) * 1000.0,
        "unique_nodes_read": float(len(ranked)),
        "payload_reads": float(len(ranked)),
        "node_records_read": float(len(ranked)),
        "edge_reads": 0.0,
        "edge_expansions": 0.0,
        "cells_touched": 0.0,
        "active_query_layers": 0.0,
        "skipped_layers": 0.0,
        "raw_final_candidate_count": float(limit),
        "final_candidate_count": float(len(ranked)),
        "vector_index_scan_count": float(built["memory_count"]) if built.get("kind") == "flat" else -1.0,
    }
    return ranked, stats, set()


def faiss_candidate_scores(
    row: dict[str, Any],
    backend: Any,
    built: dict[str, Any],
    token_index: TokenFieldIndex,
    args: argparse.Namespace,
) -> tuple[dict[int, float], dict[str, float]]:
    ranked, stats, _ = faiss_search(row, backend, built, args, int(args.hybrid_candidate_fetch))
    candidates: dict[int, float] = {}
    for node_id, score, _ in ranked:
        node_index = token_index.id_to_index.get(str(node_id))
        if node_index is None:
            continue
        candidates[node_index] = max(candidates.get(node_index, -99.0), float(score))
    return candidates, stats


def hybrid_union_candidate_scores(
    row: dict[str, Any],
    backend: Any,
    built: dict[str, Any],
    token_index: TokenFieldIndex,
    args: argparse.Namespace,
) -> tuple[dict[int, float], dict[str, float]]:
    vector_scores, vector_stats = faiss_candidate_scores(row, backend, built, token_index, args)
    token_limit = int(getattr(args, "hybrid_token_candidate_fetch", 0) or getattr(args, "max_candidates", 512))
    token_scores, token_stats, _ = token_field_candidate_scores(row, backend, token_index, args, max_candidates=token_limit)
    max_token = max((abs(score) for score in token_scores.values()), default=1.0) or 1.0
    vector_weight = float(getattr(args, "hybrid_union_vector_weight", 0.70))
    token_weight = float(getattr(args, "hybrid_union_token_weight", 0.30))
    union: dict[int, float] = {}
    for node_index, score in vector_scores.items():
        union[node_index] = max(union.get(node_index, -99.0), vector_weight * float(score))
    for node_index, score in token_scores.items():
        normalized = float(score) / max_token
        union[node_index] = max(union.get(node_index, -99.0), token_weight * normalized)
    stats = {
        "vector_candidate_latency_ms": float(vector_stats.get("latency_ms") or 0.0),
        "vector_candidate_count": float(len(vector_scores)),
        "token_candidate_latency_ms": float(token_stats.get("latency_ms") or 0.0),
        "token_candidate_count": float(len(token_scores)),
        "union_candidate_count": float(len(union)),
        "vector_index_scan_count": float(vector_stats.get("vector_index_scan_count", -1.0)),
    }
    for key, value in token_stats.items():
        if key not in stats:
            stats[f"token_{key}"] = float(value)
    return union, stats


def hybrid_faiss_token_search(
    row: dict[str, Any],
    backend: Any,
    faiss_built: dict[str, Any],
    token_index: TokenFieldIndex,
    args: argparse.Namespace,
) -> tuple[Ranked, dict[str, float], set[str]]:
    started = time.perf_counter()
    candidate_scores, faiss_stats = faiss_candidate_scores(row, backend, faiss_built, token_index, args)
    ranked, token_stats, candidate_ids = search_token_field_candidates(row, backend, token_index, args, candidate_scores)
    stats = dict(token_stats)
    stats["latency_ms"] = (time.perf_counter() - started) * 1000.0
    stats["vector_candidate_latency_ms"] = float(faiss_stats.get("latency_ms") or 0.0)
    stats["vector_candidate_count"] = float(len(candidate_scores))
    stats["vector_index_scan_count"] = float(faiss_stats.get("vector_index_scan_count", -1.0))
    stats["raw_final_candidate_count"] = float(len(candidate_scores))
    return ranked, stats, candidate_ids


def hybrid_union_token_search(
    row: dict[str, Any],
    backend: Any,
    faiss_built: dict[str, Any],
    token_index: TokenFieldIndex,
    args: argparse.Namespace,
) -> tuple[Ranked, dict[str, float], set[str]]:
    started = time.perf_counter()
    candidate_scores, union_stats = hybrid_union_candidate_scores(row, backend, faiss_built, token_index, args)
    ranked, token_stats, candidate_ids = search_token_field_candidates(row, backend, token_index, args, candidate_scores)
    stats = dict(token_stats)
    stats["latency_ms"] = (time.perf_counter() - started) * 1000.0
    stats.update(union_stats)
    stats["raw_final_candidate_count"] = float(len(candidate_scores))
    return ranked, stats, candidate_ids


def calibrator_payload(
    row: dict[str, Any],
    ranked: Ranked,
    id_to_candidate: dict[str, dict[str, Any]],
    backend: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    candidates = []
    task = row.get("retrieval_task") or {}
    query_embedding = query_embedding_for(row, backend)
    for rank, (candidate_id, score, _) in enumerate(ranked[: int(args.calibrator_max_candidates)], start=1):
        candidate = dict(id_to_candidate.get(str(candidate_id)) or {})
        if not candidate:
            continue
        candidate["base_rank"] = rank
        candidate["base_score"] = float(score)
        candidates.append(candidate)
    return {
        "query": str(task.get("query") or ""),
        "qa_id": str(task.get("qa_id") or ""),
        "question_type": str(task.get("question_type") or ""),
        "query_embedding": query_embedding,
        "budget": int(task.get("budget") or args.budget),
        "feature_ablation": str(getattr(args, "calibrator_feature_ablation", "none") or "none"),
        "candidates": candidates,
    }


def calibrated_union_search(
    row: dict[str, Any],
    backend: Any,
    faiss_built: dict[str, Any],
    token_index: TokenFieldIndex,
    calibrator: Any,
    id_to_candidate: dict[str, dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[Ranked, dict[str, float], set[str]]:
    from python.librarian.hippo_calibrator import rerank_with_calibrator

    started = time.perf_counter()
    raw_ranked, stats, candidate_ids = hybrid_union_token_search(row, backend, faiss_built, token_index, args)
    ranked = rerank_with_calibrator(
        calibrator,
        calibrator_payload(row, raw_ranked, id_to_candidate, backend, args),
        relevance_weight=args.rerank_relevance_weight,
        include_weight=args.rerank_include_weight,
        base_weight=args.rerank_base_weight,
        utility_weight=args.rerank_utility_weight,
    )
    total_latency_ms = (time.perf_counter() - started) * 1000.0
    search_latency_ms = float(stats.get("latency_ms") or 0.0)
    out = dict(stats)
    out["search_latency_ms"] = search_latency_ms
    out["latency_ms"] = total_latency_ms
    out["calibrator_latency_ms"] = max(0.0, total_latency_ms - search_latency_ms)
    return ranked, out, candidate_ids


def build_hnswlib(row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    try:
        import hnswlib
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("hnswlib system requires hnswlib to be installed") from exc

    candidates = list(row.get("candidates", []))
    if not candidates:
        raise ValueError("cannot build hnswlib index with no candidates")
    vectors = [normalized_float32([float(value) for value in candidate.get("embedding") or []]) for candidate in candidates]
    dim = int(vectors[0].shape[0])
    matrix = np.vstack(vectors).astype("float32", copy=False)
    labels = np.arange(len(candidates), dtype=np.int64)
    index = hnswlib.Index(space="cosine", dim=dim)
    index.init_index(max_elements=len(candidates), ef_construction=int(args.hnswlib_ef_construction), M=int(args.hnswlib_m))
    index.add_items(matrix, labels)
    index.set_ef(int(args.hnswlib_ef_search))
    return {
        "index": index,
        "ids": [str(candidate.get("id") or "") for candidate in candidates],
        "texts": [str(candidate.get("text") or "") for candidate in candidates],
        "memory_count": len(candidates),
        "index_bytes": int(matrix.nbytes),
    }


def hnswlib_search(row: dict[str, Any], backend: Any, built: dict[str, Any], args: argparse.Namespace) -> tuple[Ranked, dict[str, float], set[str]]:
    started = time.perf_counter()
    query = normalized_float32(query_embedding_for(row, backend)).reshape(1, -1)
    fetch_count = max(1, min(int(args.final_fetch), int(built["memory_count"])))
    labels, distances = built["index"].knn_query(query, k=fetch_count)
    ranked = []
    for index, distance in zip(labels[0].tolist(), distances[0].tolist()):
        ranked.append((built["ids"][index], 1.0 - float(distance), built["texts"][index]))
    stats = {
        "latency_ms": (time.perf_counter() - started) * 1000.0,
        "unique_nodes_read": float(len(ranked)),
        "payload_reads": float(len(ranked)),
        "node_records_read": float(len(ranked)),
        "edge_reads": 0.0,
        "edge_expansions": 0.0,
        "cells_touched": 0.0,
        "active_query_layers": 0.0,
        "skipped_layers": 0.0,
        "raw_final_candidate_count": float(fetch_count),
        "final_candidate_count": float(len(ranked)),
        "vector_index_scan_count": -1.0,
    }
    return ranked, stats, set()


def metric_cell(metrics: dict[str, Any], name: str, stat: str = "p95") -> str:
    value = float(metrics.get(name, {}).get(stat, 0.0))
    if value < 0.0:
        return "n/a"
    return f"{value:.2f}"


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
    calibrator: Any = None,
) -> dict[str, Any]:
    systems = []
    if "exact_vector" in args.systems:
        exact = run_repeated(
            "exact_vector",
            row,
            lambda item: exact_vector_search(item, backend, args),
            args,
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
        systems.append(exact)
    for system_name, kind in (("faiss_flat", "flat"), ("faiss_hnsw", "hnsw")):
        if system_name not in args.systems:
            continue
        build_started = time.perf_counter()
        faiss_built = build_faiss(row, args, kind)
        build_ms = (time.perf_counter() - build_started) * 1000.0
        faiss_result = run_repeated(
            system_name,
            row,
            lambda item: faiss_search(item, backend, faiss_built, args),
            args,
        )
        faiss_result.update(
            {
                "build_latency_ms": build_ms,
                "memory_count": faiss_built["memory_count"],
                "node_record_count": faiss_built["memory_count"],
                "cell_count": 0,
                "edge_count": 0,
                "total_index_bytes": faiss_built["index_bytes"],
            }
        )
        systems.append(faiss_result)
    if "hnswlib" in args.systems:
        build_started = time.perf_counter()
        hnswlib_built = build_hnswlib(row, args)
        build_ms = (time.perf_counter() - build_started) * 1000.0
        hnswlib_result = run_repeated(
            "hnswlib",
            row,
            lambda item: hnswlib_search(item, backend, hnswlib_built, args),
            args,
        )
        hnswlib_result.update(
            {
                "build_latency_ms": build_ms,
                "memory_count": hnswlib_built["memory_count"],
                "node_record_count": hnswlib_built["memory_count"],
                "cell_count": 0,
                "edge_count": 0,
                "total_index_bytes": hnswlib_built["index_bytes"],
            }
        )
        systems.append(hnswlib_result)
    if "hippo_rope_grid" in args.systems:
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
        systems.append(hippo)
    if "token_field" in args.systems:
        build_started = time.perf_counter()
        token_index = build_token_field_index(row, args)
        build_ms = (time.perf_counter() - build_started) * 1000.0
        token_result = run_repeated(
            "token_field",
            row,
            lambda item: search_token_field(item, backend, token_index, args),
            args,
        )
        token_result.update(
            {
                "build_latency_ms": build_ms,
                "memory_count": len(token_index.nodes),
                "node_record_count": len(token_index.nodes),
                "cell_count": int(sum(len(values) for values in token_index.layered_inverted.values())),
                "edge_count": 0,
                "total_index_bytes": int(token_index.index_bytes),
            }
        )
        systems.append(token_result)
    if "hybrid_faiss_hnsw_token" in args.systems:
        build_started = time.perf_counter()
        faiss_built = build_faiss(row, args, "hnsw")
        token_index = build_token_field_index(row, args)
        build_ms = (time.perf_counter() - build_started) * 1000.0
        hybrid_result = run_repeated(
            "hybrid_faiss_hnsw_token",
            row,
            lambda item: hybrid_faiss_token_search(item, backend, faiss_built, token_index, args),
            args,
        )
        hybrid_result.update(
            {
                "build_latency_ms": build_ms,
                "memory_count": len(token_index.nodes),
                "node_record_count": len(token_index.nodes),
                "cell_count": int(sum(len(values) for values in token_index.layered_inverted.values())),
                "edge_count": 0,
                "total_index_bytes": int(token_index.index_bytes) + int(faiss_built["index_bytes"]),
                "hybrid_candidate_fetch": int(args.hybrid_candidate_fetch),
                "hybrid_source_weight": float(args.hybrid_source_weight),
                "hybrid_semantic_weight": float(args.hybrid_semantic_weight),
                "hybrid_field_weight": float(args.hybrid_field_weight),
                "hybrid_activation_weight": float(args.hybrid_activation_weight),
            }
        )
        systems.append(hybrid_result)
    if "hybrid_union_token" in args.systems:
        build_started = time.perf_counter()
        faiss_built = build_faiss(row, args, "hnsw")
        token_index = build_token_field_index(row, args)
        build_ms = (time.perf_counter() - build_started) * 1000.0
        union_result = run_repeated(
            "hybrid_union_token",
            row,
            lambda item: hybrid_union_token_search(item, backend, faiss_built, token_index, args),
            args,
        )
        union_result.update(
            {
                "build_latency_ms": build_ms,
                "memory_count": len(token_index.nodes),
                "node_record_count": len(token_index.nodes),
                "cell_count": int(sum(len(values) for values in token_index.layered_inverted.values())),
                "edge_count": 0,
                "total_index_bytes": int(token_index.index_bytes) + int(faiss_built["index_bytes"]),
                "hybrid_candidate_fetch": int(args.hybrid_candidate_fetch),
                "hybrid_token_candidate_fetch": int(getattr(args, "hybrid_token_candidate_fetch", args.max_candidates)),
                "hybrid_union_vector_weight": float(args.hybrid_union_vector_weight),
                "hybrid_union_token_weight": float(args.hybrid_union_token_weight),
                "hybrid_source_weight": float(args.hybrid_source_weight),
                "hybrid_semantic_weight": float(args.hybrid_semantic_weight),
                "hybrid_field_weight": float(args.hybrid_field_weight),
                "hybrid_activation_weight": float(args.hybrid_activation_weight),
            }
        )
        systems.append(union_result)
    if "hippo_calibrated_union" in args.systems:
        if calibrator is None:
            raise ValueError("--calibrator-checkpoint is required when --systems includes hippo_calibrated_union")
        build_started = time.perf_counter()
        faiss_built = build_faiss(row, args, "hnsw")
        token_index = build_token_field_index(row, args)
        id_to_candidate = {str(candidate.get("id") or ""): dict(candidate) for candidate in row.get("candidates", [])}
        build_ms = (time.perf_counter() - build_started) * 1000.0
        calibrated_result = run_repeated(
            "hippo_calibrated_union",
            row,
            lambda item: calibrated_union_search(item, backend, faiss_built, token_index, calibrator, id_to_candidate, args),
            args,
        )
        calibrated_result.update(
            {
                "build_latency_ms": build_ms,
                "memory_count": len(token_index.nodes),
                "node_record_count": len(token_index.nodes),
                "cell_count": int(sum(len(values) for values in token_index.layered_inverted.values())),
                "edge_count": 0,
                "total_index_bytes": int(token_index.index_bytes) + int(faiss_built["index_bytes"]),
                "hybrid_candidate_fetch": int(args.hybrid_candidate_fetch),
                "hybrid_token_candidate_fetch": int(getattr(args, "hybrid_token_candidate_fetch", args.max_candidates)),
                "calibrator_max_candidates": int(args.calibrator_max_candidates),
                "hybrid_union_vector_weight": float(args.hybrid_union_vector_weight),
                "hybrid_union_token_weight": float(args.hybrid_union_token_weight),
            }
        )
        systems.append(calibrated_result)
    return {"name": name, "systems": systems}


def write_markdown(result: dict[str, Any], path: Path) -> None:
    lines = [
        "# Vector DB Compare",
        "",
        f"- backend: `{result['embedding_backend']}`",
        f"- pool_size: `{result['pool_size']}`",
        f"- growth_count: `{result['growth_count']}`",
        f"- growth_scenarios: `{','.join(result['growth_scenarios'])}`",
        f"- repeat_searches: `{result['repeat_searches']}`",
        f"- systems: `{','.join(result['systems'])}`",
        f"- dim_count: `{result['dim_count']}`",
        f"- max_cell_scan: `{result['max_cell_scan']}`",
        "",
        "| dataset | system | memories | index MB | build ms | p50 ms | p95 ms | search p95 | calibrator p95 | recall | precision | payload p95 | known vector scan p95 | candidates/read p95 | deterministic |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
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
                f"{metric_cell(metrics, 'search_latency_ms')} | "
                f"{metric_cell(metrics, 'calibrator_latency_ms')} | "
                f"{metrics.get('retrieval_context_recall', {}).get('avg', 0.0):.4f} | "
                f"{metrics.get('retrieval_context_precision', {}).get('avg', 0.0):.4f} | "
                f"{metric_cell(metrics, 'payload_reads')} | "
                f"{metric_cell(metrics, 'vector_index_scan_count')} | "
                f"{metric_cell(metrics, 'node_records_read')} | "
                f"{deterministic:.4f} |"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    backend = build_embedding_backend(args)
    calibrator = None
    if "hippo_calibrated_union" in args.systems:
        if not args.calibrator_checkpoint:
            raise ValueError("--calibrator-checkpoint is required when --systems includes hippo_calibrated_union")
        import torch

        from python.librarian.hippo_calibrator import load_calibrator

        device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
        calibrator = load_calibrator(args.calibrator_checkpoint, device=device)
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    base = ensure_backend_embeddings(build_large_pool_case(args.seed, args.pool_size), backend)
    datasets = [run_dataset("baseline", base, backend, work_dir, args, calibrator)]
    for scenario in args.growth_scenarios:
        grown = ensure_backend_embeddings(grow_row(base, scenario, args.growth_count, args.seed), backend)
        datasets.append(run_dataset(scenario, grown, backend, work_dir, args, calibrator))
    result = {
        "benchmark": "vector_db_compare",
        "embedding_backend": backend.name,
        "pool_size": args.pool_size,
        "growth_count": args.growth_count,
        "growth_scenarios": args.growth_scenarios,
        "repeat_searches": args.repeat_searches,
        "systems": args.systems,
        "calibrator_checkpoint": args.calibrator_checkpoint,
        "calibrator_max_candidates": args.calibrator_max_candidates,
        "calibrator_feature_ablation": args.calibrator_feature_ablation,
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
    parser.add_argument("--systems", type=parse_systems, default=["exact_vector", "faiss_flat", "faiss_hnsw", "hnswlib", "hippo_rope_grid"])
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
    parser.add_argument("--faiss-hnsw-m", type=int, default=32)
    parser.add_argument("--faiss-ef-construction", type=int, default=200)
    parser.add_argument("--faiss-ef-search", type=int, default=128)
    parser.add_argument("--hnswlib-m", type=int, default=32)
    parser.add_argument("--hnswlib-ef-construction", type=int, default=200)
    parser.add_argument("--hnswlib-ef-search", type=int, default=128)
    parser.add_argument("--min-layer-delta", type=float, default=0.0075)
    parser.add_argument("--min-query-layers", type=int, default=8)
    parser.add_argument("--max-query-layers", type=int, default=24)
    parser.add_argument("--min-node-layer-delta", type=float, default=0.0075)
    parser.add_argument("--min-node-layers", type=int, default=1)
    parser.add_argument("--max-node-layers", type=int, default=24)
    parser.add_argument("--edge-seed-count", type=int, default=48)
    parser.add_argument("--graph-depth", type=int, default=2)
    parser.add_argument("--final-fetch", type=int, default=96)
    parser.add_argument("--action-count", type=int, default=256)
    parser.add_argument("--query-token-count", type=int, default=40)
    parser.add_argument("--node-token-count", type=int, default=40)
    parser.add_argument("--projection-width", type=int, default=16)
    parser.add_argument("--bucket-width", type=float, default=0.055)
    parser.add_argument("--bucket-radius", type=int, default=2)
    parser.add_argument("--min-candidates", type=int, default=16)
    parser.add_argument("--max-candidates", type=int, default=512)
    parser.add_argument("--pre-filter-candidates", type=int, default=2048)
    parser.add_argument("--routing-layers", type=int, default=8)
    parser.add_argument("--promotion-probability", type=float, default=0.45)
    parser.add_argument("--promotion-bias", type=float, default=0.12)
    parser.add_argument("--routing-beam-width", type=int, default=32)
    parser.add_argument("--include-min-collision", type=float, default=1.0)
    parser.add_argument("--include-min-overlap", type=float, default=0.01)
    parser.add_argument("--token-encoder-checkpoint", default="")
    parser.add_argument("--token-encoder-device", default="")
    parser.add_argument("--hybrid-candidate-fetch", type=int, default=512)
    parser.add_argument("--hybrid-token-candidate-fetch", type=int, default=512)
    parser.add_argument("--hybrid-union-vector-weight", type=float, default=0.70)
    parser.add_argument("--hybrid-union-token-weight", type=float, default=0.30)
    parser.add_argument("--hybrid-source-weight", type=float, default=0.75)
    parser.add_argument("--hybrid-semantic-weight", type=float, default=0.15)
    parser.add_argument("--hybrid-field-weight", type=float, default=0.08)
    parser.add_argument("--hybrid-activation-weight", type=float, default=0.02)
    parser.add_argument("--calibrator-checkpoint", default="")
    parser.add_argument("--calibrator-max-candidates", type=int, default=64)
    parser.add_argument("--calibrator-feature-ablation", choices=["none", "metadata", "state", "state_metadata", "shortcut", "shortcuts", "no_shortcuts", "conflict_terms", "no_conflict_terms"], default="none")
    parser.add_argument("--rerank-relevance-weight", type=float, default=None)
    parser.add_argument("--rerank-include-weight", type=float, default=None)
    parser.add_argument("--rerank-base-weight", type=float, default=None)
    parser.add_argument("--rerank-utility-weight", type=float, default=None)
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
