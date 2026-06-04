from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from python.benchmarks.hippocampus_retrieval import build_embedding_backend, ensure_backend_embeddings
from python.benchmarks.hierarchical_file_ann import hierarchical_search, write_lazy_index
from python.benchmarks.large_pool_retrieval import build_large_pool_case
from python.librarian.frame_builder import (
    FrameBuilderConfig,
    build_cached_graph_frame,
    build_live_graph_frame,
    frame_recall,
    populate_frame_cache,
    select_seed_ids,
)
from python.librarian.frame_cache import FrameCache


def average(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[index]


def make_growth_noise(row: dict[str, Any], count: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    query = str((row.get("retrieval_task") or {}).get("query") or row.get("anchor", {}).get("text") or "")
    project = str((row.get("anchor") or {}).get("cluster") or "unknown")
    cards = []
    for index in range(count):
        text = (
            f"{project}: growth insert {seed}-{index} repeats query terms {query}. "
            f"This is unrelated maintenance state {rng.randint(100000, 999999)}."
        )
        cards.append(
            {
                "id": f"growth_noise_{seed}_{index}",
                "text": text,
                "summary": "",
                "embedding": [],
                "importance": 0.35,
                "cluster": project,
                "metadata": {"project": project},
                "age_days": rng.choice([0, 1, 7, 30]),
                "use_count": rng.choice([0, 1, 3]),
                "evidence_count": rng.choice([0, 1]),
                "last_outcome": "",
                "protected": False,
                "synthetic_role": "growth_noise",
            }
        )
    return cards


def stable_prefix(before: list[str], after: list[str], relevant: set[str], limit: int) -> float:
    before_relevant = [memory_id for memory_id in before[:limit] if memory_id in relevant]
    if not before_relevant:
        return 1.0
    after_top = set(after[:limit])
    return len([memory_id for memory_id in before_relevant if memory_id in after_top]) / len(before_relevant)


def frame_precision_at_k(frame: dict[str, Any], top_k: int) -> float:
    labels = [float(label) for label in frame.get("labels", [])[:top_k]]
    mask = [float(value) for value in frame.get("mask", [])[:top_k]]
    selected = [label for label, active in zip(labels, mask) if active > 0.0]
    if not selected:
        return 0.0
    return sum(1.0 for label in selected if label >= 0.5) / len(selected)


def candidate_lookup(row: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(candidate["id"]): dict(candidate) for candidate in row.get("candidates", [])}


def hierarchy_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        beam_width=args.hierarchy_beam_width,
        max_frontier=args.hierarchy_max_frontier,
        max_children_per_basin=args.hierarchy_max_children_per_basin,
        max_leaf_reads=args.hierarchy_max_leaf_reads,
        promoted_per_basin=args.hierarchy_promoted_per_basin,
        graph_seed_count=args.hierarchy_graph_seed_count,
        graph_depth=args.hierarchy_graph_depth,
        cache_size=args.hierarchy_cache_size,
        semantic_dims=args.hierarchy_semantic_dims,
        store_full_embeddings=args.hierarchy_store_full_embeddings,
        promotion_bias=args.hierarchy_promotion_bias,
        promotion_scale=args.hierarchy_promotion_scale,
        return_limit=args.hierarchy_return_limit,
        stable_growth=args.hierarchy_stable_growth,
        stable_basin_floor=args.hierarchy_stable_basin_floor,
        stable_max_basins=args.hierarchy_stable_max_basins,
        stable_max_leaf_reads=args.hierarchy_stable_max_leaf_reads,
        stable_max_promoted_reads=args.hierarchy_stable_max_promoted_reads,
    )


def select_index_seed_ids(
    row: dict[str, Any],
    backend: Any,
    work_dir: Path,
    args: argparse.Namespace,
    offset: int,
) -> tuple[list[str], dict[str, float], dict[str, float]]:
    hargs = hierarchy_args(args)
    case_dir = work_dir / f"case_{offset:05d}"
    meta = write_lazy_index(row, backend, case_dir, hargs)
    ranked, search_stats, _ = hierarchical_search(row, backend, meta, hargs)
    seeds = ranked[: max(1, args.graph_seed_count)]
    return [memory_id for memory_id, _, _ in seeds], {memory_id: score for memory_id, score, _ in ranked}, {
        "index_build_ms": float(meta.get("build_latency_ms") or 0.0),
        "index_seed_ms": float(search_stats.get("latency_ms") or 0.0),
        "index_file_reads": float(search_stats.get("file_reads") or 0.0),
        "index_unique_nodes_read": float(search_stats.get("unique_nodes_read") or 0.0),
        "index_final_candidate_count": float(search_stats.get("final_candidate_count") or 0.0),
        "index_stable_promoted_reads": float(search_stats.get("stable_promoted_reads") or 0.0),
    }


def select_query_seeds(
    row: dict[str, Any],
    backend: Any,
    config: FrameBuilderConfig,
    work_dir: Path,
    args: argparse.Namespace,
    offset: int,
) -> tuple[list[str], dict[str, float], dict[str, float]]:
    if args.seed_source == "full_scan":
        seed_ids, base_scores, seed_ms = select_seed_ids(row, config)
        return seed_ids, base_scores, {"seed_selection_ms": seed_ms}
    return select_index_seed_ids(row, backend, work_dir, args, offset)


def run(args: argparse.Namespace) -> dict[str, Any]:
    backend = build_embedding_backend(args)
    config = FrameBuilderConfig(
        frame_size=args.frame_size,
        graph_seed_count=args.graph_seed_count,
        graph_depth=args.graph_depth,
        graph_boost=args.graph_boost,
        use_role_features=not args.hide_role_features,
    )
    cache_path = Path(args.cache_path)
    if cache_path.exists() and args.reset_cache:
        cache_path.unlink()
    cache = FrameCache(cache_path)
    work_dir = Path(args.work_dir)
    rows_for_training = []
    metrics: dict[str, list[float]] = {
        "background_cache_build_ms": [],
        "seed_selection_ms": [],
        "index_build_ms": [],
        "index_seed_ms": [],
        "index_file_reads": [],
        "index_unique_nodes_read": [],
        "index_final_candidate_count": [],
        "index_stable_promoted_reads": [],
        "live_frame_ms": [],
        "cached_frame_ms": [],
        "live_frame_recall": [],
        "cached_frame_recall": [],
        "live_frame_precision_at_k": [],
        "cached_frame_precision_at_k": [],
        "growth_cached_frame_recall": [],
        "growth_cached_frame_precision_at_k": [],
        "growth_relevant_stability": [],
        "cache_hits": [],
        "cache_misses": [],
        "merged_candidates": [],
    }
    started = time.perf_counter()
    for offset in range(args.cases):
        row = build_large_pool_case(args.seed + offset, args.pool_size)
        row = ensure_backend_embeddings(row, backend)
        build_stats = populate_frame_cache(row, cache, config)
        metrics["background_cache_build_ms"].append(build_stats["cache_build_ms"])
        lookup = candidate_lookup(row)

        live_started = time.perf_counter()
        live_frame = build_live_graph_frame(row, config)
        metrics["live_frame_ms"].append((time.perf_counter() - live_started) * 1000.0)

        seed_ids, base_scores, seed_stats = select_query_seeds(row, backend, config, work_dir, args, offset)
        for key, value in seed_stats.items():
            if key in metrics:
                metrics[key].append(value)
        cached_frame, cached_stats = build_cached_graph_frame(
            row,
            cache,
            config,
            seed_ids=seed_ids,
            base_scores=base_scores,
            candidate_lookup=lookup,
        )
        metrics["cached_frame_ms"].append(cached_stats["cache_query_ms"])
        metrics["cache_hits"].append(cached_stats["cache_hits"])
        metrics["cache_misses"].append(cached_stats["cache_misses"])
        metrics["merged_candidates"].append(cached_stats["merged_candidates"])
        metrics["live_frame_recall"].append(frame_recall(live_frame))
        metrics["cached_frame_recall"].append(frame_recall(cached_frame))
        metrics["live_frame_precision_at_k"].append(frame_precision_at_k(live_frame, args.top_k))
        metrics["cached_frame_precision_at_k"].append(frame_precision_at_k(cached_frame, args.top_k))
        rows_for_training.append(cached_frame)

        if args.growth_noise > 0:
            grown_row = dict(row)
            grown_candidates = [dict(candidate) for candidate in row.get("candidates", [])]
            grown_candidates.extend(make_growth_noise(row, args.growth_noise, args.seed + offset))
            grown_row["candidates"] = grown_candidates
            grown_row = ensure_backend_embeddings(grown_row, backend)
            grown_seed_ids, grown_base_scores, _ = select_query_seeds(grown_row, backend, config, work_dir, args, offset + args.cases)
            grown_frame, _ = build_cached_graph_frame(
                grown_row,
                cache,
                config,
                seed_ids=grown_seed_ids,
                base_scores=grown_base_scores,
                candidate_lookup=candidate_lookup(grown_row),
            )
            relevant = set((row.get("retrieval_task") or {}).get("relevant_ids") or [])
            metrics["growth_cached_frame_recall"].append(frame_recall(grown_frame))
            metrics["growth_cached_frame_precision_at_k"].append(frame_precision_at_k(grown_frame, args.top_k))
            metrics["growth_relevant_stability"].append(
                stable_prefix(cached_frame.get("ids", []), grown_frame.get("ids", []), relevant, args.top_k)
            )

        if (offset + 1) % max(1, args.log_every) == 0:
            print(
                json.dumps(
                    {
                        "case": offset + 1,
                        "cached_frame_recall": average(metrics["cached_frame_recall"]),
                        "cached_frame_p95_ms": quantile(metrics["cached_frame_ms"], 0.95),
                    }
                ),
                flush=True,
            )

    result: dict[str, Any] = {
        "benchmark": "frame_cache_growth",
        "embedding_backend": backend.name,
        "cases": args.cases,
        "pool_size": args.pool_size,
        "frame_size": args.frame_size,
        "seed_source": args.seed_source,
        "growth_noise": args.growth_noise,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "cache": cache.stats(),
        "averages": {key: average(values) for key, values in metrics.items() if values},
        "p95": {key: quantile(values, 0.95) for key, values in metrics.items() if values},
    }
    cache.close()
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.output_md:
        write_markdown(result, Path(args.output_md))
    if args.train_models:
        result["training_note"] = "training is not run inside frame_cache_growth; export cached frames and use memory_frame_experiment for model sweeps"
    return result


def write_markdown(result: dict[str, Any], path: Path) -> None:
    avg = result["averages"]
    p95 = result["p95"]
    lines = [
        "# Frame Cache Growth Benchmark",
        "",
        f"- embedding_backend: `{result['embedding_backend']}`",
        f"- cases: `{result['cases']}`",
        f"- pool_size: `{result['pool_size']}`",
        f"- frame_size: `{result['frame_size']}`",
        f"- seed_source: `{result['seed_source']}`",
        f"- growth_noise: `{result['growth_noise']}`",
        f"- cache_records: `{result['cache']['frame_count']}`",
        "",
        "| metric | avg | p95 |",
        "| --- | ---: | ---: |",
    ]
    for key in sorted(avg):
        lines.append(f"| {key} | {avg[key]:.4f} | {p95.get(key, 0.0):.4f} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=int, default=50)
    parser.add_argument("--pool-size", type=int, default=5000)
    parser.add_argument("--frame-size", type=int, default=64)
    parser.add_argument("--graph-seed-count", type=int, default=16)
    parser.add_argument("--graph-depth", type=int, default=3)
    parser.add_argument("--graph-boost", type=float, default=0.85)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--growth-noise", type=int, default=0)
    parser.add_argument("--hide-role-features", action="store_true")
    parser.add_argument("--seed", type=int, default=31000)
    parser.add_argument("--cache-path", default="artifacts/frame_cache/frame_cache.sqlite")
    parser.add_argument("--work-dir", default="artifacts/frame_cache/hierarchical_seed")
    parser.add_argument("--reset-cache", action="store_true")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--train-models", default="")
    parser.add_argument("--seed-source", choices=["full_scan", "hierarchical"], default="full_scan")
    parser.add_argument("--hierarchy-return-limit", type=int, default=256)
    parser.add_argument("--hierarchy-beam-width", type=int, default=4)
    parser.add_argument("--hierarchy-max-frontier", type=int, default=256)
    parser.add_argument("--hierarchy-max-children-per-basin", type=int, default=128)
    parser.add_argument("--hierarchy-max-leaf-reads", type=int, default=384)
    parser.add_argument("--hierarchy-promoted-per-basin", type=int, default=32)
    parser.add_argument("--hierarchy-graph-seed-count", type=int, default=8)
    parser.add_argument("--hierarchy-graph-depth", type=int, default=2)
    parser.add_argument("--hierarchy-cache-size", type=int, default=256)
    parser.add_argument("--hierarchy-semantic-dims", type=int, default=128)
    parser.add_argument("--hierarchy-store-full-embeddings", action="store_true")
    parser.add_argument("--hierarchy-promotion-bias", type=float, default=0.0)
    parser.add_argument("--hierarchy-promotion-scale", type=float, default=1.0)
    parser.add_argument("--hierarchy-stable-growth", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--hierarchy-stable-basin-floor", type=float, default=0.32)
    parser.add_argument("--hierarchy-stable-max-basins", type=int, default=12)
    parser.add_argument("--hierarchy-stable-max-leaf-reads", type=int, default=0)
    parser.add_argument("--hierarchy-stable-max-promoted-reads", type=int, default=0)
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
