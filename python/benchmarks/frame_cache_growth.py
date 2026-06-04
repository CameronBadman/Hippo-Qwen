from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from python.benchmarks.hippocampus_retrieval import build_embedding_backend, ensure_backend_embeddings
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
    rows_for_training = []
    metrics: dict[str, list[float]] = {
        "background_cache_build_ms": [],
        "seed_selection_ms": [],
        "live_frame_ms": [],
        "cached_frame_ms": [],
        "live_frame_recall": [],
        "cached_frame_recall": [],
        "growth_cached_frame_recall": [],
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

        live_started = time.perf_counter()
        live_frame = build_live_graph_frame(row, config)
        metrics["live_frame_ms"].append((time.perf_counter() - live_started) * 1000.0)

        seed_ids, base_scores, seed_selection_ms = select_seed_ids(row, config)
        metrics["seed_selection_ms"].append(seed_selection_ms)
        cached_frame, cached_stats = build_cached_graph_frame(row, cache, config, seed_ids=seed_ids, base_scores=base_scores)
        metrics["cached_frame_ms"].append(cached_stats["cache_query_ms"])
        metrics["cache_hits"].append(cached_stats["cache_hits"])
        metrics["cache_misses"].append(cached_stats["cache_misses"])
        metrics["merged_candidates"].append(cached_stats["merged_candidates"])
        metrics["live_frame_recall"].append(frame_recall(live_frame))
        metrics["cached_frame_recall"].append(frame_recall(cached_frame))
        rows_for_training.append(cached_frame)

        if args.growth_noise > 0:
            grown_row = dict(row)
            grown_candidates = [dict(candidate) for candidate in row.get("candidates", [])]
            grown_candidates.extend(make_growth_noise(row, args.growth_noise, args.seed + offset))
            grown_row["candidates"] = grown_candidates
            grown_row = ensure_backend_embeddings(grown_row, backend)
            grown_seed_ids, grown_base_scores, _ = select_seed_ids(grown_row, config)
            grown_frame, _ = build_cached_graph_frame(grown_row, cache, config, seed_ids=grown_seed_ids, base_scores=grown_base_scores)
            relevant = set((row.get("retrieval_task") or {}).get("relevant_ids") or [])
            metrics["growth_cached_frame_recall"].append(frame_recall(grown_frame))
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
    parser.add_argument("--reset-cache", action="store_true")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--train-models", default="")
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
