from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from python.benchmarks.memorycraft_retrieval import (
    ADVERSARIAL_PROFILES,
    DEFAULT_HF_FILE,
    DEFAULT_HF_REPO,
    run as run_memorycraft,
)


DEFAULT_PROFILES = "clean,query_echo,answer_shaped,stale_preference,same_entity_wrong_time,superseded_conflict,near_duplicate,evidence_adjacent,mixed"
DEFAULT_BASELINE_SYSTEMS = "faiss_hnsw,hybrid_union_token"
DEFAULT_CALIBRATED_SYSTEMS = "hippo_calibrated_union"


def parse_csv(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def parse_int_csv(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def parse_calibrator(raw: str) -> tuple[str, str]:
    if "=" not in raw:
        path = Path(raw)
        return path.stem, raw
    name, path = raw.split("=", 1)
    name = name.strip()
    path = path.strip()
    if not name or not path:
        raise argparse.ArgumentTypeError("--calibrator entries must be name=/path/to/checkpoint.pt")
    return name, path


def sanitize_jsonl_text(text: str) -> str:
    return re.sub(r"\\u(?![0-9a-fA-F]{4})", r"\\\\u", text)


def prepare_dataset(args: argparse.Namespace, output_dir: Path) -> str:
    if not args.dataset:
        return ""
    source = Path(args.dataset)
    if not source.exists():
        raise FileNotFoundError(source)
    if args.record_offset <= 0 and args.record_limit <= 0 and not args.sanitize_dataset:
        return str(source)

    output = output_dir / "prepared_dataset.jsonl"
    kept = 0
    skipped = 0
    start = max(0, int(args.record_offset))
    stop = None if args.record_limit <= 0 else start + int(args.record_limit)
    output.parent.mkdir(parents=True, exist_ok=True)
    with source.open("r", encoding="utf-8") as src, output.open("w", encoding="utf-8") as dst:
        for line_index, line in enumerate(src):
            if line_index < start:
                continue
            if stop is not None and line_index >= stop:
                break
            raw = sanitize_jsonl_text(line) if args.sanitize_dataset else line
            if not raw.strip():
                continue
            try:
                record = json.loads(raw)
            except json.JSONDecodeError:
                skipped += 1
                continue
            dst.write(json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n")
            kept += 1
    if kept <= 0:
        raise ValueError(f"prepared dataset is empty: {output} skipped={skipped}")
    metadata = {
        "source": str(source),
        "output": str(output),
        "record_offset": start,
        "record_limit": args.record_limit,
        "sanitize_dataset": bool(args.sanitize_dataset),
        "kept_records": kept,
        "skipped_records": skipped,
    }
    (output_dir / "prepared_dataset.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return str(output)


def metric_value(result: dict[str, Any], system: str, metric: str, stat: str = "avg") -> float:
    return float((((result.get("rollup") or {}).get(system) or {}).get("metrics") or {}).get(metric, {}).get(stat, 0.0))


def summarize_result(
    result: dict[str, Any],
    *,
    suite_name: str,
    scenario: str,
    adversarial_negatives: int,
    candidate_pool: int,
    calibrator_name: str,
) -> list[dict[str, Any]]:
    rows = []
    for system, rollup in (result.get("rollup") or {}).items():
        metrics = rollup.get("metrics") or {}
        latency = metrics.get("latency_ms") or {}
        rows.append(
            {
                "suite": suite_name,
                "scenario": scenario,
                "adversarial_negatives": int(adversarial_negatives),
                "candidate_pool": int(candidate_pool),
                "calibrator": calibrator_name,
                "system": system,
                "records": int(rollup.get("record_count") or 0),
                "queries": int(rollup.get("query_count") or 0),
                "p95_ms": float(latency.get("p95", 0.0)),
                "search_p95_ms": float((metrics.get("search_latency_ms") or {}).get("p95", 0.0)),
                "calibrator_p95_ms": float((metrics.get("calibrator_latency_ms") or {}).get("p95", 0.0)),
                "cascade_prefilter_candidates": float((metrics.get("cascade_prefilter_candidate_count") or {}).get("avg", 0.0)),
                "cascade_survivor_candidates": float((metrics.get("cascade_survivor_count") or {}).get("avg", 0.0)),
                "typed_edge_injections": float((metrics.get("typed_edge_injections") or {}).get("avg", 0.0)),
                "recall_at_8": float((metrics.get("recall_at_k") or {}).get("avg", 0.0)),
                "precision_at_8": float((metrics.get("precision_at_k") or {}).get("avg", 0.0)),
                "context_recall": float((metrics.get("context_recall") or {}).get("avg", 0.0)),
                "context_precision": float((metrics.get("context_precision") or {}).get("avg", 0.0)),
                "evidence_in_pool": float((metrics.get("evidence_in_calibrator_pool_rate") or {}).get("avg", 0.0)),
                "false_memory_rate": float((metrics.get("false_memory_rate") or {}).get("avg", 0.0)),
                "abstention_query_rate": float((metrics.get("abstention_query") or {}).get("avg", 0.0)),
                "hard_negative_top_k_rate": float((metrics.get("hard_negative_top_k_rate") or {}).get("avg", 0.0)),
                "hard_negative_context_rate": float((metrics.get("hard_negative_context_rate") or {}).get("avg", 0.0)),
                "mrr": float((metrics.get("mrr") or {}).get("avg", 0.0)),
                "determinism_mismatches": int(rollup.get("determinism_mismatches") or 0),
            }
        )
    return rows


def quality_score(row: dict[str, Any], latency_target_ms: float) -> float:
    latency_penalty = max(0.0, (float(row["p95_ms"]) - latency_target_ms) / max(1.0, latency_target_ms))
    determinism_penalty = 1.0 if int(row["determinism_mismatches"]) else 0.0
    return (
        0.28 * float(row["recall_at_8"])
        + 0.24 * float(row["precision_at_8"])
        + 0.22 * float(row["context_precision"])
        + 0.16 * float(row["mrr"])
        + 0.10 * float(row["context_recall"])
        - 0.18 * float(row["hard_negative_top_k_rate"])
        - 0.10 * latency_penalty
        - 0.25 * determinism_penalty
    )


def memorycraft_args(
    args: argparse.Namespace,
    *,
    dataset: str,
    systems: list[str],
    suite_name: str,
    scenario: str,
    profile: str,
    adversarial_negatives: int,
    candidate_pool: int,
    calibrator_checkpoint: str,
    output_dir: Path,
) -> argparse.Namespace:
    return argparse.Namespace(
        dataset=dataset,
        hf_repo=args.hf_repo,
        hf_file=args.hf_file,
        limit_records=args.limit_records,
        limit_questions=args.limit_questions,
        include_abstention=args.include_abstention,
        unit=args.unit,
        systems=systems,
        repeat_searches=args.repeat_searches,
        adversarial_negatives=adversarial_negatives,
        adversarial_profile=profile,
        adversarial_style=args.adversarial_style,
        adversarial_families=args.adversarial_families,
        adversarial_exclude_families=args.adversarial_exclude_families,
        work_dir=str(output_dir / "work" / suite_name),
        top_k=args.top_k,
        budget=args.budget,
        context_max_items=args.context_max_items,
        layers=args.layers,
        layer_schedule=args.layer_schedule,
        dim_count=args.dim_count,
        cell_width=args.cell_width,
        radius=args.radius,
        max_cell_scan=args.max_cell_scan,
        faiss_hnsw_m=args.faiss_hnsw_m,
        faiss_ef_construction=args.faiss_ef_construction,
        faiss_ef_search=args.faiss_ef_search,
        hnswlib_m=args.hnswlib_m,
        hnswlib_ef_construction=args.hnswlib_ef_construction,
        hnswlib_ef_search=args.hnswlib_ef_search,
        min_layer_delta=args.min_layer_delta,
        min_query_layers=args.min_query_layers,
        max_query_layers=args.max_query_layers,
        min_node_layer_delta=args.min_node_layer_delta,
        min_node_layers=args.min_node_layers,
        max_node_layers=args.max_node_layers,
        edge_seed_count=args.edge_seed_count,
        graph_depth=args.graph_depth,
        final_fetch=args.final_fetch,
        calibrator_checkpoint=calibrator_checkpoint,
        calibrator_max_candidates=candidate_pool,
        cascade_prefilter_checkpoint=args.cascade_prefilter_checkpoint,
        cascade_prefilter_candidates=args.cascade_prefilter_candidates,
        cascade_survivor_candidates=candidate_pool if candidate_pool > 0 else args.cascade_survivor_candidates,
        cascade_prefilter_relevance_weight=args.cascade_prefilter_relevance_weight,
        cascade_prefilter_include_weight=args.cascade_prefilter_include_weight,
        cascade_prefilter_base_weight=args.cascade_prefilter_base_weight,
        cascade_prefilter_utility_weight=args.cascade_prefilter_utility_weight,
        typed_edge_expansion=args.typed_edge_expansion,
        typed_edge_types=args.typed_edge_types,
        typed_edge_seed_count=args.typed_edge_seed_count,
        typed_edge_max_injections=args.typed_edge_max_injections,
        typed_edge_seed_score_weight=args.typed_edge_seed_score_weight,
        entity_profiles=args.entity_profiles,
        entity_profile_max_sources=args.entity_profile_max_sources,
        anti_memories=args.anti_memories,
        calibrator_feature_ablation=args.calibrator_feature_ablation,
        rerank_relevance_weight=args.rerank_relevance_weight,
        rerank_include_weight=args.rerank_include_weight,
        rerank_base_weight=args.rerank_base_weight,
        rerank_utility_weight=args.rerank_utility_weight,
        action_count=args.action_count,
        query_token_count=args.query_token_count,
        node_token_count=args.node_token_count,
        projection_width=args.projection_width,
        bucket_width=args.bucket_width,
        bucket_radius=args.bucket_radius,
        min_candidates=args.min_candidates,
        max_candidates=args.max_candidates,
        pre_filter_candidates=args.pre_filter_candidates,
        routing_layers=args.routing_layers,
        promotion_probability=args.promotion_probability,
        promotion_bias=args.promotion_bias,
        routing_beam_width=args.routing_beam_width,
        include_min_collision=args.include_min_collision,
        include_min_overlap=args.include_min_overlap,
        token_encoder_checkpoint=args.token_encoder_checkpoint,
        token_encoder_device=args.token_encoder_device,
        hybrid_candidate_fetch=args.hybrid_candidate_fetch,
        hybrid_token_candidate_fetch=args.hybrid_token_candidate_fetch,
        hybrid_union_vector_weight=args.hybrid_union_vector_weight,
        hybrid_union_token_weight=args.hybrid_union_token_weight,
        hybrid_source_weight=args.hybrid_source_weight,
        hybrid_semantic_weight=args.hybrid_semantic_weight,
        hybrid_field_weight=args.hybrid_field_weight,
        hybrid_activation_weight=args.hybrid_activation_weight,
        memory_graph_layers=args.memory_graph_layers,
        memory_graph_route_degree=args.memory_graph_route_degree,
        memory_graph_projection_count=args.memory_graph_projection_count,
        memory_graph_projection_window=args.memory_graph_projection_window,
        memory_graph_promotion_threshold=args.memory_graph_promotion_threshold,
        memory_graph_bias_promotion=args.memory_graph_bias_promotion,
        memory_graph_bridge_degree=args.memory_graph_bridge_degree,
        memory_graph_importance_threshold=args.memory_graph_importance_threshold,
        memory_graph_reciprocal_routes=args.memory_graph_reciprocal_routes,
        memory_graph_ef=args.memory_graph_ef,
        memory_graph_beam=args.memory_graph_beam,
        memory_graph_objective_seeds=args.memory_graph_objective_seeds,
        memory_graph_truth_seeds=args.memory_graph_truth_seeds,
        memory_graph_truth_depth=args.memory_graph_truth_depth,
        memory_graph_truth_fanout=args.memory_graph_truth_fanout,
        memory_graph_min_results=args.memory_graph_min_results,
        memory_graph_cutoff_margin=args.memory_graph_cutoff_margin,
        memory_graph_min_score=args.memory_graph_min_score,
        embedding_backend=args.embedding_backend,
        hippo_checkpoint=args.hippo_checkpoint,
        hippo_encoder_src=args.hippo_encoder_src,
        hippo_max_length=args.hippo_max_length,
        hippo_batch_size=args.hippo_batch_size,
        device=args.device,
        output_json=str(output_dir / "runs" / f"{suite_name}.json"),
        output_md=str(output_dir / "runs" / f"{suite_name}.md"),
    )


def run_case(
    args: argparse.Namespace,
    *,
    dataset: str,
    systems: list[str],
    suite_name: str,
    scenario: str,
    profile: str,
    adversarial_negatives: int,
    candidate_pool: int,
    calibrator_name: str,
    calibrator_checkpoint: str,
    output_dir: Path,
) -> list[dict[str, Any]]:
    run_args = memorycraft_args(
        args,
        dataset=dataset,
        systems=systems,
        suite_name=suite_name,
        scenario=scenario,
        profile=profile,
        adversarial_negatives=adversarial_negatives,
        candidate_pool=candidate_pool,
        calibrator_checkpoint=calibrator_checkpoint,
        output_dir=output_dir,
    )
    started = time.perf_counter()
    result = run_memorycraft(run_args)
    elapsed = time.perf_counter() - started
    rows = summarize_result(
        result,
        suite_name=suite_name,
        scenario=scenario,
        adversarial_negatives=adversarial_negatives,
        candidate_pool=candidate_pool,
        calibrator_name=calibrator_name,
    )
    for row in rows:
        row["elapsed_seconds"] = round(elapsed, 3)
        row["quality_score"] = quality_score(row, args.latency_target_ms)
    return rows


def write_outputs(args: argparse.Namespace, output_dir: Path, rows: list[dict[str, Any]], run_plan: list[dict[str, Any]]) -> None:
    output_dir.joinpath("summary.json").write_text(
        json.dumps({"rows": rows, "run_plan": run_plan}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    sorted_rows = sorted(rows, key=lambda item: (-float(item["quality_score"]), item["scenario"], item["system"]))
    lines = [
        "# MemoryCraft Calibration Ablation Suite",
        "",
        f"- dataset: `{args.dataset or args.hf_repo + '/' + args.hf_file}`",
        f"- embedding backend: `{args.embedding_backend}`",
        f"- top_k: `{args.top_k}`",
        f"- budget: `{args.budget}`",
        f"- context max items: `{args.context_max_items}`",
        f"- latency target: `{args.latency_target_ms:.1f} ms`",
        f"- repeat searches: `{args.repeat_searches}`",
        f"- adversarial style: `{args.adversarial_style}`",
        f"- adversarial families: `{args.adversarial_families}`",
        f"- adversarial exclude families: `{args.adversarial_exclude_families}`",
        f"- calibrator feature ablation: `{args.calibrator_feature_ablation}`",
        "",
        "## Leaderboard",
        "",
        "| rank | scenario | system | calibrator | pool | cascade prefilter | cascade survivors | edge inject | p95 ms | recall@8 | precision@8 | ctx precision | evidence in pool | false memory | hard neg@8 | mrr | det mismatches | score |",
        "| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for rank, row in enumerate(sorted_rows, start=1):
        lines.append(
            f"| {rank} | {row['scenario']} | {row['system']} | {row['calibrator']} | "
            f"{row['candidate_pool']} | {row['cascade_prefilter_candidates']:.0f} | "
            f"{row['cascade_survivor_candidates']:.0f} | {row['typed_edge_injections']:.1f} | "
            f"{row['p95_ms']:.2f} | {row['recall_at_8']:.4f} | "
            f"{row['precision_at_8']:.4f} | {row['context_precision']:.4f} | "
            f"{row['evidence_in_pool']:.4f} | {row['false_memory_rate']:.4f} | "
            f"{row['hard_negative_top_k_rate']:.4f} | {row['mrr']:.4f} | "
            f"{row['determinism_mismatches']} | {row['quality_score']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Scenario Matrix",
            "",
            "| scenario | system | calibrator | pool | cascade prefilter | cascade survivors | edge inject | queries | p95 ms | recall@8 | precision@8 | context recall | context precision | evidence in pool | false memory | hard neg ctx | mrr |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['scenario']} | {row['system']} | {row['calibrator']} | {row['candidate_pool']} | "
            f"{row['cascade_prefilter_candidates']:.0f} | {row['cascade_survivor_candidates']:.0f} | "
            f"{row['typed_edge_injections']:.1f} | {row['queries']} | {row['p95_ms']:.2f} | {row['recall_at_8']:.4f} | "
            f"{row['precision_at_8']:.4f} | {row['context_recall']:.4f} | "
            f"{row['context_precision']:.4f} | {row['evidence_in_pool']:.4f} | "
            f"{row['false_memory_rate']:.4f} | {row['hard_negative_context_rate']:.4f} | {row['mrr']:.4f} |"
        )
    output_dir.joinpath("summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    log_lines = [
        "# Experiment Log",
        "",
        "This suite runs `python.benchmarks.memorycraft_retrieval` repeatedly with controlled ablation axes.",
        "",
        "## Ablation Axes",
        "",
        "- retrieval system: FAISS/HNSW, raw FAISS+token union, calibrated Hippo union",
        "- adversarial profile: clean or one deterministic hard-negative family",
        "- candidate pool: number of union candidates passed into the calibrator",
        "- cascade: optional pointwise prefilter over a larger pool, followed by set calibration over survivors",
        "- calibrator checkpoint: named model artifact supplied with `--calibrator name=path`",
        "",
        "## Metrics",
        "",
        "- `recall@8`: fraction of labelled evidence recovered in top 8",
        "- `precision@8`: top-8 evidence density",
        "- `context_precision`: evidence density inside the token-budgeted returned context",
        "- `hard_neg@8`: fraction of top-8 slots consumed by synthetic adversarial decoys",
        "- `mrr`: reciprocal rank of the first evidence hit",
        "- `det mismatches`: repeated search output mismatches; target is always 0",
        "",
        "## Run Plan",
        "",
        "```json",
        json.dumps(run_plan, indent=2, sort_keys=True),
        "```",
        "",
    ]
    output_dir.joinpath("experiment_log.md").write_text("\n".join(log_lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.work_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir.joinpath("runs").mkdir(parents=True, exist_ok=True)
    dataset = prepare_dataset(args, output_dir)
    profiles = parse_csv(args.profiles)
    unknown_profiles = sorted(set(profiles) - (ADVERSARIAL_PROFILES | {"clean"}))
    if unknown_profiles:
        raise ValueError(f"unknown profiles: {','.join(unknown_profiles)}")
    candidate_pools = parse_int_csv(args.candidate_pools)
    calibrators = [parse_calibrator(item) for item in args.calibrator]
    baseline_systems = parse_csv(args.baseline_systems)
    calibrated_systems = parse_csv(args.calibrated_systems)

    rows: list[dict[str, Any]] = []
    run_plan: list[dict[str, Any]] = []

    def write_checkpoint() -> None:
        if rows:
            write_outputs(args, output_dir, rows, run_plan)

    for profile in profiles:
        adversarial_negatives = 0 if profile == "clean" else int(args.adversarial_negatives)
        memorycraft_profile = "mixed" if profile == "clean" else profile
        suite_base = f"{profile}_baseline"
        plan = {
            "name": suite_base,
            "profile": profile,
            "systems": baseline_systems,
            "candidate_pool": 0,
            "calibrator": "none",
            "adversarial_negatives": adversarial_negatives,
            "adversarial_families": args.adversarial_families,
            "adversarial_exclude_families": args.adversarial_exclude_families,
        }
        run_plan.append(plan)
        print(f"RUN {suite_base}", flush=True)
        rows.extend(
            run_case(
                args,
                dataset=dataset,
                systems=baseline_systems,
                suite_name=suite_base,
                scenario=profile,
                profile=memorycraft_profile,
                adversarial_negatives=adversarial_negatives,
                candidate_pool=0,
                calibrator_name="none",
                calibrator_checkpoint="",
                output_dir=output_dir,
            )
        )
        write_checkpoint()
        for calibrator_name, checkpoint in calibrators:
            for candidate_pool in candidate_pools:
                suite_name = f"{profile}_{calibrator_name}_pool{candidate_pool}"
                plan = {
                    "name": suite_name,
                    "profile": profile,
                    "systems": calibrated_systems,
                    "candidate_pool": candidate_pool,
                    "calibrator": calibrator_name,
                    "checkpoint": checkpoint,
                    "adversarial_negatives": adversarial_negatives,
                    "adversarial_families": args.adversarial_families,
                    "adversarial_exclude_families": args.adversarial_exclude_families,
                }
                run_plan.append(plan)
                print(f"RUN {suite_name}", flush=True)
                rows.extend(
                    run_case(
                        args,
                        dataset=dataset,
                        systems=calibrated_systems,
                        suite_name=suite_name,
                        scenario=profile,
                        profile=memorycraft_profile,
                        adversarial_negatives=adversarial_negatives,
                        candidate_pool=candidate_pool,
                        calibrator_name=calibrator_name,
                        calibrator_checkpoint=checkpoint,
                        output_dir=output_dir,
                    )
                )
                write_checkpoint()
    write_outputs(args, output_dir, rows, run_plan)
    return {"summary_json": str(output_dir / "summary.json"), "summary_md": str(output_dir / "summary.md"), "rows": len(rows)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", default="artifacts/memorycraft_ablation_suite")
    parser.add_argument("--dataset", default="")
    parser.add_argument("--sanitize-dataset", action="store_true")
    parser.add_argument("--record-offset", type=int, default=0)
    parser.add_argument("--record-limit", type=int, default=0)
    parser.add_argument("--hf-repo", default=DEFAULT_HF_REPO)
    parser.add_argument("--hf-file", default=DEFAULT_HF_FILE)
    parser.add_argument("--limit-records", type=int, default=0)
    parser.add_argument("--limit-questions", type=int, default=40)
    parser.add_argument("--include-abstention", action="store_true")
    parser.add_argument("--unit", choices=["auto", "turn", "session"], default="auto")
    parser.add_argument("--profiles", default=DEFAULT_PROFILES)
    parser.add_argument("--adversarial-negatives", type=int, default=8)
    parser.add_argument("--adversarial-style", choices=["legacy", "forensic"], default="forensic")
    parser.add_argument("--adversarial-families", default="")
    parser.add_argument("--adversarial-exclude-families", default="")
    parser.add_argument("--candidate-pools", default="64,128,256")
    parser.add_argument("--calibrator", action="append", default=[])
    parser.add_argument("--baseline-systems", default=DEFAULT_BASELINE_SYSTEMS)
    parser.add_argument("--calibrated-systems", default=DEFAULT_CALIBRATED_SYSTEMS)
    parser.add_argument("--repeat-searches", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--budget", type=int, default=900)
    parser.add_argument("--context-max-items", type=int, default=0)
    parser.add_argument("--latency-target-ms", type=float, default=200.0)
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
    parser.add_argument("--final-fetch", type=int, default=128)
    parser.add_argument("--cascade-prefilter-checkpoint", default="")
    parser.add_argument("--cascade-prefilter-candidates", type=int, default=1024)
    parser.add_argument("--cascade-survivor-candidates", type=int, default=64)
    parser.add_argument("--cascade-prefilter-relevance-weight", type=float, default=None)
    parser.add_argument("--cascade-prefilter-include-weight", type=float, default=None)
    parser.add_argument("--cascade-prefilter-base-weight", type=float, default=None)
    parser.add_argument("--cascade-prefilter-utility-weight", type=float, default=None)
    parser.add_argument("--typed-edge-expansion", action="store_true")
    parser.add_argument("--typed-edge-types", default="correction,temporal_next,same_context")
    parser.add_argument("--typed-edge-seed-count", type=int, default=32)
    parser.add_argument("--typed-edge-max-injections", type=int, default=128)
    parser.add_argument("--typed-edge-seed-score-weight", type=float, default=0.82)
    parser.add_argument("--entity-profiles", action="store_true")
    parser.add_argument("--entity-profile-max-sources", type=int, default=24)
    parser.add_argument("--anti-memories", action="store_true")
    parser.add_argument("--calibrator-feature-ablation", choices=["none", "metadata", "state", "state_metadata", "shortcut", "shortcuts", "no_shortcuts", "conflict_terms", "no_conflict_terms"], default="none")
    parser.add_argument("--rerank-relevance-weight", type=float, default=None)
    parser.add_argument("--rerank-include-weight", type=float, default=None)
    parser.add_argument("--rerank-base-weight", type=float, default=None)
    parser.add_argument("--rerank-utility-weight", type=float, default=None)
    parser.add_argument("--action-count", type=int, default=256)
    parser.add_argument("--query-token-count", type=int, default=40)
    parser.add_argument("--node-token-count", type=int, default=40)
    parser.add_argument("--projection-width", type=int, default=16)
    parser.add_argument("--bucket-width", type=float, default=0.055)
    parser.add_argument("--bucket-radius", type=int, default=2)
    parser.add_argument("--min-candidates", type=int, default=16)
    parser.add_argument("--max-candidates", type=int, default=512)
    parser.add_argument("--pre-filter-candidates", type=int, default=2048)
    parser.add_argument("--routing-layers", type=int, default=1)
    parser.add_argument("--promotion-probability", type=float, default=0.45)
    parser.add_argument("--promotion-bias", type=float, default=0.12)
    parser.add_argument("--routing-beam-width", type=int, default=32)
    parser.add_argument("--include-min-collision", type=float, default=1.0)
    parser.add_argument("--include-min-overlap", type=float, default=0.01)
    parser.add_argument("--token-encoder-checkpoint", default="")
    parser.add_argument("--token-encoder-device", default="")
    parser.add_argument("--hybrid-candidate-fetch", type=int, default=1024)
    parser.add_argument("--hybrid-token-candidate-fetch", type=int, default=1024)
    parser.add_argument("--hybrid-union-vector-weight", type=float, default=0.70)
    parser.add_argument("--hybrid-union-token-weight", type=float, default=0.30)
    parser.add_argument("--hybrid-source-weight", type=float, default=0.75)
    parser.add_argument("--hybrid-semantic-weight", type=float, default=0.15)
    parser.add_argument("--hybrid-field-weight", type=float, default=0.08)
    parser.add_argument("--hybrid-activation-weight", type=float, default=0.02)
    parser.add_argument("--memory-graph-layers", type=int, default=8)
    parser.add_argument("--memory-graph-route-degree", type=int, default=24)
    parser.add_argument("--memory-graph-projection-count", type=int, default=3)
    parser.add_argument("--memory-graph-projection-window", type=int, default=48)
    parser.add_argument("--memory-graph-promotion-threshold", type=int, default=72)
    parser.add_argument("--memory-graph-bias-promotion", action="store_true")
    parser.add_argument("--memory-graph-bridge-degree", type=int, default=6)
    parser.add_argument("--memory-graph-importance-threshold", type=float, default=0.68)
    parser.add_argument("--memory-graph-reciprocal-routes", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--memory-graph-ef", type=int, default=80)
    parser.add_argument("--memory-graph-beam", type=int, default=12)
    parser.add_argument("--memory-graph-objective-seeds", type=int, default=16)
    parser.add_argument("--memory-graph-truth-seeds", type=int, default=16)
    parser.add_argument("--memory-graph-truth-depth", type=int, default=2)
    parser.add_argument("--memory-graph-truth-fanout", type=int, default=6)
    parser.add_argument("--memory-graph-min-results", type=int, default=4)
    parser.add_argument("--memory-graph-cutoff-margin", type=float, default=0.28)
    parser.add_argument("--memory-graph-min-score", type=float, default=-0.05)
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
