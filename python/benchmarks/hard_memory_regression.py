from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from python.benchmarks.benchmark_librarian import evaluate_ranked
from python.benchmarks.hierarchical_file_ann import Ranked, compact_ranked, hierarchical_search, ranked_signature, write_lazy_index
from python.benchmarks.hippocampus_retrieval import build_embedding_backend, ensure_backend_embeddings, multihop_metrics
from python.benchmarks.large_pool_retrieval import build_large_pool_case, context_decoy_metrics
from python.librarian.features import embed_text, fnv1a64


GROWTH_SCENARIOS = ("unrelated", "semantic_decoy", "conflict", "repeated", "combined")


def average(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(math.ceil(q * len(ordered)) - 1)))
    return float(ordered[index])


def bootstrap_mean_ci(values: list[float], *, iterations: int = 300, seed: int = 934711) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return float(values[0]), float(values[0])
    rng = random.Random(seed + len(values) * 1009)
    means = []
    count = len(values)
    for _ in range(max(1, int(iterations))):
        means.append(sum(values[rng.randrange(count)] for _ in range(count)) / count)
    return quantile(means, 0.025), quantile(means, 0.975)


def hargs(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        beam_width=args.beam_width,
        max_frontier=args.max_frontier,
        max_children_per_basin=args.max_children_per_basin,
        max_leaf_reads=args.max_leaf_reads,
        promoted_per_basin=args.promoted_per_basin,
        graph_seed_count=args.graph_seed_count,
        graph_depth=args.graph_depth,
        cache_size=args.cache_size,
        semantic_dims=args.semantic_dims,
        store_full_embeddings=args.store_full_embeddings,
        promotion_bias=args.promotion_bias,
        promotion_scale=args.promotion_scale,
        return_limit=args.return_limit,
        stable_growth=args.stable_growth,
        stable_basin_floor=args.stable_basin_floor,
        stable_max_basins=args.stable_max_basins,
        stable_max_leaf_reads=args.stable_max_leaf_reads,
        stable_max_promoted_reads=args.stable_max_promoted_reads,
        compact_limit=args.compact_limit,
    )


def card(
    anchor: dict[str, Any],
    memory_id: str,
    text: str,
    role: str,
    *,
    age_days: int,
    use_count: int,
    evidence_count: int,
    last_outcome: str,
    importance: float,
    protected: bool = False,
    project: str | None = None,
) -> dict[str, Any]:
    project_name = project or str((anchor.get("metadata") or {}).get("project") or anchor.get("cluster") or "unknown")
    return {
        "id": memory_id,
        "text": text,
        "summary": "",
        "embedding": embed_text(text),
        "importance": importance,
        "cluster": project_name,
        "metadata": {"project": project_name},
        "age_days": age_days,
        "use_count": use_count,
        "evidence_count": evidence_count,
        "last_outcome": last_outcome,
        "protected": protected,
        "synthetic_role": role,
    }


def stable_choice(values: list[str], key: str) -> str:
    return values[fnv1a64(key) % len(values)]


def relevant_cards(row: dict[str, Any]) -> list[dict[str, Any]]:
    relevant = set((row.get("retrieval_task") or {}).get("relevant_ids") or [])
    return [dict(candidate) for candidate in row.get("candidates", []) if candidate.get("id") in relevant]


def add_unrelated_growth(row: dict[str, Any], count: int, seed: int) -> list[dict[str, Any]]:
    anchor = row["anchor"]
    projects = ["archive", "billing", "research", "ops", "calendar", "personal"]
    topics = ["invoice cleanup", "old browser tabs", "meeting notes", "deployment checklist", "reading queue"]
    out = []
    for index in range(count):
        project = stable_choice(projects, f"{seed}:unrelated:project:{index}")
        topic = stable_choice(topics, f"{seed}:unrelated:topic:{index}")
        text = f"{project}: unrelated growth memory {seed}-{index} about {topic}; no current escalation path applies."
        out.append(
            card(
                anchor,
                f"hard_unrelated_{seed}_{index}",
                text,
                "hard_unrelated_growth",
                age_days=fnv1a64(f"{seed}:age:{index}") % 180,
                use_count=fnv1a64(f"{seed}:use:{index}") % 9,
                evidence_count=fnv1a64(f"{seed}:ev:{index}") % 4,
                last_outcome=stable_choice(["", "", "helpful", "ignored"], f"{seed}:outcome:{index}"),
                importance=0.25 + 0.25 * ((fnv1a64(f"{seed}:imp:{index}") % 100) / 100.0),
                project=project,
            )
        )
    return out


def add_semantic_decoys(row: dict[str, Any], count: int, seed: int) -> list[dict[str, Any]]:
    anchor = row["anchor"]
    query = str((row.get("retrieval_task") or {}).get("query") or anchor.get("text") or "")
    out = []
    for index in range(count):
        text = (
            f"{query} Candidate branch {seed}-{index} repeats the route words, but it is a decoy: "
            "do not use it as the accepted resolution."
        )
        out.append(
            card(
                anchor,
                f"hard_semantic_decoy_{seed}_{index}",
                text,
                "hard_semantic_decoy",
                age_days=1 + index % 7,
                use_count=18 + index % 13,
                evidence_count=6 + index % 5,
                last_outcome="ignored" if index % 2 else "corrected",
                importance=0.72,
            )
        )
    return out


def add_conflicts(row: dict[str, Any], count: int, seed: int) -> list[dict[str, Any]]:
    anchor = row["anchor"]
    query = str((row.get("retrieval_task") or {}).get("query") or anchor.get("text") or "")
    relevant_text = " ".join(candidate.get("text", "") for candidate in relevant_cards(row))
    out = []
    for index in range(count):
        text = (
            f"{query} Conflict record {seed}-{index}: older note resembles the accepted path, "
            f"but contradicts it. Relevant path said: {relevant_text[:180]}. This record is superseded."
        )
        out.append(
            card(
                anchor,
                f"hard_conflict_{seed}_{index}",
                text,
                "hard_conflict_decoy",
                age_days=220 + index % 180,
                use_count=0 if index % 3 else 2,
                evidence_count=0 if index % 2 else 1,
                last_outcome="ignored",
                importance=0.38,
            )
        )
    return out


def add_repeated_inserts(row: dict[str, Any], count: int, seed: int) -> list[dict[str, Any]]:
    anchor = row["anchor"]
    base = relevant_cards(row) or [dict(row["anchor"])]
    out = []
    for index in range(count):
        source = base[index % len(base)]
        text = (
            f"Repeated insert copy {seed}-{index}: {source.get('text', '')} "
            "This is a repeated non-authoritative duplicate."
        )
        out.append(
            card(
                anchor,
                f"hard_repeated_{seed}_{index}",
                text,
                "hard_repeated_duplicate",
                age_days=0,
                use_count=1,
                evidence_count=0,
                last_outcome="",
                importance=0.44,
            )
        )
    return out


def growth_cards(row: dict[str, Any], scenario: str, count: int, seed: int) -> list[dict[str, Any]]:
    if count <= 0:
        return []
    if scenario == "unrelated":
        return add_unrelated_growth(row, count, seed)
    if scenario == "semantic_decoy":
        return add_semantic_decoys(row, count, seed)
    if scenario == "conflict":
        return add_conflicts(row, count, seed)
    if scenario == "repeated":
        return add_repeated_inserts(row, count, seed)
    if scenario == "combined":
        buckets = [count // 4, count // 4, count // 4, count - 3 * (count // 4)]
        return (
            add_unrelated_growth(row, buckets[0], seed)
            + add_semantic_decoys(row, buckets[1], seed)
            + add_conflicts(row, buckets[2], seed)
            + add_repeated_inserts(row, buckets[3], seed)
        )
    raise ValueError(f"unknown growth scenario: {scenario}")


def grow_row(row: dict[str, Any], scenario: str, count: int, seed: int) -> dict[str, Any]:
    grown = copy.deepcopy(row)
    grown["candidates"] = [dict(candidate) for candidate in row.get("candidates", [])] + growth_cards(row, scenario, count, seed)
    return grown


def ids(ranked: Ranked) -> list[str]:
    return [memory_id for memory_id, _, _ in ranked]


def relevant_retention(base: Ranked, grown: Ranked, row: dict[str, Any]) -> float:
    relevant = set((row.get("retrieval_task") or {}).get("relevant_ids") or [])
    base_relevant = relevant & set(ids(base))
    if not base_relevant:
        return 1.0
    return len(base_relevant & set(ids(grown))) / len(base_relevant)


def top_retention(base: Ranked, grown: Ranked, top_n: int) -> float:
    base_top = set(ids(base)[:top_n])
    if not base_top:
        return 1.0
    return len(base_top & set(ids(grown))) / len(base_top)


def changed_existing_positions(base: Ranked, grown: Ranked, top_n: int) -> float:
    base_existing = ids(base)[:top_n]
    grown_positions = {memory_id: index for index, memory_id in enumerate(ids(grown))}
    changed = 0
    compared = 0
    for index, memory_id in enumerate(base_existing):
        if memory_id not in grown_positions:
            continue
        compared += 1
        if grown_positions[memory_id] != index:
            changed += 1
    return changed / max(1, compared)


def run_search(
    row: dict[str, Any],
    backend: Any,
    case_dir: Path,
    args: argparse.Namespace,
    *,
    suffix: str,
) -> tuple[Ranked, dict[str, float], dict[str, Any]]:
    meta = write_lazy_index(row, backend, case_dir / suffix, hargs(args))
    ranked, stats, protected_ids = hierarchical_search(row, backend, meta, hargs(args))
    compacted, compact_stats = compact_output(row, ranked, protected_ids, stats, args)
    return compacted, compact_stats, meta


def compact_output(
    row: dict[str, Any],
    ranked: Ranked,
    protected_ids: set[str],
    stats: dict[str, float],
    args: argparse.Namespace,
) -> tuple[Ranked, dict[str, float]]:
    compacted, compact_stats = compact_ranked(row, ranked, protected_ids, hargs(args))
    out = dict(stats)
    out["raw_final_candidate_count"] = float(out.get("final_candidate_count") or 0.0)
    out["final_candidate_count"] = float(compact_stats.get("output_count") or len(compacted))
    out["compactor_latency_ms"] = float(compact_stats.get("latency_ms") or 0.0)
    out["compactor_protected_count"] = float(compact_stats.get("protected_count") or 0.0)
    return compacted, out


def determinism_check(row: dict[str, Any], backend: Any, case_dir: Path, args: argparse.Namespace) -> tuple[Ranked, dict[str, float]]:
    meta = write_lazy_index(row, backend, case_dir / "determinism_a", hargs(args))
    raw_ranked, stats, protected_ids = hierarchical_search(row, backend, meta, hargs(args))
    ranked, stats = compact_output(row, raw_ranked, protected_ids, stats, args)
    expected = ranked_signature(raw_ranked)
    mismatches = 0
    repeat_latency = []
    for repeat in range(max(0, args.determinism_repeats - 1)):
        repeated, repeated_stats, _ = hierarchical_search(row, backend, meta, hargs(args))
        repeat_latency.append(float(repeated_stats.get("latency_ms") or 0.0))
        if ranked_signature(repeated) != expected:
            mismatches += 1
    rebuilt_meta = write_lazy_index(row, backend, case_dir / "determinism_b", hargs(args))
    rebuilt, rebuild_stats, _ = hierarchical_search(row, backend, rebuilt_meta, hargs(args))
    if ranked_signature(rebuilt) != expected:
        mismatches += 1
    out = dict(stats)
    out["determinism_mismatches"] = float(mismatches)
    out["determinism_repeat_latency_ms"] = average(repeat_latency)
    out["determinism_rebuild_latency_ms"] = float(rebuild_stats.get("latency_ms") or 0.0)
    return ranked, out


def evaluate_one(row: dict[str, Any], ranked: Ranked, stats: dict[str, float], args: argparse.Namespace) -> dict[str, float]:
    out = {
        "latency_ms": float(stats.get("latency_ms") or 0.0),
        "compactor_latency_ms": float(stats.get("compactor_latency_ms") or 0.0),
        "unique_nodes_read": float(stats.get("unique_nodes_read") or 0.0),
        "raw_final_candidate_count": float(stats.get("raw_final_candidate_count") or 0.0),
        "final_candidate_count": float(stats.get("final_candidate_count") or 0.0),
        "compactor_protected_count": float(stats.get("compactor_protected_count") or 0.0),
        "determinism_mismatches": float(stats.get("determinism_mismatches") or 0.0),
        "deterministic": 1.0 if float(stats.get("determinism_mismatches") or 0.0) == 0.0 else 0.0,
    }
    out.update({f"retrieval_{key}": value for key, value in evaluate_ranked(row, ranked, args.top_k, args.budget).items()})
    out.update({f"multihop_{key}": value for key, value in multihop_metrics(row, ranked, args.top_k, args.budget).items()})
    out.update({f"context_{key}": value for key, value in context_decoy_metrics(row, ranked, args.budget).items()})
    return out


def evaluate_growth(
    base: Ranked,
    grown: Ranked,
    grown_row: dict[str, Any],
    grown_stats: dict[str, float],
    args: argparse.Namespace,
) -> dict[str, float]:
    out = evaluate_one(grown_row, grown, grown_stats, args)
    out["growth_relevant_retention"] = relevant_retention(base, grown, grown_row)
    out["growth_topn_retention"] = top_retention(base, grown, args.stability_top_n)
    out["growth_existing_position_change_rate"] = changed_existing_positions(base, grown, args.stability_top_n)
    return out


def aggregate(rows: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    keys = sorted({key for row in rows for key in row})
    out = {}
    for key in keys:
        values = [row[key] for row in rows if key in row]
        ci_low, ci_high = bootstrap_mean_ci(values)
        out[key] = {
            "avg": average(values),
            "p50": quantile(values, 0.50),
            "p95": quantile(values, 0.95),
            "min": min(values, default=0.0),
            "max": max(values, default=0.0),
            "avg_ci95_low": ci_low,
            "avg_ci95_high": ci_high,
        }
    return out


def run(args: argparse.Namespace) -> dict[str, Any]:
    backend = build_embedding_backend(args)
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    baseline_rows: list[dict[str, float]] = []
    growth_rows: dict[str, list[dict[str, float]]] = {scenario: [] for scenario in args.growth_scenarios}
    for case_index in range(args.cases):
        seed = args.seed + case_index
        case_dir = work_dir / f"case_{seed}"
        row = ensure_backend_embeddings(build_large_pool_case(seed, args.pool_size), backend)
        base_ranked, base_stats = determinism_check(row, backend, case_dir, args)
        baseline_rows.append(evaluate_one(row, base_ranked, base_stats, args))
        for scenario in args.growth_scenarios:
            grown = grow_row(row, scenario, args.growth_count, seed)
            grown = ensure_backend_embeddings(grown, backend)
            grown_ranked, grown_stats = determinism_check(grown, backend, case_dir / f"growth_{scenario}", args)
            growth_rows[scenario].append(evaluate_growth(base_ranked, grown_ranked, grown, grown_stats, args))
        if (case_index + 1) % max(1, args.log_every) == 0:
            print(
                json.dumps(
                    {
                        "case": case_index + 1,
                        "baseline_recall": average([row["retrieval_context_recall"] for row in baseline_rows]),
                        "baseline_latency_p95": quantile([row["latency_ms"] for row in baseline_rows], 0.95),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    result = {
        "benchmark": "hard_memory_regression",
        "embedding_backend": backend.name,
        "cases": args.cases,
        "pool_size": args.pool_size,
        "growth_count": args.growth_count,
        "growth_scenarios": args.growth_scenarios,
        "determinism_repeats": args.determinism_repeats,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "baseline": aggregate(baseline_rows),
        "growth": {scenario: aggregate(rows) for scenario, rows in growth_rows.items()},
    }
    failures = regression_failures(result, args)
    result["failures"] = failures
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.output_md:
        write_markdown(result, Path(args.output_md))
    if args.fail_on_regression and failures:
        raise SystemExit("regression thresholds failed: " + "; ".join(failures))
    return result


def metric(result: dict[str, Any], section: str, key: str, stat: str = "avg", scenario: str = "") -> float:
    if section == "baseline":
        return float(result["baseline"].get(key, {}).get(stat, 0.0))
    return float(result["growth"].get(scenario, {}).get(key, {}).get(stat, 0.0))


def regression_failures(result: dict[str, Any], args: argparse.Namespace) -> list[str]:
    failures = []
    if metric(result, "baseline", "deterministic") < 1.0:
        failures.append("baseline determinism mismatch")
    if metric(result, "baseline", "retrieval_context_recall") < args.min_recall:
        failures.append("baseline recall below threshold")
    if metric(result, "baseline", "retrieval_context_precision") < args.min_precision:
        failures.append("baseline precision below threshold")
    if metric(result, "baseline", "latency_ms", "p95") > args.max_p95_ms:
        failures.append("baseline p95 latency above threshold")
    for scenario in args.growth_scenarios:
        if metric(result, "growth", "deterministic", scenario=scenario) < 1.0:
            failures.append(f"{scenario} determinism mismatch")
        if metric(result, "growth", "retrieval_context_recall", scenario=scenario) < args.min_recall:
            failures.append(f"{scenario} recall below threshold")
        if metric(result, "growth", "retrieval_context_precision", scenario=scenario) < args.min_precision:
            failures.append(f"{scenario} precision below threshold")
        if metric(result, "growth", "growth_relevant_retention", scenario=scenario) < args.min_growth_retention:
            failures.append(f"{scenario} relevant retention below threshold")
        if metric(result, "growth", "growth_topn_retention", scenario=scenario) < args.min_topn_retention:
            failures.append(f"{scenario} topN retention below threshold")
        if metric(result, "growth", "latency_ms", "p95", scenario=scenario) > args.max_p95_ms:
            failures.append(f"{scenario} p95 latency above threshold")
    return failures


def write_markdown(result: dict[str, Any], path: Path) -> None:
    lines = [
        "# Hard Memory Regression",
        "",
        f"- backend: `{result['embedding_backend']}`",
        f"- cases: `{result['cases']}`",
        f"- pool_size: `{result['pool_size']}`",
        f"- growth_count: `{result['growth_count']}`",
        f"- determinism_repeats: `{result['determinism_repeats']}`",
        f"- elapsed_seconds: `{result['elapsed_seconds']}`",
        "",
        "| section | recall | precision | latency p95 ms | deterministic | relevant retention | topN retention | position change | final candidates p95 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    def add_row(name: str, values: dict[str, dict[str, float]]) -> None:
        lines.append(
            f"| {name} | "
            f"{values.get('retrieval_context_recall', {}).get('avg', 0.0):.4f} | "
            f"{values.get('retrieval_context_precision', {}).get('avg', 0.0):.4f} | "
            f"{values.get('latency_ms', {}).get('p95', 0.0):.2f} | "
            f"{values.get('deterministic', {}).get('avg', 0.0):.4f} | "
            f"{values.get('growth_relevant_retention', {}).get('avg', 0.0):.4f} | "
            f"{values.get('growth_topn_retention', {}).get('avg', 0.0):.4f} | "
            f"{values.get('growth_existing_position_change_rate', {}).get('avg', 0.0):.4f} | "
            f"{values.get('raw_final_candidate_count', {}).get('p95', 0.0):.2f} |"
        )

    add_row("baseline", result["baseline"])
    for scenario, values in result["growth"].items():
        add_row(scenario, values)
    if result.get("failures"):
        lines.extend(["", "## Failures", ""])
        lines.extend(f"- {failure}" for failure in result["failures"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_scenarios(raw: str) -> list[str]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = sorted(set(values) - set(GROWTH_SCENARIOS))
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown growth scenarios: {', '.join(unknown)}")
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=int, default=5)
    parser.add_argument("--pool-size", type=int, default=5000)
    parser.add_argument("--growth-count", type=int, default=1000)
    parser.add_argument("--growth-scenarios", type=parse_scenarios, default=list(GROWTH_SCENARIOS))
    parser.add_argument("--seed", type=int, default=41000)
    parser.add_argument("--work-dir", default="artifacts/hard_memory_regression")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--budget", type=int, default=90)
    parser.add_argument("--return-limit", type=int, default=0)
    parser.add_argument("--beam-width", type=int, default=4)
    parser.add_argument("--max-frontier", type=int, default=256)
    parser.add_argument("--max-children-per-basin", type=int, default=128)
    parser.add_argument("--max-leaf-reads", type=int, default=384)
    parser.add_argument("--promoted-per-basin", type=int, default=32)
    parser.add_argument("--graph-seed-count", type=int, default=8)
    parser.add_argument("--graph-depth", type=int, default=2)
    parser.add_argument("--cache-size", type=int, default=256)
    parser.add_argument("--semantic-dims", type=int, default=128)
    parser.add_argument("--store-full-embeddings", action="store_true")
    parser.add_argument("--promotion-bias", type=float, default=0.0)
    parser.add_argument("--promotion-scale", type=float, default=1.0)
    parser.add_argument("--compact-limit", type=int, default=3)
    parser.add_argument("--stable-growth", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stable-basin-floor", type=float, default=0.32)
    parser.add_argument("--stable-max-basins", type=int, default=0)
    parser.add_argument("--stable-max-leaf-reads", type=int, default=0)
    parser.add_argument("--stable-max-promoted-reads", type=int, default=0)
    parser.add_argument("--stability-top-n", type=int, default=64)
    parser.add_argument("--determinism-repeats", type=int, default=2)
    parser.add_argument("--embedding-backend", choices=["hash", "hippo"], default="hash")
    parser.add_argument("--hippo-checkpoint", default="")
    parser.add_argument("--hippo-encoder-src", default="")
    parser.add_argument("--hippo-max-length", type=int, default=128)
    parser.add_argument("--hippo-batch-size", type=int, default=128)
    parser.add_argument("--device", default="")
    parser.add_argument("--min-recall", type=float, default=0.95)
    parser.add_argument("--min-precision", type=float, default=0.50)
    parser.add_argument("--min-growth-retention", type=float, default=1.0)
    parser.add_argument("--min-topn-retention", type=float, default=0.95)
    parser.add_argument("--max-p95-ms", type=float, default=200.0)
    parser.add_argument("--fail-on-regression", action="store_true")
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-md", default="")
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
