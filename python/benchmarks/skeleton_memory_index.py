from __future__ import annotations

import argparse
import json
import math
import mmap
import struct
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from python.benchmarks.benchmark_librarian import evaluate_ranked
from python.benchmarks.hard_memory_regression import (
    GROWTH_SCENARIOS,
    aggregate,
    average,
    changed_existing_positions,
    grow_row,
    quantile,
    relevant_retention,
    top_retention,
)
from python.benchmarks.hierarchical_file_ann import Ranked, compact_ranked, project_key, ranked_signature, route_key, topic_key
from python.benchmarks.hippocampus_retrieval import (
    activation_overlap,
    build_embedding_backend,
    edge_activation,
    edge_type_boost,
    ensure_backend_embeddings,
    multihop_metrics,
    state_score,
)
from python.benchmarks.large_pool_retrieval import build_large_pool_case, context_decoy_metrics
from python.librarian.features import activation_mask_for_text, fnv1a64


SKELETON_MAGIC = b"HSKEL1\0\0"
SKELETON_HEADER = struct.Struct("<8sIIII")
SKELETON_RECORD = struct.Struct("<QQQQQIIHff")
SKELETON_EDGE = struct.Struct("<IQfff")
SKELETON_VERSION = 1
FLAG_PROTECTED = 1 << 0
FLAG_CONFLICT = 1 << 1
FLAG_IGNORED = 1 << 2
FLAG_STALE = 1 << 3


def parse_scenarios(raw: str) -> list[str]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = sorted(set(values) - set(GROWTH_SCENARIOS))
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown growth scenarios: {', '.join(unknown)}")
    return values


def normalize(values: list[float], dims: int) -> list[float]:
    out = [0.0] * dims
    limit = min(len(values), dims)
    total = 0.0
    for index in range(limit):
        value = float(values[index])
        out[index] = value
        total += value * value
    if total <= 0.0:
        return out
    scale = 1.0 / math.sqrt(total)
    return [value * scale for value in out]


def quantize_unit(values: list[float], dims: int) -> bytes:
    normalized = normalize(values, dims)
    signed = [max(-127, min(127, int(round(value * 127.0)))) for value in normalized]
    return bytes((value + 256) % 256 for value in signed)


def dot_i8(left: list[int], right: memoryview) -> float:
    total = 0
    for a, b in zip(left, right):
        total += int(a) * int(b)
    return total / 16129.0


def mask_overlap(query_mask: int, candidate_mask: int) -> float:
    if not query_mask or not candidate_mask:
        return 0.0
    return (query_mask & candidate_mask).bit_count() / max(1, (query_mask | candidate_mask).bit_count())


def skeleton_flags(card: dict[str, Any]) -> int:
    text = str(card.get("text") or "").lower()
    role = str(card.get("synthetic_role") or "").lower()
    outcome = str(card.get("last_outcome") or "").lower()
    age_days = int(card.get("age_days") or 0)
    use_count = int(card.get("use_count") or 0)
    flags = 0
    if bool(card.get("protected")):
        flags |= FLAG_PROTECTED
    if "conflict" in role or "decoy" in role or "superseded" in text or "contradicts" in text:
        flags |= FLAG_CONFLICT
    if outcome == "ignored":
        flags |= FLAG_IGNORED
    if age_days >= 180 and use_count <= 0:
        flags |= FLAG_STALE
    return flags


def build_skeleton_index(row: dict[str, Any], backend: Any, output_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    row = ensure_backend_embeddings(row, backend)
    output_dir.mkdir(parents=True, exist_ok=True)
    skeleton_path = output_dir / "skeleton.hsk"
    payload_path = output_dir / "payload.json"
    meta_path = output_dir / "skeleton.json"
    dims = int(args.skeleton_dims)
    cards = [dict(card) for card in row.get("candidates", [])]
    cards.sort(key=lambda card: str(card.get("id") or ""))
    id_to_index = {str(card["id"]): index for index, card in enumerate(cards)}

    outgoing: dict[str, list[dict[str, Any]]] = {}
    for edge in (row.get("memory_graph") or {}).get("edges") or []:
        source = str(edge.get("source") or "")
        target = str(edge.get("target") or "")
        if source in id_to_index and target in id_to_index:
            outgoing.setdefault(source, []).append(edge)

    edge_frames: list[bytes] = []
    edge_offsets: dict[str, int] = {}
    edge_counts: dict[str, int] = {}
    for card in cards:
        node_id = str(card["id"])
        edge_offsets[node_id] = len(edge_frames)
        for edge in sorted(outgoing.get(node_id, []), key=lambda item: (str(item.get("target") or ""), str(item.get("type") or ""))):
            edge_frames.append(
                SKELETON_EDGE.pack(
                    id_to_index[str(edge.get("target"))],
                    int(edge_activation(edge)),
                    float(edge.get("weight") or 0.0),
                    float(edge.get("confidence") or 0.5),
                    edge_type_boost(str(edge.get("type") or "used_with")),
                )
            )
        edge_counts[node_id] = len(edge_frames) - edge_offsets[node_id]

    payload = {"ids": [], "texts": []}
    with skeleton_path.open("wb") as handle:
        handle.write(SKELETON_HEADER.pack(SKELETON_MAGIC, SKELETON_VERSION, dims, len(cards), len(edge_frames)))
        for card in cards:
            node_id = str(card["id"])
            project = project_key(card)
            topic = topic_key(card)
            route = route_key(card)
            text = str(card.get("text") or "")
            mask = activation_mask_for_text(f"{text} {card.get('summary', '')}")
            handle.write(
                SKELETON_RECORD.pack(
                    fnv1a64(node_id),
                    fnv1a64(project),
                    fnv1a64(f"{project}:{topic}"),
                    fnv1a64(f"{project}:{topic}:{route}"),
                    int(mask),
                    edge_offsets[node_id],
                    edge_counts[node_id],
                    skeleton_flags(card),
                    float(card.get("importance") or 0.5),
                    float(state_score(card)),
                )
            )
            handle.write(quantize_unit([float(value) for value in card.get("embedding") or []], dims))
            payload["ids"].append(node_id)
            payload["texts"].append(text)
        for frame in edge_frames:
            handle.write(frame)
    payload_path.write_text(json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8")
    meta = {
        "version": SKELETON_VERSION,
        "skeleton_path": str(skeleton_path),
        "payload_path": str(payload_path),
        "dims": dims,
        "record_count": len(cards),
        "edge_count": len(edge_frames),
        "segment_count": int(args.skeleton_segments),
    }
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    meta["build_latency_ms"] = (time.perf_counter() - started) * 1000.0
    meta["skeleton_bytes"] = skeleton_path.stat().st_size
    meta["payload_bytes"] = payload_path.stat().st_size
    meta["index_bytes"] = meta_path.stat().st_size
    return meta


class SkeletonIndex:
    def __init__(self, meta: dict[str, Any]):
        self.meta = meta
        self.path = Path(meta["skeleton_path"])
        self.handle = self.path.open("rb")
        self.data = mmap.mmap(self.handle.fileno(), 0, access=mmap.ACCESS_READ)
        magic, version, dims, record_count, edge_count = SKELETON_HEADER.unpack_from(self.data, 0)
        if magic != SKELETON_MAGIC or version != SKELETON_VERSION:
            raise ValueError(f"unsupported skeleton index: {self.path}")
        self.dims = int(dims)
        self.record_count = int(record_count)
        self.edge_count = int(edge_count)
        self.record_size = SKELETON_RECORD.size + self.dims
        self.records_offset = SKELETON_HEADER.size
        self.edges_offset = self.records_offset + self.record_size * self.record_count
        payload = json.loads(Path(meta["payload_path"]).read_text(encoding="utf-8"))
        self.ids = [str(value) for value in payload["ids"]]
        self.texts = [str(value) for value in payload["texts"]]

    def close(self) -> None:
        self.data.close()
        self.handle.close()

    def record_offset(self, index: int) -> int:
        return self.records_offset + index * self.record_size

    def record(self, index: int) -> tuple[int, int, int, int, int, int, int, int, float, float]:
        return SKELETON_RECORD.unpack_from(self.data, self.record_offset(index))

    def vector_view(self, index: int) -> memoryview:
        start = self.record_offset(index) + SKELETON_RECORD.size
        return memoryview(self.data)[start : start + self.dims].cast("b")

    def edge(self, edge_index: int) -> tuple[int, int, float, float, float]:
        return SKELETON_EDGE.unpack_from(self.data, self.edges_offset + edge_index * SKELETON_EDGE.size)


def score_record(
    index: SkeletonIndex,
    record_index: int,
    record: tuple[int, int, int, int, int, int, int, int, float, float],
    query_vector: list[int],
    query_mask: int,
    query_project_hash: int,
) -> float:
    _, project_hash, _, _, mask, _, _, flags, importance, state = record
    semantic = dot_i8(query_vector, index.vector_view(record_index))
    activation = mask_overlap(query_mask, int(mask))
    same_project = 1.0 if int(project_hash) == query_project_hash else 0.0
    conflict_penalty = 0.06 if flags & FLAG_CONFLICT else 0.0
    ignored_penalty = 0.04 if flags & FLAG_IGNORED else 0.0
    stale_penalty = 0.04 if flags & FLAG_STALE else 0.0
    return (
        0.48 * semantic
        + 0.22 * activation
        + 0.11 * same_project
        + 0.07 * float(importance)
        + float(state)
        - conflict_penalty
        - ignored_penalty
        - stale_penalty
    )


def insert_top(bucket: list[tuple[int, float]], item: tuple[int, float], limit: int) -> None:
    bucket.append(item)
    if len(bucket) > limit * 2:
        bucket.sort(key=lambda value: (-value[1], value[0]))
        del bucket[limit:]


def skeleton_search(row: dict[str, Any], backend: Any, meta: dict[str, Any], args: argparse.Namespace) -> tuple[Ranked, dict[str, float], set[str]]:
    task = row.get("retrieval_task") or {}
    query = str(task.get("query") or row["anchor"]["text"])
    query_embedding = task.get("query_embedding") or backend.embed_one(query)
    query_vector = list(memoryview(quantize_unit([float(value) for value in query_embedding], int(args.skeleton_dims))).cast("b"))
    query_mask = activation_mask_for_text(query)
    query_project_hash = fnv1a64(project_key(row["anchor"]))
    segment_count = max(1, int(args.skeleton_segments))
    segment_limit = max(1, int(args.skeleton_segment_limit))
    edge_seed_count = max(0, int(args.skeleton_edge_seed_count))
    final_fetch = max(1, int(args.skeleton_final_fetch))
    started = time.perf_counter()
    index = SkeletonIndex(meta)
    try:
        segment_scores: list[list[tuple[int, float]]] = [[] for _ in range(segment_count)]
        base_scores: dict[int, float] = {}
        skeleton_records_scored = 0
        for record_index in range(index.record_count):
            record = index.record(record_index)
            score = score_record(index, record_index, record, query_vector, query_mask, query_project_hash)
            base_scores[record_index] = score
            segment = int(record[0] % segment_count)
            insert_top(segment_scores[segment], (record_index, score), segment_limit)
            skeleton_records_scored += 1

        best: dict[int, float] = {}
        for bucket in segment_scores:
            bucket.sort(key=lambda value: (-value[1], value[0]))
            for record_index, score in bucket[:segment_limit]:
                if score > best.get(record_index, -99.0):
                    best[record_index] = score

        protected_indices: set[int] = set()
        edge_expansions = 0
        frontier = [(record_index, score, [record_index]) for record_index, score in sorted(best.items(), key=lambda item: (-item[1], item[0]))[:edge_seed_count]]
        for depth in range(max(0, int(args.skeleton_graph_depth))):
            next_frontier: list[tuple[int, float, list[int]]] = []
            for current_index, current_score, path in frontier:
                record = index.record(current_index)
                _, _, _, _, _, edge_offset, edge_count, _, _, _ = record
                if edge_count <= 0:
                    continue
                protected_indices.add(current_index)
                for local_edge in range(edge_count):
                    target_index, edge_mask, weight, confidence, type_boost = index.edge(edge_offset + local_edge)
                    if target_index in path:
                        continue
                    target_base = base_scores.get(target_index)
                    if target_base is None:
                        target_record = index.record(target_index)
                        target_base = score_record(index, target_index, target_record, query_vector, query_mask, query_project_hash)
                        base_scores[target_index] = target_base
                    hop_gain = (
                        float(weight)
                        * float(type_boost)
                        * (0.70 + 0.45 * activation_overlap(query_mask, int(edge_mask)))
                        * (0.70 + 0.30 * float(confidence))
                        / (1.25 + depth)
                    )
                    score = 0.58 * current_score + 0.42 * target_base + hop_gain
                    if score > best.get(target_index, -99.0):
                        best[target_index] = score
                        protected_indices.add(target_index)
                        edge_expansions += 1
                        next_frontier.append((target_index, score, path + [target_index]))
            frontier = next_frontier

        ordered = sorted(best.items(), key=lambda item: (-item[1], item[0]))
        fetched = ordered[:final_fetch]
        ranked = [(index.ids[record_index], score, index.texts[record_index]) for record_index, score in fetched]
        protected_ids = {index.ids[record_index] for record_index in protected_indices}
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        stats = {
            "latency_ms": elapsed_ms,
            "skeleton_records_scored": float(skeleton_records_scored),
            "skeleton_segments": float(segment_count),
            "skeleton_segment_limit": float(segment_limit),
            "edge_expansions": float(edge_expansions),
            "raw_final_candidate_count": float(len(best)),
            "final_candidate_count": float(len(ranked)),
            "payload_reads": float(len(ranked)),
            "unique_nodes_read": float(len(ranked)),
        }
        return ranked, stats, protected_ids
    finally:
        index.close()


def compact_output(row: dict[str, Any], ranked: Ranked, protected_ids: set[str], stats: dict[str, float], args: argparse.Namespace) -> tuple[Ranked, dict[str, float]]:
    compacted, compact_stats = compact_ranked(row, ranked, protected_ids, SimpleNamespace(compact_limit=args.compact_limit))
    out = dict(stats)
    out["raw_final_candidate_count"] = float(stats.get("raw_final_candidate_count") or len(ranked))
    out["final_candidate_count"] = float(compact_stats.get("output_count") or len(compacted))
    out["compactor_latency_ms"] = float(compact_stats.get("latency_ms") or 0.0)
    out["compactor_protected_count"] = float(compact_stats.get("protected_count") or 0.0)
    return compacted, out


def determinism_check(row: dict[str, Any], backend: Any, case_dir: Path, args: argparse.Namespace) -> tuple[Ranked, dict[str, float]]:
    meta = build_skeleton_index(row, backend, case_dir / "determinism_a", args)
    raw_ranked, stats, protected_ids = skeleton_search(row, backend, meta, args)
    ranked, stats = compact_output(row, raw_ranked, protected_ids, stats, args)
    expected = ranked_signature(raw_ranked)
    mismatches = 0
    repeat_latency = []
    for _ in range(max(0, int(args.determinism_repeats) - 1)):
        repeated, repeated_stats, _ = skeleton_search(row, backend, meta, args)
        repeat_latency.append(float(repeated_stats.get("latency_ms") or 0.0))
        if ranked_signature(repeated) != expected:
            mismatches += 1
    rebuilt_meta = build_skeleton_index(row, backend, case_dir / "determinism_b", args)
    rebuilt, rebuild_stats, _ = skeleton_search(row, backend, rebuilt_meta, args)
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
        "payload_reads": float(stats.get("payload_reads") or 0.0),
        "skeleton_records_scored": float(stats.get("skeleton_records_scored") or 0.0),
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


def evaluate_growth(base: Ranked, grown: Ranked, grown_row: dict[str, Any], grown_stats: dict[str, float], args: argparse.Namespace) -> dict[str, float]:
    out = evaluate_one(grown_row, grown, grown_stats, args)
    out["growth_relevant_retention"] = relevant_retention(base, grown, grown_row)
    out["growth_topn_retention"] = top_retention(base, grown, args.stability_top_n)
    out["growth_existing_position_change_rate"] = changed_existing_positions(base, grown, args.stability_top_n)
    return out


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
        "benchmark": "skeleton_memory_index",
        "embedding_backend": backend.name,
        "cases": args.cases,
        "pool_size": args.pool_size,
        "growth_count": args.growth_count,
        "growth_scenarios": args.growth_scenarios,
        "determinism_repeats": args.determinism_repeats,
        "skeleton_dims": args.skeleton_dims,
        "skeleton_segments": args.skeleton_segments,
        "skeleton_segment_limit": args.skeleton_segment_limit,
        "skeleton_final_fetch": args.skeleton_final_fetch,
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


def write_markdown(result: dict[str, Any], path: Path) -> None:
    lines = [
        "# Skeleton Memory Index",
        "",
        f"- backend: `{result['embedding_backend']}`",
        f"- cases: `{result['cases']}`",
        f"- pool_size: `{result['pool_size']}`",
        f"- growth_count: `{result['growth_count']}`",
        f"- skeleton_dims: `{result['skeleton_dims']}`",
        f"- skeleton_segments: `{result['skeleton_segments']}`",
        f"- skeleton_segment_limit: `{result['skeleton_segment_limit']}`",
        f"- skeleton_final_fetch: `{result['skeleton_final_fetch']}`",
        f"- elapsed_seconds: `{result['elapsed_seconds']}`",
        "",
        "| section | recall | precision | latency p95 ms | deterministic | payload reads p95 | raw candidates p95 | relevant retention | topN retention |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    def add_row(name: str, values: dict[str, dict[str, float]]) -> None:
        lines.append(
            f"| {name} | "
            f"{values.get('retrieval_context_recall', {}).get('avg', 0.0):.4f} | "
            f"{values.get('retrieval_context_precision', {}).get('avg', 0.0):.4f} | "
            f"{values.get('latency_ms', {}).get('p95', 0.0):.2f} | "
            f"{values.get('deterministic', {}).get('avg', 0.0):.4f} | "
            f"{values.get('payload_reads', {}).get('p95', 0.0):.2f} | "
            f"{values.get('raw_final_candidate_count', {}).get('p95', 0.0):.2f} | "
            f"{values.get('growth_relevant_retention', {}).get('avg', 0.0):.4f} | "
            f"{values.get('growth_topn_retention', {}).get('avg', 0.0):.4f} |"
        )

    add_row("baseline", result["baseline"])
    for scenario, values in result["growth"].items():
        add_row(scenario, values)
    if result.get("failures"):
        lines.extend(["", "## Failures", ""])
        lines.extend(f"- {failure}" for failure in result["failures"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=int, default=5)
    parser.add_argument("--pool-size", type=int, default=5000)
    parser.add_argument("--growth-count", type=int, default=1000)
    parser.add_argument("--growth-scenarios", type=parse_scenarios, default=list(GROWTH_SCENARIOS))
    parser.add_argument("--seed", type=int, default=51000)
    parser.add_argument("--work-dir", default="artifacts/hippocampus/skeleton_memory_index")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--budget", type=int, default=90)
    parser.add_argument("--compact-limit", type=int, default=3)
    parser.add_argument("--stability-top-n", type=int, default=64)
    parser.add_argument("--determinism-repeats", type=int, default=2)
    parser.add_argument("--skeleton-dims", type=int, default=32)
    parser.add_argument("--skeleton-segments", type=int, default=16)
    parser.add_argument("--skeleton-segment-limit", type=int, default=64)
    parser.add_argument("--skeleton-edge-seed-count", type=int, default=48)
    parser.add_argument("--skeleton-graph-depth", type=int, default=2)
    parser.add_argument("--skeleton-final-fetch", type=int, default=64)
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
