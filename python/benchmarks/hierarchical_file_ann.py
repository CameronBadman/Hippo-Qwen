from __future__ import annotations

import argparse
import json
import math
import struct
import sys
import time
import zlib
from collections import OrderedDict
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from python.benchmarks.benchmark_librarian import evaluate_ranked
from python.benchmarks.hippocampus_retrieval import (
    CachedEmbeddingBackend,
    activation_overlap,
    build_embedding_backend,
    edge_activation,
    edge_type_boost,
    ensure_backend_embeddings,
    multihop_metrics,
    sparse_basin_scores,
    state_score,
    vector_scores_with_backend,
)
from python.benchmarks.large_pool_retrieval import build_large_pool_case, context_decoy_metrics
from python.librarian.features import (
    activation_mask_for_text,
    clamp,
    cosine,
    fnv1a64,
    jaccard,
    normalize,
    tokens,
)


Ranked = list[tuple[str, float, str]]
LEVEL_ROOT = 0
LEVEL_PROJECT = 1
LEVEL_TOPIC = 2
LEVEL_MEMORY = 3
NODE_STORE_MAGIC = b"HGFANN1\x00"
NODE_HEADER = struct.Struct("<II")


def stable_unit(value: str) -> float:
    return (fnv1a64(value) % 10_000_000) / 10_000_000.0


def mean_embedding(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        return []
    dims = len(vectors[0])
    total = [0.0] * dims
    for vector in vectors:
        if len(vector) != dims:
            continue
        for idx, value in enumerate(vector):
            total[idx] += value
    return normalize([value / max(1, len(vectors)) for value in total])


def project_key(card: dict[str, Any]) -> str:
    metadata = card.get("metadata") or {}
    return str(metadata.get("project") or card.get("cluster") or "unknown").lower().replace(" ", "_")


def topic_key(card: dict[str, Any]) -> str:
    text = str(card.get("text") or "").lower()
    if "accepted resolution" in text or "support note" in text or "succeeded" in text:
        return "resolution"
    if "escalation" in text:
        return "escalation"
    if "wrong" in text or "conflict" in text or "superseded" in text or "decoy" in text:
        return "conflict"
    token_list = [token for token in tokens(text) if len(token) > 3]
    if not token_list:
        return "misc"
    return f"bucket_{fnv1a64(token_list[0]) % 16:02d}"


def route_key(card: dict[str, Any]) -> str:
    text = str(card.get("text") or "").lower()
    mask = activation_mask_for_text(text)
    return f"a{mask % 32:02d}"


def promotion_score(card: dict[str, Any], bias: float, scale: float) -> float:
    use_count = max(0.0, float(card.get("use_count") or 0.0))
    evidence_count = max(0.0, float(card.get("evidence_count") or 0.0))
    age_days = max(0.0, float(card.get("age_days") or 0.0))
    outcome = str(card.get("last_outcome") or "").lower()
    outcome_value = {"helpful": 0.22, "corrected": 0.07, "ignored": -0.20}.get(outcome, 0.0)
    protected = 0.12 if bool(card.get("protected")) else 0.0
    stale_penalty = 0.18 if age_days >= 180 and use_count <= 0 and not protected else 0.0
    raw = (
        0.20
        + 0.30 * float(card.get("importance") or 0.5)
        + 0.18 * clamp(math.log1p(use_count) / math.log1p(100.0), 0.0, 1.0)
        + 0.16 * clamp(math.log1p(evidence_count) / math.log1p(50.0), 0.0, 1.0)
        + 0.09 * clamp(1.0 - age_days / 365.0, 0.0, 1.0)
        + outcome_value
        + protected
        - stale_penalty
        + bias
    )
    return clamp(raw * scale, 0.0, 1.0)


def promotion_probability(card: dict[str, Any], level: int, bias: float, scale: float) -> float:
    level_multiplier = {LEVEL_ROOT: 0.12, LEVEL_PROJECT: 0.28, LEVEL_TOPIC: 0.58}.get(level, 0.0)
    return clamp(promotion_score(card, bias, scale) * level_multiplier, 0.0, 0.95)


def should_promote(card: dict[str, Any], level: int, bias: float, scale: float) -> bool:
    node_id = str(card.get("id") or "")
    return stable_unit(f"{node_id}:promote:{level}") <= promotion_probability(card, level, bias, scale)


def memory_priority(card: dict[str, Any], bias: float, scale: float) -> float:
    return (
        promotion_score(card, bias, scale)
        + 0.06 * float(card.get("importance") or 0.5)
        + 0.02 * stable_unit(str(card.get("id") or ""))
    )


def sort_ids_by_priority(ids: list[str], cards: dict[str, dict[str, Any]], bias: float, scale: float) -> list[str]:
    return sorted(ids, key=lambda node_id: (-memory_priority(cards[node_id], bias, scale), node_id))


def basin_text(kind: str, key: str, count: int) -> str:
    return f"{kind} memory basin {key} containing {count} memories"


def make_basin_node(node_id: str, level: int, kind: str, key: str, children: list[str], embeddings: list[list[float]]) -> dict[str, Any]:
    text = basin_text(kind, key, len(children))
    return {
        "id": node_id,
        "node_type": "basin",
        "level": level,
        "kind": kind,
        "key": key,
        "text": text,
        "summary": "",
        "embedding": mean_embedding(embeddings),
        "importance": 0.5,
        "cluster": key.split(":")[0],
        "metadata": {"project": key.split(":")[0]},
        "children": children,
        "promoted_children": [],
        "child_count": len(children),
    }


def encode_node_record(node: dict[str, Any]) -> bytes:
    raw = json.dumps(node, separators=(",", ":"), sort_keys=True).encode("utf-8")
    payload = zlib.compress(raw, level=3)
    return NODE_HEADER.pack(len(payload), len(raw)) + payload


def decode_node_record(frame: bytes, raw_size: int) -> dict[str, Any]:
    raw = zlib.decompress(frame)
    if len(raw) != raw_size:
        raise ValueError(f"corrupt node frame: expected {raw_size} raw bytes, got {len(raw)}")
    return json.loads(raw.decode("utf-8"))


def write_lazy_index(row: dict[str, Any], backend: CachedEmbeddingBackend, output_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    row = ensure_backend_embeddings(row, backend)
    output_dir.mkdir(parents=True, exist_ok=True)
    nodes_path = output_dir / "nodes.hgb"
    meta_path = output_dir / "index.json"

    memory_cards = {str(card["id"]): dict(card) for card in row.get("candidates", [])}
    for card in memory_cards.values():
        card["node_type"] = "memory"
        card["level"] = LEVEL_MEMORY
        card["children"] = []
        card["promoted_children"] = []
        card["edges"] = []

    for edge in (row.get("memory_graph") or {}).get("edges") or []:
        source = str(edge.get("source") or "")
        target = str(edge.get("target") or "")
        if source in memory_cards and target in memory_cards:
            memory_cards[source]["edges"].append(edge)

    project_groups: dict[str, list[str]] = {}
    topic_groups: dict[tuple[str, str], list[str]] = {}
    route_groups: dict[tuple[str, str, str], list[str]] = {}
    for node_id, card in memory_cards.items():
        project = project_key(card)
        topic = topic_key(card)
        route = route_key(card)
        project_groups.setdefault(project, []).append(node_id)
        topic_groups.setdefault((project, topic), []).append(node_id)
        route_groups.setdefault((project, topic, route), []).append(node_id)

    level2_nodes: dict[tuple[str, str, str], dict[str, Any]] = {}
    for key, ids in route_groups.items():
        ids = sort_ids_by_priority(ids, memory_cards, args.promotion_bias, args.promotion_scale)
        project, topic, route = key
        level2_nodes[key] = make_basin_node(
            f"basin:l2:{project}:{topic}:{route}",
            LEVEL_TOPIC,
            "route",
            f"{project}:{topic}:{route}",
            ids,
            [memory_cards[node_id]["embedding"] for node_id in ids],
        )

    level1_nodes: dict[tuple[str, str], dict[str, Any]] = {}
    for key, ids in topic_groups.items():
        project, topic = key
        child_keys = sorted(route_key(memory_cards[node_id]) for node_id in ids)
        child_ids = [f"basin:l2:{project}:{topic}:{route}" for route in OrderedDict.fromkeys(child_keys)]
        child_embeddings = [level2_nodes[(project, topic, route)]["embedding"] for route in OrderedDict.fromkeys(child_keys)]
        level1_nodes[key] = make_basin_node(
            f"basin:l1:{project}:{topic}",
            LEVEL_PROJECT,
            "topic",
            f"{project}:{topic}",
            child_ids,
            child_embeddings,
        )

    level0_nodes: dict[str, dict[str, Any]] = {}
    for project, ids in project_groups.items():
        topics = sorted(topic_key(memory_cards[node_id]) for node_id in ids)
        child_ids = [f"basin:l1:{project}:{topic}" for topic in OrderedDict.fromkeys(topics)]
        child_embeddings = [level1_nodes[(project, topic)]["embedding"] for topic in OrderedDict.fromkeys(topics)]
        level0_nodes[project] = make_basin_node(
            f"basin:l0:{project}",
            LEVEL_ROOT,
            "project",
            project,
            child_ids,
            child_embeddings,
        )

    promoted_count = 0
    for node_id, card in memory_cards.items():
        project = project_key(card)
        topic = topic_key(card)
        route = route_key(card)
        targets = [
            (LEVEL_ROOT, level0_nodes[project]),
            (LEVEL_PROJECT, level1_nodes[(project, topic)]),
            (LEVEL_TOPIC, level2_nodes[(project, topic, route)]),
        ]
        for level, basin in targets:
            if should_promote(card, level, args.promotion_bias, args.promotion_scale):
                basin["promoted_children"].append(node_id)
                promoted_count += 1

    all_nodes: list[dict[str, Any]] = []
    all_nodes.extend(level0_nodes.values())
    all_nodes.extend(level1_nodes.values())
    all_nodes.extend(level2_nodes.values())
    all_nodes.extend(memory_cards.values())

    offsets: dict[str, int] = {}
    levels: dict[str, list[str]] = {str(level): [] for level in range(LEVEL_MEMORY + 1)}
    with nodes_path.open("wb") as handle:
        handle.write(NODE_STORE_MAGIC)
        for node in all_nodes:
            node["promoted_children"] = list(OrderedDict.fromkeys(node.get("promoted_children") or []))
            offsets[node["id"]] = handle.tell()
            levels[str(int(node["level"]))].append(node["id"])
            handle.write(encode_node_record(node))

    meta = {
        "version": 1,
        "nodes_path": str(nodes_path),
        "levels": levels,
        "top_ids": levels[str(LEVEL_ROOT)],
        "offsets": offsets,
        "memory_count": len(memory_cards),
        "basin_count": len(all_nodes) - len(memory_cards),
        "promoted_count": promoted_count,
        "promotion_bias": args.promotion_bias,
        "promotion_scale": args.promotion_scale,
    }
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return meta


class LazyNodeStore:
    def __init__(self, meta: dict[str, Any], cache_size: int):
        self.meta = meta
        self.path = Path(meta["nodes_path"])
        self.cache_size = max(0, cache_size)
        self.cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self.handle = self.path.open("rb")
        magic = self.handle.read(len(NODE_STORE_MAGIC))
        if magic != NODE_STORE_MAGIC:
            raise ValueError(f"unsupported node store format: {self.path}")
        self.reads = 0
        self.cache_hits = 0
        self.unique_reads: set[str] = set()

    def close(self) -> None:
        self.handle.close()

    def read(self, node_id: str) -> dict[str, Any]:
        if node_id in self.cache:
            self.cache_hits += 1
            node = self.cache.pop(node_id)
            self.cache[node_id] = node
            return node
        offset = int(self.meta["offsets"][node_id])
        self.handle.seek(offset)
        header = self.handle.read(NODE_HEADER.size)
        if len(header) != NODE_HEADER.size:
            raise ValueError(f"short node header for {node_id}")
        payload_size, raw_size = NODE_HEADER.unpack(header)
        frame = self.handle.read(payload_size)
        if len(frame) != payload_size:
            raise ValueError(f"short node payload for {node_id}")
        self.reads += 1
        self.unique_reads.add(node_id)
        node = decode_node_record(frame, raw_size)
        if self.cache_size > 0:
            self.cache[node_id] = node
            while len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
        return node


def score_node(row: dict[str, Any], query_text: str, query_embedding: list[float], query_mask: int, node: dict[str, Any]) -> float:
    anchor = row["anchor"]
    semantic = cosine(query_embedding, node.get("embedding") or [])
    lexical = jaccard(query_text, f"{node.get('text', '')} {node.get('summary', '')}")
    activation = activation_overlap(query_mask, activation_mask_for_text(node.get("text") or ""))
    same_cluster = 1.0 if str(anchor.get("cluster") or "").lower() == str(node.get("cluster") or "").lower() else 0.0
    child_penalty = 0.012 * math.log1p(float(node.get("child_count") or len(node.get("children") or [])))
    if node.get("node_type") == "memory":
        state = state_score(node)
        promotion = 0.06 * float(node.get("importance") or 0.5)
    else:
        state = 0.0
        promotion = 0.025 * min(1.0, len(node.get("promoted_children") or []) / 12.0)
    return (
        0.36 * semantic
        + 0.22 * lexical
        + 0.18 * activation
        + 0.11 * same_cluster
        + 0.07 * float(node.get("importance") or 0.5)
        + state
        + promotion
        - child_penalty
    )


def unique_limited(ids: list[str], limit: int) -> list[str]:
    out = []
    seen = set()
    for node_id in ids:
        if node_id in seen:
            continue
        seen.add(node_id)
        out.append(node_id)
        if len(out) >= limit:
            break
    return out


def ranked_signature(ranked: Ranked) -> list[tuple[str, float]]:
    return [(node_id, round(float(score), 12)) for node_id, score, _ in ranked]


def hierarchical_search(row: dict[str, Any], backend: CachedEmbeddingBackend, meta: dict[str, Any], args: argparse.Namespace) -> tuple[Ranked, dict[str, float]]:
    task = row.get("retrieval_task") or {}
    query_text = task.get("query") or row["anchor"]["text"]
    query_embedding = backend.embed_one(query_text)
    query_mask = activation_mask_for_text(query_text)
    started = time.perf_counter()
    store = LazyNodeStore(meta, args.cache_size)
    try:
        frontier = list(meta.get("top_ids") or [])
        leaf_scores: dict[str, tuple[float, str]] = {}
        basin_nodes_scored = 0
        promoted_scored = 0
        edge_expansions = 0
        visited_ids: set[str] = set()
        for level in [LEVEL_ROOT, LEVEL_PROJECT, LEVEL_TOPIC]:
            scored: list[tuple[str, float, dict[str, Any]]] = []
            for node_id in frontier:
                if node_id in visited_ids:
                    continue
                visited_ids.add(node_id)
                node = store.read(node_id)
                basin_nodes_scored += 1
                scored.append((node_id, score_node(row, query_text, query_embedding, query_mask, node), node))
                promoted_ids = unique_limited(node.get("promoted_children") or [], args.promoted_per_basin)
                for promoted_id in promoted_ids:
                    promoted = store.read(promoted_id)
                    promoted_scored += 1
                    score = score_node(row, query_text, query_embedding, query_mask, promoted) + 0.03 * (LEVEL_TOPIC - level + 1)
                    prior = leaf_scores.get(promoted_id)
                    if prior is None or score > prior[0]:
                        leaf_scores[promoted_id] = (score, promoted.get("text", ""))
            scored.sort(key=lambda item: (-item[1], item[0]))
            kept = scored[: args.beam_width]
            next_ids: list[str] = []
            for _, _, node in kept:
                child_ids = list(node.get("children") or [])
                if level == LEVEL_TOPIC:
                    child_ids = child_ids[: args.max_children_per_basin]
                next_ids.extend(child_ids)
            frontier = unique_limited(next_ids, args.max_frontier)

        for node_id in frontier[: args.max_leaf_reads]:
            node = store.read(node_id)
            score = score_node(row, query_text, query_embedding, query_mask, node)
            prior = leaf_scores.get(node_id)
            if prior is None or score > prior[0]:
                leaf_scores[node_id] = (score, node.get("text", ""))

        frontier_edges = [
            (node_id, score, [node_id])
            for node_id, (score, _) in sorted(leaf_scores.items(), key=lambda item: (-item[1][0], item[0]))[: args.graph_seed_count]
        ]
        for depth in range(args.graph_depth):
            next_frontier: list[tuple[str, float, list[str]]] = []
            for current_id, current_score, path in frontier_edges:
                source = store.read(current_id)
                for edge in source.get("edges") or []:
                    target_id = str(edge.get("target") or "")
                    if target_id in path or target_id not in meta["offsets"]:
                        continue
                    target = store.read(target_id)
                    hop_gain = (
                        float(edge.get("weight") or 0.0)
                        * edge_type_boost(str(edge.get("type") or "used_with"))
                        * (0.70 + 0.45 * activation_overlap(query_mask, edge_activation(edge)))
                        * (0.70 + 0.30 * float(edge.get("confidence") or 0.5))
                        / (1.25 + depth)
                    )
                    target_score = score_node(row, query_text, query_embedding, query_mask, target)
                    score = 0.58 * current_score + 0.42 * target_score + hop_gain
                    prior = leaf_scores.get(target_id)
                    if prior is None or score > prior[0]:
                        edge_expansions += 1
                        leaf_scores[target_id] = (score, target.get("text", ""))
                        next_frontier.append((target_id, score, path + [target_id]))
            frontier_edges = next_frontier
    finally:
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        reads = store.reads
        cache_hits = store.cache_hits
        unique_reads = len(store.unique_reads)
        store.close()

    ranked = sorted(
        [(node_id, score, text) for node_id, (score, text) in leaf_scores.items()],
        key=lambda item: (-item[1], item[0]),
    )
    stats = {
        "latency_ms": elapsed_ms,
        "file_reads": float(reads),
        "cache_hits": float(cache_hits),
        "unique_nodes_read": float(unique_reads),
        "basins_scored": float(basin_nodes_scored),
        "promoted_scored": float(promoted_scored),
        "edge_expansions": float(edge_expansions),
        "final_candidate_count": float(len(ranked)),
    }
    return ranked[: args.return_limit], stats


def flatten(prefix: str, values: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": float(value) for key, value in values.items()}


def evaluate_case(row: dict[str, Any], backend: CachedEmbeddingBackend, case_dir: Path, args: argparse.Namespace) -> dict[str, float]:
    row = ensure_backend_embeddings(row, backend)
    meta = write_lazy_index(row, backend, case_dir, args)
    hierarchical, io_stats = hierarchical_search(row, backend, meta, args)
    determinism_mismatches = 0
    repeat_latency_ms = 0.0
    if args.determinism_repeats > 1:
        expected = ranked_signature(hierarchical)
        for _ in range(args.determinism_repeats - 1):
            repeated, repeated_stats = hierarchical_search(row, backend, meta, args)
            repeat_latency_ms += repeated_stats["latency_ms"]
            if ranked_signature(repeated) != expected:
                determinism_mismatches += 1
    sparse = sparse_basin_scores(row, backend)[: args.return_limit]
    vector = vector_scores_with_backend(row, backend)[: args.return_limit]
    flat: dict[str, float] = {
        "memory_count": float(meta["memory_count"]),
        "basin_count": float(meta["basin_count"]),
        "promoted_count": float(meta["promoted_count"]),
        "promotion_rate": float(meta["promoted_count"]) / max(1.0, float(meta["memory_count"])),
        "deterministic": 1.0 if determinism_mismatches == 0 else 0.0,
        "determinism_mismatches": float(determinism_mismatches),
        "determinism_repeats": float(args.determinism_repeats),
        "determinism_repeat_latency_ms": repeat_latency_ms / max(1, args.determinism_repeats - 1),
    }
    flat.update(flatten("hierarchical_retrieval", evaluate_ranked(row, hierarchical, args.top_k, args.budget)))
    flat.update(flatten("hierarchical_multihop", multihop_metrics(row, hierarchical, args.top_k, args.budget)))
    flat.update(flatten("hierarchical_context_exposure", context_decoy_metrics(row, hierarchical, args.budget)))
    flat.update(flatten("hierarchical_io", io_stats))
    flat.update(flatten("sparse_retrieval", evaluate_ranked(row, sparse, args.top_k, args.budget)))
    flat.update(flatten("sparse_multihop", multihop_metrics(row, sparse, args.top_k, args.budget)))
    flat.update(flatten("vector_retrieval", evaluate_ranked(row, vector, args.top_k, args.budget)))
    flat.update(flatten("vector_multihop", multihop_metrics(row, vector, args.top_k, args.budget)))
    return flat


def run(args: argparse.Namespace) -> dict[str, Any]:
    backend = build_embedding_backend(args)
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    rows = []
    for offset in range(args.cases):
        seed = args.seed + offset
        row = build_large_pool_case(seed, args.pool_size)
        case_dir = work_dir / f"case_{seed}"
        rows.append(evaluate_case(row, backend, case_dir, args))
    keys = sorted(rows[0]) if rows else []
    averages = {key: sum(row[key] for row in rows) / max(1, len(rows)) for key in keys}
    return {
        "benchmark": "hierarchical_file_ann",
        "embedding_backend": backend.name,
        "hippo_checkpoint": args.hippo_checkpoint if backend.name == "hippo" else "",
        "cases": args.cases,
        "pool_size": args.pool_size,
        "seed": args.seed,
        "beam_width": args.beam_width,
        "max_children_per_basin": args.max_children_per_basin,
        "max_leaf_reads": args.max_leaf_reads,
        "promotion_bias": args.promotion_bias,
        "promotion_scale": args.promotion_scale,
        "determinism_repeats": args.determinism_repeats,
        "elapsed_seconds": round(time.time() - started, 2),
        "averages": averages,
        "rows": rows if args.include_rows else [],
    }


def write_markdown(result: dict[str, Any], path: Path) -> None:
    avg = result["averages"]
    lines = [
        "# Hierarchical File ANN Benchmark",
        "",
        f"- backend: `{result['embedding_backend']}`",
        f"- cases: `{result['cases']}`",
        f"- pool_size: `{result['pool_size']}`",
        f"- beam_width: `{result['beam_width']}`",
        f"- max_children_per_basin: `{result['max_children_per_basin']}`",
        f"- promotion_bias: `{result['promotion_bias']}`",
        f"- promotion_scale: `{result['promotion_scale']}`",
        f"- determinism_repeats: `{result['determinism_repeats']}`",
        f"- elapsed_seconds: `{result['elapsed_seconds']}`",
        "",
        "| method | precision | recall | target ctx | path ctx | noise |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| hierarchical | {avg.get('hierarchical_retrieval_context_precision', 0.0):.4f} | "
            f"{avg.get('hierarchical_retrieval_context_recall', 0.0):.4f} | "
            f"{avg.get('hierarchical_multihop_target_context_recall', 0.0):.4f} | "
            f"{avg.get('hierarchical_multihop_path_context_success', 0.0):.4f} | "
            f"{avg.get('hierarchical_retrieval_noise', 0.0):.2f} |"
        ),
        (
            f"| sparse topN | {avg.get('sparse_retrieval_context_precision', 0.0):.4f} | "
            f"{avg.get('sparse_retrieval_context_recall', 0.0):.4f} | "
            f"{avg.get('sparse_multihop_target_context_recall', 0.0):.4f} | "
            f"{avg.get('sparse_multihop_path_context_success', 0.0):.4f} | "
            f"{avg.get('sparse_retrieval_noise', 0.0):.2f} |"
        ),
        (
            f"| vector topN | {avg.get('vector_retrieval_context_precision', 0.0):.4f} | "
            f"{avg.get('vector_retrieval_context_recall', 0.0):.4f} | "
            f"{avg.get('vector_multihop_target_context_recall', 0.0):.4f} | "
            f"{avg.get('vector_multihop_path_context_success', 0.0):.4f} | "
            f"{avg.get('vector_retrieval_noise', 0.0):.2f} |"
        ),
        "",
        "| IO | value |",
        "| --- | ---: |",
        f"| file reads | {avg.get('hierarchical_io_file_reads', 0.0):.2f} |",
        f"| unique nodes read | {avg.get('hierarchical_io_unique_nodes_read', 0.0):.2f} |",
        f"| cache hits | {avg.get('hierarchical_io_cache_hits', 0.0):.2f} |",
        f"| basins scored | {avg.get('hierarchical_io_basins_scored', 0.0):.2f} |",
        f"| promoted leaves scored | {avg.get('hierarchical_io_promoted_scored', 0.0):.2f} |",
        f"| edge expansions | {avg.get('hierarchical_io_edge_expansions', 0.0):.2f} |",
        f"| final candidates | {avg.get('hierarchical_io_final_candidate_count', 0.0):.2f} |",
        f"| latency ms | {avg.get('hierarchical_io_latency_ms', 0.0):.2f} |",
        f"| promoted links | {avg.get('promoted_count', 0.0):.2f} |",
        f"| promotion rate | {avg.get('promotion_rate', 0.0):.4f} |",
        "",
        "| Determinism | value |",
        "| --- | ---: |",
        f"| deterministic cases | {avg.get('deterministic', 0.0):.4f} |",
        f"| repeat mismatches | {avg.get('determinism_mismatches', 0.0):.2f} |",
        f"| repeat latency ms | {avg.get('determinism_repeat_latency_ms', 0.0):.2f} |",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=int, default=20)
    parser.add_argument("--pool-size", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=12000)
    parser.add_argument("--work-dir", default="artifacts/hippocampus/hierarchical_file_ann")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--budget", type=int, default=90)
    parser.add_argument("--return-limit", type=int, default=128)
    parser.add_argument("--beam-width", type=int, default=4)
    parser.add_argument("--max-frontier", type=int, default=256)
    parser.add_argument("--max-children-per-basin", type=int, default=128)
    parser.add_argument("--max-leaf-reads", type=int, default=384)
    parser.add_argument("--promoted-per-basin", type=int, default=32)
    parser.add_argument("--graph-seed-count", type=int, default=8)
    parser.add_argument("--graph-depth", type=int, default=2)
    parser.add_argument("--cache-size", type=int, default=256)
    parser.add_argument("--promotion-bias", type=float, default=0.0)
    parser.add_argument("--promotion-scale", type=float, default=1.0)
    parser.add_argument("--determinism-repeats", type=int, default=1)
    parser.add_argument("--embedding-backend", choices=["hash", "hippo"], default="hash")
    parser.add_argument("--hippo-checkpoint", default="")
    parser.add_argument("--hippo-encoder-src", default="")
    parser.add_argument("--hippo-max-length", type=int, default=128)
    parser.add_argument("--hippo-batch-size", type=int, default=128)
    parser.add_argument("--device", default="")
    parser.add_argument("--include-rows", action="store_true")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-md", default="")
    args = parser.parse_args()
    result = run(args)
    body = json.dumps(result, indent=2)
    print(body)
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(body + "\n", encoding="utf-8")
    if args.output_md:
        write_markdown(result, Path(args.output_md))


if __name__ == "__main__":
    main()
