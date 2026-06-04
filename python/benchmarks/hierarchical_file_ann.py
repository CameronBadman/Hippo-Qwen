from __future__ import annotations

import argparse
import json
import math
import mmap
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
    embed_text,
    fnv1a64,
    jaccard,
    tokens,
)


Ranked = list[tuple[str, float, str]]
SearchResult = tuple[Ranked, dict[str, float], set[str]]
LEVEL_ROOT = 0
LEVEL_PROJECT = 1
LEVEL_TOPIC = 2
LEVEL_MEMORY = 3
NODE_STORE_MAGIC = b"HGFANN1\x00"
NODE_HEADER = struct.Struct("<II")
ROUTING_FLOAT = struct.Struct("<f")


def stable_unit(value: str) -> float:
    return (fnv1a64(value) % 10_000_000) / 10_000_000.0


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


def basin_text(kind: str, key: str) -> str:
    return f"{kind} memory basin {key}"


def activation_union(cards: list[dict[str, Any]]) -> int:
    mask = 0
    for card in cards:
        mask |= activation_mask_for_text(f"{card.get('text', '')} {card.get('summary', '')}")
    return mask


def centroid_routing_embedding(child_cards: list[dict[str, Any]], routing_dims: int) -> list[float]:
    dims = routing_dims
    if dims <= 0:
        dims = max((len(card.get("routing_embedding") or card.get("embedding") or []) for card in child_cards), default=0)
    if dims <= 0:
        return []
    total = [0.0] * dims
    count = 0
    for card in child_cards:
        embedding = card.get("routing_embedding") or card.get("embedding") or []
        if not embedding:
            continue
        local_dims = min(dims, len(embedding))
        for index in range(local_dims):
            total[index] += float(embedding[index])
        count += 1
    if count <= 0:
        return []
    return [value / count for value in total]


def make_basin_node(
    node_id: str,
    level: int,
    kind: str,
    key: str,
    children: list[str],
    child_cards: list[dict[str, Any]],
    routing_dims: int,
) -> dict[str, Any]:
    text = basin_text(kind, key)
    return {
        "id": node_id,
        "node_type": "basin",
        "level": level,
        "kind": kind,
        "key": key,
        "text": text,
        "summary": "",
        "embedding": embed_text(text),
        "routing_embedding": centroid_routing_embedding(child_cards, routing_dims),
        "routing_mask": activation_union(child_cards),
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


def activation_coverage(query_mask: int, candidate_mask: int) -> float:
    if not query_mask or not candidate_mask:
        return 0.0
    return (query_mask & candidate_mask).bit_count() / max(1, query_mask.bit_count())


def cosine_prefix(a: list[float], b: list[float], max_dims: int) -> float:
    if not a or not b:
        return 0.0
    dims = min(len(a), len(b))
    if max_dims > 0:
        dims = min(dims, max_dims)
    if dims <= 0:
        return 0.0
    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for index in range(dims):
        left = float(a[index])
        right = float(b[index])
        dot += left * right
        left_norm += left * left
        right_norm += right * right
    if left_norm <= 0.0 or right_norm <= 0.0:
        return 0.0
    return dot / math.sqrt(left_norm * right_norm)


def write_lazy_index(row: dict[str, Any], backend: CachedEmbeddingBackend, output_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    row = ensure_backend_embeddings(row, backend)
    output_dir.mkdir(parents=True, exist_ok=True)
    nodes_path = output_dir / "nodes.hgb"
    routing_path = output_dir / "routing.f32"
    meta_path = output_dir / "index.json"

    memory_cards = {str(card["id"]): dict(card) for card in row.get("candidates", [])}
    for node_id in sorted(memory_cards):
        card = memory_cards[node_id]
        embedding = [float(value) for value in card.get("embedding") or []]
        routing_dims = int(args.semantic_dims) if int(args.semantic_dims) > 0 else len(embedding)
        card["embedding"] = embedding
        card["routing_embedding"] = embedding[:routing_dims]
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
    routing_dims = int(args.semantic_dims)
    for key in sorted(route_groups):
        ids = route_groups[key]
        ids = sort_ids_by_priority(ids, memory_cards, args.promotion_bias, args.promotion_scale)
        project, topic, route = key
        level2_nodes[key] = make_basin_node(
            f"basin:l2:{project}:{topic}:{route}",
            LEVEL_TOPIC,
            "route",
            f"{project}:{topic}:{route}",
            ids,
            [memory_cards[node_id] for node_id in ids],
            routing_dims,
        )

    level1_nodes: dict[tuple[str, str], dict[str, Any]] = {}
    for key in sorted(topic_groups):
        ids = topic_groups[key]
        project, topic = key
        child_keys = sorted(route_key(memory_cards[node_id]) for node_id in ids)
        child_ids = [f"basin:l2:{project}:{topic}:{route}" for route in OrderedDict.fromkeys(child_keys)]
        child_cards = [level2_nodes[(project, topic, route)] for route in OrderedDict.fromkeys(child_keys)]
        level1_nodes[key] = make_basin_node(
            f"basin:l1:{project}:{topic}",
            LEVEL_PROJECT,
            "topic",
            f"{project}:{topic}",
            child_ids,
            child_cards,
            routing_dims,
        )

    level0_nodes: dict[str, dict[str, Any]] = {}
    for project in sorted(project_groups):
        ids = project_groups[project]
        topics = sorted(topic_key(memory_cards[node_id]) for node_id in ids)
        child_ids = [f"basin:l1:{project}:{topic}" for topic in OrderedDict.fromkeys(topics)]
        child_cards = [level1_nodes[(project, topic)] for topic in OrderedDict.fromkeys(topics)]
        level0_nodes[project] = make_basin_node(
            f"basin:l0:{project}",
            LEVEL_ROOT,
            "project",
            project,
            child_ids,
            child_cards,
            routing_dims,
        )

    promoted_count = 0
    for node_id in sorted(memory_cards):
        card = memory_cards[node_id]
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
    all_nodes.extend(level0_nodes[key] for key in sorted(level0_nodes))
    all_nodes.extend(level1_nodes[key] for key in sorted(level1_nodes))
    all_nodes.extend(level2_nodes[key] for key in sorted(level2_nodes))
    all_nodes.extend(memory_cards[key] for key in sorted(memory_cards))

    with routing_path.open("wb") as routing_handle:
        for node in all_nodes:
            routing = [float(value) for value in node.get("routing_embedding") or []]
            node["routing_embedding_offset"] = routing_handle.tell()
            node["routing_embedding_dims"] = len(routing)
            for value in routing:
                routing_handle.write(ROUTING_FLOAT.pack(value))
            node.pop("routing_embedding", None)

    if not args.store_full_embeddings:
        for node in all_nodes:
            node["embedding"] = []

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
        "routing_path": str(routing_path),
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
    meta["build_latency_ms"] = (time.perf_counter() - started) * 1000.0
    meta["node_store_bytes"] = nodes_path.stat().st_size
    meta["routing_store_bytes"] = routing_path.stat().st_size
    meta["index_bytes"] = meta_path.stat().st_size
    return meta


class LazyNodeStore:
    def __init__(self, meta: dict[str, Any], cache_size: int):
        self.meta = meta
        self.path = Path(meta["nodes_path"])
        self.routing_path = Path(meta["routing_path"]) if meta.get("routing_path") else None
        self.cache_size = max(0, cache_size)
        self.cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self.handle = self.path.open("rb")
        self.data = mmap.mmap(self.handle.fileno(), 0, access=mmap.ACCESS_READ)
        magic = self.data[: len(NODE_STORE_MAGIC)]
        if magic != NODE_STORE_MAGIC:
            raise ValueError(f"unsupported node store format: {self.path}")
        self.routing_handle = self.routing_path.open("rb") if self.routing_path and self.routing_path.exists() else None
        self.routing_data = (
            mmap.mmap(self.routing_handle.fileno(), 0, access=mmap.ACCESS_READ)
            if self.routing_handle and self.routing_path and self.routing_path.stat().st_size > 0
            else None
        )
        self.reads = 0
        self.cache_hits = 0
        self.unique_reads: set[str] = set()

    def close(self) -> None:
        if self.routing_data is not None:
            self.routing_data.close()
        if self.routing_handle is not None:
            self.routing_handle.close()
        self.data.close()
        self.handle.close()

    def _attach_routing_embedding(self, node: dict[str, Any]) -> dict[str, Any]:
        if self.routing_data is None or node.get("routing_embedding") is not None:
            return node
        dims = int(node.get("routing_embedding_dims") or 0)
        offset = int(node.get("routing_embedding_offset") or 0)
        if dims <= 0:
            return node
        size = dims * ROUTING_FLOAT.size
        frame = self.routing_data[offset : offset + size]
        if len(frame) != size:
            raise ValueError(f"short routing vector for {node.get('id')}")
        node["routing_embedding"] = list(struct.unpack(f"<{dims}f", frame))
        return node

    def read(self, node_id: str) -> dict[str, Any]:
        if node_id in self.cache:
            self.cache_hits += 1
            node = self.cache.pop(node_id)
            self.cache[node_id] = node
            return node
        offset = int(self.meta["offsets"][node_id])
        header = self.data[offset : offset + NODE_HEADER.size]
        if len(header) != NODE_HEADER.size:
            raise ValueError(f"short node header for {node_id}")
        payload_size, raw_size = NODE_HEADER.unpack(header)
        frame_start = offset + NODE_HEADER.size
        frame = self.data[frame_start : frame_start + payload_size]
        if len(frame) != payload_size:
            raise ValueError(f"short node payload for {node_id}")
        self.reads += 1
        self.unique_reads.add(node_id)
        node = self._attach_routing_embedding(decode_node_record(frame, raw_size))
        if self.cache_size > 0:
            self.cache[node_id] = node
            while len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
        return node


def score_node(row: dict[str, Any], query_text: str, query_embedding: list[float], query_mask: int, node: dict[str, Any], semantic_dims: int) -> float:
    anchor = row["anchor"]
    lexical = jaccard(query_text, f"{node.get('text', '')} {node.get('summary', '')}")
    same_cluster = 1.0 if str(anchor.get("cluster") or "").lower() == str(node.get("cluster") or "").lower() else 0.0
    if node.get("node_type") == "memory":
        semantic = cosine_prefix(query_embedding, node.get("routing_embedding") or node.get("embedding") or [], semantic_dims)
        activation = activation_overlap(query_mask, activation_mask_for_text(node.get("text") or ""))
        child_penalty = 0.0
        state = state_score(node)
        promotion = 0.06 * float(node.get("importance") or 0.5)
    else:
        semantic = cosine_prefix(query_embedding, node.get("routing_embedding") or node.get("embedding") or [], semantic_dims)
        activation = activation_coverage(query_mask, int(node.get("routing_mask") or 0))
        child_penalty = 0.0
        state = 0.0
        promotion = 0.0
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


def unique_ids(ids: list[str], limit: int = 0) -> list[str]:
    out = []
    seen = set()
    for node_id in ids:
        if node_id in seen:
            continue
        seen.add(node_id)
        out.append(node_id)
        if limit > 0 and len(out) >= limit:
            break
    return out


def keep_basin_candidates(scored: list[tuple[str, float, dict[str, Any]]], args: argparse.Namespace) -> list[tuple[str, float, dict[str, Any]]]:
    scored.sort(key=lambda item: (-item[1], item[0]))
    if not args.stable_growth:
        return scored[: args.beam_width]
    kept = [item for item in scored if item[1] >= args.stable_basin_floor]
    if not kept:
        return scored[: args.beam_width]
    if args.stable_max_basins > 0:
        return kept[: args.stable_max_basins]
    return kept


def stable_limited(ids: list[str], limit: int) -> list[str]:
    if limit <= 0:
        return ids
    return ids[:limit]


def ranked_signature(ranked: Ranked) -> list[tuple[str, float]]:
    return [(node_id, round(float(score), 12)) for node_id, score, _ in ranked]


def hierarchical_search(row: dict[str, Any], backend: CachedEmbeddingBackend, meta: dict[str, Any], args: argparse.Namespace) -> SearchResult:
    task = row.get("retrieval_task") or {}
    query_text = task.get("query") or row["anchor"]["text"]
    query_embedding = task.get("query_embedding") or backend.embed_one(query_text)
    query_embedding = [float(value) for value in query_embedding]
    query_mask = activation_mask_for_text(query_text)
    started = time.perf_counter()
    store = LazyNodeStore(meta, args.cache_size)
    try:
        frontier = list(meta.get("top_ids") or [])
        leaf_scores: dict[str, tuple[float, str]] = {}
        basin_nodes_scored = 0
        promoted_scored = 0
        edge_expansions = 0
        protected_ids: set[str] = set()
        edge_source_scores: dict[str, tuple[float, str]] = {}
        visited_ids: set[str] = set()
        stable_promoted_reads = 0
        for level in [LEVEL_ROOT, LEVEL_PROJECT, LEVEL_TOPIC]:
            scored: list[tuple[str, float, dict[str, Any]]] = []
            for node_id in frontier:
                if node_id in visited_ids:
                    continue
                visited_ids.add(node_id)
                node = store.read(node_id)
                basin_nodes_scored += 1
                scored.append((node_id, score_node(row, query_text, query_embedding, query_mask, node, args.semantic_dims), node))
                promoted_ids = unique_limited(node.get("promoted_children") or [], args.promoted_per_basin)
                for promoted_id in promoted_ids:
                    if args.stable_growth and args.stable_max_promoted_reads > 0 and stable_promoted_reads >= args.stable_max_promoted_reads:
                        break
                    promoted = store.read(promoted_id)
                    stable_promoted_reads += 1
                    promoted_scored += 1
                    score = score_node(row, query_text, query_embedding, query_mask, promoted, args.semantic_dims) + 0.03 * (LEVEL_TOPIC - level + 1)
                    prior = leaf_scores.get(promoted_id)
                    if prior is None or score > prior[0]:
                        leaf_scores[promoted_id] = (score, promoted.get("text", ""))
                    if promoted.get("edges"):
                        prior_source = edge_source_scores.get(promoted_id)
                        if prior_source is None or score > prior_source[0]:
                            edge_source_scores[promoted_id] = (score, promoted.get("text", ""))
            kept = keep_basin_candidates(scored, args)
            next_ids: list[str] = []
            for _, _, node in kept:
                child_ids = list(node.get("children") or [])
                if level == LEVEL_TOPIC and not args.stable_growth:
                    child_ids = child_ids[: args.max_children_per_basin]
                next_ids.extend(child_ids)
            frontier = unique_ids(next_ids, 0 if args.stable_growth else args.max_frontier)

        leaf_frontier = stable_limited(frontier, args.stable_max_leaf_reads) if args.stable_growth else frontier[: args.max_leaf_reads]
        for node_id in leaf_frontier:
            node = store.read(node_id)
            score = score_node(row, query_text, query_embedding, query_mask, node, args.semantic_dims)
            prior = leaf_scores.get(node_id)
            if prior is None or score > prior[0]:
                leaf_scores[node_id] = (score, node.get("text", ""))
            if node.get("edges"):
                prior_source = edge_source_scores.get(node_id)
                if prior_source is None or score > prior_source[0]:
                    edge_source_scores[node_id] = (score, node.get("text", ""))

        edge_seed_items = edge_source_scores if args.stable_growth else leaf_scores
        edge_seed_limit = 0 if args.stable_growth else args.graph_seed_count
        frontier_edges = []
        for node_id, (score, _) in sorted(edge_seed_items.items(), key=lambda item: (-item[1][0], item[0])):
            frontier_edges.append((node_id, score, [node_id]))
            if edge_seed_limit > 0 and len(frontier_edges) >= edge_seed_limit:
                break
        for depth in range(args.graph_depth):
            next_frontier: list[tuple[str, float, list[str]]] = []
            for current_id, current_score, path in frontier_edges:
                protected_ids.add(current_id)
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
                    target_score = score_node(row, query_text, query_embedding, query_mask, target, args.semantic_dims)
                    score = 0.58 * current_score + 0.42 * target_score + hop_gain
                    prior = leaf_scores.get(target_id)
                    if prior is None or score > prior[0]:
                        edge_expansions += 1
                        protected_ids.add(target_id)
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
        "stable_promoted_reads": float(stable_promoted_reads),
    }
    if args.return_limit > 0:
        ranked = ranked[: args.return_limit]
    return ranked, stats, protected_ids


def flatten(prefix: str, values: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": float(value) for key, value in values.items()}


def make_growth_noise(row: dict[str, Any], backend: CachedEmbeddingBackend, seed: int, count: int) -> list[dict[str, Any]]:
    cards = []
    texts = []
    project = project_key(row.get("anchor") or {})
    query = str((row.get("retrieval_task") or {}).get("query") or row.get("anchor", {}).get("text") or "")
    for index in range(count):
        text = (
            f"{project}: growth-noise-{seed}-{index} mentions current lookup terms: {query}. "
            f"This is an unrelated maintenance decoy with random ticket {900000 + seed * 17 + index}."
        )
        texts.append(text)
        cards.append(
            {
                "id": f"growth_noise_{seed}_{index}",
                "text": text,
                "summary": "",
                "embedding": [],
                "importance": 0.25,
                "cluster": project,
                "metadata": {"project": project},
                "age_days": 1,
                "use_count": 0,
                "evidence_count": 0,
                "last_outcome": "",
                "protected": False,
                "synthetic_role": "growth_noise",
            }
        )
    if cards:
        for card, embedding in zip(cards, backend.embed_many(texts)):
            card["embedding"] = embedding
    return cards


def maybe_limit_ranked(ranked: Ranked, limit: int) -> Ranked:
    if limit > 0:
        return ranked[:limit]
    return ranked


def compact_score(row: dict[str, Any], item: tuple[str, float, str], protected_ids: set[str]) -> float:
    node_id, retrieval_score, text = item
    task = row.get("retrieval_task") or {}
    query = task.get("query") or row["anchor"]["text"]
    query_mask = activation_mask_for_text(query)
    text_mask = activation_mask_for_text(text)
    lexical = jaccard(query, text)
    activation = activation_overlap(query_mask, text_mask)
    protected = 1.0 if node_id in protected_ids else 0.0
    lower = text.lower()
    conflict_penalty = 0.18 if any(term in lower for term in ("decoy", "wrong", "conflict", "superseded", "growth-noise")) else 0.0
    path_bonus = 0.20 if any(term in lower for term in ("accepted resolution", "support note", "succeeded", "escalation")) else 0.0
    return (
        0.42 * retrieval_score
        + 0.24 * lexical
        + 0.18 * activation
        + 0.28 * protected
        + path_bonus
        - conflict_penalty
    )


def compact_ranked(row: dict[str, Any], ranked: Ranked, protected_ids: set[str], args: argparse.Namespace) -> tuple[Ranked, dict[str, float]]:
    started = time.perf_counter()
    if args.compact_limit <= 0:
        return ranked, {"latency_ms": 0.0, "output_count": float(len(ranked)), "protected_count": float(len(protected_ids))}
    protected_ranked = [item for item in ranked if item[0] in protected_ids]
    protected_ranked.sort(key=lambda item: (-item[1], item[0]))
    if len(protected_ranked) >= args.compact_limit:
        output = protected_ranked[: args.compact_limit]
        return output, {
            "latency_ms": (time.perf_counter() - started) * 1000.0,
            "output_count": float(len(output)),
            "protected_count": float(len(output)),
            "input_count": float(len(ranked)),
        }
    scored = []
    for item in ranked:
        candidate_id, _, text = item
        if candidate_id in protected_ids:
            continue
        scored.append((candidate_id, compact_score(row, item, protected_ids), text))
    normal = scored
    selected = []
    seen = set()
    for candidate_id, retrieval_score, text in protected_ranked:
        selected.append((candidate_id, retrieval_score, text))
        seen.add(candidate_id)
    normal.sort(key=lambda item: (-item[1], item[0]))
    for item in normal:
        if item[0] in seen:
            continue
        seen.add(item[0])
        selected.append(item)
        if len(selected) >= args.compact_limit:
            break
    output = [(candidate_id, score, text) for candidate_id, score, text in selected]
    return output, {
        "latency_ms": (time.perf_counter() - started) * 1000.0,
        "output_count": float(len(output)),
        "protected_count": float(sum(1 for candidate_id, _, _ in output if candidate_id in protected_ids)),
        "input_count": float(len(ranked)),
    }


def growth_stability_metrics(base: Ranked, grown: Ranked, row: dict[str, Any], top_n: int) -> dict[str, float]:
    relevant = set((row.get("retrieval_task") or {}).get("relevant_ids") or [])
    base_ids = [node_id for node_id, _, _ in base]
    grown_ids = [node_id for node_id, _, _ in grown]
    base_set = set(base_ids)
    grown_set = set(grown_ids)
    base_relevant = relevant & base_set
    retained_relevant = base_relevant & grown_set
    protected_base = set(base_ids[:top_n]) if top_n > 0 else base_set
    protected_retained = protected_base & grown_set
    return {
        "relevant_retention": 1.0 if not base_relevant else len(retained_relevant) / len(base_relevant),
        "relevant_dropped": float(len(base_relevant - grown_set)),
        "topn_retention": len(protected_retained) / max(1, len(protected_base)),
        "topn_dropped": float(len(protected_base - grown_set)),
        "candidate_growth": float(len(grown_ids) - len(base_ids)),
        "base_candidates": float(len(base_ids)),
        "grown_candidates": float(len(grown_ids)),
    }


def evaluate_case(row: dict[str, Any], backend: CachedEmbeddingBackend, case_dir: Path, args: argparse.Namespace) -> dict[str, float]:
    row = ensure_backend_embeddings(row, backend)
    meta = write_lazy_index(row, backend, case_dir, args)
    hierarchical, io_stats, protected_ids = hierarchical_search(row, backend, meta, args)
    compacted, compact_stats = compact_ranked(row, hierarchical, protected_ids, args)
    determinism_mismatches = 0
    repeat_latency_ms = 0.0
    if args.determinism_repeats > 1:
        expected = ranked_signature(hierarchical)
        for _ in range(args.determinism_repeats - 1):
            repeated, repeated_stats, _ = hierarchical_search(row, backend, meta, args)
            repeat_latency_ms += repeated_stats["latency_ms"]
            if ranked_signature(repeated) != expected:
                determinism_mismatches += 1
    sparse = maybe_limit_ranked(sparse_basin_scores(row, backend), args.return_limit)
    vector = maybe_limit_ranked(vector_scores_with_backend(row, backend), args.return_limit)
    flat: dict[str, float] = {
        "memory_count": float(meta["memory_count"]),
        "basin_count": float(meta["basin_count"]),
        "promoted_count": float(meta["promoted_count"]),
        "promotion_rate": float(meta["promoted_count"]) / max(1.0, float(meta["memory_count"])),
        "index_build_latency_ms": float(meta["build_latency_ms"]),
        "index_node_store_bytes": float(meta["node_store_bytes"]),
        "index_sidecar_bytes": float(meta["index_bytes"]),
        "index_bytes_per_memory": (float(meta["node_store_bytes"]) + float(meta["index_bytes"])) / max(1.0, float(meta["memory_count"])),
        "deterministic": 1.0 if determinism_mismatches == 0 else 0.0,
        "determinism_mismatches": float(determinism_mismatches),
        "determinism_repeats": float(args.determinism_repeats),
        "determinism_repeat_latency_ms": repeat_latency_ms / max(1, args.determinism_repeats - 1),
    }
    flat.update(flatten("hierarchical_retrieval", evaluate_ranked(row, hierarchical, args.top_k, args.budget)))
    flat.update(flatten("hierarchical_multihop", multihop_metrics(row, hierarchical, args.top_k, args.budget)))
    flat.update(flatten("hierarchical_context_exposure", context_decoy_metrics(row, hierarchical, args.budget)))
    flat.update(flatten("hierarchical_io", io_stats))
    flat.update(flatten("compacted_retrieval", evaluate_ranked(row, compacted, args.top_k, args.budget)))
    flat.update(flatten("compacted_multihop", multihop_metrics(row, compacted, args.top_k, args.budget)))
    flat.update(flatten("compacted_context_exposure", context_decoy_metrics(row, compacted, args.budget)))
    flat.update(flatten("compactor", compact_stats))
    if args.growth_noise_count > 0:
        grown_row = dict(row)
        grown_row["candidates"] = list(row.get("candidates") or []) + make_growth_noise(row, backend, args.seed, args.growth_noise_count)
        grown_meta = write_lazy_index(grown_row, backend, case_dir / "growth", args)
        grown, grown_io, _ = hierarchical_search(grown_row, backend, grown_meta, args)
        flat.update(flatten("growth_stability", growth_stability_metrics(hierarchical, grown, row, args.stability_top_n)))
        flat.update(flatten("growth_io", grown_io))
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
    quantile_keys = [
        "hierarchical_io_latency_ms",
        "hierarchical_io_file_reads",
        "hierarchical_io_final_candidate_count",
        "compactor_latency_ms",
        "compactor_output_count",
        "compacted_retrieval_context_precision",
        "compacted_retrieval_context_recall",
        "hierarchical_retrieval_context_precision",
        "hierarchical_retrieval_context_recall",
        "index_build_latency_ms",
        "index_bytes_per_memory",
        "growth_stability_relevant_retention",
        "growth_stability_topn_retention",
    ]
    return {
        "benchmark": "hierarchical_file_ann",
        "embedding_backend": backend.name,
        "hippo_checkpoint": args.hippo_checkpoint if backend.name == "hippo" else "",
        "cases": args.cases,
        "pool_size": args.pool_size,
        "seed": args.seed,
        "stable_growth": args.stable_growth,
        "stable_basin_floor": args.stable_basin_floor,
        "growth_noise_count": args.growth_noise_count,
        "beam_width": args.beam_width,
        "max_children_per_basin": args.max_children_per_basin,
        "max_leaf_reads": args.max_leaf_reads,
        "promotion_bias": args.promotion_bias,
        "promotion_scale": args.promotion_scale,
        "compact_limit": args.compact_limit,
        "determinism_repeats": args.determinism_repeats,
        "elapsed_seconds": round(time.time() - started, 2),
        "averages": averages,
        "quantiles": {key: quantiles([row[key] for row in rows if key in row]) for key in quantile_keys},
        "rows": rows if args.include_rows else [],
    }


def quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values)

    def pick(q: float) -> float:
        index = min(len(ordered) - 1, max(0, int(math.ceil(q * len(ordered)) - 1)))
        return float(ordered[index])

    return {
        "min": float(ordered[0]),
        "p50": pick(0.50),
        "p95": pick(0.95),
        "p99": pick(0.99),
        "max": float(ordered[-1]),
    }


def write_markdown(result: dict[str, Any], path: Path) -> None:
    avg = result["averages"]
    lines = [
        "# Hierarchical File ANN Benchmark",
        "",
        f"- backend: `{result['embedding_backend']}`",
        f"- cases: `{result['cases']}`",
        f"- pool_size: `{result['pool_size']}`",
        f"- stable_growth: `{result['stable_growth']}`",
        f"- stable_basin_floor: `{result['stable_basin_floor']}`",
        f"- growth_noise_count: `{result['growth_noise_count']}`",
        f"- beam_width: `{result['beam_width']}`",
        f"- max_children_per_basin: `{result['max_children_per_basin']}`",
        f"- promotion_bias: `{result['promotion_bias']}`",
        f"- promotion_scale: `{result['promotion_scale']}`",
        f"- compact_limit: `{result['compact_limit']}`",
        f"- determinism_repeats: `{result['determinism_repeats']}`",
        f"- elapsed_seconds: `{result['elapsed_seconds']}`",
        "",
        "| method | precision | recall | target ctx | path ctx | noise |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| stable raw | {avg.get('hierarchical_retrieval_context_precision', 0.0):.4f} | "
            f"{avg.get('hierarchical_retrieval_context_recall', 0.0):.4f} | "
            f"{avg.get('hierarchical_multihop_target_context_recall', 0.0):.4f} | "
            f"{avg.get('hierarchical_multihop_path_context_success', 0.0):.4f} | "
            f"{avg.get('hierarchical_retrieval_noise', 0.0):.2f} |"
        ),
        (
            f"| compacted | {avg.get('compacted_retrieval_context_precision', 0.0):.4f} | "
            f"{avg.get('compacted_retrieval_context_recall', 0.0):.4f} | "
            f"{avg.get('compacted_multihop_target_context_recall', 0.0):.4f} | "
            f"{avg.get('compacted_multihop_path_context_success', 0.0):.4f} | "
            f"{avg.get('compacted_retrieval_noise', 0.0):.2f} |"
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
        f"| compactor latency ms | {avg.get('compactor_latency_ms', 0.0):.2f} |",
        f"| compactor output count | {avg.get('compactor_output_count', 0.0):.2f} |",
        f"| compactor protected count | {avg.get('compactor_protected_count', 0.0):.2f} |",
        f"| promoted links | {avg.get('promoted_count', 0.0):.2f} |",
        f"| promotion rate | {avg.get('promotion_rate', 0.0):.4f} |",
        f"| build latency ms | {avg.get('index_build_latency_ms', 0.0):.2f} |",
        f"| bytes per memory | {avg.get('index_bytes_per_memory', 0.0):.2f} |",
        "",
        "| Latency Quantiles | min | p50 | p95 | p99 | max |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key in [
        "hierarchical_io_latency_ms",
        "hierarchical_io_file_reads",
        "hierarchical_io_final_candidate_count",
        "compactor_latency_ms",
        "compactor_output_count",
        "index_build_latency_ms",
    ]:
        q = result.get("quantiles", {}).get(key, {})
        lines.append(
            f"| {key} | {q.get('min', 0.0):.2f} | {q.get('p50', 0.0):.2f} | "
            f"{q.get('p95', 0.0):.2f} | {q.get('p99', 0.0):.2f} | {q.get('max', 0.0):.2f} |"
        )
    lines.extend(
        [
        "",
        "| Determinism | value |",
        "| --- | ---: |",
        f"| deterministic cases | {avg.get('deterministic', 0.0):.4f} |",
        f"| repeat mismatches | {avg.get('determinism_mismatches', 0.0):.2f} |",
        f"| repeat latency ms | {avg.get('determinism_repeat_latency_ms', 0.0):.2f} |",
        "",
        "| Growth Stability | value |",
        "| --- | ---: |",
        f"| relevant retention | {avg.get('growth_stability_relevant_retention', 0.0):.4f} |",
        f"| relevant dropped | {avg.get('growth_stability_relevant_dropped', 0.0):.2f} |",
        f"| topN retention | {avg.get('growth_stability_topn_retention', 0.0):.4f} |",
        f"| topN dropped | {avg.get('growth_stability_topn_dropped', 0.0):.2f} |",
        f"| candidate growth | {avg.get('growth_stability_candidate_growth', 0.0):.2f} |",
        f"| grown latency ms | {avg.get('growth_io_latency_ms', 0.0):.2f} |",
        "",
        ]
    )
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
    parser.add_argument("--determinism-repeats", type=int, default=1)
    parser.add_argument("--stable-growth", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stable-basin-floor", type=float, default=0.32)
    parser.add_argument("--stable-max-basins", type=int, default=0)
    parser.add_argument("--stable-max-leaf-reads", type=int, default=0)
    parser.add_argument("--stable-max-promoted-reads", type=int, default=0)
    parser.add_argument("--growth-noise-count", type=int, default=1)
    parser.add_argument("--stability-top-n", type=int, default=64)
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
