from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from python.benchmarks.hierarchical_file_ann import Ranked, project_key
from python.benchmarks.hippocampus_retrieval import activation_overlap, edge_activation, edge_type_boost, ensure_backend_embeddings, state_score
from python.benchmarks.rope_delta_grid import query_embedding_for, read_vector
from python.librarian.features import activation_mask_for_text, cosine, fnv1a64


@dataclass(frozen=True)
class MemoryGraphNode:
    index: int
    node_id: str
    text: str
    embedding: tuple[float, ...]
    mask: int
    project_hash: int
    importance: float
    state: float
    max_level: int


@dataclass(frozen=True)
class MemoryGraphEdge:
    target: int
    edge_type: str
    mask: int
    weight: float
    confidence: float
    type_boost: float


@dataclass
class AgentMemoryGraphIndex:
    nodes: list[MemoryGraphNode]
    truth_edges: list[list[MemoryGraphEdge]]
    routing_edges: list[list[list[int]]]
    entrypoints: list[int]
    build_latency_ms: float
    index_bytes: int


def deterministic_level(card: dict[str, Any], truth_degree: int, args: argparse.Namespace) -> int:
    max_level = max(1, int(args.memory_graph_layers)) - 1
    threshold = max(1, min(255, int(args.memory_graph_promotion_threshold)))
    value = fnv1a64(str(card.get("id") or ""))
    level = 0
    while level < max_level and ((value >> (level * 8)) & 0xFF) < threshold:
        level += 1
    importance = float(card.get("importance") or 0.5)
    if bool(getattr(args, "memory_graph_bias_promotion", False)):
        if truth_degree >= int(args.memory_graph_bridge_degree):
            level += 1
        if importance >= float(args.memory_graph_importance_threshold):
            level += 1
    return min(max_level, level)


def normalized(values: list[float]) -> tuple[float, ...]:
    total = math.sqrt(sum(value * value for value in values))
    if total <= 0.0 or not math.isfinite(total):
        return tuple(float(value) for value in values)
    return tuple(float(value) / total for value in values)


def route_score(left: MemoryGraphNode, right: MemoryGraphNode) -> float:
    semantic = cosine(list(left.embedding), list(right.embedding))
    activation = activation_overlap(left.mask, right.mask)
    same_project = 1.0 if left.project_hash == right.project_hash else 0.0
    return 0.78 * semantic + 0.08 * activation + 0.06 * same_project + 0.04 * right.importance + right.state


def projection_value(node: MemoryGraphNode, layer: int, projection: int, dim_count: int) -> float:
    if dim_count <= 0:
        return 0.0
    value = 0.0
    # A deterministic sparse random projection. Sixteen dimensions is enough
    # for routing candidates without turning build time into an all-pairs scan.
    for offset in range(min(16, dim_count)):
        hashed = fnv1a64(f"route:{layer}:{projection}:{offset}")
        dim = hashed % dim_count
        sign = -1.0 if ((hashed >> 17) & 1) else 1.0
        value += sign * node.embedding[dim]
    return value


def query_node_score(query: str, query_embedding: list[float], query_mask: int, query_project_hash: int, node: MemoryGraphNode) -> float:
    semantic = cosine(query_embedding, node.embedding)
    activation = activation_overlap(query_mask, node.mask)
    same_project = 1.0 if query_project_hash == node.project_hash else 0.0
    return 0.66 * semantic + 0.18 * activation + 0.06 * same_project + 0.04 * node.importance + node.state


def objective_seed_nodes(
    index: AgentMemoryGraphIndex,
    query: str,
    query_embedding: list[float],
    query_mask: int,
    query_project_hash: int,
    args: argparse.Namespace,
) -> list[int]:
    count = max(0, int(getattr(args, "memory_graph_objective_seeds", 0)))
    if count <= 0:
        return []
    scored = []
    for node in index.nodes:
        activation = activation_overlap(query_mask, node.mask)
        if activation <= 0.0:
            continue
        same_project = 1.0 if node.project_hash == query_project_hash else 0.0
        score = 0.62 * activation + 0.08 * same_project + 0.05 * node.importance + node.state
        scored.append((score, node.index))
    if len(scored) < count:
        for node in index.nodes:
            if any(node.index == existing for _, existing in scored):
                continue
            score = query_node_score(query, query_embedding, query_mask, query_project_hash, node)
            scored.append((score, node.index))
    scored.sort(key=lambda item: (-item[0], index.nodes[item[1]].node_id))
    return [node_index for _, node_index in scored[:count]]


def build_routing_edges(nodes: list[MemoryGraphNode], args: argparse.Namespace) -> list[list[list[int]]]:
    layer_count = max(1, int(args.memory_graph_layers))
    max_degree = max(1, int(args.memory_graph_route_degree))
    projection_count = max(1, int(getattr(args, "memory_graph_projection_count", 3)))
    projection_window = max(max_degree, int(getattr(args, "memory_graph_projection_window", max_degree * 2)))
    dim_count = len(nodes[0].embedding) if nodes else 0
    routing: list[list[list[int]]] = [[[] for _ in nodes] for _ in range(layer_count)]
    for layer in range(layer_count):
        layer_nodes = [node.index for node in nodes if node.max_level >= layer]
        if len(layer_nodes) <= 1:
            continue
        candidate_scores: dict[int, dict[int, float]] = {node_index: {} for node_index in layer_nodes}
        for projection in range(projection_count):
            ordered = sorted(
                layer_nodes,
                key=lambda node_index: (projection_value(nodes[node_index], layer, projection, dim_count), nodes[node_index].node_id),
            )
            positions = {node_index: position for position, node_index in enumerate(ordered)}
            for source in layer_nodes:
                position = positions[source]
                start = max(0, position - projection_window)
                end = min(len(ordered), position + projection_window + 1)
                for target_position in range(start, end):
                    if target_position == position:
                        continue
                    target = ordered[target_position]
                    distance = abs(position - target_position)
                    candidate_scores[source][target] = candidate_scores[source].get(target, 0.0) + 1.0 / float(1 + distance)
        for source in layer_nodes:
            source_node = nodes[source]
            scored = []
            for target, proximity in candidate_scores[source].items():
                target_node = nodes[target]
                jitter = (fnv1a64(f"{source_node.node_id}->{target_node.node_id}:{layer}") & 0xFFFF) / 1_000_000_000.0
                same_project = 1.0 if source_node.project_hash == target_node.project_hash else 0.0
                score = proximity + 0.06 * same_project + 0.04 * target_node.importance + target_node.state + jitter
                scored.append((score, target))
            scored.sort(key=lambda item: (-item[0], nodes[item[1]].node_id))
            routing[layer][source] = [target for _, target in scored[:max_degree]]
    if bool(getattr(args, "memory_graph_reciprocal_routes", True)):
        for layer in range(layer_count):
            for source, targets in enumerate(list(routing[layer])):
                for target in targets:
                    if source not in routing[layer][target]:
                        routing[layer][target].append(source)
                        routing[layer][target].sort(key=lambda item: nodes[item].node_id)
                        if len(routing[layer][target]) > max_degree * 2:
                            routing[layer][target] = sorted(
                                routing[layer][target],
                                key=lambda item: (-(nodes[item].importance + nodes[item].state), nodes[item].node_id),
                            )[: max_degree * 2]
    return routing


def choose_entrypoints(nodes: list[MemoryGraphNode], args: argparse.Namespace) -> list[int]:
    layer_count = max(1, int(args.memory_graph_layers))
    entrypoints = []
    for layer in range(layer_count):
        candidates = [node for node in nodes if node.max_level >= layer]
        if not candidates:
            candidates = nodes
        chosen = sorted(
            candidates,
            key=lambda node: (
                -(node.importance + node.state + 0.01 * node.max_level),
                fnv1a64(f"entry:{layer}:{node.node_id}"),
                node.node_id,
            ),
        )[0]
        entrypoints.append(chosen.index)
    return entrypoints


def estimate_index_bytes(index: AgentMemoryGraphIndex, dim_count: int) -> int:
    node_bytes = len(index.nodes) * (64 + 4 * dim_count)
    truth_bytes = sum(len(edges) for edges in index.truth_edges) * 32
    route_bytes = sum(len(edges) for layer in index.routing_edges for edges in layer) * 4
    text_bytes = sum(len(node.node_id.encode("utf-8")) + len(node.text.encode("utf-8")) for node in index.nodes)
    return node_bytes + truth_bytes + route_bytes + text_bytes


def build_agent_memory_graph(row: dict[str, Any], backend: Any, output_dir: Path, args: argparse.Namespace) -> AgentMemoryGraphIndex:
    started = time.perf_counter()
    row = ensure_backend_embeddings(row, backend)
    output_dir.mkdir(parents=True, exist_ok=True)
    dim_count = max(3, int(args.dim_count))
    cards = [dict(card) for card in row.get("candidates", [])]
    cards.sort(key=lambda card: str(card.get("id") or ""))
    if not cards:
        raise ValueError("cannot build an empty agent memory graph")
    id_to_index = {str(card.get("id") or ""): index for index, card in enumerate(cards)}
    outgoing_raw: dict[int, list[dict[str, Any]]] = {}
    truth_degree = [0 for _ in cards]
    for edge in (row.get("memory_graph") or {}).get("edges") or []:
        source = str(edge.get("source") or "")
        target = str(edge.get("target") or "")
        if source in id_to_index and target in id_to_index:
            source_index = id_to_index[source]
            target_index = id_to_index[target]
            outgoing_raw.setdefault(source_index, []).append(edge)
            truth_degree[source_index] += 1
            truth_degree[target_index] += 1
    nodes = []
    for index, card in enumerate(cards):
        text = str(card.get("text") or "")
        nodes.append(
            MemoryGraphNode(
                index=index,
                node_id=str(card.get("id") or ""),
                text=text,
                embedding=normalized(read_vector(card, dim_count)),
                mask=activation_mask_for_text(f"{text} {card.get('summary', '')}"),
                project_hash=fnv1a64(project_key(card)),
                importance=float(card.get("importance") or 0.5),
                state=float(state_score(card)),
                max_level=deterministic_level(card, truth_degree[index], args),
            )
        )
    truth_edges: list[list[MemoryGraphEdge]] = [[] for _ in nodes]
    for source, edges in outgoing_raw.items():
        for edge in sorted(edges, key=lambda item: (str(item.get("target") or ""), str(item.get("type") or ""))):
            target = id_to_index[str(edge.get("target") or "")]
            edge_type = str(edge.get("type") or "used_with")
            truth_edges[source].append(
                MemoryGraphEdge(
                    target=target,
                    edge_type=edge_type,
                    mask=int(edge_activation(edge)),
                    weight=float(edge.get("weight") or 0.0),
                    confidence=float(edge.get("confidence") or 0.5),
                    type_boost=edge_type_boost(edge_type),
                )
            )
    routing_edges = build_routing_edges(nodes, args)
    index = AgentMemoryGraphIndex(
        nodes=nodes,
        truth_edges=truth_edges,
        routing_edges=routing_edges,
        entrypoints=choose_entrypoints(nodes, args),
        build_latency_ms=(time.perf_counter() - started) * 1000.0,
        index_bytes=0,
    )
    index.index_bytes = estimate_index_bytes(index, dim_count)
    return index


def search_layer(
    index: AgentMemoryGraphIndex,
    layer: int,
    seeds: list[int],
    query: str,
    query_embedding: list[float],
    query_mask: int,
    query_project_hash: int,
    score_cache: dict[int, float],
    args: argparse.Namespace,
) -> tuple[list[tuple[int, float]], int]:
    ef = max(1, int(args.memory_graph_ef))
    beam = max(1, int(args.memory_graph_beam))
    visited: set[int] = set()
    scored: dict[int, float] = {}
    frontier = []
    for seed in seeds:
        if seed < 0 or seed >= len(index.nodes):
            continue
        if seed not in score_cache:
            score_cache[seed] = query_node_score(query, query_embedding, query_mask, query_project_hash, index.nodes[seed])
        score = score_cache[seed]
        scored[seed] = max(scored.get(seed, -99.0), score)
        frontier.append((score, seed))
    expansions = 0
    while frontier and expansions < ef:
        frontier.sort(key=lambda item: (-item[0], index.nodes[item[1]].node_id))
        score, current = frontier.pop(0)
        if current in visited:
            continue
        visited.add(current)
        expansions += 1
        for target in index.routing_edges[layer][current]:
            if target in visited:
                continue
            if target not in score_cache:
                score_cache[target] = query_node_score(query, query_embedding, query_mask, query_project_hash, index.nodes[target])
            target_score = score_cache[target]
            if target_score > scored.get(target, -99.0):
                scored[target] = target_score
                frontier.append((target_score, target))
        frontier = sorted(frontier, key=lambda item: (-item[0], index.nodes[item[1]].node_id))[:ef]
    ranked = sorted(scored.items(), key=lambda item: (-item[1], index.nodes[item[0]].node_id))[:beam]
    return ranked, len(visited)


def apply_truth_expansion(
    index: AgentMemoryGraphIndex,
    seeds: list[tuple[int, float]],
    query_mask: int,
    args: argparse.Namespace,
) -> tuple[dict[int, float], int, int]:
    best = {node_index: score for node_index, score in seeds}
    frontier = [(node_index, score, 0, {node_index}) for node_index, score in seeds[: max(1, int(args.memory_graph_truth_seeds))]]
    edge_reads = 0
    edge_expansions = 0
    max_depth = max(0, int(args.memory_graph_truth_depth))
    while frontier:
        node_index, score, depth, path = frontier.pop(0)
        if depth >= max_depth:
            continue
        next_items = []
        for edge in index.truth_edges[node_index]:
            edge_reads += 1
            if edge.target in path:
                continue
            gain = (
                float(edge.weight)
                * float(edge.type_boost)
                * (0.62 + 0.50 * activation_overlap(query_mask, edge.mask))
                * (0.70 + 0.30 * edge.confidence)
                / (1.35 + depth)
            )
            target_score = 0.78 * score + gain + 0.02 * index.nodes[edge.target].importance + index.nodes[edge.target].state
            if target_score > best.get(edge.target, -99.0):
                best[edge.target] = target_score
                edge_expansions += 1
                next_items.append((edge.target, target_score, depth + 1, path | {edge.target}))
        next_items.sort(key=lambda item: (-item[1], index.nodes[item[0]].node_id))
        frontier.extend(next_items[: max(1, int(args.memory_graph_truth_fanout))])
    return best, edge_reads, edge_expansions


def cutoff_ranked(ranked: Ranked, args: argparse.Namespace) -> Ranked:
    if not ranked:
        return ranked
    minimum = max(1, int(args.memory_graph_min_results))
    if len(ranked) <= minimum:
        return ranked
    top_score = float(ranked[0][1])
    margin = float(args.memory_graph_cutoff_margin)
    floor = float(args.memory_graph_min_score)
    kept = []
    for item in ranked:
        if len(kept) < minimum or (float(item[1]) >= floor and top_score - float(item[1]) <= margin):
            kept.append(item)
    return kept


def search_agent_memory_graph(row: dict[str, Any], backend: Any, index: AgentMemoryGraphIndex, args: argparse.Namespace) -> tuple[Ranked, dict[str, float], set[str]]:
    task = row.get("retrieval_task") or {}
    query = str(task.get("query") or row["anchor"]["text"])
    query_embedding = normalized(query_embedding_for(row, backend))
    query_embedding_list = list(query_embedding)
    query_mask = activation_mask_for_text(query)
    query_project_hash = fnv1a64(project_key(row["anchor"]))
    started = time.perf_counter()
    score_cache: dict[int, float] = {}
    total_node_reads = 0
    layer_count = len(index.routing_edges)
    objective_seeds = objective_seed_nodes(index, query, query_embedding_list, query_mask, query_project_hash, args)
    seeds = [index.entrypoints[-1]] + objective_seeds[: max(0, int(args.memory_graph_beam))]
    candidates: list[tuple[int, float]] = []
    for layer in range(layer_count - 1, -1, -1):
        entry = index.entrypoints[layer]
        layer_seeds = sorted(set(seeds + [entry]))
        if layer == 0:
            layer_seeds = sorted(set(layer_seeds + objective_seeds))
        candidates, reads = search_layer(
            index,
            layer,
            layer_seeds,
            query,
            query_embedding_list,
            query_mask,
            query_project_hash,
            score_cache,
            args,
        )
        total_node_reads += reads
        seeds = [node_index for node_index, _ in candidates[: max(1, int(args.memory_graph_beam))]]
    best, edge_reads, edge_expansions = apply_truth_expansion(index, candidates, query_mask, args)
    ranked_pairs = sorted(best.items(), key=lambda item: (-item[1], index.nodes[item[0]].node_id))
    raw_count = len(ranked_pairs)
    ranked: Ranked = [(index.nodes[node_index].node_id, float(score), index.nodes[node_index].text) for node_index, score in ranked_pairs]
    ranked = cutoff_ranked(ranked, args)
    fetch = ranked[: max(1, int(args.final_fetch))]
    stats = {
        "latency_ms": (time.perf_counter() - started) * 1000.0,
        "unique_nodes_read": float(total_node_reads),
        "payload_reads": float(len(fetch)),
        "node_records_read": float(total_node_reads),
        "edge_reads": float(edge_reads),
        "edge_expansions": float(edge_expansions),
        "raw_final_candidate_count": float(raw_count),
        "final_candidate_count": float(len(fetch)),
        "calibrator_latency_ms": 0.0,
    }
    return fetch, stats, set()
