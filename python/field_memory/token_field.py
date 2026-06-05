from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

from python.librarian.features import activation_mask_for_text, cosine, fnv1a64, token_set


@dataclass(frozen=True)
class FieldToken:
    action_id: int
    bucket: int
    weight: float


@dataclass(frozen=True)
class FieldNode:
    index: int
    node_id: str
    text: str
    embedding: tuple[float, ...]
    mask: int
    tokens: tuple[FieldToken, ...]
    level: int


@dataclass
class TokenFieldIndex:
    nodes: list[FieldNode]
    inverted: dict[tuple[int, int], list[int]]
    action_inverted: dict[int, list[int]]
    layered_inverted: dict[tuple[int, int, int], list[int]]
    projection_plan: tuple[tuple[tuple[int, float], ...], ...]
    token_emitter: Any | None
    layer_count: int
    build_latency_ms: float
    index_bytes: int


def fit_embedding(values: list[float], dim_count: int) -> tuple[float, ...]:
    out = [float(value) for value in values[:dim_count]]
    if len(out) < dim_count:
        out.extend([0.0] * (dim_count - len(out)))
    norm = math.sqrt(sum(value * value for value in out))
    if norm > 0.0 and math.isfinite(norm):
        out = [value / norm for value in out]
    return tuple(out)


def make_projection_plan(
    dim_count: int,
    action_count: int,
    projection_width: int,
) -> tuple[tuple[tuple[int, float], ...], ...]:
    dim_count = max(1, int(dim_count))
    action_count = max(1, int(action_count))
    width = min(max(1, int(projection_width)), dim_count)
    plan = []
    for action_id in range(action_count):
        points = []
        for offset in range(width):
            hashed = fnv1a64(f"field-action:{action_id}:{offset}")
            dim = hashed % dim_count
            sign = -1.0 if ((hashed >> 23) & 1) else 1.0
            points.append((dim, sign))
        plan.append(tuple(points))
    return tuple(plan)


def action_projection_from_plan(
    embedding: tuple[float, ...],
    action_points: tuple[tuple[int, float], ...],
) -> float:
    if not embedding:
        return 0.0
    total = 0.0
    for dim, sign in action_points:
        total += sign * embedding[dim]
    return total / math.sqrt(max(1, len(action_points)))


def bucket_for(value: float, bucket_width: float) -> int:
    width = max(1e-6, float(bucket_width))
    return int(math.floor(float(value) / width))


def hash_unit(key: str) -> float:
    return (fnv1a64(key) >> 11) / float(1 << 53)


def promoted_level(
    node_id: str,
    text: str,
    tokens: tuple[FieldToken, ...],
    layer_count: int,
    promotion_probability: float,
    promotion_bias: float,
) -> int:
    layer_count = max(1, int(layer_count))
    probability = min(0.95, max(0.0, float(promotion_probability)))
    if layer_count <= 1 or probability <= 0.0:
        return 0
    mean_weight = sum(token.weight for token in tokens) / max(1, len(tokens))
    text_bias = (fnv1a64(f"field-importance:{node_id}:{len(text)}") % 1000) / 1000.0
    biased_probability = min(0.95, probability * (1.0 + max(0.0, promotion_bias) * (0.5 * mean_weight + 0.5 * text_bias)))
    level = 0
    for candidate_level in range(1, layer_count):
        if hash_unit(f"field-promote:{node_id}:{candidate_level}") > biased_probability:
            break
        level = candidate_level
    return level


def lexical_action_boosts(text: str, action_count: int) -> tuple[float, ...]:
    # Sparse lexical boosts keep this as a random action-field retriever while
    # giving exact names, dates, and entities a stable way to influence tokens.
    boosts = [0.0] * max(1, int(action_count))
    for token in token_set(text):
        for offset in range(3):
            hashed = fnv1a64(f"field-lex:{token}:{offset}")
            boosts[hashed % len(boosts)] += 0.035
    return tuple(min(0.25, value) for value in boosts)


def token_weight(action_id: int, value: float, lexical_boosts: tuple[float, ...]) -> float:
    lexical = lexical_boosts[action_id] if action_id < len(lexical_boosts) else 0.0
    return abs(value) + lexical


def field_tokens_for(
    text: str,
    embedding: tuple[float, ...],
    projection_plan: tuple[tuple[tuple[int, float], ...], ...],
    token_count: int,
    bucket_width: float,
) -> tuple[FieldToken, ...]:
    scored = []
    lexical_boosts = lexical_action_boosts(text, len(projection_plan))
    for action_id, action_points in enumerate(projection_plan):
        value = action_projection_from_plan(embedding, action_points)
        scored.append((token_weight(action_id, value, lexical_boosts), action_id, bucket_for(value, bucket_width)))
    scored.sort(key=lambda item: (-item[0], item[1], item[2]))
    return tuple(FieldToken(action_id, bucket, weight) for weight, action_id, bucket in scored[: max(1, int(token_count))])


def build_token_field_index(row: dict[str, Any], args: Any) -> TokenFieldIndex:
    started = time.perf_counter()
    dim_count = max(8, int(args.dim_count))
    projection_plan = make_projection_plan(dim_count, int(args.action_count), int(args.projection_width))
    token_emitter = load_configured_token_emitter(args)
    layer_count = max(1, int(getattr(args, "routing_layers", 1)))
    promotion_probability = float(getattr(args, "promotion_probability", 0.35))
    promotion_bias = float(getattr(args, "promotion_bias", 0.0))
    nodes = []
    inverted: dict[tuple[int, int], list[int]] = {}
    action_inverted_sets: dict[int, set[int]] = {}
    layered_inverted: dict[tuple[int, int, int], list[int]] = {}
    sorted_candidates = sorted(row.get("candidates") or [], key=lambda item: str(item.get("id") or ""))
    embeddings = [
        fit_embedding([float(value) for value in candidate.get("embedding") or []], dim_count)
        for candidate in sorted_candidates
    ]
    encoded_tokens = None
    if token_emitter is not None and embeddings:
        encoded_tokens = token_emitter.tokens_for_embeddings(embeddings, "node", token_count=int(args.node_token_count))
    for index, candidate in enumerate(sorted_candidates):
        node_id = str(candidate.get("id") or "")
        text = str(candidate.get("text") or "")
        embedding = embeddings[index]
        if encoded_tokens is not None:
            tokens = encoded_tokens[index]
        else:
            tokens = field_tokens_for(
                text,
                embedding,
                projection_plan,
                int(args.node_token_count),
                float(args.bucket_width),
            )
        level = promoted_level(node_id, text, tokens, layer_count, promotion_probability, promotion_bias)
        node = FieldNode(
            index=index,
            node_id=node_id,
            text=text,
            embedding=embedding,
            mask=activation_mask_for_text(text),
            tokens=tokens,
            level=level,
        )
        nodes.append(node)
        for token in tokens:
            inverted.setdefault((token.action_id, token.bucket), []).append(index)
            action_inverted_sets.setdefault(token.action_id, set()).add(index)
            for layer in range(level + 1):
                layered_inverted.setdefault((layer, token.action_id, token.bucket), []).append(index)
    index_bytes = (
        len(nodes) * (64 + 4 * dim_count)
        + sum(len(values) for values in inverted.values()) * 8
        + sum(len(values) for values in action_inverted_sets.values()) * 4
        + sum(len(values) for values in layered_inverted.values()) * 8
        + sum(len(node.text.encode("utf-8")) + len(node.node_id.encode("utf-8")) for node in nodes)
    )
    return TokenFieldIndex(
        nodes=nodes,
        inverted={key: sorted(values) for key, values in inverted.items()},
        action_inverted={key: sorted(values) for key, values in action_inverted_sets.items()},
        layered_inverted={key: sorted(values) for key, values in layered_inverted.items()},
        projection_plan=projection_plan,
        token_emitter=token_emitter,
        layer_count=layer_count,
        build_latency_ms=(time.perf_counter() - started) * 1000.0,
        index_bytes=index_bytes,
    )


def query_tokens(
    query: str,
    query_embedding: tuple[float, ...],
    projection_plan: tuple[tuple[tuple[int, float], ...], ...],
    args: Any,
) -> tuple[FieldToken, ...]:
    return field_tokens_for(
        query,
        query_embedding,
        projection_plan,
        int(args.query_token_count),
        float(args.bucket_width),
    )


def load_configured_token_emitter(args: Any) -> Any | None:
    checkpoint = str(getattr(args, "token_encoder_checkpoint", "") or "")
    if not checkpoint:
        return None
    from python.field_memory.token_encoder import TokenFieldEmitter

    return TokenFieldEmitter(checkpoint, device=str(getattr(args, "token_encoder_device", "") or getattr(args, "device", "") or ""))


def overlap_score(query_tokens_: tuple[FieldToken, ...], node: FieldNode, bucket_radius: int) -> float:
    node_lookup = {(token.action_id, token.bucket): token.weight for token in node.tokens}
    score = 0.0
    for token in query_tokens_:
        for delta in range(-bucket_radius, bucket_radius + 1):
            node_weight = node_lookup.get((token.action_id, token.bucket + delta))
            if node_weight is None:
                continue
            score += token.weight * node_weight / float(1 + abs(delta))
    return score


def layer_collisions(
    index: TokenFieldIndex,
    tokens: tuple[FieldToken, ...],
    layer: int,
    bucket_radius: int,
    allowed: set[int] | None = None,
) -> dict[int, float]:
    collisions: dict[int, float] = {}
    for token in tokens:
        for delta in range(-bucket_radius, bucket_radius + 1):
            key = (layer, token.action_id, token.bucket + delta)
            for node_index in index.layered_inverted.get(key, []):
                if allowed is not None and node_index not in allowed:
                    continue
                collisions[node_index] = collisions.get(node_index, 0.0) + token.weight / float(1 + abs(delta))
    return collisions


def deterministic_take(candidates: dict[int, float], limit: int) -> dict[int, float]:
    limit = max(1, int(limit))
    if len(candidates) <= limit:
        return dict(sorted(candidates.items(), key=lambda item: item[0]))
    return dict(sorted(candidates.items(), key=lambda item: (-item[1], item[0]))[:limit])


def route_layers(
    index: TokenFieldIndex,
    tokens: tuple[FieldToken, ...],
    bucket_radius: int,
    args: Any,
) -> tuple[dict[int, float], dict[str, float]]:
    if index.layer_count <= 1:
        return {}, {"routing_layer_reads": 0.0, "routing_candidate_count": 0.0}
    beam_width = max(1, int(getattr(args, "routing_beam_width", 48)))
    frontier: dict[int, float] = {}
    layer_reads = 0
    for layer in range(index.layer_count - 1, 0, -1):
        collisions = layer_collisions(index, tokens, layer, bucket_radius)
        layer_reads += 1
        if frontier:
            for node_index, score in frontier.items():
                if index.nodes[node_index].level >= layer - 1:
                    collisions[node_index] = collisions.get(node_index, 0.0) + 0.5 * score
        if collisions:
            frontier = deterministic_take(collisions, beam_width)
    return frontier, {
        "routing_layer_reads": float(layer_reads),
        "routing_candidate_count": float(len(frontier)),
    }


def action_fallback_candidates(index: TokenFieldIndex, tokens: tuple[FieldToken, ...]) -> dict[int, float]:
    candidates: dict[int, float] = {}
    for token in tokens:
        for node_index in index.action_inverted.get(token.action_id, []):
            candidates[node_index] = candidates.get(node_index, 0.0) + 0.05 * token.weight
    return candidates


def search_token_field(row: dict[str, Any], backend: Any, index: TokenFieldIndex, args: Any) -> tuple[list[tuple[str, float, str]], dict[str, float], set[str]]:
    started = time.perf_counter()
    task = row.get("retrieval_task") or {}
    query = str(task.get("query") or row["anchor"]["text"])
    raw_embedding = task.get("query_embedding") or backend.embed_one(query)
    query_embedding = fit_embedding([float(value) for value in raw_embedding], int(args.dim_count))
    if index.token_emitter is not None:
        tokens = index.token_emitter.tokens_for_embeddings([query_embedding], "query", token_count=int(args.query_token_count))[0]
    else:
        tokens = query_tokens(query, query_embedding, index.projection_plan, args)
    bucket_radius = max(0, int(args.bucket_radius))
    routing_frontier, route_stats = route_layers(index, tokens, bucket_radius, args)
    layer_zero = layer_collisions(index, tokens, 0, bucket_radius)
    min_collision = max(0.0, float(getattr(args, "include_min_collision", 0.0)))
    min_overlap = max(0.0, float(getattr(args, "include_min_overlap", 0.0)))
    candidates = {}
    for node_index, collision in layer_zero.items():
        routed_bonus = routing_frontier.get(node_index, 0.0)
        node = index.nodes[node_index]
        if collision < min_collision:
            continue
        if min_overlap > 0.0 and overlap_score(tokens, node, bucket_radius) < min_overlap:
            continue
        candidates[node_index] = collision + 0.35 * routed_bonus
    if routing_frontier:
        for node_index, score in routing_frontier.items():
            if node_index in layer_zero:
                candidates.setdefault(node_index, layer_zero[node_index] + 0.35 * score)
    if len(candidates) < int(args.min_candidates):
        # Deterministic fallback: widen the field by using action-id collisions
        # only. This preserves the "shape token" path while avoiding empty hits.
        candidates.update(action_fallback_candidates(index, tokens))
    raw_candidate_count = len(candidates)
    max_candidates = max(1, int(getattr(args, "max_candidates", 192)))
    candidates = deterministic_take(candidates, max_candidates)
    scored = []
    query_mask = activation_mask_for_text(query)
    for node_index, collision in candidates.items():
        node = index.nodes[node_index]
        field = overlap_score(tokens, node, bucket_radius)
        semantic = cosine(query_embedding, node.embedding)
        activation = (query_mask & node.mask).bit_count() / max(1, query_mask.bit_count())
        score = 0.58 * field + 0.28 * semantic + 0.14 * activation + 0.02 * collision
        scored.append((node.node_id, float(score), node.text))
    scored.sort(key=lambda item: (-item[1], item[0]))
    fetch = scored[: max(1, int(args.final_fetch))]
    stats = {
        "latency_ms": (time.perf_counter() - started) * 1000.0,
        "unique_nodes_read": float(len(candidates)),
        "payload_reads": float(len(fetch)),
        "node_records_read": float(len(candidates)),
        "edge_reads": 0.0,
        "edge_expansions": route_stats["routing_layer_reads"],
        "routing_layer_reads": route_stats["routing_layer_reads"],
        "routing_candidate_count": route_stats["routing_candidate_count"],
        "raw_final_candidate_count": float(raw_candidate_count),
        "final_candidate_count": float(len(fetch)),
        "calibrator_latency_ms": 0.0,
    }
    return fetch, stats, {index.nodes[node_index].node_id for node_index in candidates}
