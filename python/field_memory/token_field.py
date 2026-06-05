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


@dataclass
class TokenFieldIndex:
    nodes: list[FieldNode]
    inverted: dict[tuple[int, int], list[int]]
    projection_plan: tuple[tuple[tuple[int, float], ...], ...]
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
    nodes = []
    inverted: dict[tuple[int, int], list[int]] = {}
    for index, candidate in enumerate(sorted(row.get("candidates") or [], key=lambda item: str(item.get("id") or ""))):
        node_id = str(candidate.get("id") or "")
        text = str(candidate.get("text") or "")
        embedding = fit_embedding([float(value) for value in candidate.get("embedding") or []], dim_count)
        tokens = field_tokens_for(
            text,
            embedding,
            projection_plan,
            int(args.node_token_count),
            float(args.bucket_width),
        )
        node = FieldNode(
            index=index,
            node_id=node_id,
            text=text,
            embedding=embedding,
            mask=activation_mask_for_text(text),
            tokens=tokens,
        )
        nodes.append(node)
        for token in tokens:
            inverted.setdefault((token.action_id, token.bucket), []).append(index)
    index_bytes = (
        len(nodes) * (64 + 4 * dim_count)
        + sum(len(values) for values in inverted.values()) * 8
        + sum(len(node.text.encode("utf-8")) + len(node.node_id.encode("utf-8")) for node in nodes)
    )
    return TokenFieldIndex(
        nodes=nodes,
        inverted={key: sorted(values) for key, values in inverted.items()},
        projection_plan=projection_plan,
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


def search_token_field(row: dict[str, Any], backend: Any, index: TokenFieldIndex, args: Any) -> tuple[list[tuple[str, float, str]], dict[str, float], set[str]]:
    started = time.perf_counter()
    task = row.get("retrieval_task") or {}
    query = str(task.get("query") or row["anchor"]["text"])
    raw_embedding = task.get("query_embedding") or backend.embed_one(query)
    query_embedding = fit_embedding([float(value) for value in raw_embedding], int(args.dim_count))
    tokens = query_tokens(query, query_embedding, index.projection_plan, args)
    candidates: dict[int, float] = {}
    bucket_radius = max(0, int(args.bucket_radius))
    for token in tokens:
        for delta in range(-bucket_radius, bucket_radius + 1):
            for node_index in index.inverted.get((token.action_id, token.bucket + delta), []):
                candidates[node_index] = candidates.get(node_index, 0.0) + token.weight / float(1 + abs(delta))
    if len(candidates) < int(args.min_candidates):
        # Deterministic fallback: widen the field by using action-id collisions
        # only. This preserves the "shape token" path while avoiding empty hits.
        query_action_ids = {token.action_id for token in tokens}
        for node in index.nodes:
            if any(token.action_id in query_action_ids for token in node.tokens):
                candidates.setdefault(node.index, 0.0)
    raw_candidate_count = len(candidates)
    max_candidates = max(1, int(getattr(args, "max_candidates", 192)))
    if len(candidates) > max_candidates:
        candidates = dict(sorted(candidates.items(), key=lambda item: (-item[1], item[0]))[:max_candidates])
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
        "edge_expansions": 0.0,
        "raw_final_candidate_count": float(raw_candidate_count),
        "final_candidate_count": float(len(fetch)),
        "calibrator_latency_ms": 0.0,
    }
    return fetch, stats, set()
