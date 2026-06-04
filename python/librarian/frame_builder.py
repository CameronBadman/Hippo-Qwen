from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from python.librarian.features import (
    DEFAULT_DIMS,
    activation_mask_for_text,
    clamp,
    cosine,
    jaccard,
    state_features,
    tokens,
)
from python.librarian.frame_cache import CachedNeighbor, FrameCache


FRAME_SCALARS = 14


@dataclass(frozen=True)
class FrameBuilderConfig:
    frame_size: int = 64
    graph_seed_count: int = 16
    graph_depth: int = 3
    graph_boost: float = 0.85
    use_role_features: bool = False


def query_text(row: dict[str, Any]) -> str:
    return str((row.get("retrieval_task") or {}).get("query") or row["anchor"]["text"])


def query_embedding(row: dict[str, Any]) -> list[float]:
    from python.librarian.features import embed_text

    return embed_text(query_text(row))


def frame_score(row: dict[str, Any], candidate: dict[str, Any], query_vec: list[float], use_role_features: bool) -> float:
    query = query_text(row)
    anchor = row["anchor"]
    qmask = activation_mask_for_text(query)
    cmask = activation_mask_for_text(candidate.get("text", ""))
    activation = (qmask & cmask).bit_count() / max(1, (qmask | cmask).bit_count())
    state = state_features(anchor, candidate)
    role = str(candidate.get("synthetic_role") or "") if use_role_features else ""
    conflict_penalty = 0.25 if any(term in role for term in ("decoy", "wrong", "stale", "background")) else 0.0
    return (
        0.36 * cosine(query_vec, candidate.get("embedding") or [])
        + 0.22 * jaccard(query, candidate.get("text", ""))
        + 0.16 * activation
        + 0.10 * float(candidate.get("importance") or 0.5)
        + 0.09 * state["candidate_use_norm"]
        + 0.07 * state["candidate_evidence_norm"]
        + 0.08 * max(0.0, state["last_outcome_value"])
        - 0.12 * state["stale_unused_flag"]
        - conflict_penalty
    )


def edge_strength(edge: dict[str, Any], query_mask: int, depth: int, graph_boost: float) -> float:
    activation_text = str(edge.get("activation_text") or "")
    edge_mask = activation_mask_for_text(activation_text)
    activation = (query_mask & edge_mask).bit_count() / max(1, (query_mask | edge_mask).bit_count())
    weight = float(edge.get("weight") or 0.0)
    confidence = float(edge.get("confidence") or 0.0)
    return graph_boost * (0.48 * weight + 0.34 * confidence + 0.18 * activation) / (depth + 1)


def graph_adjacency(row: dict[str, Any], candidates: dict[str, dict[str, Any]]) -> dict[str, list[tuple[str, dict[str, Any]]]]:
    adjacency: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    for edge in (row.get("memory_graph") or {}).get("edges") or []:
        source = str(edge.get("source") or "")
        target = str(edge.get("target") or "")
        if source not in candidates or target not in candidates:
            continue
        adjacency.setdefault(source, []).append((target, dict(edge)))
        reverse = dict(edge)
        reverse["source"] = target
        reverse["target"] = source
        reverse["weight"] = float(edge.get("weight") or 0.0) * 0.72
        reverse["confidence"] = float(edge.get("confidence") or 0.0) * 0.72
        adjacency.setdefault(target, []).append((source, reverse))
    for edges in adjacency.values():
        edges.sort(key=lambda item: item[0])
    return adjacency


def graph_expanded_scores(
    row: dict[str, Any],
    candidates: list[dict[str, Any]],
    base_scores: dict[str, float],
    config: FrameBuilderConfig,
) -> dict[str, float]:
    scores = dict(base_scores)
    by_id = {candidate["id"]: candidate for candidate in candidates}
    adjacency = graph_adjacency(row, by_id)
    query_mask = activation_mask_for_text(query_text(row))
    seeds = sorted(base_scores.items(), key=lambda item: (-item[1], item[0]))[: max(1, config.graph_seed_count)]
    frontier = [(candidate_id, score, 0) for candidate_id, score in seeds]
    best_depth = {candidate_id: 0 for candidate_id, _, _ in frontier}
    while frontier:
        source_id, source_score, depth = frontier.pop(0)
        if depth >= config.graph_depth:
            continue
        for target_id, edge in adjacency.get(source_id, []):
            hop = edge_strength(edge, query_mask, depth, config.graph_boost)
            candidate_score = base_scores.get(target_id, -99.0) + 0.35 * source_score + hop
            if candidate_score > scores.get(target_id, -99.0):
                scores[target_id] = candidate_score
            next_depth = depth + 1
            if next_depth < best_depth.get(target_id, 999):
                best_depth[target_id] = next_depth
                frontier.append((target_id, scores[target_id], next_depth))
    return scores


def cached_neighbor_rows(row: dict[str, Any], config: FrameBuilderConfig) -> list[tuple[str, list[CachedNeighbor]]]:
    candidates = {str(candidate["id"]): dict(candidate) for candidate in row.get("candidates", [])}
    adjacency = graph_adjacency(row, candidates)
    rows = []
    for memory_id in sorted(candidates):
        neighbors: dict[str, CachedNeighbor] = {memory_id: CachedNeighbor(memory_id, 1.0, 0)}
        frontier = [(memory_id, 1.0, 0, [memory_id])]
        while frontier:
            source_id, source_strength, depth, path = frontier.pop(0)
            if depth >= config.graph_depth:
                continue
            for target_id, edge in adjacency.get(source_id, []):
                if target_id in path:
                    continue
                weight = float(edge.get("weight") or 0.0)
                confidence = float(edge.get("confidence") or 0.0)
                step_strength = source_strength * max(0.0, 0.55 * weight + 0.45 * confidence) / (depth + 1)
                prior = neighbors.get(target_id)
                if prior is None or step_strength > prior.boost:
                    neighbors[target_id] = CachedNeighbor(target_id, step_strength, depth + 1)
                    frontier.append((target_id, step_strength, depth + 1, path + [target_id]))
        rows.append((memory_id, list(neighbors.values())))
    return rows


def populate_frame_cache(row: dict[str, Any], cache: FrameCache, config: FrameBuilderConfig) -> dict[str, float]:
    started = time.perf_counter()
    rows = cached_neighbor_rows(row, config)
    cache.put_many(rows)
    return {"cache_build_ms": (time.perf_counter() - started) * 1000.0, "cache_records": float(len(rows))}


def build_live_graph_frame(row: dict[str, Any], config: FrameBuilderConfig) -> dict[str, Any]:
    qvec = query_embedding(row)
    candidates = [dict(candidate) for candidate in row.get("candidates", [])]
    base_scores = {candidate["id"]: frame_score(row, candidate, qvec, config.use_role_features) for candidate in candidates}
    scores = graph_expanded_scores(row, candidates, base_scores, config)
    return materialize_frame(row, candidates, scores, qvec, config)


def select_seed_ids(row: dict[str, Any], config: FrameBuilderConfig) -> tuple[list[str], dict[str, float], float]:
    started = time.perf_counter()
    qvec = query_embedding(row)
    candidates = [dict(candidate) for candidate in row.get("candidates", [])]
    base_scores = {candidate["id"]: frame_score(row, candidate, qvec, config.use_role_features) for candidate in candidates}
    seeds = sorted(base_scores.items(), key=lambda item: (-item[1], item[0]))[: max(1, config.graph_seed_count)]
    return [memory_id for memory_id, _ in seeds], base_scores, (time.perf_counter() - started) * 1000.0


def build_cached_graph_frame(
    row: dict[str, Any],
    cache: FrameCache,
    config: FrameBuilderConfig,
    seed_ids: list[str] | None = None,
    base_scores: dict[str, float] | None = None,
    candidate_lookup: dict[str, dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], dict[str, float]]:
    started = time.perf_counter()
    qvec = query_embedding(row)
    if candidate_lookup is None:
        candidates = [dict(candidate) for candidate in row.get("candidates", [])]
        by_id = {candidate["id"]: candidate for candidate in candidates}
    else:
        by_id = candidate_lookup
    if seed_ids is None:
        seed_ids, base_scores, seed_selection_ms = select_seed_ids(row, config)
    else:
        seed_selection_ms = 0.0
    base_scores = base_scores or {}
    merged_scores: dict[str, float] = {}
    hits = 0
    misses = 0
    pending_score_ids = set(seed_ids)
    for seed_id in seed_ids:
        cached = cache.get(seed_id)
        if cached is None:
            misses += 1
            pending_score_ids.add(seed_id)
            continue
        hits += 1
        for neighbor in cached.neighbors:
            if neighbor.memory_id not in by_id:
                continue
            pending_score_ids.add(neighbor.memory_id)

    missing_score_ids = [memory_id for memory_id in sorted(pending_score_ids) if memory_id not in base_scores and memory_id in by_id]
    if missing_score_ids:
        for candidate in [by_id[memory_id] for memory_id in missing_score_ids]:
            base_scores[candidate["id"]] = frame_score(row, candidate, qvec, config.use_role_features)

    for seed_id in seed_ids:
        seed_score = base_scores.get(seed_id, -99.0)
        cached = cache.get(seed_id)
        if cached is None:
            merged_scores[seed_id] = max(merged_scores.get(seed_id, -99.0), seed_score)
            continue
        for neighbor in cached.neighbors:
            if neighbor.memory_id not in by_id:
                continue
            score = base_scores.get(neighbor.memory_id, -99.0) + 0.35 * seed_score + config.graph_boost * neighbor.boost / (neighbor.hop + 1)
            if score > merged_scores.get(neighbor.memory_id, -99.0):
                merged_scores[neighbor.memory_id] = score
    selected_candidates = [by_id[memory_id] for memory_id in merged_scores if memory_id in by_id]
    frame = materialize_frame(row, selected_candidates, merged_scores, qvec, config)
    stats = {
        "cache_query_ms": (time.perf_counter() - started) * 1000.0,
        "seed_selection_ms": seed_selection_ms,
        "cache_hits": float(hits),
        "cache_misses": float(misses),
        "merged_candidates": float(len(merged_scores)),
        "scored_candidates": float(len(pending_score_ids)),
    }
    return frame, stats


def materialize_frame(
    row: dict[str, Any],
    candidates: list[dict[str, Any]],
    scores: dict[str, float],
    qvec: list[float],
    config: FrameBuilderConfig,
) -> dict[str, Any]:
    relevant = set((row.get("retrieval_task") or {}).get("relevant_ids") or [])
    ordered = sorted(candidates, key=lambda item: (-scores.get(item["id"], -99.0), item["id"]))
    selected = ordered[: config.frame_size]
    embeddings = []
    scalars = []
    labels = []
    mask = []
    ids = []
    qmask = activation_mask_for_text(query_text(row))
    for candidate in selected:
        ids.append(candidate["id"])
        embeddings.append(candidate.get("embedding") or [0.0] * DEFAULT_DIMS)
        scalars.append(frame_scalars(row, candidate, qvec, qmask, config.use_role_features))
        labels.append(1.0 if candidate["id"] in relevant else 0.0)
        mask.append(1.0)
    while len(embeddings) < config.frame_size:
        ids.append("")
        embeddings.append([0.0] * DEFAULT_DIMS)
        scalars.append([0.0] * FRAME_SCALARS)
        labels.append(0.0)
        mask.append(0.0)
    return {
        "ids": ids,
        "query": qvec,
        "anchor": row["anchor"].get("embedding") or [0.0] * DEFAULT_DIMS,
        "embeddings": embeddings,
        "scalars": scalars,
        "labels": labels,
        "mask": mask,
        "gold_total": float(len(relevant)),
    }


def frame_scalars(row: dict[str, Any], candidate: dict[str, Any], query_vec: list[float], query_mask: int, use_role_features: bool) -> list[float]:
    query = query_text(row)
    state = state_features(row["anchor"], candidate)
    text = candidate.get("text", "")
    candidate_mask = activation_mask_for_text(text)
    candidate_len = max(1, len(tokens(text)))
    role = str(candidate.get("synthetic_role") or "") if use_role_features else ""
    values = [
        cosine(query_vec, candidate.get("embedding") or []),
        jaccard(query, text),
        (query_mask & candidate_mask).bit_count() / max(1, (query_mask | candidate_mask).bit_count()),
        float(candidate.get("importance") or 0.5),
        clamp(candidate_len / 90.0, 0.0, 1.0),
        state["candidate_age_norm"],
        state["candidate_use_norm"],
        state["candidate_evidence_norm"],
        state["last_outcome_value"],
        state["protected_flag"],
        state["stale_unused_flag"],
        state["recency_score"],
        1.0 if "decoy" in role or "wrong" in role else 0.0,
        1.0 if "target" in role or "support" in role or "bridge" in role else 0.0,
    ]
    return [clamp(float(value), -1.0, 1.0) for value in values]


def frame_recall(frame: dict[str, Any]) -> float:
    total = max(1.0, float(frame.get("gold_total") or 0.0))
    return sum(1.0 for label in frame.get("labels", []) if float(label) >= 0.5) / total
