from __future__ import annotations

import math
import re
from typing import Any

DEFAULT_DIMS = 64
DEFAULT_FEATURE_DIMS = 16
EDGE_TYPES = [
    "same_cluster",
    "same_context",
    "preference",
    "correction",
    "same_topic",
    "temporal_next",
    "used_with",
]
EDGE_TYPE_TO_ID = {name: idx for idx, name in enumerate(EDGE_TYPES)}

TOKEN_RE = re.compile(r"[a-z0-9]+")
FNV_OFFSET = 14695981039346656037
FNV_PRIME = 1099511628211
UINT64_MASK = (1 << 64) - 1


def tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def token_set(text: str) -> set[str]:
    return {token for token in tokens(text) if len(token) > 1}


def fnv1a64(text: str) -> int:
    value = FNV_OFFSET
    for byte in text.encode("utf-8"):
        value ^= byte
        value = (value * FNV_PRIME) & UINT64_MASK
    return value


def embed_text(text: str, dims: int = DEFAULT_DIMS) -> list[float]:
    vec = [0.0] * dims
    for token in tokens(text):
        hashed = fnv1a64(token)
        idx = hashed % dims
        sign = -1.0 if ((hashed >> 8) & 1) else 1.0
        vec[idx] += sign
    return normalize(vec)


def normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vec))
    if norm == 0:
        return vec
    return [value / norm for value in vec]


def cosine(a: list[float], b: list[float]) -> float:
    if not a or len(a) != len(b):
        return 0.0
    return sum(left * right for left, right in zip(a, b))


def jaccard(left: str, right: str) -> float:
    a = token_set(left)
    b = token_set(right)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def activation_mask_for_text(text: str) -> int:
    mask = 0
    for token in token_set(text):
        mask |= 1 << (fnv1a64(token) % 64)
    return mask


def activation_mask_for_edge(anchor: dict[str, Any], candidate: dict[str, Any], edge_type: str) -> int:
    parts = [
        anchor.get("text", ""),
        anchor.get("summary", ""),
        anchor.get("cluster", ""),
        candidate.get("text", ""),
        candidate.get("summary", ""),
        candidate.get("cluster", ""),
        edge_type,
    ]
    for item in (anchor.get("metadata") or {}).items():
        parts.extend(item)
    for item in (candidate.get("metadata") or {}).items():
        parts.extend(item)
    return activation_mask_for_text(" ".join(parts))


def metadata_score(anchor: dict[str, Any], candidate: dict[str, Any]) -> float:
    left = anchor.get("metadata") or {}
    right = candidate.get("metadata") or {}
    if not left or not right:
        return 0.0
    total = 0
    matches = 0
    for key, value in left.items():
        total += 1
        if right.get(key) == value:
            matches += 1
    return matches / total if total else 0.0


def cluster_score(anchor: dict[str, Any], candidate: dict[str, Any]) -> float:
    left = anchor.get("cluster") or ""
    right = candidate.get("cluster") or ""
    return 1.0 if left and left == right else 0.0


def infer_edge_type(anchor: dict[str, Any], candidate: dict[str, Any], lexical: float, meta: float) -> str:
    if cluster_score(anchor, candidate) > 0:
        return "same_cluster"
    if meta > 0.5:
        return "same_context"
    text = f"{anchor.get('text', '')} {candidate.get('text', '')}".lower()
    if any(word in text for word in ("prefer", "preference", "like", "dislike", "always", "never")):
        return "preference"
    if any(word in text for word in ("correct", "instead", "actually", "wrong")):
        return "correction"
    if lexical > 0.18:
        return "same_topic"
    return "used_with"


def heuristic_action(anchor: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    ensure_embedding(anchor)
    ensure_embedding(candidate)
    semantic = cosine(anchor["embedding"], candidate["embedding"])
    lexical = jaccard(
        f"{anchor.get('text', '')} {anchor.get('summary', '')}",
        f"{candidate.get('text', '')} {candidate.get('summary', '')}",
    )
    meta = metadata_score(anchor, candidate)
    cluster = cluster_score(anchor, candidate)
    state = state_features(anchor, candidate)
    state_adjustment = (
        0.08 * state["candidate_use_norm"]
        + 0.06 * state["candidate_evidence_norm"]
        + 0.05 * state["protected_flag"]
        + 0.08 * max(0.0, state["last_outcome_value"])
        - 0.16 * state["stale_unused_flag"]
        - 0.10 * max(0.0, -state["last_outcome_value"])
    )
    score = 0.35 * semantic + 0.20 * lexical + 0.20 * meta + 0.15 * cluster + 0.05 + state_adjustment
    score = clamp(score, 0.0, 1.0)
    edge_type = infer_edge_type(anchor, candidate, lexical, meta)
    return {
        "candidate_id": candidate["id"],
        "connect_score": score,
        "edge_type": edge_type,
        "edge_type_id": EDGE_TYPE_TO_ID[edge_type],
        "weight": clamp(0.2 + score, 0.05, 1.2),
        "confidence": clamp(score, 0.05, 1.0),
        "activation_mask": activation_mask_for_edge(anchor, candidate, edge_type),
        "decay_rate": decay_for_edge(edge_type),
        "importance_delta": clamp((score - 0.5) * 0.08, -0.04, 0.08),
        "attach": 1.0 if score >= 0.28 else 0.0,
    }


def candidate_features(anchor: dict[str, Any], candidate: dict[str, Any], feature_dim: int = DEFAULT_FEATURE_DIMS) -> list[float]:
    ensure_embedding(anchor)
    ensure_embedding(candidate)
    semantic = cosine(anchor["embedding"], candidate["embedding"])
    lexical = jaccard(anchor.get("text", ""), candidate.get("text", ""))
    meta = metadata_score(anchor, candidate)
    cluster = cluster_score(anchor, candidate)
    anchor_importance = float(anchor.get("importance") or 0.5)
    candidate_importance = float(candidate.get("importance") or 0.5)
    anchor_len = max(1, len(tokens(anchor.get("text", ""))))
    candidate_len = max(1, len(tokens(candidate.get("text", ""))))
    state = state_features(anchor, candidate)
    features = [
        semantic,
        lexical,
        meta,
        cluster,
        anchor_importance,
        candidate_importance,
        min(anchor_len, candidate_len) / max(anchor_len, candidate_len),
        min(candidate_len / 64.0, 1.0),
        state["anchor_age_norm"],
        state["candidate_age_norm"],
        state["candidate_use_norm"],
        state["candidate_evidence_norm"],
        state["last_outcome_value"],
        state["protected_flag"],
        state["stale_unused_flag"],
        state["recency_score"],
    ]
    if feature_dim <= len(features):
        return features[:feature_dim]
    return features + [0.0] * (feature_dim - len(features))


def state_features(anchor: dict[str, Any], candidate: dict[str, Any]) -> dict[str, float]:
    anchor_age = numeric_value(anchor, "age_days", 0.0)
    candidate_age = numeric_value(candidate, "age_days", 0.0)
    use_count = numeric_value(candidate, "use_count", 0.0)
    evidence_count = numeric_value(candidate, "evidence_count", 0.0)
    last_outcome = str(candidate.get("last_outcome") or "").lower()
    last_outcome_value = {
        "helpful": 1.0,
        "corrected": 0.35,
        "ignored": -0.65,
    }.get(last_outcome, 0.0)
    protected = 1.0 if bool(candidate.get("protected")) else 0.0
    candidate_age_norm = clamp(candidate_age / 365.0, 0.0, 1.0)
    candidate_use_norm = clamp(math.log1p(use_count) / math.log1p(100.0), 0.0, 1.0)
    candidate_evidence_norm = clamp(math.log1p(evidence_count) / math.log1p(50.0), 0.0, 1.0)
    stale_unused = 1.0 if candidate_age >= 180.0 and use_count <= 0 and protected == 0.0 else 0.0
    return {
        "anchor_age_norm": clamp(anchor_age / 365.0, 0.0, 1.0),
        "candidate_age_norm": candidate_age_norm,
        "candidate_use_norm": candidate_use_norm,
        "candidate_evidence_norm": candidate_evidence_norm,
        "last_outcome_value": last_outcome_value,
        "protected_flag": protected,
        "stale_unused_flag": stale_unused,
        "recency_score": 1.0 - candidate_age_norm,
    }


def numeric_value(card: dict[str, Any], key: str, default: float) -> float:
    value = card.get(key)
    if value is None:
        metadata = card.get("metadata") or {}
        value = metadata.get(key)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def ensure_embedding(card: dict[str, Any], dims: int = DEFAULT_DIMS) -> None:
    if not card.get("embedding"):
        card["embedding"] = embed_text(f"{card.get('text', '')} {card.get('summary', '')}", dims)


def decay_for_edge(edge_type: str) -> float:
    if edge_type in ("preference", "correction"):
        return 0.005
    if edge_type in ("same_cluster", "same_context"):
        return 0.01
    if edge_type == "temporal_next":
        return 0.03
    return 0.02


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
