from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from python.librarian.features import DEFAULT_DIMS, clamp, cosine, jaccard, tokens
from python.librarian.hippo_calibrator import (
    apply_feature_ablation,
    calibration_features,
    fit_embedding,
)


RELATION_TYPES = [
    "higher_rank",
    "same_user",
    "same_project",
    "same_brand",
    "same_session",
    "temporal_neighbor",
    "correction",
    "supersedes",
    "lexical_neighbor",
    "vector_neighbor",
    "same_source",
    "field_overlap",
]
RELATION_TYPE_TO_ID = {name: idx for idx, name in enumerate(RELATION_TYPES)}


@dataclass
class RelationalFrameCalibratorConfig:
    embedding_dim: int = DEFAULT_DIMS
    node_feature_dim: int = 48
    edge_feature_dim: int = 24
    node_frame_dim: int = 256
    small_edge_dim: int = 64
    large_edge_dim: int = 256
    d_model: int = 128
    edge_layers: int = 2
    candidate_layers: int = 0
    num_heads: int = 4
    dropout: float = 0.1
    max_candidates: int = 256
    max_edges_per_candidate: int = 16
    use_edges: bool = True


class NodeSummaryEncoder(nn.Module):
    def __init__(self, config: RelationalFrameCalibratorConfig):
        super().__init__()
        token_dim = config.embedding_dim * 4 + config.node_feature_dim
        self.net = nn.Sequential(
            nn.Linear(token_dim, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.node_frame_dim),
            nn.LayerNorm(config.node_frame_dim),
            nn.GELU(),
        )

    def forward(self, query: torch.Tensor, candidates: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        query_expanded = query.unsqueeze(1).expand(-1, candidates.shape[1], -1)
        diff = candidates - query_expanded
        product = candidates * query_expanded
        return self.net(torch.cat([query_expanded, candidates, diff, product, features], dim=-1))


class EdgeFrameEncoder(nn.Module):
    def __init__(self, config: RelationalFrameCalibratorConfig):
        super().__init__()
        token_dim = config.node_frame_dim * 2 + config.edge_feature_dim
        self.input = nn.Sequential(
            nn.Linear(token_dim, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.GELU(),
        )
        self.encoder: nn.Module | None = None
        if config.edge_layers > 0:
            layer = nn.TransformerEncoderLayer(
                d_model=config.d_model,
                nhead=config.num_heads,
                dim_feedforward=config.d_model * 4,
                dropout=config.dropout,
                batch_first=True,
                activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=config.edge_layers)
        self.small = nn.Sequential(nn.Linear(config.d_model, config.small_edge_dim), nn.LayerNorm(config.small_edge_dim), nn.GELU())
        self.large = nn.Sequential(nn.Linear(config.d_model, config.large_edge_dim), nn.LayerNorm(config.large_edge_dim), nn.GELU())
        self.aux_head = nn.Linear(config.d_model, 1)

    def forward(
        self,
        source_frames: torch.Tensor,
        neighbor_frames: torch.Tensor,
        edge_features: torch.Tensor,
        edge_mask: torch.Tensor,
        group_size: int = 0,
    ) -> dict[str, torch.Tensor]:
        hidden = self.input(torch.cat([source_frames, neighbor_frames, edge_features], dim=-1))
        if self.encoder is not None and hidden.shape[1] > 0:
            if group_size > 0 and hidden.shape[1] % group_size == 0:
                batch_size, edge_count, width = hidden.shape
                grouped_hidden = hidden.reshape(batch_size * (edge_count // group_size), group_size, width)
                grouped_mask = edge_mask.reshape(batch_size * (edge_count // group_size), group_size).bool()
                valid_groups = grouped_mask.any(dim=1)
                encoded = grouped_hidden.clone()
                if valid_groups.any():
                    encoded[valid_groups] = self.encoder(
                        grouped_hidden[valid_groups],
                        src_key_padding_mask=~grouped_mask[valid_groups],
                    )
                hidden = encoded.reshape(batch_size, edge_count, width)
            else:
                hidden = self.encoder(hidden, src_key_padding_mask=~edge_mask.bool())
        return {
            "small_edge_frame": self.small(hidden),
            "large_edge_frame": self.large(hidden),
            "edge_aux_logits": self.aux_head(hidden).squeeze(-1),
        }


class RelationalFrameCalibrator(nn.Module):
    def __init__(self, config: RelationalFrameCalibratorConfig):
        super().__init__()
        self.config = config
        self.node_encoder = NodeSummaryEncoder(config)
        self.edge_encoder = EdgeFrameEncoder(config)
        self.small_pool = nn.Linear(config.small_edge_dim, config.d_model)
        score_dim = config.node_frame_dim + config.large_edge_dim + config.d_model
        self.candidate_encoder: nn.Module | None = None
        if config.candidate_layers > 0:
            layer = nn.TransformerEncoderLayer(
                d_model=score_dim,
                nhead=config.num_heads,
                dim_feedforward=score_dim * 2,
                dropout=config.dropout,
                batch_first=True,
                activation="gelu",
            )
            self.candidate_encoder = nn.TransformerEncoder(layer, num_layers=config.candidate_layers)
        self.scorer = nn.Sequential(
            nn.Linear(score_dim, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.relevance_head = nn.Linear(config.d_model, 1)
        self.include_head = nn.Linear(config.d_model, 1)
        self.utility_head = nn.Linear(config.d_model, 1)

    def forward(
        self,
        query: torch.Tensor,
        candidates: torch.Tensor,
        node_features: torch.Tensor,
        mask: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
        edge_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        node_frames = self.node_encoder(query, candidates, node_features)
        batch_size, max_candidates, _ = node_frames.shape
        max_edges = edge_index.shape[1]
        large_pool = torch.zeros(
            (batch_size, max_candidates, self.config.large_edge_dim),
            dtype=node_frames.dtype,
            device=node_frames.device,
        )
        small_pool = torch.zeros(
            (batch_size, max_candidates, self.config.small_edge_dim),
            dtype=node_frames.dtype,
            device=node_frames.device,
        )
        edge_counts = torch.zeros((batch_size, max_candidates, 1), dtype=node_frames.dtype, device=node_frames.device)
        edge_aux_logits = torch.zeros((batch_size, max_edges), dtype=node_frames.dtype, device=node_frames.device)
        if self.config.use_edges and max_edges > 0 and edge_mask.bool().any():
            source_index = edge_index[:, :, 0].clamp(0, max_candidates - 1)
            neighbor_index = edge_index[:, :, 1].clamp(0, max_candidates - 1)
            gather_shape = (-1, -1, node_frames.shape[-1])
            source_frames = torch.gather(node_frames, 1, source_index.unsqueeze(-1).expand(gather_shape))
            neighbor_frames = torch.gather(node_frames, 1, neighbor_index.unsqueeze(-1).expand(gather_shape))
            edge_outputs = self.edge_encoder(
                source_frames,
                neighbor_frames,
                edge_features,
                edge_mask,
                group_size=max(0, int(self.config.max_edges_per_candidate)),
            )
            large_edges = edge_outputs["large_edge_frame"] * edge_mask.unsqueeze(-1).to(node_frames.dtype)
            small_edges = edge_outputs["small_edge_frame"] * edge_mask.unsqueeze(-1).to(node_frames.dtype)
            edge_aux_logits = edge_outputs["edge_aux_logits"]
            for batch_index in range(batch_size):
                valid = edge_mask[batch_index].bool()
                if not valid.any():
                    continue
                targets = source_index[batch_index, valid]
                large_pool[batch_index].index_add_(0, targets, large_edges[batch_index, valid])
                small_pool[batch_index].index_add_(0, targets, small_edges[batch_index, valid])
                ones = torch.ones((int(valid.sum().item()), 1), dtype=node_frames.dtype, device=node_frames.device)
                edge_counts[batch_index].index_add_(0, targets, ones)
            edge_counts = edge_counts.clamp_min(1.0)
            large_pool = large_pool / edge_counts
            small_pool = small_pool / edge_counts
        hidden = torch.cat([node_frames, large_pool, self.small_pool(small_pool)], dim=-1)
        if self.candidate_encoder is not None:
            hidden = self.candidate_encoder(hidden, src_key_padding_mask=~mask.bool())
        hidden = self.scorer(hidden)
        return {
            "relevance_logits": self.relevance_head(hidden).squeeze(-1),
            "include_logits": self.include_head(hidden).squeeze(-1),
            "utility": torch.tanh(self.utility_head(hidden).squeeze(-1)),
            "summary_frame": node_frames,
            "edge_aux_logits": edge_aux_logits,
        }


def candidate_sources(candidate: dict[str, Any]) -> set[str]:
    return {str(source) for source in candidate.get("candidate_sources") or []}


def derived_values(candidate: dict[str, Any]) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    metadata = candidate.get("metadata") or {}
    for key in ("user_id", "project", "brand", "session_id"):
        value = metadata.get(key)
        if value:
            out.setdefault(key, set()).add(str(value).lower())
    for value in metadata.get("entities") or []:
        if value:
            out.setdefault("entity", set()).add(str(value).lower())
    for key, entries in (((candidate.get("derived_metadata") or {}).get("fields") or {}).items()):
        for entry in entries or []:
            if isinstance(entry, dict) and entry.get("value") is not None:
                field_name = str(key)
                if field_name == "entities":
                    field_name = "entity"
                out.setdefault(field_name, set()).add(str(entry["value"]).lower())
    return out


def field_overlap(left: dict[str, Any], right: dict[str, Any]) -> float:
    left_values = derived_values(left)
    right_values = derived_values(right)
    keys = set(left_values) | set(right_values)
    if not keys:
        return 0.0
    hits = 0
    total = 0
    for key in keys:
        a = left_values.get(key, set())
        b = right_values.get(key, set())
        if a or b:
            total += 1
            if a & b:
                hits += 1
    return hits / max(1, total)


def same_field(left: dict[str, Any], right: dict[str, Any], field: str) -> bool:
    return bool(derived_values(left).get(field, set()) & derived_values(right).get(field, set()))


def turn_number(candidate: dict[str, Any]) -> int | None:
    turn = str((candidate.get("metadata") or {}).get("turn_id") or "")
    digits = ""
    for char in reversed(turn):
        if char.isdigit():
            digits = char + digits
        elif digits:
            break
    if digits:
        return int(digits)
    return None


def timestamp_seconds(candidate: dict[str, Any]) -> float | None:
    raw = str((candidate.get("metadata") or {}).get("timestamp") or "")
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def temporal_gap(left: dict[str, Any], right: dict[str, Any]) -> float:
    left_turn = turn_number(left)
    right_turn = turn_number(right)
    if left_turn is not None and right_turn is not None:
        return float(abs(left_turn - right_turn))
    left_ts = timestamp_seconds(left)
    right_ts = timestamp_seconds(right)
    if left_ts is not None and right_ts is not None:
        return abs(left_ts - right_ts) / 3600.0
    return 0.0


def correction_like(candidate: dict[str, Any]) -> bool:
    text = str(candidate.get("text") or "").lower()
    fields = (candidate.get("derived_metadata") or {}).get("fields") or {}
    return "correction" in fields or any(term in text for term in ("correct", "instead", "superseded", "override", "wrong"))


def edge_feature_values(
    source: dict[str, Any],
    neighbor: dict[str, Any],
    relation_type: str,
    max_candidates: int,
    edge_feature_dim: int,
) -> list[float]:
    source_rank = int(source.get("base_rank") or 0)
    neighbor_rank = int(neighbor.get("base_rank") or 0)
    source_score = clamp(float(source.get("base_score") or 0.0), -2.0, 2.0) / 2.0
    neighbor_score = clamp(float(neighbor.get("base_score") or 0.0), -2.0, 2.0) / 2.0
    source_embedding = fit_embedding(source.get("embedding") or [], len(neighbor.get("embedding") or source.get("embedding") or []))
    neighbor_embedding = fit_embedding(neighbor.get("embedding") or [], len(source_embedding))
    source_sources = candidate_sources(source)
    neighbor_sources = candidate_sources(neighbor)
    gap = temporal_gap(source, neighbor)
    relation_id = RELATION_TYPE_TO_ID.get(relation_type, 0)
    values = [
        clamp(source_rank / max(1, max_candidates), 0.0, 1.0),
        clamp(neighbor_rank / max(1, max_candidates), 0.0, 1.0),
        clamp((source_rank - neighbor_rank) / max(1, max_candidates), -1.0, 1.0),
        source_score,
        neighbor_score,
        clamp(source_score - neighbor_score, -1.0, 1.0),
        cosine(source_embedding, neighbor_embedding),
        jaccard(str(source.get("text") or ""), str(neighbor.get("text") or "")),
        field_overlap(source, neighbor),
        1.0 if same_field(source, neighbor, "user_id") else 0.0,
        1.0 if same_field(source, neighbor, "project") else 0.0,
        1.0 if same_field(source, neighbor, "brand") else 0.0,
        1.0 if same_field(source, neighbor, "session_id") else 0.0,
        1.0 if bool(source_sources & neighbor_sources) else 0.0,
        1.0 if "vector" in source_sources and "vector" in neighbor_sources else 0.0,
        1.0 if "token" in source_sources and "token" in neighbor_sources else 0.0,
        1.0 if any(item.startswith("metadata:") for item in source_sources) and any(item.startswith("metadata:") for item in neighbor_sources) else 0.0,
        1.0 if any(item.startswith("graph:") for item in source_sources) and any(item.startswith("graph:") for item in neighbor_sources) else 0.0,
        1.0 if any(item.startswith("derived_metadata:") for item in source_sources) and any(item.startswith("derived_metadata:") for item in neighbor_sources) else 0.0,
        clamp(gap / 32.0, 0.0, 1.0),
        1.0 if neighbor_rank and source_rank and neighbor_rank < source_rank else 0.0,
        1.0 if relation_type == "temporal_neighbor" else 0.0,
        1.0 if correction_like(source) or correction_like(neighbor) else 0.0,
        relation_id / max(1, len(RELATION_TYPES) - 1),
    ]
    if edge_feature_dim <= len(values):
        return values[:edge_feature_dim]
    return values + [0.0] * (edge_feature_dim - len(values))


def relation_priority(relation_type: str) -> int:
    return {
        "correction": 0,
        "supersedes": 1,
        "same_session": 2,
        "temporal_neighbor": 3,
        "same_project": 4,
        "same_user": 5,
        "same_brand": 6,
        "field_overlap": 7,
        "vector_neighbor": 8,
        "lexical_neighbor": 9,
        "same_source": 10,
        "higher_rank": 11,
    }.get(relation_type, 99)


def build_candidate_edges(
    candidates: list[dict[str, Any]],
    *,
    max_edges_per_candidate: int = 16,
    lexical_neighbors: int = 4,
    vector_neighbors: int = 4,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    embeddings = [fit_embedding(candidate.get("embedding") or [], len(candidate.get("embedding") or [])) for candidate in candidates]
    lexical_by_target: dict[int, list[tuple[float, int]]] = {idx: [] for idx in range(len(candidates))}
    vector_by_target: dict[int, list[tuple[float, int]]] = {idx: [] for idx in range(len(candidates))}
    for left_index, left in enumerate(candidates):
        for right_index, right in enumerate(candidates):
            if left_index == right_index:
                continue
            lexical_by_target[left_index].append((jaccard(str(left.get("text") or ""), str(right.get("text") or "")), right_index))
            vector_by_target[left_index].append((cosine(embeddings[left_index], embeddings[right_index]), right_index))
    for values in lexical_by_target.values():
        values.sort(key=lambda item: (-item[0], int(candidates[item[1]].get("base_rank") or item[1] + 1), str(candidates[item[1]].get("id") or "")))
    for values in vector_by_target.values():
        values.sort(key=lambda item: (-item[0], int(candidates[item[1]].get("base_rank") or item[1] + 1), str(candidates[item[1]].get("id") or "")))

    for source_index, source in enumerate(candidates):
        proposals: dict[int, tuple[str, int, float]] = {}

        def add(neighbor_index: int, relation_type: str, strength: float = 1.0) -> None:
            if neighbor_index == source_index or not (0 <= neighbor_index < len(candidates)):
                return
            priority = relation_priority(relation_type)
            current = proposals.get(neighbor_index)
            key = (priority, -float(strength), int(candidates[neighbor_index].get("base_rank") or neighbor_index + 1), str(candidates[neighbor_index].get("id") or ""))
            if current is None:
                proposals[neighbor_index] = (relation_type, priority, float(strength))
                return
            current_key = (current[1], -current[2], int(candidates[neighbor_index].get("base_rank") or neighbor_index + 1), str(candidates[neighbor_index].get("id") or ""))
            if key < current_key:
                proposals[neighbor_index] = (relation_type, priority, float(strength))

        source_rank = int(source.get("base_rank") or source_index + 1)
        for neighbor_index, neighbor in enumerate(candidates):
            neighbor_rank = int(neighbor.get("base_rank") or neighbor_index + 1)
            if neighbor_rank < source_rank:
                add(neighbor_index, "higher_rank", 1.0 / max(1, source_rank - neighbor_rank))
            if same_field(source, neighbor, "session_id"):
                add(neighbor_index, "same_session", 1.0)
            if same_field(source, neighbor, "user_id"):
                add(neighbor_index, "same_user", 0.8)
            if same_field(source, neighbor, "project"):
                add(neighbor_index, "same_project", 0.9)
            if same_field(source, neighbor, "brand"):
                add(neighbor_index, "same_brand", 0.7)
            if same_field(source, neighbor, "entity"):
                add(neighbor_index, "field_overlap", 0.8)
            overlap = field_overlap(source, neighbor)
            if overlap > 0.0:
                add(neighbor_index, "field_overlap", overlap)
            if candidate_sources(source) & candidate_sources(neighbor):
                add(neighbor_index, "same_source", 0.4)
            if correction_like(source) or correction_like(neighbor):
                if same_field(source, neighbor, "project") or same_field(source, neighbor, "user_id"):
                    add(neighbor_index, "correction", 1.0)
                if "superseded" in f"{source.get('text', '')} {neighbor.get('text', '')}".lower():
                    add(neighbor_index, "supersedes", 1.0)

        same_session = [
            (temporal_gap(source, neighbor), neighbor_index)
            for neighbor_index, neighbor in enumerate(candidates)
            if neighbor_index != source_index and same_field(source, neighbor, "session_id")
        ]
        same_session.sort(key=lambda item: (item[0], int(candidates[item[1]].get("base_rank") or item[1] + 1), str(candidates[item[1]].get("id") or "")))
        for _, neighbor_index in same_session[:2]:
            add(neighbor_index, "temporal_neighbor", 1.0)

        for score, neighbor_index in lexical_by_target[source_index][:lexical_neighbors]:
            if score > 0.0:
                add(neighbor_index, "lexical_neighbor", score)
        for score, neighbor_index in vector_by_target[source_index][:vector_neighbors]:
            if score > 0.0:
                add(neighbor_index, "vector_neighbor", score)

        ordered = sorted(
            proposals.items(),
            key=lambda item: (item[1][1], -item[1][2], int(candidates[item[0]].get("base_rank") or item[0] + 1), str(candidates[item[0]].get("id") or "")),
        )
        for neighbor_index, (relation_type, _, strength) in ordered[: max(0, int(max_edges_per_candidate))]:
            rows.append(
                {
                    "source_index": source_index,
                    "neighbor_index": neighbor_index,
                    "source_id": str(source.get("id") or ""),
                    "neighbor_id": str(candidates[neighbor_index].get("id") or ""),
                    "relation_type": relation_type,
                    "strength": float(strength),
                }
            )
    return rows


def tensorize_relational_payload(
    payload: dict[str, Any],
    max_candidates: int,
    node_feature_dim: int,
    embedding_dim: int,
    edge_feature_dim: int,
    max_edges_per_candidate: int,
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]], list[dict[str, Any]]]:
    query = str(payload.get("query") or "")
    query_embedding = fit_embedding([float(value) for value in payload.get("query_embedding") or []], embedding_dim)
    candidates = [dict(item) for item in payload.get("candidates", [])[:max_candidates]]
    feature_ablation = str(payload.get("feature_ablation") or "none")
    candidate_tensor = torch.zeros((1, max_candidates, embedding_dim), dtype=torch.float32)
    node_feature_tensor = torch.zeros((1, max_candidates, node_feature_dim), dtype=torch.float32)
    mask = torch.zeros((1, max_candidates), dtype=torch.bool)
    for idx, candidate in enumerate(candidates):
        candidate_tensor[0, idx] = torch.tensor(fit_embedding(candidate.get("embedding") or [], embedding_dim), dtype=torch.float32)
        node_feature_tensor[0, idx] = torch.tensor(
            apply_feature_ablation(
                calibration_features(
                    query,
                    candidate,
                    int(candidate.get("base_rank") or idx + 1),
                    float(candidate.get("base_score") or 0.0),
                    int(payload.get("budget") or 900),
                    node_feature_dim,
                ),
                feature_ablation,
            ),
            dtype=torch.float32,
        )
        mask[0, idx] = True
    if max_edges_per_candidate > 0:
        edges = build_candidate_edges(candidates, max_edges_per_candidate=max_edges_per_candidate)
    else:
        edges = []
    max_edges = max(1, max_candidates * max(0, int(max_edges_per_candidate)))
    edge_index = torch.zeros((1, max_edges, 2), dtype=torch.long)
    edge_features = torch.zeros((1, max_edges, edge_feature_dim), dtype=torch.float32)
    edge_mask = torch.zeros((1, max_edges), dtype=torch.bool)
    slot_counts = [0] * max_candidates
    for edge in edges:
        source_index = int(edge["source_index"])
        neighbor_index = int(edge["neighbor_index"])
        if not (0 <= source_index < max_candidates):
            continue
        slot = slot_counts[source_index]
        if slot >= max_edges_per_candidate:
            continue
        idx = source_index * max_edges_per_candidate + slot if max_edges_per_candidate > 0 else 0
        if idx >= max_edges:
            continue
        slot_counts[source_index] += 1
        edge_index[0, idx, 0] = source_index
        edge_index[0, idx, 1] = neighbor_index
        edge_features[0, idx] = torch.tensor(
            edge_feature_values(
                candidates[source_index],
                candidates[neighbor_index],
                str(edge["relation_type"]),
                max_candidates,
                edge_feature_dim,
            ),
            dtype=torch.float32,
        )
        edge_mask[0, idx] = True
    return {
        "query": torch.tensor(query_embedding, dtype=torch.float32).unsqueeze(0),
        "candidates": candidate_tensor,
        "node_features": node_feature_tensor,
        "mask": mask,
        "edge_index": edge_index,
        "edge_features": edge_features,
        "edge_mask": edge_mask,
    }, candidates, edges


def score_with_relational_calibrator(
    model: RelationalFrameCalibrator,
    payload: dict[str, Any],
    *,
    max_candidates: int | None = None,
    relevance_weight: float | None = None,
    include_weight: float | None = None,
    base_weight: float | None = None,
    utility_weight: float | None = None,
) -> list[dict[str, Any]]:
    metadata = getattr(model, "metadata", {}) or {}
    if relevance_weight is None:
        relevance_weight = float(metadata.get("rerank_relevance_weight", 0.30))
    if include_weight is None:
        include_weight = float(metadata.get("rerank_include_weight", 0.65))
    if base_weight is None:
        base_weight = float(metadata.get("rerank_base_weight", 0.03))
    if utility_weight is None:
        utility_weight = float(metadata.get("rerank_utility_weight", 0.02))
    with torch.no_grad():
        device = next(model.parameters()).device
        tensors, candidates, _ = tensorize_relational_payload(
            payload,
            int(max_candidates or model.config.max_candidates),
            model.config.node_feature_dim,
            model.config.embedding_dim,
            model.config.edge_feature_dim,
            model.config.max_edges_per_candidate if model.config.use_edges else 0,
        )
        tensors = {key: value.to(device) for key, value in tensors.items()}
        outputs = model(**tensors)
        logits = outputs["relevance_logits"][0].detach().cpu().tolist()
        include_logits = outputs["include_logits"][0].detach().cpu().tolist()
        utility = outputs["utility"][0].detach().cpu().tolist()
    ranked: list[dict[str, Any]] = []
    for idx, candidate in enumerate(candidates):
        base_score = float(candidate.get("base_score") or 0.0)
        include_logit = float(include_logits[idx])
        score = (
            relevance_weight * float(logits[idx])
            + include_weight * include_logit
            + base_weight * base_score
            + utility_weight * float(utility[idx])
        )
        ranked.append(
            {
                "id": str(candidate.get("id") or ""),
                "score": score,
                "text": str(candidate.get("text") or ""),
                "relevance_logit": float(logits[idx]),
                "include_logit": include_logit,
                "include_probability": float(torch.sigmoid(torch.tensor(include_logit)).item()),
                "utility": float(utility[idx]),
                "base_score": base_score,
            }
        )
    return sorted(ranked, key=lambda item: (-float(item["score"]), str(item["id"])))


def rerank_with_relational_calibrator(
    model: RelationalFrameCalibrator,
    payload: dict[str, Any],
    *,
    max_candidates: int | None = None,
) -> list[tuple[str, float, str]]:
    ranked = score_with_relational_calibrator(model, payload, max_candidates=max_candidates)
    return [(item["id"], item["score"], item["text"]) for item in ranked]


def save_relational_calibrator(model: RelationalFrameCalibrator, path: str | Path, **metadata: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": asdict(model.config), "state_dict": model.state_dict(), "metadata": metadata}, path)


def load_relational_calibrator(path: str | Path, device: torch.device | str = "cpu") -> RelationalFrameCalibrator:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    model = RelationalFrameCalibrator(RelationalFrameCalibratorConfig(**checkpoint["config"])).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.metadata = dict(checkpoint.get("metadata") or {})
    model.eval()
    return model
