from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from python.librarian.features import DEFAULT_DIMS, clamp, jaccard, state_features, tokens


@dataclass
class HippoCalibratorConfig:
    embedding_dim: int = DEFAULT_DIMS
    feature_dim: int = 16
    d_model: int = 128
    num_layers: int = 3
    num_heads: int = 4
    dropout: float = 0.1
    max_candidates: int = 128


class HippoCalibrationTransformer(nn.Module):
    def __init__(self, config: HippoCalibratorConfig):
        super().__init__()
        self.config = config
        token_dim = config.embedding_dim * 4 + config.feature_dim
        self.input = nn.Sequential(
            nn.Linear(token_dim, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.GELU(),
        )
        self.encoder: nn.Module | None = None
        if config.num_layers > 0:
            layer = nn.TransformerEncoderLayer(
                d_model=config.d_model,
                nhead=config.num_heads,
                dim_feedforward=config.d_model * 4,
                dropout=config.dropout,
                batch_first=True,
                activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=config.num_layers)
        self.relevance_head = nn.Linear(config.d_model, 1)
        self.include_head = nn.Linear(config.d_model, 1)
        self.utility_head = nn.Linear(config.d_model, 1)

    def forward(
        self,
        query: torch.Tensor,
        candidates: torch.Tensor,
        features: torch.Tensor,
        mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        query_expanded = query.unsqueeze(1).expand(-1, candidates.shape[1], -1)
        diff = candidates - query_expanded
        product = candidates * query_expanded
        hidden = self.input(torch.cat([query_expanded, candidates, diff, product, features], dim=-1))
        if self.encoder is not None:
            hidden = self.encoder(hidden, src_key_padding_mask=~mask.bool())
        return {
            "relevance_logits": self.relevance_head(hidden).squeeze(-1),
            "include_logits": self.include_head(hidden).squeeze(-1),
            "utility": torch.tanh(self.utility_head(hidden).squeeze(-1)),
        }


def calibration_features(query: str, candidate: dict[str, Any], rank: int, score: float, budget: int, feature_dim: int) -> list[float]:
    text = str(candidate.get("text") or "")
    candidate_len = max(1, len(tokens(text)))
    query_len = max(1, len(tokens(query)))
    metadata = candidate.get("metadata") or {}
    state = state_features({}, candidate)
    outcome = state["last_outcome_value"]
    lower_text = text.lower()
    conflict_terms = ("decoy", "wrong", "conflict", "contradict", "superseded", "obsolete", "ignored")
    values = [
        clamp(float(score), -2.0, 2.0) / 2.0,
        1.0 / float(max(1, rank)),
        clamp(rank / 256.0, 0.0, 1.0),
        jaccard(query, text),
        min(query_len, candidate_len) / max(query_len, candidate_len),
        min(candidate_len / max(1, budget), 1.0),
        min(candidate_len / 128.0, 1.0),
        float(candidate.get("importance") or 0.5),
        clamp(float(candidate.get("use_count") or 0.0) / 32.0, 0.0, 1.0),
        clamp(float(candidate.get("evidence_count") or 0.0) / 16.0, 0.0, 1.0),
        1.0 if metadata.get("session_id") else 0.0,
        1.0 if metadata.get("turn_id") else 0.0,
        1.0 if metadata.get("has_answer") else 0.0,
        1.0 if metadata.get("speaker") else 0.0,
        1.0 if metadata.get("timestamp") else 0.0,
        1.0 if candidate.get("cluster") else 0.0,
        state["candidate_age_norm"],
        state["stale_unused_flag"],
        state["recency_score"],
        max(0.0, -outcome),
        max(0.0, outcome),
        1.0 if any(term in lower_text for term in conflict_terms) else 0.0,
        clamp(jaccard(query, str(candidate.get("summary") or "")), 0.0, 1.0),
        clamp(float(candidate.get("base_score_gap") or 0.0), -2.0, 2.0) / 2.0,
    ]
    if feature_dim <= len(values):
        return values[:feature_dim]
    return values + [0.0] * (feature_dim - len(values))


def apply_feature_ablation(values: list[float], ablation: str) -> list[float]:
    out = list(values)

    def zero_range(start: int, stop: int) -> None:
        for index in range(start, min(stop, len(out))):
            out[index] = 0.0

    def zero_index(index: int) -> None:
        if 0 <= index < len(out):
            out[index] = 0.0

    mode = (ablation or "none").strip().lower()
    if mode in {"", "none"}:
        return out
    if mode in {"metadata", "no_metadata"}:
        zero_range(10, 15)
    elif mode in {"state", "no_state"}:
        zero_index(8)
        zero_index(9)
        zero_range(16, 21)
    elif mode in {"state_metadata", "shortcut", "shortcuts", "no_shortcuts"}:
        zero_range(8, 16)
        zero_range(16, 21)
        zero_index(21)
    elif mode in {"conflict_terms", "no_conflict_terms"}:
        zero_index(21)
    else:
        raise ValueError(f"unknown calibrator feature ablation: {ablation}")
    return out


def tensorize_calibration_payload(
    payload: dict[str, Any],
    max_candidates: int,
    feature_dim: int,
    embedding_dim: int,
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]]]:
    query = str(payload.get("query") or "")
    query_embedding = [float(value) for value in payload.get("query_embedding") or []]
    query_embedding = fit_embedding(query_embedding, embedding_dim)
    candidates = [dict(item) for item in payload.get("candidates", [])[:max_candidates]]
    feature_ablation = str(payload.get("feature_ablation") or "none")
    candidate_tensor = torch.zeros((1, max_candidates, embedding_dim), dtype=torch.float32)
    feature_tensor = torch.zeros((1, max_candidates, feature_dim), dtype=torch.float32)
    mask = torch.zeros((1, max_candidates), dtype=torch.bool)
    for idx, candidate in enumerate(candidates):
        candidate_tensor[0, idx] = torch.tensor(fit_embedding(candidate.get("embedding") or [], embedding_dim), dtype=torch.float32)
        feature_tensor[0, idx] = torch.tensor(
            apply_feature_ablation(
                calibration_features(
                    query,
                    candidate,
                    int(candidate.get("base_rank") or idx + 1),
                    float(candidate.get("base_score") or 0.0),
                    int(payload.get("budget") or 900),
                    feature_dim,
                ),
                feature_ablation,
            ),
            dtype=torch.float32,
        )
        mask[0, idx] = True
    return {
        "query": torch.tensor(query_embedding, dtype=torch.float32).unsqueeze(0),
        "candidates": candidate_tensor,
        "features": feature_tensor,
        "mask": mask,
    }, candidates


def fit_embedding(values: list[float], dim: int) -> list[float]:
    out = [float(value) for value in values[:dim]]
    if len(out) < dim:
        out.extend([0.0] * (dim - len(out)))
    return out


def rerank_with_calibrator(
    model: HippoCalibrationTransformer,
    payload: dict[str, Any],
    *,
    max_candidates: int | None = None,
    relevance_weight: float | None = None,
    include_weight: float | None = None,
    base_weight: float | None = None,
    utility_weight: float | None = None,
) -> list[tuple[str, float, str]]:
    metadata = getattr(model, "metadata", {}) or {}
    has_trained_include_head = bool(getattr(model, "has_trained_include_head", True))
    if relevance_weight is None:
        relevance_weight = float(metadata.get("rerank_relevance_weight", 0.35 if has_trained_include_head else 0.90))
    if include_weight is None:
        include_weight = float(metadata.get("rerank_include_weight", 0.60 if has_trained_include_head else 0.0))
    if base_weight is None:
        base_weight = float(metadata.get("rerank_base_weight", 0.05))
    if utility_weight is None:
        utility_weight = float(metadata.get("rerank_utility_weight", 0.05))
    with torch.no_grad():
        device = next(model.parameters()).device
        tensors, candidates = tensorize_calibration_payload(
            payload,
            int(max_candidates or model.config.max_candidates),
            model.config.feature_dim,
            model.config.embedding_dim,
        )
        tensors = {key: value.to(device) for key, value in tensors.items()}
        outputs = model(**tensors)
        logits = outputs["relevance_logits"][0].detach().cpu().tolist()
        include_logits = outputs.get("include_logits", outputs["relevance_logits"])[0].detach().cpu().tolist()
        utility = outputs["utility"][0].detach().cpu().tolist()
    ranked = []
    for idx, candidate in enumerate(candidates):
        base_score = float(candidate.get("base_score") or 0.0)
        score = (
            relevance_weight * float(logits[idx])
            + include_weight * float(include_logits[idx])
            + base_weight * float(base_score)
            + utility_weight * float(utility[idx])
        )
        ranked.append((str(candidate.get("id") or ""), score, str(candidate.get("text") or "")))
    return sorted(ranked, key=lambda item: (-item[1], item[0]))


def save_calibrator(model: HippoCalibrationTransformer, path: str | Path, **metadata: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": asdict(model.config), "state_dict": model.state_dict(), "metadata": metadata}, path)


def load_calibrator(path: str | Path, device: torch.device | str = "cpu") -> HippoCalibrationTransformer:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    model = HippoCalibrationTransformer(HippoCalibratorConfig(**checkpoint["config"])).to(device)
    state_dict = checkpoint["state_dict"]
    model.has_trained_include_head = any(str(key).startswith("include_head.") for key in state_dict)
    model.load_state_dict(state_dict, strict=True)
    model.metadata = dict(checkpoint.get("metadata") or {})
    model.eval()
    return model
