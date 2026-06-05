from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from python.field_memory.token_field import FieldToken, bucket_for


@dataclass
class TokenFieldEncoderConfig:
    embedding_dim: int = 1024
    action_count: int = 512
    bucket_count: int = 65
    hidden_dim: int = 512
    dropout: float = 0.1
    query_token_count: int = 48
    node_token_count: int = 48
    bucket_width: float = 0.055


class TokenFieldEncoder(nn.Module):
    def __init__(self, config: TokenFieldEncoderConfig):
        super().__init__()
        self.config = config
        self.input = nn.Sequential(
            nn.Linear(config.embedding_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
        )
        self.query_action = nn.Linear(config.hidden_dim, config.action_count)
        self.node_action = nn.Linear(config.hidden_dim, config.action_count)
        self.query_bucket = nn.Linear(config.hidden_dim, config.action_count * config.bucket_count)
        self.node_bucket = nn.Linear(config.hidden_dim, config.action_count * config.bucket_count)
        self.query_weight = nn.Linear(config.hidden_dim, config.action_count)
        self.node_weight = nn.Linear(config.hidden_dim, config.action_count)

    def forward(self, embeddings: torch.Tensor, role: str) -> dict[str, torch.Tensor]:
        hidden = self.input(embeddings)
        if role == "query":
            action_logits = self.query_action(hidden)
            bucket_logits = self.query_bucket(hidden)
            weight_logits = self.query_weight(hidden)
        elif role == "node":
            action_logits = self.node_action(hidden)
            bucket_logits = self.node_bucket(hidden)
            weight_logits = self.node_weight(hidden)
        else:
            raise ValueError(f"unknown token-field role: {role}")
        bucket_logits = bucket_logits.view(*action_logits.shape, self.config.bucket_count)
        return {
            "action_logits": action_logits,
            "bucket_logits": bucket_logits,
            "weight_logits": weight_logits,
        }

    def export_tokens(
        self,
        embeddings: torch.Tensor,
        role: str,
        token_count: int | None = None,
    ) -> list[tuple[FieldToken, ...]]:
        outputs = self.forward(embeddings, role)
        action_logits = outputs["action_logits"].detach().cpu()
        bucket_logits = outputs["bucket_logits"].detach().cpu()
        weights = torch.sigmoid(outputs["weight_logits"]).detach().cpu()
        count = int(token_count or (self.config.query_token_count if role == "query" else self.config.node_token_count))
        bucket_offset = self.config.bucket_count // 2
        batches = []
        for item_actions, item_buckets, item_weights in zip(action_logits, bucket_logits, weights):
            ranked = []
            for action_id, action_logit in enumerate(item_actions.tolist()):
                bucket = int(torch.argmax(item_buckets[action_id]).item()) - bucket_offset
                weight = float(item_weights[action_id].item())
                ranked.append((float(action_logit) * max(0.01, weight), action_id, bucket, weight))
            ranked.sort(key=lambda item: (-item[0], item[1], item[2]))
            batches.append(tuple(FieldToken(action_id, bucket, weight) for _, action_id, bucket, weight in ranked[:count]))
        return batches


def fit_tensor_embeddings(values: torch.Tensor, dim: int) -> torch.Tensor:
    if values.shape[-1] > dim:
        values = values[..., :dim]
    elif values.shape[-1] < dim:
        values = F.pad(values, (0, dim - values.shape[-1]))
    return F.normalize(values.float(), dim=-1, eps=1e-8)


def bucket_index_to_value(bucket_index: torch.Tensor, bucket_count: int) -> torch.Tensor:
    return bucket_index.float() - float(bucket_count // 2)


def expected_collision_score(
    query_outputs: dict[str, torch.Tensor],
    node_outputs: dict[str, torch.Tensor],
    bucket_radius: int,
    temperature: float,
) -> torch.Tensor:
    query_action = F.softmax(query_outputs["action_logits"] / max(1e-4, temperature), dim=-1)
    node_action = F.softmax(node_outputs["action_logits"] / max(1e-4, temperature), dim=-1)
    query_bucket = F.softmax(query_outputs["bucket_logits"], dim=-1)
    node_bucket = F.softmax(node_outputs["bucket_logits"], dim=-1)
    query_weight = torch.sigmoid(query_outputs["weight_logits"])
    node_weight = torch.sigmoid(node_outputs["weight_logits"])

    bucket_count = query_bucket.shape[-1]
    bucket_kernel = query_bucket.new_zeros((bucket_count, bucket_count))
    radius = max(0, int(bucket_radius))
    for left in range(bucket_count):
        for right in range(max(0, left - radius), min(bucket_count, left + radius + 1)):
            bucket_kernel[left, right] = 1.0 / float(1 + abs(left - right))
    bucket_overlap = torch.einsum("bac,cd,bad->ba", query_bucket, bucket_kernel, node_bucket)
    action_overlap = query_action * node_action
    weight_overlap = query_weight * node_weight
    return (action_overlap * bucket_overlap * weight_overlap).sum(dim=-1)


def deterministic_embedding_tokens(
    model: TokenFieldEncoder,
    embeddings: list[list[float]] | list[tuple[float, ...]],
    role: str,
    device: torch.device | str,
    token_count: int | None = None,
) -> list[tuple[FieldToken, ...]]:
    tensor = torch.tensor(embeddings, dtype=torch.float32, device=device)
    tensor = fit_tensor_embeddings(tensor, model.config.embedding_dim)
    model.eval()
    with torch.no_grad():
        return model.export_tokens(tensor, role, token_count=token_count)


def save_token_field_encoder(model: TokenFieldEncoder, path: str | Path, **metadata: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": asdict(model.config), "state_dict": model.state_dict(), "metadata": metadata}, path)


def load_token_field_encoder(path: str | Path, device: torch.device | str = "cpu") -> TokenFieldEncoder:
    checkpoint = torch.load(path, map_location=device)
    model = TokenFieldEncoder(TokenFieldEncoderConfig(**checkpoint["config"])).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


class TokenFieldEmitter:
    def __init__(self, checkpoint: str | Path, device: str = ""):
        self.torch = torch
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = load_token_field_encoder(checkpoint, self.device)
        self.config = self.model.config

    def tokens_for_embeddings(
        self,
        embeddings: list[list[float]] | list[tuple[float, ...]],
        role: str,
        token_count: int | None = None,
    ) -> list[tuple[FieldToken, ...]]:
        return deterministic_embedding_tokens(self.model, embeddings, role, self.device, token_count=token_count)


def bucket_for_embedding_value(value: float, bucket_width: float) -> int:
    return bucket_for(float(value), bucket_width)

