from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn

from .features import DEFAULT_DIMS, EDGE_TYPES


@dataclass
class ModelConfig:
    embedding_dim: int = DEFAULT_DIMS
    feature_dim: int = 8
    d_model: int = 128
    num_layers: int = 4
    num_heads: int = 4
    dropout: float = 0.1
    max_candidates: int = 32
    edge_types: tuple[str, ...] = tuple(EDGE_TYPES)


class NeighborhoodTransformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        token_dim = config.embedding_dim * 3 + config.feature_dim
        self.input = nn.Sequential(
            nn.Linear(token_dim, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.GELU(),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.num_heads,
            dim_feedforward=config.d_model * 4,
            dropout=config.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.num_layers)
        self.attach_head = nn.Linear(config.d_model, 1)
        self.edge_type_head = nn.Linear(config.d_model, len(config.edge_types))
        self.weight_head = nn.Linear(config.d_model, 1)
        self.confidence_head = nn.Linear(config.d_model, 1)
        self.decay_head = nn.Linear(config.d_model, 1)
        self.importance_head = nn.Linear(config.d_model, 1)

    def forward(
        self,
        anchor: torch.Tensor,
        candidates: torch.Tensor,
        pair_features: torch.Tensor,
        mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        anchor_expanded = anchor.unsqueeze(1).expand(-1, candidates.shape[1], -1)
        diff = candidates - anchor_expanded
        tokens = torch.cat([anchor_expanded, candidates, diff, pair_features], dim=-1)
        hidden = self.input(tokens)
        hidden = self.encoder(hidden, src_key_padding_mask=~mask.bool())
        return {
            "attach_logits": self.attach_head(hidden).squeeze(-1),
            "edge_type_logits": self.edge_type_head(hidden),
            "weight": torch.sigmoid(self.weight_head(hidden).squeeze(-1)) * 1.2,
            "confidence": torch.sigmoid(self.confidence_head(hidden).squeeze(-1)),
            "decay_rate": torch.sigmoid(self.decay_head(hidden).squeeze(-1)) * 0.05,
            "importance_delta": torch.tanh(self.importance_head(hidden).squeeze(-1)) * 0.08,
        }


def save_checkpoint(model: NeighborhoodTransformer, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": asdict(model.config),
            "state_dict": model.state_dict(),
        },
        path,
    )


def load_checkpoint(path: str | Path, device: torch.device | str = "cpu") -> NeighborhoodTransformer:
    checkpoint = torch.load(path, map_location=device)
    config_payload = checkpoint["config"]
    if isinstance(config_payload.get("edge_types"), list):
        config_payload["edge_types"] = tuple(config_payload["edge_types"])
    model = NeighborhoodTransformer(ModelConfig(**config_payload)).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model

