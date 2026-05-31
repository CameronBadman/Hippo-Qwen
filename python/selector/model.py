from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn

from python.librarian.features import DEFAULT_DIMS


@dataclass
class SelectorConfig:
    embedding_dim: int = DEFAULT_DIMS
    feature_dim: int = 16
    d_model: int = 128
    num_layers: int = 4
    num_heads: int = 4
    dropout: float = 0.1
    max_candidates: int = 32
    budget_tokens: int = 90
    use_anchor_seed: bool = True


class MultiSeedContextSelector(nn.Module):
    def __init__(self, config: SelectorConfig):
        super().__init__()
        self.config = config
        token_dim = config.embedding_dim * 5 + config.feature_dim
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
        self.select_head = nn.Linear(config.d_model, 1)

    def forward(
        self,
        query: torch.Tensor,
        anchor: torch.Tensor,
        candidates: torch.Tensor,
        features: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        if not self.config.use_anchor_seed:
            anchor = query
        query_expanded = query.unsqueeze(1).expand(-1, candidates.shape[1], -1)
        anchor_expanded = anchor.unsqueeze(1).expand(-1, candidates.shape[1], -1)
        query_diff = candidates - query_expanded
        anchor_diff = candidates - anchor_expanded
        tokens = torch.cat([query_expanded, anchor_expanded, candidates, query_diff, anchor_diff, features], dim=-1)
        hidden = self.input(tokens)
        hidden = self.encoder(hidden, src_key_padding_mask=~mask.bool())
        return self.select_head(hidden).squeeze(-1)


def save_selector(model: MultiSeedContextSelector, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": asdict(model.config), "state_dict": model.state_dict()}, path)


def load_selector(path: str | Path, device: torch.device | str = "cpu") -> MultiSeedContextSelector:
    checkpoint = torch.load(path, map_location=device)
    model = MultiSeedContextSelector(SelectorConfig(**checkpoint["config"])).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model
