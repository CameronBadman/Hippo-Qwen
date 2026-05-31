from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from python.librarian.features import EDGE_TYPES, candidate_features, ensure_embedding


class NeighborhoodDataset(Dataset):
    def __init__(self, path: str | Path, max_candidates: int):
        self.path = Path(path)
        self.max_candidates = max_candidates
        with self.path.open("r", encoding="utf-8") as handle:
            self.rows = [json.loads(line) for line in handle if line.strip()]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        anchor = row["anchor"]
        ensure_embedding(anchor)
        candidates = row["candidates"][: self.max_candidates]
        labels = row["labels"]
        emb_dim = len(anchor["embedding"])

        candidate_embeddings = torch.zeros((self.max_candidates, emb_dim), dtype=torch.float32)
        pair_features = torch.zeros((self.max_candidates, 8), dtype=torch.float32)
        mask = torch.zeros((self.max_candidates,), dtype=torch.bool)
        attach = torch.zeros((self.max_candidates,), dtype=torch.float32)
        edge_type = torch.zeros((self.max_candidates,), dtype=torch.long)
        weight = torch.zeros((self.max_candidates,), dtype=torch.float32)
        confidence = torch.zeros((self.max_candidates,), dtype=torch.float32)
        decay_rate = torch.zeros((self.max_candidates,), dtype=torch.float32)
        importance_delta = torch.zeros((self.max_candidates,), dtype=torch.float32)

        for idx, candidate in enumerate(candidates):
            ensure_embedding(candidate)
            candidate_embeddings[idx] = torch.tensor(candidate["embedding"], dtype=torch.float32)
            pair_features[idx] = torch.tensor(candidate_features(anchor, candidate), dtype=torch.float32)
            mask[idx] = True
            attach[idx] = float(labels["attach"][idx])
            edge_type[idx] = int(labels["edge_type"][idx])
            weight[idx] = float(labels["weight"][idx])
            confidence[idx] = float(labels["confidence"][idx])
            decay_rate[idx] = float(labels["decay_rate"][idx])
            importance_delta[idx] = float(labels["importance_delta"][idx])

        return {
            "anchor": torch.tensor(anchor["embedding"], dtype=torch.float32),
            "candidates": candidate_embeddings,
            "pair_features": pair_features,
            "mask": mask,
            "attach": attach,
            "edge_type": edge_type.clamp(0, len(EDGE_TYPES) - 1),
            "weight": weight,
            "confidence": confidence,
            "decay_rate": decay_rate,
            "importance_delta": importance_delta,
        }

