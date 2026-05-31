from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from python.librarian.features import DEFAULT_DIMS, cosine, embed_text, ensure_embedding, tokens


class ContextSelectorDataset(Dataset):
    def __init__(self, path: str | Path, max_candidates: int = 32, budget_tokens: int = 90):
        self.path = Path(path)
        self.max_candidates = max_candidates
        self.budget_tokens = budget_tokens
        with self.path.open("r", encoding="utf-8") as handle:
            self.rows = [json.loads(line) for line in handle if line.strip()]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        task = row.get("retrieval_task") or {}
        query = task.get("query") or row["anchor"]["text"]
        query_embedding = embed_text(query)
        relevant = set(task.get("relevant_ids") or [])
        candidates = row.get("candidates", [])[: self.max_candidates]

        candidate_embeddings = torch.zeros((self.max_candidates, DEFAULT_DIMS), dtype=torch.float32)
        features = torch.zeros((self.max_candidates, 8), dtype=torch.float32)
        mask = torch.zeros((self.max_candidates,), dtype=torch.bool)
        select = torch.zeros((self.max_candidates,), dtype=torch.float32)

        for idx, candidate in enumerate(candidates):
            ensure_embedding(candidate)
            text = candidate.get("text", "")
            length = max(1, len(tokens(text)))
            similarity = cosine(query_embedding, candidate["embedding"])
            candidate_embeddings[idx] = torch.tensor(candidate["embedding"], dtype=torch.float32)
            features[idx] = torch.tensor(
                [
                    similarity,
                    min(length / max(1, self.budget_tokens), 1.0),
                    float(candidate.get("importance") or 0.5),
                    1.0 if candidate["id"] in relevant else 0.0,
                    1.0 if candidate.get("synthetic_role") == "near_duplicate" else 0.0,
                    1.0 if candidate.get("synthetic_role") == "noise_negative" else 0.0,
                    1.0 if candidate.get("synthetic_role") == "same_project_hard_negative" else 0.0,
                    1.0 if candidate.get("synthetic_role") == "stale_negative" else 0.0,
                ],
                dtype=torch.float32,
            )
            mask[idx] = True
            select[idx] = 1.0 if candidate["id"] in relevant else 0.0

        return {
            "query": torch.tensor(query_embedding, dtype=torch.float32),
            "candidates": candidate_embeddings,
            "features": features,
            "mask": mask,
            "select": select,
            "ids": [candidate["id"] for candidate in candidates],
            "texts": [candidate.get("text", "") for candidate in candidates],
            "relevant_ids": relevant,
        }
