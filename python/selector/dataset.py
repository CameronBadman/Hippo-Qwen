from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from python.librarian.features import (
    DEFAULT_DIMS,
    clamp,
    cluster_score,
    cosine,
    embed_text,
    ensure_embedding,
    jaccard,
    metadata_score,
    state_features,
    tokens,
)


class ContextSelectorDataset(Dataset):
    def __init__(self, path: str | Path, max_candidates: int = 32, budget_tokens: int = 90, feature_dim: int = 16):
        self.path = Path(path)
        self.max_candidates = max_candidates
        self.budget_tokens = budget_tokens
        self.feature_dim = feature_dim
        with self.path.open("r", encoding="utf-8") as handle:
            self.rows = [json.loads(line) for line in handle if line.strip()]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        task = row.get("retrieval_task") or {}
        query = task.get("query") or row["anchor"]["text"]
        query_embedding = embed_text(query)
        anchor = dict(row["anchor"])
        ensure_embedding(anchor)
        relevant = set(task.get("relevant_ids") or [])
        candidates = row.get("candidates", [])[: self.max_candidates]

        anchor_embedding = torch.tensor(anchor["embedding"], dtype=torch.float32)
        candidate_embeddings = torch.zeros((self.max_candidates, DEFAULT_DIMS), dtype=torch.float32)
        features = torch.zeros((self.max_candidates, self.feature_dim), dtype=torch.float32)
        mask = torch.zeros((self.max_candidates,), dtype=torch.bool)
        select = torch.zeros((self.max_candidates,), dtype=torch.float32)

        for idx, candidate in enumerate(candidates):
            ensure_embedding(candidate)
            candidate_embeddings[idx] = torch.tensor(candidate["embedding"], dtype=torch.float32)
            features[idx] = torch.tensor(selector_features(query, query_embedding, anchor, candidate, self.budget_tokens, self.feature_dim))
            mask[idx] = True
            select[idx] = 1.0 if candidate["id"] in relevant else 0.0

        return {
            "query": torch.tensor(query_embedding, dtype=torch.float32),
            "anchor": anchor_embedding,
            "candidates": candidate_embeddings,
            "features": features,
            "mask": mask,
            "select": select,
            "ids": [candidate["id"] for candidate in candidates],
            "texts": [candidate.get("text", "") for candidate in candidates],
            "relevant_ids": relevant,
        }


def selector_features(
    query: str,
    query_embedding: list[float],
    anchor: dict[str, Any],
    candidate: dict[str, Any],
    budget_tokens: int,
    feature_dim: int,
) -> list[float]:
    ensure_embedding(anchor)
    ensure_embedding(candidate)
    state = state_features(anchor, candidate)
    candidate_len = max(1, len(tokens(candidate.get("text", ""))))
    anchor_len = max(1, len(tokens(anchor.get("text", ""))))
    features = [
        cosine(query_embedding, candidate["embedding"]),
        jaccard(query, candidate.get("text", "")),
        cosine(anchor["embedding"], candidate["embedding"]),
        jaccard(anchor.get("text", ""), candidate.get("text", "")),
        metadata_score(anchor, candidate),
        cluster_score(anchor, candidate),
        float(candidate.get("importance") or 0.5),
        min(candidate_len / max(1, budget_tokens), 1.0),
        float(anchor.get("importance") or 0.5),
        min(anchor_len, candidate_len) / max(anchor_len, candidate_len),
        state["candidate_age_norm"],
        state["candidate_use_norm"],
        state["candidate_evidence_norm"],
        state["last_outcome_value"],
        state["protected_flag"],
        state["stale_unused_flag"],
        state["recency_score"],
    ]
    features = [clamp(float(value), -1.0, 1.0) for value in features]
    if feature_dim <= len(features):
        return features[:feature_dim]
    return features + [0.0] * (feature_dim - len(features))
