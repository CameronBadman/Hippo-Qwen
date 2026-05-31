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

PREFERENCE_MARKERS = {"preference", "prefer", "prefers", "preferred", "wants", "likes", "avoids", "needs"}


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
    query_len = max(1, len(tokens(query)))
    metadata = candidate.get("metadata") or {}
    preference = preference_features(anchor.get("text", ""), candidate.get("text", ""))
    cluster = cluster_score(anchor, candidate)
    context = context_features(anchor, candidate, preference, cluster, state)
    features = [
        cosine(query_embedding, candidate["embedding"]),
        jaccard(query, candidate.get("text", "")),
        float(candidate.get("importance") or 0.5),
        min(candidate_len / max(1, budget_tokens), 1.0),
        min(candidate_len / 64.0, 1.0),
        min(query_len, candidate_len) / max(query_len, candidate_len),
        1.0 if metadata.get("project") else 0.0,
        1.0 if candidate.get("cluster") else 0.0,
        cosine(anchor["embedding"], candidate["embedding"]),
        jaccard(anchor.get("text", ""), candidate.get("text", "")),
        metadata_score(anchor, candidate),
        cluster,
        preference["overlap"],
        preference["conflict"],
        preference["same_context_conflict"] * cluster,
        float(anchor.get("importance") or 0.5),
        min(anchor_len, candidate_len) / max(anchor_len, candidate_len),
        context["mismatch"],
        context["mismatch_lexical"],
        context["mismatch_preference_overlap"],
        context["mismatch_preference_conflict"],
        state["candidate_age_norm"],
        state["candidate_use_norm"],
        state["candidate_evidence_norm"],
        state["last_outcome_value"],
        state["protected_flag"],
        state["stale_unused_flag"],
        state["recency_score"],
        context["mismatch_positive_state"],
        context["same_context_stale"],
        context["same_context_positive_state"],
    ]
    features = [clamp(float(value), -1.0, 1.0) for value in features]
    if feature_dim <= len(features):
        return features[:feature_dim]
    return features + [0.0] * (feature_dim - len(features))


def preference_terms(text: str) -> set[str]:
    raw_tokens = tokens(text)
    terms: set[str] = set()
    for idx, token in enumerate(raw_tokens):
        if token not in PREFERENCE_MARKERS:
            continue
        terms.update(raw_tokens[idx + 1 : idx + 5])
    return {term for term in terms if len(term) > 1}


def preference_features(anchor_text: str, candidate_text: str) -> dict[str, float]:
    anchor_terms = preference_terms(anchor_text)
    candidate_terms = preference_terms(candidate_text)
    if not anchor_terms or not candidate_terms:
        return {
            "overlap": 0.0,
            "conflict": 0.0,
            "same_context_conflict": 0.0,
        }
    overlap = len(anchor_terms & candidate_terms) / len(anchor_terms | candidate_terms)
    conflict = 1.0 if overlap < 0.34 else 0.0
    return {
        "overlap": overlap,
        "conflict": conflict,
        "same_context_conflict": conflict,
    }


def context_features(
    anchor: dict[str, Any],
    candidate: dict[str, Any],
    preference: dict[str, float],
    cluster: float,
    state: dict[str, float],
) -> dict[str, float]:
    anchor_cluster = str(anchor.get("cluster") or "")
    candidate_cluster = str(candidate.get("cluster") or "")
    mismatch = 1.0 if anchor_cluster and candidate_cluster and anchor_cluster != candidate_cluster else 0.0
    lexical = jaccard(anchor.get("text", ""), candidate.get("text", ""))
    positive_state = (
        state["candidate_use_norm"]
        + state["candidate_evidence_norm"]
        + max(0.0, state["last_outcome_value"])
        + float(candidate.get("importance") or 0.5)
    ) / 4.0
    return {
        "mismatch": mismatch,
        "mismatch_lexical": mismatch * lexical,
        "mismatch_preference_overlap": mismatch * preference["overlap"],
        "mismatch_preference_conflict": mismatch * preference["conflict"],
        "mismatch_positive_state": mismatch * positive_state,
        "same_context_stale": cluster * state["stale_unused_flag"],
        "same_context_positive_state": cluster * positive_state,
    }
