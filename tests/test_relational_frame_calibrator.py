from __future__ import annotations

import argparse
import json

import pytest

torch = pytest.importorskip("torch")

from python.librarian.relational_frame_calibrator import (
    RelationalFrameCalibrator,
    RelationalFrameCalibratorConfig,
    build_candidate_edges,
    load_relational_calibrator,
    save_relational_calibrator,
    score_with_relational_calibrator,
    tensorize_relational_payload,
)
from python.librarian.train_relational_frame_calibrator import train


def candidate(candidate_id: str, text: str, rank: int, score: float, embedding: list[float]) -> dict:
    return {
        "id": candidate_id,
        "text": text,
        "embedding": embedding,
        "base_rank": rank,
        "base_score": score,
        "candidate_sources": ["vector", "metadata:project"],
        "metadata": {
            "user_id": "user_001",
            "project": "project_001",
            "brand": "aurora",
            "session_id": "session_001",
            "turn_id": f"session_001_t{rank}",
            "timestamp": f"2026-05-01T0{rank}:00:00Z",
        },
        "derived_metadata": {
            "fields": {
                "user_id": [{"value": "user_001", "confidence": 0.9}],
                "project": [{"value": "project_001", "confidence": 0.9}],
                "brand": [{"value": "aurora", "confidence": 0.9}],
                "session_id": [{"value": "session_001", "confidence": 0.9}],
            }
        },
        "include_label": 1.0 if candidate_id.endswith("a") else 0.0,
    }


def payload() -> dict:
    return {
        "query": "What is the current aurora project preference?",
        "query_embedding": [0.2, 0.1, 0.0, 0.3, 0.4, 0.0, 0.1, 0.2],
        "budget": 128,
        "relevant_ids": ["mem_a"],
        "candidates": [
            candidate("mem_a", "Current aurora preference uses teal and concise tone.", 1, 0.9, [0.9, 0.1, 0.0, 0.0, 0.2, 0.0, 0.0, 0.1]),
            candidate("mem_b", "Older aurora note was superseded by the current preference.", 2, 0.7, [0.7, 0.2, 0.0, 0.1, 0.2, 0.0, 0.0, 0.1]),
            candidate("mem_c", "Unrelated archive for the same user and project.", 3, 0.3, [0.2, 0.2, 0.6, 0.0, 0.1, 0.0, 0.0, 0.0]),
        ],
    }


def test_edge_construction_is_deterministic_and_capped() -> None:
    candidates = payload()["candidates"]
    first = build_candidate_edges(candidates, max_edges_per_candidate=2)
    second = build_candidate_edges(candidates, max_edges_per_candidate=2)
    assert first == second
    counts: dict[int, int] = {}
    for edge in first:
        counts[edge["source_index"]] = counts.get(edge["source_index"], 0) + 1
    assert counts
    assert max(counts.values()) <= 2


def test_tensorization_shapes() -> None:
    tensors, candidates, edges = tensorize_relational_payload(
        payload(),
        max_candidates=4,
        node_feature_dim=8,
        embedding_dim=8,
        edge_feature_dim=8,
        max_edges_per_candidate=2,
    )
    assert len(candidates) == 3
    assert tensors["query"].shape == (1, 8)
    assert tensors["candidates"].shape == (1, 4, 8)
    assert tensors["node_features"].shape == (1, 4, 8)
    assert tensors["edge_index"].shape == (1, 8, 2)
    assert tensors["edge_features"].shape == (1, 8, 8)
    assert int(tensors["mask"].sum().item()) == 3
    assert int(tensors["edge_mask"].sum().item()) == len(edges)


def small_config() -> RelationalFrameCalibratorConfig:
    return RelationalFrameCalibratorConfig(
        embedding_dim=8,
        node_feature_dim=8,
        edge_feature_dim=8,
        node_frame_dim=16,
        small_edge_dim=8,
        large_edge_dim=16,
        d_model=16,
        edge_layers=0,
        candidate_layers=0,
        num_heads=2,
        dropout=0.0,
        max_candidates=4,
        max_edges_per_candidate=2,
    )


def test_checkpoint_roundtrip_and_deterministic_ranking(tmp_path) -> None:
    torch.manual_seed(7)
    model = RelationalFrameCalibrator(small_config())
    model.eval()
    first = score_with_relational_calibrator(model, payload(), max_candidates=4)
    second = score_with_relational_calibrator(model, payload(), max_candidates=4)
    assert [item["id"] for item in first] == [item["id"] for item in second]
    path = tmp_path / "relational.pt"
    save_relational_calibrator(model, path, test_marker=True)
    loaded = load_relational_calibrator(path)
    loaded_ranked = score_with_relational_calibrator(loaded, payload(), max_candidates=4)
    assert [item["id"] for item in first] == [item["id"] for item in loaded_ranked]


def test_smoke_train_one_cpu_epoch(tmp_path) -> None:
    dataset = tmp_path / "train.jsonl"
    rows = [payload(), payload()]
    rows[1] = dict(rows[1])
    rows[1]["query"] = "Which aurora memory is superseded?"
    rows[1]["relevant_ids"] = ["mem_b"]
    with dataset.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    checkpoint = tmp_path / "v2.pt"
    args = argparse.Namespace(
        dataset=str(dataset),
        output=str(checkpoint),
        epochs=1,
        batch_size=2,
        val_fraction=0.5,
        max_candidates=4,
        embedding_dim=8,
        node_feature_dim=8,
        edge_feature_dim=8,
        node_frame_dim=16,
        small_edge_dim=8,
        large_edge_dim=16,
        d_model=16,
        edge_layers=0,
        candidate_layers=0,
        heads=2,
        dropout=0.0,
        max_edges_per_candidate=2,
        disable_edges=False,
        lr=1e-3,
        weight_decay=0.0,
        rank_loss_weight=0.1,
        include_loss_weight=1.0,
        include_rank_loss_weight=0.1,
        false_positive_loss_weight=0.0,
        false_positive_margin=0.0,
        edge_aux_loss_weight=0.0,
        max_pos_weight=8.0,
        negative_weight=1.0,
        include_negative_weight=1.0,
        metric_head="include",
        selection_metric="mrr",
        feature_ablation="none",
        rerank_relevance_weight=0.30,
        rerank_include_weight=0.65,
        rerank_base_weight=0.03,
        rerank_utility_weight=0.02,
        grad_clip=1.0,
        top_k=2,
        seed=99,
        cpu=True,
    )
    train(args)
    assert checkpoint.exists()
    loaded = load_relational_calibrator(checkpoint)
    ranked = score_with_relational_calibrator(loaded, payload(), max_candidates=4)
    assert len(ranked) == 3
    assert {item["id"] for item in ranked} == {"mem_a", "mem_b", "mem_c"}
