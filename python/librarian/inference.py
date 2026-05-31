from __future__ import annotations

from typing import Any

from .features import EDGE_TYPES, activation_mask_for_edge, candidate_features, ensure_embedding, heuristic_action


def tensorize_payload(payload: dict[str, Any], max_candidates: int) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]]]:
    import torch

    anchor = dict(payload["anchor"])
    ensure_embedding(anchor)
    candidates = [dict(item) for item in payload.get("candidates", [])[:max_candidates]]
    for candidate in candidates:
        ensure_embedding(candidate)

    emb_dim = len(anchor["embedding"])
    anchor_tensor = torch.tensor(anchor["embedding"], dtype=torch.float32).unsqueeze(0)
    candidate_tensor = torch.zeros((1, max_candidates, emb_dim), dtype=torch.float32)
    feature_tensor = torch.zeros((1, max_candidates, 8), dtype=torch.float32)
    mask = torch.zeros((1, max_candidates), dtype=torch.bool)
    for idx, candidate in enumerate(candidates):
        candidate_tensor[0, idx] = torch.tensor(candidate["embedding"], dtype=torch.float32)
        feature_tensor[0, idx] = torch.tensor(candidate_features(anchor, candidate), dtype=torch.float32)
        mask[0, idx] = True
    return {
        "anchor": anchor_tensor,
        "candidates": candidate_tensor,
        "pair_features": feature_tensor,
        "mask": mask,
    }, candidates


def score_with_model(model: Any, payload: dict[str, Any], threshold: float = 0.5) -> dict[str, Any]:
    import torch

    tensors, candidates = tensorize_payload(payload, model.config.max_candidates)
    with torch.no_grad():
        device = next(model.parameters()).device
        tensors = {key: value.to(device) for key, value in tensors.items()}
        outputs = model(**tensors)
    probs = torch.sigmoid(outputs["attach_logits"])[0].cpu()
    edge_type_ids = outputs["edge_type_logits"][0].argmax(dim=-1).cpu()
    weights = outputs["weight"][0].cpu()
    confidences = outputs["confidence"][0].cpu()
    decay_rates = outputs["decay_rate"][0].cpu()
    importance_deltas = outputs["importance_delta"][0].cpu()

    anchor = payload["anchor"]
    actions: list[dict[str, Any]] = []
    for idx, candidate in enumerate(candidates):
        score = float(probs[idx].item())
        if score < threshold:
            continue
        edge_type = EDGE_TYPES[int(edge_type_ids[idx].item())]
        actions.append(
            {
                "candidate_id": candidate["id"],
                "connect_score": score,
                "edge_type": edge_type,
                "weight": float(weights[idx].item()),
                "confidence": float(confidences[idx].item()),
                "activation_mask": activation_mask_for_edge(anchor, candidate, edge_type),
                "decay_rate": float(decay_rates[idx].item()),
                "importance_delta": float(importance_deltas[idx].item()),
            }
        )
    actions.sort(key=lambda item: (-item["connect_score"], item["candidate_id"]))
    return {
        "actions": actions,
        "create_new_cluster": len(actions) == 0,
        "new_cluster_score": 1.0 - (actions[0]["connect_score"] if actions else 0.0),
    }


def score_with_heuristic(payload: dict[str, Any], threshold: float = 0.28, max_edges: int = 8) -> dict[str, Any]:
    anchor = dict(payload["anchor"])
    ensure_embedding(anchor)
    actions = []
    for candidate in payload.get("candidates", []):
        item = dict(candidate)
        ensure_embedding(item)
        action = heuristic_action(anchor, item)
        if action["attach"] >= 1.0 and action["connect_score"] >= threshold:
            action.pop("attach", None)
            action.pop("edge_type_id", None)
            actions.append(action)
    actions.sort(key=lambda item: (-item["connect_score"], item["candidate_id"]))
    actions = actions[:max_edges]
    return {
        "actions": actions,
        "create_new_cluster": len(actions) == 0,
        "new_cluster_score": 1.0 - (actions[0]["connect_score"] if actions else 0.0),
    }
