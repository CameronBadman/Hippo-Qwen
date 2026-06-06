from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, random_split

from python.librarian.hippo_calibrator import (
    HippoCalibrationTransformer,
    HippoCalibratorConfig,
    calibration_features,
    fit_embedding,
    save_calibrator,
)


class HippoCalibrationDataset(Dataset):
    def __init__(self, path: str | Path, max_candidates: int, feature_dim: int, embedding_dim: int):
        self.path = Path(path)
        self.max_candidates = int(max_candidates)
        self.feature_dim = int(feature_dim)
        self.embedding_dim = int(embedding_dim)
        with self.path.open("r", encoding="utf-8") as handle:
            self.rows = [json.loads(line) for line in handle if line.strip()]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        query = str(row.get("query") or "")
        relevant = {str(item) for item in row.get("relevant_ids") or []}
        query_embedding = torch.tensor(fit_embedding(row.get("query_embedding") or [], self.embedding_dim), dtype=torch.float32)
        candidates = torch.zeros((self.max_candidates, self.embedding_dim), dtype=torch.float32)
        features = torch.zeros((self.max_candidates, self.feature_dim), dtype=torch.float32)
        labels = torch.zeros((self.max_candidates,), dtype=torch.float32)
        include_labels = torch.zeros((self.max_candidates,), dtype=torch.float32)
        weights = torch.ones((self.max_candidates,), dtype=torch.float32)
        include_weights = torch.ones((self.max_candidates,), dtype=torch.float32)
        mask = torch.zeros((self.max_candidates,), dtype=torch.bool)
        for idx, candidate in enumerate((row.get("candidates") or [])[: self.max_candidates]):
            candidates[idx] = torch.tensor(fit_embedding(candidate.get("embedding") or [], self.embedding_dim), dtype=torch.float32)
            features[idx] = torch.tensor(
                calibration_features(
                    query,
                    candidate,
                    int(candidate.get("base_rank") or idx + 1),
                    float(candidate.get("base_score") or 0.0),
                    int(row.get("budget") or 900),
                    self.feature_dim,
                ),
                dtype=torch.float32,
            )
            is_relevant = str(candidate.get("id") or "") in relevant
            labels[idx] = 1.0 if is_relevant else 0.0
            include_labels[idx] = float(candidate.get("include_label", 1.0 if is_relevant else 0.0))
            weights[idx] = max(0.05, float(candidate.get("label_weight") or 1.0))
            include_weights[idx] = max(0.05, float(candidate.get("include_weight") or candidate.get("label_weight") or 1.0))
            mask[idx] = True
        return {
            "query": query_embedding,
            "candidates": candidates,
            "features": features,
            "labels": labels,
            "include_labels": include_labels,
            "weights": weights,
            "include_weights": include_weights,
            "mask": mask,
        }


def collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {
        "query": torch.stack([item["query"] for item in batch]),
        "candidates": torch.stack([item["candidates"] for item in batch]),
        "features": torch.stack([item["features"] for item in batch]),
        "labels": torch.stack([item["labels"] for item in batch]),
        "include_labels": torch.stack([item["include_labels"] for item in batch]),
        "weights": torch.stack([item["weights"] for item in batch]),
        "include_weights": torch.stack([item["include_weights"] for item in batch]),
        "mask": torch.stack([item["mask"] for item in batch]),
    }


def ranking_loss(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    losses = []
    if weights is None:
        weights = torch.ones_like(labels)
    for item_logits, item_labels, item_mask, item_weights in zip(logits, labels, mask, weights):
        valid_logits = item_logits[item_mask]
        valid_labels = item_labels[item_mask]
        valid_weights = item_weights[item_mask]
        positive = valid_labels >= 0.5
        negative = valid_labels <= 0.0
        if not positive.any() or not negative.any():
            continue
        diffs = valid_logits[positive].unsqueeze(1) - valid_logits[negative].unsqueeze(0)
        pair_weights = valid_weights[positive].unsqueeze(1) * valid_weights[negative].unsqueeze(0)
        losses.append((F.softplus(-diffs) * pair_weights).sum() / pair_weights.sum().clamp_min(1.0))
    if not losses:
        return logits.sum() * 0
    return torch.stack(losses).mean()


def topk_false_positive_loss(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor, top_k: int, margin: float) -> torch.Tensor:
    losses = []
    for item_logits, item_labels, item_mask in zip(logits, labels, mask):
        valid_logits = item_logits[item_mask]
        valid_labels = item_labels[item_mask]
        if valid_logits.numel() == 0:
            continue
        order = torch.argsort(valid_logits, descending=True)[: max(1, int(top_k))]
        top_logits = valid_logits[order]
        top_labels = valid_labels[order]
        negative_logits = top_logits[top_labels <= 0.0]
        if negative_logits.numel() == 0:
            continue
        losses.append(F.softplus(negative_logits + float(margin)).mean())
    if not losses:
        return logits.sum() * 0
    return torch.stack(losses).mean()


def metrics(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor, top_k: int) -> dict[str, float]:
    recall = []
    precision = []
    mrr = []
    for item_logits, item_labels, item_mask in zip(logits, labels, mask):
        valid_logits = item_logits[item_mask]
        valid_labels = item_labels[item_mask]
        positives = torch.nonzero(valid_labels >= 0.5, as_tuple=False).flatten()
        if valid_logits.numel() == 0 or positives.numel() == 0:
            continue
        order = torch.argsort(valid_logits, descending=True)
        top = order[:top_k]
        hits = torch.isin(top, positives).sum().item()
        recall.append(hits / max(1, positives.numel()))
        precision.append(hits / max(1, top.numel()))
        ordered_positive = torch.isin(order, positives)
        first = torch.nonzero(ordered_positive, as_tuple=False).flatten()
        mrr.append(1.0 / (int(first[0].item()) + 1) if first.numel() else 0.0)
    if not recall:
        return {"recall_at_k": 0.0, "precision_at_k": 0.0, "f1_at_k": 0.0, "mrr": 0.0}
    avg_recall = sum(recall) / len(recall)
    avg_precision = sum(precision) / len(precision)
    f1 = 2.0 * avg_recall * avg_precision / max(1e-6, avg_recall + avg_precision)
    return {
        "recall_at_k": avg_recall,
        "precision_at_k": avg_precision,
        "f1_at_k": f1,
        "mrr": sum(mrr) / len(mrr),
    }


def selection_score(metric: dict[str, float], name: str) -> float:
    if name == "mrr":
        return metric["mrr"]
    if name == "precision":
        return metric["precision_at_k"]
    if name == "recall":
        return metric["recall_at_k"]
    if name == "f1":
        return metric["f1_at_k"]
    return 0.50 * metric["f1_at_k"] + 0.30 * metric["mrr"] + 0.20 * metric["recall_at_k"]


def positive_weight(dataset: HippoCalibrationDataset, max_weight: float) -> float:
    positives = 1.0
    negatives = 1.0
    for row in dataset.rows:
        relevant = {str(item) for item in row.get("relevant_ids") or []}
        for candidate in (row.get("candidates") or [])[: dataset.max_candidates]:
            if str(candidate.get("id") or "") in relevant:
                positives += 1.0
            else:
                negatives += 1.0
    return min(float(max_weight), negatives / positives)


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    dataset = HippoCalibrationDataset(args.dataset, args.max_candidates, args.feature_dim, args.embedding_dim)
    val_count = max(1, int(len(dataset) * args.val_fraction))
    train_count = max(1, len(dataset) - val_count)
    train_set, val_set = random_split(dataset, [train_count, val_count])
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, collate_fn=collate)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = HippoCalibrationTransformer(
        HippoCalibratorConfig(
            embedding_dim=args.embedding_dim,
            feature_dim=args.feature_dim,
            d_model=args.d_model,
            num_layers=args.layers,
            num_heads=args.heads,
            dropout=args.dropout,
            max_candidates=args.max_candidates,
        )
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    pos_weight = torch.tensor(positive_weight(dataset, args.max_pos_weight), dtype=torch.float32, device=device)

    best_score = -1.0
    best_metrics: dict[str, float] = {"recall_at_k": 0.0, "precision_at_k": 0.0, "f1_at_k": 0.0, "mrr": 0.0}
    best_state = None
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(batch["query"], batch["candidates"], batch["features"], batch["mask"])
            logits = outputs["relevance_logits"]
            include_logits = outputs["include_logits"]
            valid = batch["mask"].bool()
            effective_weights = batch["weights"] * torch.where(
                batch["labels"] >= 0.5,
                torch.ones_like(batch["labels"]),
                torch.full_like(batch["labels"], float(args.negative_weight)),
            )
            include_effective_weights = batch["include_weights"] * torch.where(
                batch["include_labels"] >= 0.5,
                torch.ones_like(batch["include_labels"]),
                torch.full_like(batch["include_labels"], float(args.include_negative_weight)),
            )
            relevance_bce = F.binary_cross_entropy_with_logits(
                logits[valid],
                batch["labels"][valid],
                pos_weight=pos_weight,
                weight=effective_weights[valid],
            )
            include_bce = F.binary_cross_entropy_with_logits(
                include_logits[valid],
                batch["include_labels"][valid],
                pos_weight=pos_weight,
                weight=include_effective_weights[valid],
            )
            rank = ranking_loss(logits, batch["labels"], batch["mask"], effective_weights)
            include_rank = ranking_loss(include_logits, batch["include_labels"], batch["mask"], include_effective_weights)
            false_positive = topk_false_positive_loss(
                include_logits,
                batch["include_labels"],
                batch["mask"],
                args.top_k,
                args.false_positive_margin,
            )
            loss = (
                relevance_bce
                + args.include_loss_weight * include_bce
                + args.rank_loss_weight * rank
                + args.include_rank_loss_weight * include_rank
                + args.false_positive_loss_weight * false_positive
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            train_loss += float(loss.detach().cpu().item())

        model.eval()
        val_losses = []
        val_metrics = []
        with torch.no_grad():
            for batch in val_loader:
                batch = {key: value.to(device) for key, value in batch.items()}
                outputs = model(batch["query"], batch["candidates"], batch["features"], batch["mask"])
                logits = outputs["relevance_logits"]
                include_logits = outputs["include_logits"]
                valid = batch["mask"].bool()
                effective_weights = batch["weights"] * torch.where(
                    batch["labels"] >= 0.5,
                    torch.ones_like(batch["labels"]),
                    torch.full_like(batch["labels"], float(args.negative_weight)),
                )
                include_effective_weights = batch["include_weights"] * torch.where(
                    batch["include_labels"] >= 0.5,
                    torch.ones_like(batch["include_labels"]),
                    torch.full_like(batch["include_labels"], float(args.include_negative_weight)),
                )
                relevance_bce = F.binary_cross_entropy_with_logits(
                    logits[valid],
                    batch["labels"][valid],
                    pos_weight=pos_weight,
                    weight=effective_weights[valid],
                )
                include_bce = F.binary_cross_entropy_with_logits(
                    include_logits[valid],
                    batch["include_labels"][valid],
                    pos_weight=pos_weight,
                    weight=include_effective_weights[valid],
                )
                rank = ranking_loss(logits, batch["labels"], batch["mask"], effective_weights)
                include_rank = ranking_loss(include_logits, batch["include_labels"], batch["mask"], include_effective_weights)
                false_positive = topk_false_positive_loss(
                    include_logits,
                    batch["include_labels"],
                    batch["mask"],
                    args.top_k,
                    args.false_positive_margin,
                )
                val_loss = (
                    relevance_bce
                    + args.include_loss_weight * include_bce
                    + args.rank_loss_weight * rank
                    + args.include_rank_loss_weight * include_rank
                    + args.false_positive_loss_weight * false_positive
                )
                val_losses.append(float(val_loss.detach().cpu().item()))
                val_metrics.append(metrics(include_logits, batch["include_labels"], batch["mask"], args.top_k))
        recall = sum(item["recall_at_k"] for item in val_metrics) / max(1, len(val_metrics))
        precision = sum(item["precision_at_k"] for item in val_metrics) / max(1, len(val_metrics))
        f1 = sum(item["f1_at_k"] for item in val_metrics) / max(1, len(val_metrics))
        mrr = sum(item["mrr"] for item in val_metrics) / max(1, len(val_metrics))
        current_metrics = {"recall_at_k": recall, "precision_at_k": precision, "f1_at_k": f1, "mrr": mrr}
        current_score = selection_score(current_metrics, args.selection_metric)
        if current_score > best_score:
            best_score = current_score
            best_metrics = current_metrics
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        print(
            f"epoch={epoch + 1} train_loss={train_loss / max(1, len(train_loader)):.4f} "
            f"val_loss={sum(val_losses) / max(1, len(val_losses)):.4f} "
            f"val_recall@{args.top_k}={recall:.3f} val_precision@{args.top_k}={precision:.3f} "
            f"val_f1@{args.top_k}={f1:.3f} val_mrr={mrr:.3f}",
            flush=True,
        )
    if best_state is not None:
        model.load_state_dict(best_state)
    save_calibrator(
        model,
        args.output,
        dataset=args.dataset,
        best_val_score=best_score,
        best_val_metrics=best_metrics,
        selection_metric=args.selection_metric,
        max_candidates=args.max_candidates,
        feature_dim=args.feature_dim,
        rerank_relevance_weight=args.rerank_relevance_weight,
        rerank_include_weight=args.rerank_include_weight,
        rerank_base_weight=args.rerank_base_weight,
        rerank_utility_weight=args.rerank_utility_weight,
    )
    print(f"saved {args.output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="artifacts/hippo_calibrator/memorycraft_train.jsonl")
    parser.add_argument("--output", default="artifacts/hippo_calibrator/calibrator.pt")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--max-candidates", type=int, default=128)
    parser.add_argument("--embedding-dim", type=int, default=1024)
    parser.add_argument("--feature-dim", type=int, default=16)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--rank-loss-weight", type=float, default=0.6)
    parser.add_argument("--include-loss-weight", type=float, default=1.0)
    parser.add_argument("--include-rank-loss-weight", type=float, default=0.6)
    parser.add_argument("--false-positive-loss-weight", type=float, default=0.0)
    parser.add_argument("--false-positive-margin", type=float, default=0.0)
    parser.add_argument("--max-pos-weight", type=float, default=16.0)
    parser.add_argument("--negative-weight", type=float, default=1.0)
    parser.add_argument("--include-negative-weight", type=float, default=1.0)
    parser.add_argument("--selection-metric", choices=["mrr", "precision", "recall", "f1", "balanced"], default="mrr")
    parser.add_argument("--rerank-relevance-weight", type=float, default=0.35)
    parser.add_argument("--rerank-include-weight", type=float, default=0.60)
    parser.add_argument("--rerank-base-weight", type=float, default=0.05)
    parser.add_argument("--rerank-utility-weight", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--seed", type=int, default=9101)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
