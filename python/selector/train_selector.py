from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split

from python.selector.dataset import AUXILIARY_LABELS, CONTEXT_REASONS, ContextSelectorDataset, reason_label
from python.selector.model import MultiSeedContextSelector, SelectorConfig, save_selector


def collate(batch: list[dict]) -> dict:
    return {
        "query": torch.stack([item["query"] for item in batch]),
        "anchor": torch.stack([item["anchor"] for item in batch]),
        "candidates": torch.stack([item["candidates"] for item in batch]),
        "features": torch.stack([item["features"] for item in batch]),
        "mask": torch.stack([item["mask"] for item in batch]),
        "select": torch.stack([item["select"] for item in batch]),
        "reason": torch.stack([item["reason"] for item in batch]),
        "auxiliary": torch.stack([item["auxiliary"] for item in batch]),
    }


def ranking_loss(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    losses = []
    for item_logits, item_labels, item_mask in zip(logits, labels, mask):
        valid_logits = item_logits[item_mask]
        valid_labels = item_labels[item_mask]
        positive = valid_labels >= 0.5
        negative = valid_labels <= 0.0
        if not positive.any() or not negative.any():
            continue
        diffs = valid_logits[positive].unsqueeze(1) - valid_logits[negative].unsqueeze(0)
        losses.append(F.softplus(-diffs).mean())
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
        top_positive = torch.isin(top, positives).sum().item()
        recall.append(top_positive / max(1, positives.numel()))
        precision.append(top_positive / max(1, top.numel()))
        ordered_positive = torch.isin(order, positives)
        first = torch.nonzero(ordered_positive, as_tuple=False).flatten()
        mrr.append(1.0 / (int(first[0].item()) + 1) if first.numel() else 0.0)
    if not recall:
        return {"recall_at_k": 0.0, "precision_at_k": 0.0, "mrr": 0.0}
    return {
        "recall_at_k": sum(recall) / len(recall),
        "precision_at_k": sum(precision) / len(precision),
        "mrr": sum(mrr) / len(mrr),
    }


def reason_metrics(reason_logits: torch.Tensor, reason: torch.Tensor, mask: torch.Tensor) -> dict[str, float]:
    valid = mask & (reason >= 0)
    if not valid.any():
        return {"reason_accuracy": 0.0}
    predicted = reason_logits.argmax(dim=-1)
    return {"reason_accuracy": float((predicted[valid] == reason[valid]).float().mean().detach().cpu().item())}


def auxiliary_loss(auxiliary_logits: torch.Tensor, auxiliary: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = mask.bool()
    if not valid.any():
        return auxiliary_logits.sum() * 0
    return F.binary_cross_entropy_with_logits(auxiliary_logits[valid], auxiliary[valid])


def auxiliary_metrics(auxiliary_logits: torch.Tensor, auxiliary: torch.Tensor, mask: torch.Tensor) -> dict[str, float]:
    valid = mask.bool()
    if not valid.any():
        return {"auxiliary_macro_f1": 0.0}
    predicted = torch.sigmoid(auxiliary_logits[valid]) >= 0.5
    expected = auxiliary[valid] >= 0.5
    f1_scores = []
    for idx in range(len(AUXILIARY_LABELS)):
        pred = predicted[:, idx]
        exp = expected[:, idx]
        tp = (pred & exp).sum().float()
        fp = (pred & ~exp).sum().float()
        fn = (~pred & exp).sum().float()
        precision = tp / torch.clamp(tp + fp, min=1.0)
        recall = tp / torch.clamp(tp + fn, min=1.0)
        f1_scores.append(2.0 * precision * recall / torch.clamp(precision + recall, min=1e-6))
    return {"auxiliary_macro_f1": float(torch.stack(f1_scores).mean().detach().cpu().item())}


def reason_class_weights(dataset: ContextSelectorDataset, device: torch.device, enabled: bool) -> torch.Tensor | None:
    if not enabled:
        return None
    counts = torch.ones((len(CONTEXT_REASONS),), dtype=torch.float32)
    for row in dataset.rows:
        task = row.get("retrieval_task") or {}
        relevant = set(task.get("relevant_ids") or [])
        anchor = row.get("anchor") or {}
        for candidate in row.get("candidates", [])[: dataset.max_candidates]:
            counts[reason_label(anchor, candidate, candidate.get("id") in relevant)] += 1.0
    weights = counts.sum() / (counts * len(CONTEXT_REASONS))
    return torch.clamp(weights, max=8.0).to(device)


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    dataset = ContextSelectorDataset(args.dataset, args.max_candidates, args.budget, args.feature_dim)
    val_count = max(1, int(len(dataset) * args.val_fraction))
    train_count = max(1, len(dataset) - val_count)
    train_set, val_set = random_split(dataset, [train_count, val_count])
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, collate_fn=collate)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    config = SelectorConfig(
        d_model=args.d_model,
        num_layers=args.layers,
        num_heads=args.heads,
        dropout=args.dropout,
        max_candidates=args.max_candidates,
        budget_tokens=args.budget,
        feature_dim=args.feature_dim,
        use_anchor_seed=not args.query_only,
    )
    model = MultiSeedContextSelector(config).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    reason_weights = reason_class_weights(dataset, device, args.reason_class_balance)

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(batch["query"], batch["anchor"], batch["candidates"], batch["features"], batch["mask"])
            logits = outputs["select_logits"]
            bce = F.binary_cross_entropy_with_logits(logits[batch["mask"]], batch["select"][batch["mask"]])
            rank = ranking_loss(logits, batch["select"], batch["mask"])
            reason_loss = F.cross_entropy(outputs["reason_logits"][batch["mask"]], batch["reason"][batch["mask"]], weight=reason_weights)
            aux = auxiliary_loss(outputs["auxiliary_logits"], batch["auxiliary"], batch["mask"])
            loss = bce + args.rank_loss_weight * rank + args.reason_loss_weight * reason_loss + args.auxiliary_loss_weight * aux
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            train_loss += float(loss.detach().cpu().item())

        model.eval()
        val_losses = []
        val_metrics = []
        val_reason = []
        val_auxiliary = []
        with torch.no_grad():
            for batch in val_loader:
                batch = {key: value.to(device) for key, value in batch.items()}
                outputs = model(batch["query"], batch["anchor"], batch["candidates"], batch["features"], batch["mask"])
                logits = outputs["select_logits"]
                bce = F.binary_cross_entropy_with_logits(logits[batch["mask"]], batch["select"][batch["mask"]])
                rank = ranking_loss(logits, batch["select"], batch["mask"])
                reason_loss = F.cross_entropy(outputs["reason_logits"][batch["mask"]], batch["reason"][batch["mask"]], weight=reason_weights)
                aux = auxiliary_loss(outputs["auxiliary_logits"], batch["auxiliary"], batch["mask"])
                val_losses.append(float((bce + args.rank_loss_weight * rank + args.reason_loss_weight * reason_loss + args.auxiliary_loss_weight * aux).detach().cpu().item()))
                val_metrics.append(metrics(logits, batch["select"], batch["mask"], args.top_k))
                val_reason.append(reason_metrics(outputs["reason_logits"], batch["reason"], batch["mask"]))
                val_auxiliary.append(auxiliary_metrics(outputs["auxiliary_logits"], batch["auxiliary"], batch["mask"]))
        recall = sum(item["recall_at_k"] for item in val_metrics) / max(1, len(val_metrics))
        precision = sum(item["precision_at_k"] for item in val_metrics) / max(1, len(val_metrics))
        mrr = sum(item["mrr"] for item in val_metrics) / max(1, len(val_metrics))
        reason_acc = sum(item["reason_accuracy"] for item in val_reason) / max(1, len(val_reason))
        aux_macro_f1 = sum(item["auxiliary_macro_f1"] for item in val_auxiliary) / max(1, len(val_auxiliary))
        print(
            f"epoch={epoch + 1} train_loss={train_loss / max(1, len(train_loader)):.4f} "
            f"val_loss={sum(val_losses) / max(1, len(val_losses)):.4f} "
            f"val_recall@{args.top_k}={recall:.3f} val_precision@{args.top_k}={precision:.3f} "
            f"val_mrr={mrr:.3f} val_reason_acc={reason_acc:.3f} val_aux_macro_f1={aux_macro_f1:.3f}",
            flush=True,
        )

    save_selector(model, args.output)
    print(f"saved {args.output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/synthetic/librarian_hard_cases.jsonl")
    parser.add_argument("--output", default="artifacts/librarian/context_selector.pt")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-candidates", type=int, default=32)
    parser.add_argument("--budget", type=int, default=90)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--feature-dim", type=int, default=16)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--rank-loss-weight", type=float, default=0.25)
    parser.add_argument("--reason-loss-weight", type=float, default=0.1)
    parser.add_argument("--auxiliary-loss-weight", type=float, default=0.05)
    parser.add_argument("--reason-class-balance", action="store_true")
    parser.add_argument("--no-reason-class-balance", dest="reason_class_balance", action="store_false")
    parser.set_defaults(reason_class_balance=False)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--query-only", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    train(parser.parse_args())


if __name__ == "__main__":
    main()
