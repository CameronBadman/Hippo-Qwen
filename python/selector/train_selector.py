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

from python.selector.dataset import ContextSelectorDataset
from python.selector.model import MultiSeedContextSelector, SelectorConfig, save_selector


def collate(batch: list[dict]) -> dict:
    return {
        "query": torch.stack([item["query"] for item in batch]),
        "candidates": torch.stack([item["candidates"] for item in batch]),
        "features": torch.stack([item["features"] for item in batch]),
        "mask": torch.stack([item["mask"] for item in batch]),
        "select": torch.stack([item["select"] for item in batch]),
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


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    dataset = ContextSelectorDataset(args.dataset, args.max_candidates, args.budget)
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
    )
    model = MultiSeedContextSelector(config).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            logits = model(batch["query"], batch["candidates"], batch["features"], batch["mask"])
            bce = F.binary_cross_entropy_with_logits(logits[batch["mask"]], batch["select"][batch["mask"]])
            rank = ranking_loss(logits, batch["select"], batch["mask"])
            loss = bce + args.rank_loss_weight * rank
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
                logits = model(batch["query"], batch["candidates"], batch["features"], batch["mask"])
                bce = F.binary_cross_entropy_with_logits(logits[batch["mask"]], batch["select"][batch["mask"]])
                rank = ranking_loss(logits, batch["select"], batch["mask"])
                val_losses.append(float((bce + args.rank_loss_weight * rank).detach().cpu().item()))
                val_metrics.append(metrics(logits, batch["select"], batch["mask"], args.top_k))
        recall = sum(item["recall_at_k"] for item in val_metrics) / max(1, len(val_metrics))
        precision = sum(item["precision_at_k"] for item in val_metrics) / max(1, len(val_metrics))
        mrr = sum(item["mrr"] for item in val_metrics) / max(1, len(val_metrics))
        print(
            f"epoch={epoch + 1} train_loss={train_loss / max(1, len(train_loader)):.4f} "
            f"val_loss={sum(val_losses) / max(1, len(val_losses)):.4f} "
            f"val_recall@{args.top_k}={recall:.3f} val_precision@{args.top_k}={precision:.3f} val_mrr={mrr:.3f}",
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
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--rank-loss-weight", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--cpu", action="store_true")
    train(parser.parse_args())


if __name__ == "__main__":
    main()
