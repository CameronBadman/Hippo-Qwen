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

from python.librarian.model import ModelConfig, NeighborhoodTransformer, save_checkpoint
from python.training.dataset import NeighborhoodDataset


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked = values * mask.float()
    return masked.sum() / mask.float().sum().clamp(min=1.0)


def compute_loss(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
    mask = batch["mask"]
    attach_loss = F.binary_cross_entropy_with_logits(outputs["attach_logits"][mask], batch["attach"][mask])
    rank_loss = pairwise_ranking_loss(outputs["attach_logits"], batch["rank"], mask)
    positive = mask & (batch["attach"] > 0.5)
    if positive.any():
        edge_loss = F.cross_entropy(outputs["edge_type_logits"][positive], batch["edge_type"][positive])
        weight_loss = F.mse_loss(outputs["weight"][positive], batch["weight"][positive])
        confidence_loss = F.mse_loss(outputs["confidence"][positive], batch["confidence"][positive])
        decay_loss = F.mse_loss(outputs["decay_rate"][positive], batch["decay_rate"][positive])
        importance_loss = F.mse_loss(outputs["importance_delta"][positive], batch["importance_delta"][positive])
    else:
        zero = outputs["attach_logits"].sum() * 0
        edge_loss = weight_loss = confidence_loss = decay_loss = importance_loss = zero
    total = attach_loss + 0.35 * edge_loss + 0.25 * rank_loss + weight_loss + confidence_loss + decay_loss + importance_loss
    metrics = {
        "loss": float(total.detach().cpu().item()),
        "attach_loss": float(attach_loss.detach().cpu().item()),
        "edge_loss": float(edge_loss.detach().cpu().item()),
        "rank_loss": float(rank_loss.detach().cpu().item()),
    }
    return total, metrics


def pairwise_ranking_loss(logits: torch.Tensor, rank: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    losses = []
    for item_logits, item_rank, item_mask in zip(logits, rank, mask):
        valid_logits = item_logits[item_mask]
        valid_rank = item_rank[item_mask]
        positive = valid_rank >= 0.5
        negative = valid_rank <= 0.0
        if not positive.any() or not negative.any():
            continue
        diffs = valid_logits[positive].unsqueeze(1) - valid_logits[negative].unsqueeze(0)
        losses.append(F.softplus(-diffs).mean())
    if not losses:
        return logits.sum() * 0
    return torch.stack(losses).mean()


def accuracy(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> float:
    mask = batch["mask"]
    pred = torch.sigmoid(outputs["attach_logits"]) > 0.5
    correct = pred[mask] == (batch["attach"][mask] > 0.5)
    return float(correct.float().mean().detach().cpu().item()) if correct.numel() else 0.0


def ranking_metrics(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> dict[str, float]:
    logits = outputs["attach_logits"].detach()
    ranks = batch["rank"]
    masks = batch["mask"]
    reciprocal = []
    top1 = []
    top3 = []
    for item_logits, item_rank, item_mask in zip(logits, ranks, masks):
        valid_logits = item_logits[item_mask]
        valid_rank = item_rank[item_mask]
        positive_indexes = torch.nonzero(valid_rank >= 0.5, as_tuple=False).flatten()
        if valid_logits.numel() == 0 or positive_indexes.numel() == 0:
            continue
        order = torch.argsort(valid_logits, descending=True)
        ordered_positive = torch.isin(order, positive_indexes)
        first_positive = torch.nonzero(ordered_positive, as_tuple=False).flatten()
        if first_positive.numel() == 0:
            reciprocal.append(0.0)
            top1.append(0.0)
            top3.append(0.0)
            continue
        rank_position = int(first_positive[0].item()) + 1
        reciprocal.append(1.0 / rank_position)
        top1.append(1.0 if rank_position <= 1 else 0.0)
        top3.append(1.0 if rank_position <= 3 else 0.0)
    if not reciprocal:
        return {"mrr": 0.0, "top1": 0.0, "top3": 0.0}
    return {
        "mrr": sum(reciprocal) / len(reciprocal),
        "top1": sum(top1) / len(top1),
        "top3": sum(top3) / len(top3),
    }


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    dataset = NeighborhoodDataset(args.dataset, args.max_candidates, args.feature_dim)
    val_count = max(1, int(len(dataset) * args.val_fraction))
    train_count = max(1, len(dataset) - val_count)
    train_set, val_set = random_split(dataset, [train_count, val_count])
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    config = ModelConfig(
        d_model=args.d_model,
        feature_dim=args.feature_dim,
        num_layers=args.layers,
        num_heads=args.heads,
        dropout=args.dropout,
        max_candidates=args.max_candidates,
    )
    model = NeighborhoodTransformer(config).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(batch["anchor"], batch["candidates"], batch["pair_features"], batch["mask"])
            loss, _ = compute_loss(outputs, batch)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            total_loss += float(loss.detach().cpu().item())

        model.eval()
        val_losses = []
        val_accs = []
        val_mrrs = []
        val_top3s = []
        with torch.no_grad():
            for batch in val_loader:
                batch = {key: value.to(device) for key, value in batch.items()}
                outputs = model(batch["anchor"], batch["candidates"], batch["pair_features"], batch["mask"])
                loss, _ = compute_loss(outputs, batch)
                val_losses.append(float(loss.detach().cpu().item()))
                val_accs.append(accuracy(outputs, batch))
                rank = ranking_metrics(outputs, batch)
                val_mrrs.append(rank["mrr"])
                val_top3s.append(rank["top3"])
        print(
            f"epoch={epoch + 1} train_loss={total_loss / max(1, len(train_loader)):.4f} "
            f"val_loss={sum(val_losses) / max(1, len(val_losses)):.4f} "
            f"val_attach_acc={sum(val_accs) / max(1, len(val_accs)):.3f} "
            f"val_mrr={sum(val_mrrs) / max(1, len(val_mrrs)):.3f} "
            f"val_top3={sum(val_top3s) / max(1, len(val_top3s)):.3f}",
            flush=True,
        )

    output = Path(args.output)
    save_checkpoint(model, output)
    print(f"saved {output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/synthetic/librarian_cases.jsonl")
    parser.add_argument("--output", default="artifacts/librarian/neighborhood_transformer.pt")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-candidates", type=int, default=32)
    parser.add_argument("--feature-dim", type=int, default=16)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--cpu", action="store_true")
    train(parser.parse_args())


if __name__ == "__main__":
    main()
