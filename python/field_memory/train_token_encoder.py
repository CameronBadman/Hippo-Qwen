from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, random_split

from python.field_memory.token_encoder import (
    TokenFieldEncoder,
    TokenFieldEncoderConfig,
    expected_collision_score,
    fit_tensor_embeddings,
    save_token_field_encoder,
)


class TokenEncoderDataset(Dataset):
    def __init__(self, path: str | Path, embedding_dim: int, max_positives: int, max_negatives: int, seed: int):
        self.path = Path(path)
        self.embedding_dim = int(embedding_dim)
        self.max_positives = int(max_positives)
        self.max_negatives = int(max_negatives)
        self.seed = int(seed)
        with self.path.open("r", encoding="utf-8") as handle:
            self.rows = [json.loads(line) for line in handle if line.strip()]

    def __len__(self) -> int:
        return len(self.rows)

    def _embedding(self, payload: dict[str, Any]) -> torch.Tensor:
        return torch.tensor([float(value) for value in payload.get("embedding") or []], dtype=torch.float32)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        rng = random.Random(self.seed + index)
        positives = list(row.get("positives") or [])
        negatives = list(row.get("hard_negatives") or []) + list(row.get("random_negatives") or [])
        rng.shuffle(positives)
        rng.shuffle(negatives)
        positives = positives[: self.max_positives]
        negatives = negatives[: self.max_negatives]
        candidates = positives + negatives
        labels = [1.0] * len(positives) + [0.0] * len(negatives)
        return {
            "query": torch.tensor([float(value) for value in row.get("query_embedding") or []], dtype=torch.float32),
            "candidates": [self._embedding(candidate) for candidate in candidates],
            "labels": torch.tensor(labels, dtype=torch.float32),
        }


def collate(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    embedding_dim = max([item["query"].numel() for item in batch] + [candidate.numel() for item in batch for candidate in item["candidates"]])
    max_candidates = max(1, max(len(item["candidates"]) for item in batch))
    queries = torch.zeros((len(batch), embedding_dim), dtype=torch.float32)
    candidates = torch.zeros((len(batch), max_candidates, embedding_dim), dtype=torch.float32)
    labels = torch.zeros((len(batch), max_candidates), dtype=torch.float32)
    mask = torch.zeros((len(batch), max_candidates), dtype=torch.bool)
    for row_index, item in enumerate(batch):
        query = item["query"]
        queries[row_index, : query.numel()] = query
        for candidate_index, candidate in enumerate(item["candidates"][:max_candidates]):
            candidates[row_index, candidate_index, : candidate.numel()] = candidate
            labels[row_index, candidate_index] = item["labels"][candidate_index]
            mask[row_index, candidate_index] = True
    return {"query": queries, "candidates": candidates, "labels": labels, "mask": mask}


def pair_scores(model: TokenFieldEncoder, query: torch.Tensor, candidates: torch.Tensor, bucket_radius: int, temperature: float) -> torch.Tensor:
    batch, count, _ = candidates.shape
    query = fit_tensor_embeddings(query, model.config.embedding_dim)
    candidates = fit_tensor_embeddings(candidates, model.config.embedding_dim)
    query_outputs = model(query, "query")
    flat_candidates = candidates.reshape(batch * count, model.config.embedding_dim)
    node_outputs = model(flat_candidates, "node")
    repeated_query_outputs = {key: value.repeat_interleave(count, dim=0) for key, value in query_outputs.items()}
    scores = expected_collision_score(repeated_query_outputs, node_outputs, bucket_radius, temperature)
    return scores.view(batch, count)


def multi_positive_infonce(scores: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    losses = []
    for item_scores, item_labels, item_mask in zip(scores, labels, mask):
        valid_scores = item_scores[item_mask]
        valid_labels = item_labels[item_mask]
        positive = valid_labels >= 0.5
        if valid_scores.numel() == 0 or not positive.any():
            continue
        log_probs = F.log_softmax(valid_scores, dim=0)
        target = positive.float() / positive.float().sum().clamp_min(1.0)
        losses.append(-(target * log_probs).sum())
    if not losses:
        return scores.sum() * 0.0
    return torch.stack(losses).mean()


def margin_loss(scores: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor, margin: float) -> torch.Tensor:
    losses = []
    for item_scores, item_labels, item_mask in zip(scores, labels, mask):
        valid_scores = item_scores[item_mask]
        valid_labels = item_labels[item_mask]
        positives = valid_scores[valid_labels >= 0.5]
        negatives = valid_scores[valid_labels <= 0.0]
        if positives.numel() == 0 or negatives.numel() == 0:
            continue
        hardest_negative = negatives.max()
        losses.append(F.relu(float(margin) - positives + hardest_negative).mean())
    if not losses:
        return scores.sum() * 0.0
    return torch.stack(losses).mean()


def saturation_loss(model: TokenFieldEncoder, embeddings: torch.Tensor) -> torch.Tensor:
    flat = fit_tensor_embeddings(embeddings.reshape(-1, embeddings.shape[-1]), model.config.embedding_dim)
    outputs = model(flat, "node")
    return torch.sigmoid(outputs["weight_logits"]).mean()


def metrics(scores: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor, candidate_cap: int, top_k: int) -> dict[str, float]:
    candidate_recalls = []
    recalls = []
    precisions = []
    for item_scores, item_labels, item_mask in zip(scores, labels, mask):
        valid_scores = item_scores[item_mask]
        valid_labels = item_labels[item_mask]
        positive_indexes = torch.nonzero(valid_labels >= 0.5, as_tuple=False).flatten()
        if valid_scores.numel() == 0 or positive_indexes.numel() == 0:
            continue
        order = torch.argsort(valid_scores, descending=True)
        candidate_top = order[: min(candidate_cap, order.numel())]
        top = order[: min(top_k, order.numel())]
        candidate_hits = torch.isin(candidate_top, positive_indexes).sum().item()
        top_hits = torch.isin(top, positive_indexes).sum().item()
        candidate_recalls.append(candidate_hits / max(1, positive_indexes.numel()))
        recalls.append(top_hits / max(1, positive_indexes.numel()))
        precisions.append(top_hits / max(1, top.numel()))
    if not candidate_recalls:
        return {"candidate_recall": 0.0, "recall_at_k": 0.0, "precision_at_k": 0.0}
    return {
        "candidate_recall": sum(candidate_recalls) / len(candidate_recalls),
        "recall_at_k": sum(recalls) / len(recalls),
        "precision_at_k": sum(precisions) / len(precisions),
    }


def hard_token_scores(model: TokenFieldEncoder, query: torch.Tensor, candidates: torch.Tensor, bucket_radius: int) -> torch.Tensor:
    batch, count, _ = candidates.shape
    query = fit_tensor_embeddings(query, model.config.embedding_dim)
    candidates = fit_tensor_embeddings(candidates, model.config.embedding_dim)
    query_tokens = model.export_tokens(query, "query", token_count=model.config.query_token_count)
    node_tokens = model.export_tokens(
        candidates.reshape(batch * count, model.config.embedding_dim),
        "node",
        token_count=model.config.node_token_count,
    )
    node_tokens_by_row = [node_tokens[index * count : (index + 1) * count] for index in range(batch)]
    scores = torch.zeros((batch, count), dtype=torch.float32, device=query.device)
    radius = max(0, int(bucket_radius))
    for row_index, (row_query_tokens, row_node_tokens) in enumerate(zip(query_tokens, node_tokens_by_row)):
        query_by_action: dict[int, list[tuple[int, float]]] = {}
        for token in row_query_tokens:
            query_by_action.setdefault(token.action_id, []).append((token.bucket, token.weight))
        for candidate_index, candidate_tokens in enumerate(row_node_tokens):
            score = 0.0
            for token in candidate_tokens:
                for query_bucket, query_weight in query_by_action.get(token.action_id, []):
                    delta = abs(int(query_bucket) - int(token.bucket))
                    if delta <= radius:
                        score += (float(query_weight) * float(token.weight)) / float(1 + delta)
            scores[row_index, candidate_index] = score
    return scores


def average(items: list[dict[str, float]], key: str) -> float:
    return sum(item.get(key, 0.0) for item in items) / max(1, len(items))


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    dataset = TokenEncoderDataset(args.dataset, args.embedding_dim, args.max_positives, args.max_negatives, args.seed)
    if len(dataset) < 2:
        raise ValueError("token encoder training needs at least two rows")
    val_count = max(1, int(len(dataset) * args.val_fraction))
    train_count = max(1, len(dataset) - val_count)
    train_set, val_set = random_split(dataset, [train_count, val_count], generator=torch.Generator().manual_seed(args.seed))
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, collate_fn=collate)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = TokenFieldEncoder(
        TokenFieldEncoderConfig(
            embedding_dim=args.embedding_dim,
            action_count=args.action_count,
            bucket_count=args.bucket_count,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            query_token_count=args.query_token_count,
            node_token_count=args.node_token_count,
            bucket_width=args.bucket_width,
        )
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_recall = -1.0
    best_hard_recall = -1.0
    best_hard_recall_at_k = -1.0
    best_state = None
    for epoch in range(args.epochs):
        model.train()
        train_losses = []
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            scores = pair_scores(model, batch["query"], batch["candidates"], args.bucket_radius, args.temperature)
            scores = scores.masked_fill(~batch["mask"], -1e9)
            loss = (
                multi_positive_infonce(scores, batch["labels"], batch["mask"])
                + args.margin_loss_weight * margin_loss(scores, batch["labels"], batch["mask"], args.margin)
                + args.saturation_loss_weight * saturation_loss(model, batch["candidates"])
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu().item()))

        model.eval()
        val_metrics = []
        hard_val_metrics = []
        with torch.no_grad():
            for batch in val_loader:
                batch = {key: value.to(device) for key, value in batch.items()}
                scores = pair_scores(model, batch["query"], batch["candidates"], args.bucket_radius, args.temperature)
                scores = scores.masked_fill(~batch["mask"], -1e9)
                val_metrics.append(metrics(scores, batch["labels"], batch["mask"], args.candidate_cap, args.top_k))
                hard_scores = hard_token_scores(model, batch["query"], batch["candidates"], args.bucket_radius)
                hard_scores = hard_scores.masked_fill(~batch["mask"], -1e9)
                hard_val_metrics.append(metrics(hard_scores, batch["labels"], batch["mask"], args.candidate_cap, args.top_k))
        candidate_recall = average(val_metrics, "candidate_recall")
        hard_candidate_recall = average(hard_val_metrics, "candidate_recall")
        hard_recall_at_k = average(hard_val_metrics, "recall_at_k")
        if (hard_candidate_recall, hard_recall_at_k, candidate_recall) > (best_hard_recall, best_hard_recall_at_k, best_recall):
            best_recall = candidate_recall
            best_hard_recall = hard_candidate_recall
            best_hard_recall_at_k = hard_recall_at_k
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        print(
            f"epoch={epoch + 1} train_loss={sum(train_losses) / max(1, len(train_losses)):.4f} "
            f"val_candidate_recall={candidate_recall:.4f} "
            f"val_recall@{args.top_k}={average(val_metrics, 'recall_at_k'):.4f} "
            f"val_precision@{args.top_k}={average(val_metrics, 'precision_at_k'):.4f} "
            f"hard_candidate_recall={hard_candidate_recall:.4f} "
            f"hard_recall@{args.top_k}={hard_recall_at_k:.4f} "
            f"hard_precision@{args.top_k}={average(hard_val_metrics, 'precision_at_k'):.4f}",
            flush=True,
        )
    if best_state is not None:
        model.load_state_dict(best_state)
    save_token_field_encoder(
        model,
        args.output,
        dataset=args.dataset,
        best_candidate_recall=best_recall,
        best_hard_candidate_recall=best_hard_recall,
        best_hard_recall_at_k=best_hard_recall_at_k,
        candidate_cap=args.candidate_cap,
        bucket_radius=args.bucket_radius,
    )
    print(f"saved {args.output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="artifacts/token_field_encoder/train.jsonl")
    parser.add_argument("--output", default="artifacts/token_field_encoder/token_encoder.pt")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--embedding-dim", type=int, default=1024)
    parser.add_argument("--action-count", type=int, default=512)
    parser.add_argument("--bucket-count", type=int, default=65)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--query-token-count", type=int, default=48)
    parser.add_argument("--node-token-count", type=int, default=48)
    parser.add_argument("--bucket-width", type=float, default=0.055)
    parser.add_argument("--bucket-radius", type=int, default=2)
    parser.add_argument("--candidate-cap", type=int, default=384)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--max-positives", type=int, default=16)
    parser.add_argument("--max-negatives", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--margin", type=float, default=0.05)
    parser.add_argument("--margin-loss-weight", type=float, default=0.35)
    parser.add_argument("--saturation-loss-weight", type=float, default=0.02)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
