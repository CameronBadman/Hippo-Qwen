from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

from python.benchmarks.large_pool_retrieval import build_large_pool_case
from python.librarian.features import (
    DEFAULT_DIMS,
    activation_mask_for_text,
    clamp,
    cosine,
    jaccard,
    state_features,
    tokens,
)


FRAME_SCALARS = 14


@dataclass
class FrameConfig:
    frame_size: int = 64
    embedding_dim: int = DEFAULT_DIMS
    scalar_dim: int = FRAME_SCALARS
    d_model: int = 128
    prefix_tokens: int = 4
    num_heads: int = 4
    dropout: float = 0.05
    top_k: int = 3


class MemoryFrameDataset(Dataset):
    def __init__(self, frames: list[dict[str, Any]]):
        self.frames = frames

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        frame = self.frames[index]
        return {
            "query": torch.tensor(frame["query"], dtype=torch.float32),
            "anchor": torch.tensor(frame["anchor"], dtype=torch.float32),
            "embeddings": torch.tensor(frame["embeddings"], dtype=torch.float32),
            "scalars": torch.tensor(frame["scalars"], dtype=torch.float32),
            "labels": torch.tensor(frame["labels"], dtype=torch.float32),
            "mask": torch.tensor(frame["mask"], dtype=torch.bool),
            "gold_total": torch.tensor(frame["gold_total"], dtype=torch.float32),
        }


def build_frames(cases: int, pool_size: int, frame_size: int, seed: int, use_role_features: bool) -> list[dict[str, Any]]:
    frames = []
    for offset in range(cases):
        row = build_large_pool_case(seed + offset, pool_size)
        frames.append(build_frame(row, frame_size, use_role_features))
    return frames


def query_embedding(row: dict[str, Any]) -> list[float]:
    from python.librarian.features import embed_text

    query = (row.get("retrieval_task") or {}).get("query") or row["anchor"]["text"]
    return embed_text(query)


def frame_score(row: dict[str, Any], candidate: dict[str, Any], query_vec: list[float], use_role_features: bool) -> float:
    query = (row.get("retrieval_task") or {}).get("query") or row["anchor"]["text"]
    anchor = row["anchor"]
    qmask = activation_mask_for_text(query)
    cmask = activation_mask_for_text(candidate.get("text", ""))
    activation = (qmask & cmask).bit_count() / max(1, (qmask | cmask).bit_count())
    state = state_features(anchor, candidate)
    role = str(candidate.get("synthetic_role") or "") if use_role_features else ""
    conflict_penalty = 0.25 if any(term in role for term in ("decoy", "wrong", "stale", "background")) else 0.0
    return (
        0.36 * cosine(query_vec, candidate.get("embedding") or [])
        + 0.22 * jaccard(query, candidate.get("text", ""))
        + 0.16 * activation
        + 0.10 * float(candidate.get("importance") or 0.5)
        + 0.09 * state["candidate_use_norm"]
        + 0.07 * state["candidate_evidence_norm"]
        + 0.08 * max(0.0, state["last_outcome_value"])
        - 0.12 * state["stale_unused_flag"]
        - conflict_penalty
    )


def build_frame(row: dict[str, Any], frame_size: int, use_role_features: bool) -> dict[str, Any]:
    qvec = query_embedding(row)
    anchor = dict(row["anchor"])
    relevant = set((row.get("retrieval_task") or {}).get("relevant_ids") or [])
    candidates = [dict(candidate) for candidate in row.get("candidates", [])]
    candidates.sort(key=lambda item: (-frame_score(row, item, qvec, use_role_features), item["id"]))
    selected = candidates[:frame_size]
    embeddings = []
    scalars = []
    labels = []
    mask = []
    query = (row.get("retrieval_task") or {}).get("query") or row["anchor"]["text"]
    qmask = activation_mask_for_text(query)
    for candidate in selected:
        embedding = candidate.get("embedding") or [0.0] * DEFAULT_DIMS
        embeddings.append(embedding)
        scalars.append(frame_scalars(row, candidate, qvec, qmask, use_role_features))
        labels.append(1.0 if candidate["id"] in relevant else 0.0)
        mask.append(1.0)
    while len(embeddings) < frame_size:
        embeddings.append([0.0] * DEFAULT_DIMS)
        scalars.append([0.0] * FRAME_SCALARS)
        labels.append(0.0)
        mask.append(0.0)
    return {
        "query": qvec,
        "anchor": anchor.get("embedding") or [0.0] * DEFAULT_DIMS,
        "embeddings": embeddings,
        "scalars": scalars,
        "labels": labels,
        "mask": mask,
        "gold_total": float(len(relevant)),
    }


def frame_scalars(row: dict[str, Any], candidate: dict[str, Any], query_vec: list[float], query_mask: int, use_role_features: bool) -> list[float]:
    query = (row.get("retrieval_task") or {}).get("query") or row["anchor"]["text"]
    state = state_features(row["anchor"], candidate)
    text = candidate.get("text", "")
    candidate_mask = activation_mask_for_text(text)
    candidate_len = max(1, len(tokens(text)))
    role = str(candidate.get("synthetic_role") or "") if use_role_features else ""
    values = [
        cosine(query_vec, candidate.get("embedding") or []),
        jaccard(query, text),
        (query_mask & candidate_mask).bit_count() / max(1, (query_mask | candidate_mask).bit_count()),
        float(candidate.get("importance") or 0.5),
        clamp(candidate_len / 90.0, 0.0, 1.0),
        state["candidate_age_norm"],
        state["candidate_use_norm"],
        state["candidate_evidence_norm"],
        state["last_outcome_value"],
        state["protected_flag"],
        state["stale_unused_flag"],
        state["recency_score"],
        1.0 if "decoy" in role or "wrong" in role else 0.0,
        1.0 if "target" in role or "support" in role or "bridge" in role else 0.0,
    ]
    return [clamp(float(value), -1.0, 1.0) for value in values]


class BaseFrameModel(nn.Module):
    def __init__(self, config: FrameConfig):
        super().__init__()
        self.config = config
        token_dim = config.embedding_dim * 4 + config.scalar_dim
        self.input = nn.Sequential(nn.Linear(token_dim, config.d_model), nn.LayerNorm(config.d_model), nn.GELU())

    def token_input(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        query = batch["query"].unsqueeze(1).expand_as(batch["embeddings"])
        anchor = batch["anchor"].unsqueeze(1).expand_as(batch["embeddings"])
        candidate = batch["embeddings"]
        return torch.cat([query, anchor, candidate, candidate - query, batch["scalars"]], dim=-1)


class StructuredFrameMLP(BaseFrameModel):
    def __init__(self, config: FrameConfig):
        super().__init__(config)
        self.head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, 1),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.head(self.input(self.token_input(batch))).squeeze(-1)


class PrefixFrameAdapter(BaseFrameModel):
    def __init__(self, config: FrameConfig):
        super().__init__(config)
        self.prefix = nn.Linear(config.d_model, config.prefix_tokens * config.d_model)
        self.mix = nn.Sequential(nn.Linear(config.d_model * 2, config.d_model), nn.GELU(), nn.Linear(config.d_model, 1))

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        tokens = self.input(self.token_input(batch))
        mask = batch["mask"].float().unsqueeze(-1)
        pooled = (tokens * mask).sum(dim=1) / torch.clamp(mask.sum(dim=1), min=1.0)
        prefix = self.prefix(pooled).view(tokens.shape[0], self.config.prefix_tokens, self.config.d_model).mean(dim=1)
        prefix = prefix.unsqueeze(1).expand_as(tokens)
        return self.mix(torch.cat([tokens, prefix], dim=-1)).squeeze(-1)


class CrossAttentionFrameAdapter(BaseFrameModel):
    def __init__(self, config: FrameConfig):
        super().__init__(config)
        self.attn = nn.MultiheadAttention(config.d_model, config.num_heads, dropout=config.dropout, batch_first=True)
        self.norm = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, 1)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        tokens = self.input(self.token_input(batch))
        attended, _ = self.attn(tokens, tokens, tokens, key_padding_mask=~batch["mask"].bool(), need_weights=False)
        return self.head(self.norm(tokens + attended)).squeeze(-1)


class LowRankRouterAdapter(BaseFrameModel):
    def __init__(self, config: FrameConfig, rank: int = 8, experts: int = 4):
        super().__init__(config)
        self.rank = rank
        self.experts = experts
        self.base = nn.Linear(config.d_model, 1)
        self.a = nn.Parameter(torch.randn(experts, config.d_model, rank) * 0.02)
        self.b = nn.Parameter(torch.randn(experts, rank, 1) * 0.02)
        self.router = nn.Linear(config.d_model, experts)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        tokens = self.input(self.token_input(batch))
        mask = batch["mask"].float().unsqueeze(-1)
        pooled = (tokens * mask).sum(dim=1) / torch.clamp(mask.sum(dim=1), min=1.0)
        weights = torch.softmax(self.router(pooled), dim=-1)
        deltas = []
        for idx in range(self.experts):
            deltas.append(tokens.matmul(self.a[idx]).matmul(self.b[idx]).squeeze(-1))
        delta = torch.stack(deltas, dim=-1).mul(weights.unsqueeze(1)).sum(dim=-1)
        return self.base(tokens).squeeze(-1) + delta


def build_model(name: str, config: FrameConfig) -> nn.Module:
    if name == "structured":
        return StructuredFrameMLP(config)
    if name == "prefix":
        return PrefixFrameAdapter(config)
    if name == "cross_attention":
        return CrossAttentionFrameAdapter(config)
    if name == "lowrank_router":
        return LowRankRouterAdapter(config)
    raise ValueError(f"unknown model: {name}")


def collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {key: torch.stack([item[key] for item in batch]) for key in batch[0]}


def masked_loss(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = mask.bool()
    bce = F.binary_cross_entropy_with_logits(logits[valid], labels[valid])
    rank_losses = []
    for item_logits, item_labels, item_mask in zip(logits, labels, mask):
        valid_logits = item_logits[item_mask.bool()]
        valid_labels = item_labels[item_mask.bool()]
        pos = valid_labels >= 0.5
        neg = valid_labels <= 0.0
        if pos.any() and neg.any():
            rank_losses.append(F.softplus(-(valid_logits[pos].unsqueeze(1) - valid_logits[neg].unsqueeze(0))).mean())
    rank = torch.stack(rank_losses).mean() if rank_losses else logits.sum() * 0.0
    return bce + 0.35 * rank


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, top_k: int) -> dict[str, float]:
    model.eval()
    rows = []
    latencies = []
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            started = time.perf_counter()
            logits = model(batch)
            if device.type == "cuda":
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - started) * 1000.0 / max(1, logits.shape[0]))
            rows.append(
                score_metrics(
                    logits.detach().cpu(),
                    batch["labels"].detach().cpu(),
                    batch["mask"].detach().cpu(),
                    batch["gold_total"].detach().cpu(),
                    top_k,
                )
            )
    return average(rows) | quantile_metrics(latencies, "latency_ms")


def score_metrics(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor, gold_total: torch.Tensor, top_k: int) -> dict[str, float]:
    available_recall = []
    frame_recall = []
    recall = []
    precision = []
    mrr = []
    for item_logits, item_labels, item_mask, item_gold_total in zip(logits, labels, mask, gold_total):
        valid_logits = item_logits[item_mask.bool()]
        valid_labels = item_labels[item_mask.bool()]
        positives = torch.nonzero(valid_labels >= 0.5, as_tuple=False).flatten()
        total_gold = max(1.0, float(item_gold_total.item()))
        frame_recall.append(float(positives.numel()) / total_gold)
        if positives.numel() == 0:
            continue
        order = torch.argsort(valid_logits, descending=True)
        top = order[:top_k]
        hits = torch.isin(top, positives).sum().item()
        available_recall.append(hits / max(1, positives.numel()))
        recall.append(hits / total_gold)
        precision.append(hits / max(1, top.numel()))
        first = torch.nonzero(torch.isin(order, positives), as_tuple=False).flatten()
        mrr.append(1.0 / (int(first[0].item()) + 1) if first.numel() else 0.0)
    return {
        "recall_at_k": sum(recall) / max(1, len(recall)),
        "available_recall_at_k": sum(available_recall) / max(1, len(available_recall)),
        "frame_recall": sum(frame_recall) / max(1, len(frame_recall)),
        "precision_at_k": sum(precision) / max(1, len(precision)),
        "mrr": sum(mrr) / max(1, len(mrr)),
    }


def average(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = sorted({key for row in rows for key in row})
    return {key: sum(row.get(key, 0.0) for row in rows) / max(1, len(rows)) for key in keys}


def quantile_metrics(values: list[float], prefix: str) -> dict[str, float]:
    if not values:
        return {f"{prefix}_p50": 0.0, f"{prefix}_p95": 0.0}
    ordered = sorted(values)
    return {
        f"{prefix}_p50": ordered[min(len(ordered) - 1, int(math.ceil(0.50 * len(ordered)) - 1))],
        f"{prefix}_p95": ordered[min(len(ordered) - 1, int(math.ceil(0.95 * len(ordered)) - 1))],
    }


def train_one(name: str, config: FrameConfig, train_loader: DataLoader, val_loader: DataLoader, args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    torch.manual_seed(args.seed)
    model = build_model(name, config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    history = []
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            logits = model(batch)
            loss = masked_loss(logits, batch["labels"], batch["mask"])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        metrics = evaluate(model, val_loader, device, config.top_k)
        metrics["train_loss"] = sum(losses) / max(1, len(losses))
        history.append(metrics)
        print(f"model={name} epoch={epoch + 1} " + " ".join(f"{k}={v:.4f}" for k, v in metrics.items()), flush=True)
    return {"name": name, "final": history[-1], "history": history, "parameters": sum(p.numel() for p in model.parameters())}


def run(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    build_started = time.perf_counter()
    use_role_features = not args.hide_role_features
    frames = build_frames(args.cases, args.pool_size, args.frame_size, args.seed, use_role_features)
    build_seconds = time.perf_counter() - build_started
    print(f"built_frames={len(frames)} build_seconds={build_seconds:.2f}", flush=True)
    dataset = MemoryFrameDataset(frames)
    val_count = max(1, int(len(dataset) * args.val_fraction))
    train_count = max(1, len(dataset) - val_count)
    train_set, val_set = random_split(dataset, [train_count, val_count], generator=torch.Generator().manual_seed(args.seed))
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, collate_fn=collate)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    config = FrameConfig(
        frame_size=args.frame_size,
        d_model=args.d_model,
        prefix_tokens=args.prefix_tokens,
        num_heads=args.heads,
        dropout=args.dropout,
        top_k=args.top_k,
    )
    results = [train_one(name, config, train_loader, val_loader, args, device) for name in args.models.split(",") if name]
    return {
        "device": str(device),
        "cases": args.cases,
        "pool_size": args.pool_size,
        "frame_size": args.frame_size,
        "role_features": use_role_features,
        "build_seconds": build_seconds,
        "config": asdict(config),
        "results": results,
    }


def write_markdown(result: dict[str, Any], path: Path) -> None:
    lines = [
        "# Memory Frame Adapter Experiment",
        "",
        f"- device: `{result['device']}`",
        f"- cases: `{result['cases']}`",
        f"- pool_size: `{result['pool_size']}`",
        f"- frame_size: `{result['frame_size']}`",
        f"- role_features: `{result.get('role_features', True)}`",
        f"- build_seconds: `{result.get('build_seconds', 0.0):.2f}`",
        "",
        "| model | params | precision@k | recall@k | frame recall | available recall@k | mrr | p50 ms/case | p95 ms/case |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in result["results"]:
        final = item["final"]
        lines.append(
            f"| {item['name']} | {item['parameters']} | "
            f"{final.get('precision_at_k', 0.0):.4f} | {final.get('recall_at_k', 0.0):.4f} | "
            f"{final.get('frame_recall', 0.0):.4f} | {final.get('available_recall_at_k', 0.0):.4f} | "
            f"{final.get('mrr', 0.0):.4f} | {final.get('latency_ms_p50', 0.0):.4f} | {final.get('latency_ms_p95', 0.0):.4f} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=int, default=6000)
    parser.add_argument("--pool-size", type=int, default=5000)
    parser.add_argument("--frame-size", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--models", default="structured,prefix,cross_attention,lowrank_router")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--prefix-tokens", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=17000)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--hide-role-features", action="store_true")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-md", default="")
    args = parser.parse_args()
    result = run(args)
    body = json.dumps(result, indent=2)
    print(body)
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(body + "\n", encoding="utf-8")
    if args.output_md:
        write_markdown(result, Path(args.output_md))


if __name__ == "__main__":
    main()
