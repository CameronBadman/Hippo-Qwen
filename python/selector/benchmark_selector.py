from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from python.benchmarks.benchmark_librarian import average_metrics, evaluate_ranked, load_rows
from python.selector.dataset import CONTEXT_REASONS, ContextSelectorDataset
from python.selector.model import load_selector


def selector_scores(model, row: dict, budget: int) -> list[tuple[str, float, str]]:
    import torch

    dataset = ContextSelectorDataset.__new__(ContextSelectorDataset)
    dataset.max_candidates = model.config.max_candidates
    dataset.budget_tokens = budget
    dataset.feature_dim = model.config.feature_dim
    dataset.rows = [row]
    item = dataset[0]
    batch = {
        "query": item["query"].unsqueeze(0),
        "anchor": item["anchor"].unsqueeze(0),
        "candidates": item["candidates"].unsqueeze(0),
        "features": item["features"].unsqueeze(0),
        "mask": item["mask"].unsqueeze(0),
    }
    device = next(model.parameters()).device
    with torch.no_grad():
        batch = {key: value.to(device) for key, value in batch.items()}
        outputs = model(batch["query"], batch["anchor"], batch["candidates"], batch["features"], batch["mask"])
        probs = torch.sigmoid(outputs["select_logits"])[0].detach().cpu().tolist()
    scored = [(candidate_id, float(probs[idx]), item["texts"][idx]) for idx, candidate_id in enumerate(item["ids"])]
    return sorted(scored, key=lambda entry: (-entry[1], entry[0]))


def reason_report(model, row: dict, budget: int) -> dict:
    import torch

    dataset = ContextSelectorDataset.__new__(ContextSelectorDataset)
    dataset.max_candidates = model.config.max_candidates
    dataset.budget_tokens = budget
    dataset.feature_dim = model.config.feature_dim
    dataset.rows = [row]
    item = dataset[0]
    batch = {
        "query": item["query"].unsqueeze(0),
        "anchor": item["anchor"].unsqueeze(0),
        "candidates": item["candidates"].unsqueeze(0),
        "features": item["features"].unsqueeze(0),
        "mask": item["mask"].unsqueeze(0),
    }
    device = next(model.parameters()).device
    with torch.no_grad():
        batch = {key: value.to(device) for key, value in batch.items()}
        outputs = model(batch["query"], batch["anchor"], batch["candidates"], batch["features"], batch["mask"])
        predicted = outputs["reason_logits"].argmax(dim=-1)[0].detach().cpu()
    valid = item["mask"] & (item["reason"] >= 0)
    total = int(valid.sum().item())
    correct = int((predicted[valid] == item["reason"][valid]).sum().item()) if total else 0
    confusion: dict[str, dict[str, int]] = {}
    for expected_id, predicted_id in zip(item["reason"][valid].tolist(), predicted[valid].tolist()):
        expected = CONTEXT_REASONS[int(expected_id)]
        actual = CONTEXT_REASONS[int(predicted_id)]
        confusion.setdefault(expected, {})
        confusion[expected][actual] = confusion[expected].get(actual, 0) + 1
    return {"total": total, "correct": correct, "confusion": confusion}


def run(args: argparse.Namespace) -> dict:
    import torch

    rows = load_rows(Path(args.dataset), args.limit)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = load_selector(args.checkpoint, device=device)
    metrics = []
    reason_totals = {"total": 0, "correct": 0}
    confusion: dict[str, dict[str, int]] = {}
    for row in rows:
        ranked = selector_scores(model, row, args.budget)
        metrics.append(evaluate_ranked(row, ranked, args.top_k, args.budget))
        report = reason_report(model, row, args.budget)
        reason_totals["total"] += report["total"]
        reason_totals["correct"] += report["correct"]
        for expected, predicted_counts in report["confusion"].items():
            confusion.setdefault(expected, {})
            for predicted, count in predicted_counts.items():
                confusion[expected][predicted] = confusion[expected].get(predicted, 0) + count
    return {
        "dataset": args.dataset,
        "checkpoint": args.checkpoint,
        "cases": len(rows),
        "top_k": args.top_k,
        "budget": args.budget,
        "metrics": {"context_selector": average_metrics(metrics)},
        "reason_metrics": summarize_reasons(reason_totals, confusion),
    }


def summarize_reasons(reason_totals: dict[str, int], confusion: dict[str, dict[str, int]]) -> dict:
    per_class_recall = {}
    for expected in CONTEXT_REASONS:
        predicted_counts = confusion.get(expected, {})
        total = sum(predicted_counts.values())
        if total > 0:
            per_class_recall[expected] = predicted_counts.get(expected, 0) / total
    return {
        "accuracy": reason_totals["correct"] / max(1, reason_totals["total"]),
        "macro_recall": sum(per_class_recall.values()) / max(1, len(per_class_recall)),
        "per_class_recall": per_class_recall,
        "correct": reason_totals["correct"],
        "total": reason_totals["total"],
        "confusion": confusion,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/synthetic/librarian_hard_cases.jsonl")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--budget", type=int, default=90)
    parser.add_argument("--output-json", default="")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    result = run(args)
    body = json.dumps(result, indent=2)
    print(body)
    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
