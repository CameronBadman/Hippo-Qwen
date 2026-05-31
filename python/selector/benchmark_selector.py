from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from python.benchmarks.benchmark_librarian import average_metrics, evaluate_ranked, load_rows
from python.selector.dataset import ContextSelectorDataset
from python.selector.model import load_selector


def selector_scores(model, row: dict, budget: int) -> list[tuple[str, float, str]]:
    import torch

    dataset = ContextSelectorDataset.__new__(ContextSelectorDataset)
    dataset.max_candidates = model.config.max_candidates
    dataset.budget_tokens = budget
    dataset.rows = [row]
    item = dataset[0]
    batch = {
        "query": item["query"].unsqueeze(0),
        "candidates": item["candidates"].unsqueeze(0),
        "features": item["features"].unsqueeze(0),
        "mask": item["mask"].unsqueeze(0),
    }
    device = next(model.parameters()).device
    with torch.no_grad():
        batch = {key: value.to(device) for key, value in batch.items()}
        logits = model(batch["query"], batch["candidates"], batch["features"], batch["mask"])
        probs = torch.sigmoid(logits)[0].detach().cpu().tolist()
    scored = [(candidate_id, float(probs[idx]), item["texts"][idx]) for idx, candidate_id in enumerate(item["ids"])]
    return sorted(scored, key=lambda entry: (-entry[1], entry[0]))


def run(args: argparse.Namespace) -> dict:
    import torch

    rows = load_rows(Path(args.dataset), args.limit)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = load_selector(args.checkpoint, device=device)
    metrics = []
    for row in rows:
        ranked = selector_scores(model, row, args.budget)
        metrics.append(evaluate_ranked(row, ranked, args.top_k, args.budget))
    return {
        "dataset": args.dataset,
        "checkpoint": args.checkpoint,
        "cases": len(rows),
        "top_k": args.top_k,
        "budget": args.budget,
        "metrics": {"context_selector": average_metrics(metrics)},
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
