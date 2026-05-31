from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from python.benchmarks.benchmark_librarian import average_metrics, evaluate_ranked, load_rows
from python.librarian.features import tokens
from python.selector.calibration import default_auxiliary_thresholds, summarize_auxiliary, tune_auxiliary_thresholds
from python.selector.dataset import AUXILIARY_LABELS, CONTEXT_REASONS, ContextSelectorDataset
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


def role_exposure(row: dict, ranked: list[tuple[str, float, str]], top_k: int, default_budget: int) -> dict[str, dict[str, int]]:
    relevant = relevant_ids_for_row(row)
    candidates = row.get("candidates", [])
    roles = {candidate["id"]: str(candidate.get("synthetic_role") or "history") for candidate in candidates}
    totals: dict[str, dict[str, int]] = {}
    for candidate in candidates:
        role = roles[candidate["id"]]
        totals.setdefault(role, {"total": 0, "relevant": 0, "top_k": 0, "budget": 0})
        totals[role]["total"] += 1
        if candidate["id"] in relevant:
            totals[role]["relevant"] += 1

    for candidate_id, _, _ in ranked[:top_k]:
        role = roles.get(candidate_id, "unknown")
        totals.setdefault(role, {"total": 0, "relevant": 0, "top_k": 0, "budget": 0})
        totals[role]["top_k"] += 1

    budget = int((row.get("retrieval_task") or {}).get("budget") or default_budget)
    used = 0
    for candidate_id, _, text in ranked:
        cost = max(1, len(tokens(text)))
        if used + cost > budget:
            break
        role = roles.get(candidate_id, "unknown")
        totals.setdefault(role, {"total": 0, "relevant": 0, "top_k": 0, "budget": 0})
        totals[role]["budget"] += 1
        used += cost
    return totals


def relevant_ids_for_row(row: dict) -> set[str]:
    task = row.get("retrieval_task") or {}
    return set(task.get("relevant_ids") or [])


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


def auxiliary_report(model, row: dict, budget: int, thresholds: dict[str, float]) -> dict:
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
        scores = torch.sigmoid(outputs["auxiliary_logits"])[0].detach().cpu()
    valid = item["mask"]
    expected = item["auxiliary"] >= 0.5
    report: dict[str, dict[str, int]] = {}
    samples: dict[str, list[tuple[float, bool]]] = {}
    for idx, label in enumerate(AUXILIARY_LABELS):
        threshold = thresholds.get(label, 0.5)
        pred = scores[valid, idx] >= threshold
        exp = expected[valid, idx]
        report[label] = {
            "tp": int((pred & exp).sum().item()),
            "fp": int((pred & ~exp).sum().item()),
            "fn": int((~pred & exp).sum().item()),
            "tn": int((~pred & ~exp).sum().item()),
        }
        samples[label] = [(float(score), bool(value)) for score, value in zip(scores[valid, idx].tolist(), exp.tolist())]
    return {"counts": report, "samples": samples}


def run(args: argparse.Namespace) -> dict:
    import torch

    rows = load_rows(Path(args.dataset), args.limit)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = load_selector(args.checkpoint, device=device)
    metrics = []
    reason_totals = {"total": 0, "correct": 0}
    confusion: dict[str, dict[str, int]] = {}
    auxiliary_totals = {label: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for label in AUXILIARY_LABELS}
    auxiliary_samples: dict[str, list[tuple[float, bool]]] = {label: [] for label in AUXILIARY_LABELS}
    role_totals: dict[str, dict[str, int]] = {}
    if args.use_checkpoint_auxiliary_thresholds:
        auxiliary_thresholds = getattr(model, "auxiliary_thresholds", None) or default_auxiliary_thresholds(args.auxiliary_threshold)
    else:
        auxiliary_thresholds = default_auxiliary_thresholds(args.auxiliary_threshold)
    for row in rows:
        ranked = selector_scores(model, row, args.budget)
        metrics.append(evaluate_ranked(row, ranked, args.top_k, args.budget))
        exposure = role_exposure(row, ranked, args.top_k, args.budget)
        for role, counts in exposure.items():
            role_totals.setdefault(role, {"total": 0, "relevant": 0, "top_k": 0, "budget": 0})
            for key, count in counts.items():
                role_totals[role][key] += count
        report = reason_report(model, row, args.budget)
        reason_totals["total"] += report["total"]
        reason_totals["correct"] += report["correct"]
        for expected, predicted_counts in report["confusion"].items():
            confusion.setdefault(expected, {})
            for predicted, count in predicted_counts.items():
                confusion[expected][predicted] = confusion[expected].get(predicted, 0) + count
        auxiliary = auxiliary_report(model, row, args.budget, auxiliary_thresholds)
        for label, counts in auxiliary["counts"].items():
            for key, count in counts.items():
                auxiliary_totals[label][key] += count
        for label, samples in auxiliary["samples"].items():
            auxiliary_samples[label].extend(samples)
    auxiliary_metrics = summarize_auxiliary(auxiliary_totals)
    auxiliary_metrics["thresholds"] = auxiliary_thresholds
    if args.tune_auxiliary_thresholds:
        auxiliary_metrics["tuned"] = tune_auxiliary_thresholds(auxiliary_samples)
    return {
        "dataset": args.dataset,
        "checkpoint": args.checkpoint,
        "cases": len(rows),
        "top_k": args.top_k,
        "budget": args.budget,
        "metrics": {"context_selector": average_metrics(metrics)},
        "reason_metrics": summarize_reasons(reason_totals, confusion),
        "auxiliary_metrics": auxiliary_metrics,
        "role_metrics": summarize_roles(role_totals),
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


def summarize_roles(role_totals: dict[str, dict[str, int]]) -> dict[str, dict[str, float | int]]:
    summary = {}
    for role, counts in sorted(role_totals.items()):
        total = max(1, counts["total"])
        summary[role] = {
            "total": counts["total"],
            "relevant": counts["relevant"],
            "top_k": counts["top_k"],
            "budget": counts["budget"],
            "relevant_rate": counts["relevant"] / total,
            "top_k_rate": counts["top_k"] / total,
            "budget_rate": counts["budget"] / total,
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/synthetic/librarian_hard_cases.jsonl")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--budget", type=int, default=90)
    parser.add_argument("--auxiliary-threshold", type=float, default=0.5)
    parser.add_argument("--use-checkpoint-auxiliary-thresholds", action="store_true")
    parser.add_argument("--no-use-checkpoint-auxiliary-thresholds", dest="use_checkpoint_auxiliary_thresholds", action="store_false")
    parser.set_defaults(use_checkpoint_auxiliary_thresholds=True)
    parser.add_argument("--tune-auxiliary-thresholds", action="store_true")
    parser.add_argument("--no-tune-auxiliary-thresholds", dest="tune_auxiliary_thresholds", action="store_false")
    parser.set_defaults(tune_auxiliary_thresholds=True)
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
