from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from python.librarian.features import cosine, embed_text, ensure_embedding, heuristic_action, tokens


def load_rows(path: Path, limit: int) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit > 0 and len(rows) >= limit:
                break
    return rows


def relevant_ids(row: dict[str, Any]) -> set[str]:
    task = row.get("retrieval_task") or {}
    explicit = set(task.get("relevant_ids") or [])
    if explicit:
        return explicit
    labels = row.get("labels") or {}
    attach = labels.get("attach") or []
    rank = labels.get("rank") or attach
    return {
        candidate["id"]
        for candidate, attach_value, rank_value in zip(row.get("candidates", []), attach, rank)
        if attach_value >= 0.5 and rank_value >= 0.5
    }


def vector_scores(row: dict[str, Any]) -> list[tuple[str, float, str]]:
    query = ((row.get("retrieval_task") or {}).get("query") or row["anchor"]["text"])
    query_embedding = embed_text(query)
    scored = []
    for candidate in row.get("candidates", []):
        ensure_embedding(candidate)
        scored.append((candidate["id"], cosine(query_embedding, candidate["embedding"]), candidate.get("text", "")))
    return sorted(scored, key=lambda item: (-item[1], item[0]))


def heuristic_scores(row: dict[str, Any]) -> list[tuple[str, float, str]]:
    anchor = dict(row["anchor"])
    ensure_embedding(anchor)
    scored = []
    for candidate in row.get("candidates", []):
        item = dict(candidate)
        ensure_embedding(item)
        action = heuristic_action(anchor, item)
        scored.append((candidate["id"], float(action["connect_score"]), candidate.get("text", "")))
    return sorted(scored, key=lambda item: (-item[1], item[0]))


def transformer_scores(model: Any, row: dict[str, Any]) -> list[tuple[str, float, str]]:
    import torch

    from python.librarian.inference import tensorize_payload

    tensors, candidates = tensorize_payload(
        {"anchor": row["anchor"], "candidates": row.get("candidates", [])},
        model.config.max_candidates,
        model.config.feature_dim,
    )
    with torch.no_grad():
        device = next(model.parameters()).device
        tensors = {key: value.to(device) for key, value in tensors.items()}
        outputs = model(**tensors)
    probs = torch.sigmoid(outputs["attach_logits"])[0].detach().cpu().tolist()
    scored = [(candidate["id"], float(probs[idx]), candidate.get("text", "")) for idx, candidate in enumerate(candidates)]
    return sorted(scored, key=lambda item: (-item[1], item[0]))


def evaluate_ranked(row: dict[str, Any], ranked: list[tuple[str, float, str]], top_k: int, default_budget: int) -> dict[str, float]:
    relevant = relevant_ids(row)
    if not relevant:
        return {
            "recall_at_k": 0.0,
            "mrr": 0.0,
            "context_precision": 0.0,
            "context_recall": 0.0,
            "budget_hit": 1.0,
            "noise": 0.0,
        }

    top = [item[0] for item in ranked[:top_k]]
    recall_at_k = len(relevant & set(top)) / len(relevant)
    mrr = 0.0
    for position, (candidate_id, _, _) in enumerate(ranked, start=1):
        if candidate_id in relevant:
            mrr = 1.0 / position
            break

    budget = int((row.get("retrieval_task") or {}).get("budget") or default_budget)
    included: list[str] = []
    used = 0
    for candidate_id, _, text in ranked:
        cost = max(1, len(tokens(text)))
        if used + cost > budget:
            break
        included.append(candidate_id)
        used += cost
    included_set = set(included)
    relevant_in_context = len(relevant & included_set)
    context_precision = relevant_in_context / max(1, len(included))
    context_recall = relevant_in_context / len(relevant)
    return {
        "recall_at_k": recall_at_k,
        "mrr": mrr,
        "context_precision": context_precision,
        "context_recall": context_recall,
        "budget_hit": 1.0 if used <= budget else 0.0,
        "noise": float(len(included) - relevant_in_context),
    }


def average_metrics(items: list[dict[str, float]]) -> dict[str, float]:
    if not items:
        return {}
    keys = sorted(items[0])
    return {key: sum(item[key] for item in items) / len(items) for key in keys}


def run(args: argparse.Namespace) -> dict[str, Any]:
    rows = load_rows(Path(args.dataset), args.limit)
    scorers = {
        "vector_only": vector_scores,
        "heuristic_graph": heuristic_scores,
    }
    model = None
    if args.checkpoint:
        import torch

        from python.librarian.model import load_checkpoint

        device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
        model = load_checkpoint(args.checkpoint, device=device)
        scorers["transformer_graph"] = lambda row: transformer_scores(model, row)

    results: dict[str, list[dict[str, float]]] = {name: [] for name in scorers}
    for row in rows:
        for name, scorer in scorers.items():
            ranked = scorer(row)
            results[name].append(evaluate_ranked(row, ranked, args.top_k, args.budget))

    return {
        "dataset": args.dataset,
        "checkpoint": args.checkpoint,
        "cases": len(rows),
        "top_k": args.top_k,
        "budget": args.budget,
        "metrics": {name: average_metrics(items) for name, items in results.items()},
    }


def write_markdown(result: dict[str, Any], path: Path) -> None:
    lines = [
        "# Librarian Retrieval Benchmark",
        "",
        f"- dataset: `{result['dataset']}`",
        f"- checkpoint: `{result['checkpoint'] or 'none'}`",
        f"- cases: `{result['cases']}`",
        f"- top_k: `{result['top_k']}`",
        "",
        "| method | recall@k | mrr | context precision | context recall | noise |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, metrics in result["metrics"].items():
        lines.append(
            "| "
            + " | ".join(
                [
                    name,
                    f"{metrics.get('recall_at_k', 0.0):.4f}",
                    f"{metrics.get('mrr', 0.0):.4f}",
                    f"{metrics.get('context_precision', 0.0):.4f}",
                    f"{metrics.get('context_recall', 0.0):.4f}",
                    f"{metrics.get('noise', 0.0):.2f}",
                ]
            )
            + " |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/synthetic/librarian_cases.jsonl")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--budget", type=int, default=90)
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-md", default="")
    parser.add_argument("--cpu", action="store_true")
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
