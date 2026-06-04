from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from python.benchmarks.benchmark_librarian import average_metrics, evaluate_ranked, load_rows, vector_scores
from python.librarian.features import (
    activation_mask_for_text,
    cluster_score,
    cosine,
    embed_text,
    ensure_embedding,
    jaccard,
    metadata_score,
    state_features,
    tokens,
)


Ranked = list[tuple[str, float, str]]


def activation_overlap(left: int, right: int) -> float:
    if not left or not right:
        return 0.0
    intersection = (left & right).bit_count()
    union = (left | right).bit_count()
    if union == 0:
        return 0.0
    return intersection / union


def state_score(card: dict[str, Any]) -> float:
    state = state_features({}, card)
    return (
        0.18 * state["candidate_use_norm"]
        + 0.16 * state["candidate_evidence_norm"]
        + 0.12 * max(0.0, state["last_outcome_value"])
        + 0.08 * state["protected_flag"]
        + 0.07 * state["recency_score"]
        - 0.22 * state["stale_unused_flag"]
        - 0.10 * max(0.0, -state["last_outcome_value"])
    )


def sparse_basin_scores(row: dict[str, Any]) -> Ranked:
    task = row.get("retrieval_task") or {}
    query = task.get("query") or row["anchor"]["text"]
    query_embedding = embed_text(query)
    query_mask = activation_mask_for_text(query)
    anchor = dict(row["anchor"])
    ensure_embedding(anchor)
    scored: Ranked = []
    for candidate in row.get("candidates", []):
        item = dict(candidate)
        ensure_embedding(item)
        semantic = cosine(query_embedding, item["embedding"])
        lexical = jaccard(query, item.get("text", ""))
        activation = activation_overlap(query_mask, activation_mask_for_text(item.get("text", "")))
        cluster = cluster_score(anchor, item)
        meta = metadata_score(anchor, item)
        mismatch_penalty = 0.18 if anchor.get("cluster") and item.get("cluster") and anchor.get("cluster") != item.get("cluster") else 0.0
        score = (
            0.27 * semantic
            + 0.24 * activation
            + 0.18 * lexical
            + 0.13 * cluster
            + 0.09 * meta
            + 0.09 * float(item.get("importance") or 0.5)
            + state_score(item)
            - mismatch_penalty
        )
        scored.append((item["id"], float(score), item.get("text", "")))
    return sorted(scored, key=lambda item: (-item[1], item[0]))


def edge_activation(edge: dict[str, Any]) -> int:
    if edge.get("activation_mask") is not None:
        try:
            return int(edge["activation_mask"])
        except (TypeError, ValueError):
            return 0
    return activation_mask_for_text(str(edge.get("activation_text") or ""))


def edge_type_boost(edge_type: str) -> float:
    if edge_type in {"preference", "correction"}:
        return 1.22
    if edge_type in {"same_context", "same_cluster"}:
        return 1.12
    if edge_type == "temporal_next":
        return 0.84
    return 1.0


def associative_recall_scores(row: dict[str, Any], seed_count: int = 8) -> Ranked:
    candidates = {candidate["id"]: dict(candidate) for candidate in row.get("candidates", [])}
    sparse = sparse_basin_scores(row)
    dense = vector_scores(row)
    base_scores: dict[str, float] = {}
    texts: dict[str, str] = {}
    for rank, (candidate_id, score, text) in enumerate(sparse):
        base_scores[candidate_id] = max(base_scores.get(candidate_id, -99.0), score + 0.012 * max(0, seed_count - rank))
        texts[candidate_id] = text
    for rank, (candidate_id, score, text) in enumerate(dense):
        base_scores[candidate_id] = max(base_scores.get(candidate_id, -99.0), 0.72 * score + 0.008 * max(0, seed_count - rank))
        texts[candidate_id] = text

    query = (row.get("retrieval_task") or {}).get("query") or row["anchor"]["text"]
    query_mask = activation_mask_for_text(query)
    graph = row.get("memory_graph") or {}
    max_depth = int(graph.get("max_depth") or 3)
    outgoing: dict[str, list[dict[str, Any]]] = {}
    for edge in graph.get("edges") or []:
        source = str(edge.get("source") or "")
        target = str(edge.get("target") or "")
        if source in candidates and target in candidates:
            outgoing.setdefault(source, []).append(edge)

    best = dict(base_scores)
    seeds = sorted(base_scores.items(), key=lambda item: (-item[1], item[0]))[:seed_count]
    frontier = [(candidate_id, score, [candidate_id]) for candidate_id, score in seeds]
    for depth in range(max_depth):
        next_frontier: list[tuple[str, float, list[str]]] = []
        for current_id, current_score, path in frontier:
            for edge in outgoing.get(current_id, []):
                target_id = str(edge.get("target") or "")
                if target_id in path:
                    continue
                target = candidates[target_id]
                activation = activation_overlap(query_mask, edge_activation(edge))
                edge_weight = float(edge.get("weight") or 0.0)
                confidence = float(edge.get("confidence") or 0.5)
                hop_gain = (
                    edge_weight
                    * edge_type_boost(str(edge.get("type") or "used_with"))
                    * (0.70 + 0.45 * activation)
                    * (0.70 + 0.30 * confidence)
                    / (1.25 + depth)
                )
                target_bonus = 0.07 * float(target.get("importance") or 0.5) + state_score(target)
                score = 0.62 * current_score + hop_gain + target_bonus
                if score > best.get(target_id, -99.0):
                    best[target_id] = score
                    texts[target_id] = target.get("text", "")
                    next_frontier.append((target_id, score, path + [target_id]))
        frontier = next_frontier

    return sorted([(candidate_id, score, texts.get(candidate_id, "")) for candidate_id, score in best.items()], key=lambda item: (-item[1], item[0]))


def budgeted_ids(row: dict[str, Any], ranked: Ranked, default_budget: int) -> list[str]:
    budget = int((row.get("retrieval_task") or {}).get("budget") or default_budget)
    used = 0
    included = []
    for candidate_id, _, text in ranked:
        cost = max(1, len(tokens(text)))
        if used + cost > budget:
            break
        included.append(candidate_id)
        used += cost
    return included


def role_by_id(row: dict[str, Any]) -> dict[str, str]:
    return {candidate["id"]: str(candidate.get("synthetic_role") or "history") for candidate in row.get("candidates", [])}


def multihop_metrics(row: dict[str, Any], ranked: Ranked, top_k: int, default_budget: int) -> dict[str, float]:
    task = row.get("retrieval_task") or {}
    bridge_ids = set(task.get("bridge_ids") or [])
    target_ids = set(task.get("target_ids") or [])
    top_ids = [candidate_id for candidate_id, _, _ in ranked[:top_k]]
    budget_ids = budgeted_ids(row, ranked, default_budget)
    top_set = set(top_ids)
    budget_set = set(budget_ids)
    roles = role_by_id(row)
    stale_roles = {"stale_high_similarity_negative", "obsolete_preference_negative", "old_helpful_preference_negative"}
    wrong_roles = {
        "popular_wrong_project_negative",
        "popular_wrong_context_negative",
        "contradicted_preference_negative",
        "same_project_wrong_preference_negative",
    }
    return {
        "bridge_recall_at_k": len(bridge_ids & top_set) / max(1, len(bridge_ids)),
        "target_recall_at_k": len(target_ids & top_set) / max(1, len(target_ids)),
        "path_success_at_k": 1.0 if bridge_ids & top_set and target_ids & top_set else 0.0,
        "bridge_context_recall": len(bridge_ids & budget_set) / max(1, len(bridge_ids)),
        "target_context_recall": len(target_ids & budget_set) / max(1, len(target_ids)),
        "path_context_success": 1.0 if bridge_ids & budget_set and target_ids & budget_set else 0.0,
        "stale_exposure": float(sum(1 for candidate_id in budget_ids if roles.get(candidate_id) in stale_roles)),
        "wrong_context_exposure": float(sum(1 for candidate_id in budget_ids if roles.get(candidate_id) in wrong_roles)),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    rows = load_rows(Path(args.dataset), args.limit)
    scorers = {
        "vector_only": lambda row: vector_scores(row),
        "sparse_basin": lambda row: sparse_basin_scores(row),
        "associative_recall": lambda row: associative_recall_scores(row, args.seed_count),
    }
    retrieval: dict[str, list[dict[str, float]]] = {name: [] for name in scorers}
    multihop: dict[str, list[dict[str, float]]] = {name: [] for name in scorers}
    for row in rows:
        for name, scorer in scorers.items():
            ranked = scorer(row)
            retrieval[name].append(evaluate_ranked(row, ranked, args.top_k, args.budget))
            multihop[name].append(multihop_metrics(row, ranked, args.top_k, args.budget))
    return {
        "dataset": args.dataset,
        "cases": len(rows),
        "top_k": args.top_k,
        "budget": args.budget,
        "seed_count": args.seed_count,
        "metrics": {name: average_metrics(items) for name, items in retrieval.items()},
        "multihop_metrics": {name: average_metrics(items) for name, items in multihop.items()},
    }


def write_markdown(result: dict[str, Any], path: Path) -> None:
    lines = [
        "# Hippocampus Retrieval Scorecard",
        "",
        f"- dataset: `{result['dataset']}`",
        f"- cases: `{result['cases']}`",
        f"- top_k: `{result['top_k']}`",
        f"- budget: `{result['budget']}`",
        f"- seed_count: `{result['seed_count']}`",
        "",
        "## Retrieval",
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
    lines.extend(
        [
            "",
            "## Associative Recall",
            "",
            "| method | bridge@k | target@k | path@k | bridge ctx | target ctx | path ctx | stale | wrong ctx |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for name, metrics in result["multihop_metrics"].items():
        lines.append(
            "| "
            + " | ".join(
                [
                    name,
                    f"{metrics.get('bridge_recall_at_k', 0.0):.4f}",
                    f"{metrics.get('target_recall_at_k', 0.0):.4f}",
                    f"{metrics.get('path_success_at_k', 0.0):.4f}",
                    f"{metrics.get('bridge_context_recall', 0.0):.4f}",
                    f"{metrics.get('target_context_recall', 0.0):.4f}",
                    f"{metrics.get('path_context_success', 0.0):.4f}",
                    f"{metrics.get('stale_exposure', 0.0):.2f}",
                    f"{metrics.get('wrong_context_exposure', 0.0):.2f}",
                ]
            )
            + " |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/synthetic/associative_multihop.jsonl")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--budget", type=int, default=90)
    parser.add_argument("--seed-count", type=int, default=8)
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
