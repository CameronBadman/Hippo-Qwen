from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from python.benchmarks.benchmark_librarian import evaluate_ranked, load_rows, relevant_ids
from python.librarian.features import tokens
from python.selector.benchmark_selector import selector_scores
from python.selector.model import load_selector


def context_ids(row: dict[str, Any], ranked: list[tuple[str, float, str]], default_budget: int) -> list[str]:
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


def candidate_map(row: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {candidate["id"]: candidate for candidate in row.get("candidates", [])}


def role_for(candidate: dict[str, Any] | None) -> str:
    if not candidate:
        return "missing"
    return str(candidate.get("synthetic_role") or "history")


def short_text(value: str, length: int = 140) -> str:
    value = " ".join(value.split())
    return value if len(value) <= length else value[: length - 3] + "..."


def analyze_row(row: dict[str, Any], ranked: list[tuple[str, float, str]], top_k: int, budget: int) -> dict[str, Any]:
    relevant = relevant_ids(row)
    candidates = candidate_map(row)
    top_ids = [candidate_id for candidate_id, _, _ in ranked[:top_k]]
    included = context_ids(row, ranked, budget)
    included_set = set(included)
    false_context = [candidate_id for candidate_id in included if candidate_id not in relevant]
    missed_context = sorted(relevant - included_set)
    metrics = evaluate_ranked(row, ranked, top_k, budget)
    return {
        "metrics": metrics,
        "false_context": false_context,
        "missed_context": missed_context,
        "false_top_k": [candidate_id for candidate_id in top_ids if candidate_id not in relevant],
        "missed_top_k": sorted(relevant - set(top_ids)),
        "false_context_roles": Counter(role_for(candidates.get(candidate_id)) for candidate_id in false_context),
        "missed_context_roles": Counter(role_for(candidates.get(candidate_id)) for candidate_id in missed_context),
        "included": included,
    }


def detail_case(row: dict[str, Any], ranked: list[tuple[str, float, str]], analysis: dict[str, Any], limit: int) -> dict[str, Any]:
    candidates = candidate_map(row)
    relevant = relevant_ids(row)
    scores = {candidate_id: score for candidate_id, score, _ in ranked}
    selected = []
    for candidate_id in analysis["included"][:limit]:
        candidate = candidates.get(candidate_id, {})
        selected.append(
            {
                "id": candidate_id,
                "score": round(scores.get(candidate_id, 0.0), 4),
                "role": role_for(candidate),
                "relevant": candidate_id in relevant,
                "age_days": candidate.get("age_days"),
                "use_count": candidate.get("use_count"),
                "last_outcome": candidate.get("last_outcome"),
                "text": short_text(candidate.get("text", "")),
            }
        )
    missed = []
    for candidate_id in analysis["missed_context"][:limit]:
        candidate = candidates.get(candidate_id, {})
        missed.append(
            {
                "id": candidate_id,
                "score": round(scores.get(candidate_id, 0.0), 4),
                "role": role_for(candidate),
                "age_days": candidate.get("age_days"),
                "use_count": candidate.get("use_count"),
                "last_outcome": candidate.get("last_outcome"),
                "text": short_text(candidate.get("text", "")),
            }
        )
    return {
        "query": (row.get("retrieval_task") or {}).get("query", ""),
        "anchor": short_text((row.get("anchor") or {}).get("text", "")),
        "metrics": analysis["metrics"],
        "selected": selected,
        "missed": missed,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    rows = load_rows(Path(args.dataset), args.limit)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = load_selector(args.checkpoint, device=device)

    false_roles: Counter[str] = Counter()
    missed_roles: Counter[str] = Counter()
    false_top_roles: Counter[str] = Counter()
    missed_top_roles: Counter[str] = Counter()
    role_totals: Counter[str] = Counter()
    role_relevant: Counter[str] = Counter()
    role_context_hits: Counter[str] = Counter()
    cases = []
    metrics = []

    for index, row in enumerate(rows):
        candidates = candidate_map(row)
        relevant = relevant_ids(row)
        for candidate in candidates.values():
            role_totals[role_for(candidate)] += 1
        for candidate_id in relevant:
            role_relevant[role_for(candidates.get(candidate_id))] += 1

        ranked = selector_scores(model, row, args.budget)
        analysis = analyze_row(row, ranked, args.top_k, args.budget)
        metrics.append(analysis["metrics"])
        false_roles.update(analysis["false_context_roles"])
        missed_roles.update(analysis["missed_context_roles"])
        false_top_roles.update(role_for(candidates.get(candidate_id)) for candidate_id in analysis["false_top_k"])
        missed_top_roles.update(role_for(candidates.get(candidate_id)) for candidate_id in analysis["missed_top_k"])
        for candidate_id in set(analysis["included"]) & relevant:
            role_context_hits[role_for(candidates.get(candidate_id))] += 1
        cases.append((index, analysis["metrics"]["context_recall"], analysis["metrics"]["context_precision"], analysis["metrics"]["noise"], row, ranked, analysis))

    worst = sorted(cases, key=lambda item: (item[1], item[2], -item[3]))[: args.worst_cases]
    return {
        "dataset": args.dataset,
        "checkpoint": args.checkpoint,
        "cases": len(rows),
        "top_k": args.top_k,
        "budget": args.budget,
        "average_metrics": average(metrics),
        "role_totals": dict(role_totals),
        "role_relevant": dict(role_relevant),
        "role_context_hits": dict(role_context_hits),
        "false_context_roles": dict(false_roles),
        "missed_context_roles": dict(missed_roles),
        "false_top_k_roles": dict(false_top_roles),
        "missed_top_k_roles": dict(missed_top_roles),
        "worst_cases": [
            {"index": index, **detail_case(row, ranked, analysis, args.case_items)}
            for index, _, _, _, row, ranked, analysis in worst
        ],
    }


def average(items: list[dict[str, float]]) -> dict[str, float]:
    if not items:
        return {}
    keys = sorted(items[0])
    return {key: sum(item[key] for item in items) / len(items) for key in keys}


def write_markdown(result: dict[str, Any], path: Path) -> None:
    lines = [
        "# Selector Error Analysis",
        "",
        f"- dataset: `{result['dataset']}`",
        f"- checkpoint: `{result['checkpoint']}`",
        f"- cases: `{result['cases']}`",
        "",
        "## Average Metrics",
        "",
        "| recall@k | mrr | context precision | context recall | noise |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    metrics = result["average_metrics"]
    lines.append(
        f"| {metrics.get('recall_at_k', 0.0):.4f} | {metrics.get('mrr', 0.0):.4f} | "
        f"{metrics.get('context_precision', 0.0):.4f} | {metrics.get('context_recall', 0.0):.4f} | "
        f"{metrics.get('noise', 0.0):.2f} |"
    )
    for title, key in [
        ("False Context Roles", "false_context_roles"),
        ("Missed Context Roles", "missed_context_roles"),
        ("Context Hits By Relevant Role", "role_context_hits"),
    ]:
        lines.extend(["", f"## {title}", "", "| role | count |", "| --- | ---: |"])
        for role, count in sorted(result[key].items(), key=lambda item: (-item[1], item[0])):
            lines.append(f"| {role} | {count} |")

    lines.extend(["", "## Worst Cases", ""])
    for case in result["worst_cases"]:
        lines.append(f"### Case {case['index']}")
        lines.append(f"- context recall: `{case['metrics']['context_recall']:.4f}`")
        lines.append(f"- context precision: `{case['metrics']['context_precision']:.4f}`")
        lines.append(f"- anchor: {case['anchor']}")
        lines.append("")
        lines.append("| selected role | relevant | score | text |")
        lines.append("| --- | ---: | ---: | --- |")
        for item in case["selected"]:
            lines.append(f"| {item['role']} | {str(item['relevant']).lower()} | {item['score']:.4f} | {item['text']} |")
        if case["missed"]:
            lines.append("")
            lines.append("| missed role | score | text |")
            lines.append("| --- | ---: | --- |")
            for item in case["missed"]:
                lines.append(f"| {item['role']} | {item['score']:.4f} | {item['text']} |")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--budget", type=int, default=90)
    parser.add_argument("--worst-cases", type=int, default=5)
    parser.add_argument("--case-items", type=int, default=10)
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
