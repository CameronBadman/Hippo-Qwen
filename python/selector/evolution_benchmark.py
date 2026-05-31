from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from python.benchmarks.benchmark_librarian import average_metrics, evaluate_ranked, heuristic_scores, load_rows, vector_scores
from python.librarian.features import clamp, jaccard, tokens
from python.synthetic.generate import build_cases

Ranked = list[tuple[str, float, str]]
Scorer = Callable[[dict[str, Any]], Ranked]
PREFERENCE_MARKERS = {"preference", "prefer", "prefers", "preferred", "wants", "likes", "avoids", "needs"}
EVOLUTION_POLICIES = {"off", "always", "uncertainty_gated", "low_confidence_only"}


class MemoryState:
    def __init__(self) -> None:
        self.items: dict[str, dict[str, Any]] = {}

    def apply(self, row: dict[str, Any]) -> dict[str, Any]:
        evolved = copy.deepcopy(row)
        evolved["anchor"] = self.apply_card(evolved["anchor"])
        evolved["candidates"] = [self.apply_card(candidate) for candidate in evolved.get("candidates", [])]
        return evolved

    def apply_card(self, card: dict[str, Any]) -> dict[str, Any]:
        memory_id = str(card.get("id") or "")
        state = self.items.get(memory_id)
        if not state:
            return card
        card["use_count"] = int(card.get("use_count") or 0) + state["use_count_delta"]
        card["evidence_count"] = int(card.get("evidence_count") or 0) + state["evidence_count_delta"]
        card["importance"] = clamp(float(card.get("importance") or 0.5) + state["importance_delta"], 0.05, 0.95)
        if state["last_outcome"]:
            card["last_outcome"] = state["last_outcome"]
        if state["protected"]:
            card["protected"] = True
        return card

    def update(self, memory_id: str, outcome: str, protected: bool = False) -> None:
        state = self.items.setdefault(
            memory_id,
            {
                "use_count_delta": 0,
                "evidence_count_delta": 0,
                "importance_delta": 0.0,
                "score_bias": 0.0,
                "last_outcome": "",
                "protected": False,
                "feedback": Counter(),
            },
        )
        state["feedback"][outcome] += 1
        state["last_outcome"] = outcome
        state["protected"] = bool(state["protected"] or protected)
        if outcome == "helpful":
            state["use_count_delta"] += 1
            state["evidence_count_delta"] += 1
            state["importance_delta"] = clamp(state["importance_delta"] + 0.04, -0.45, 0.45)
            state["score_bias"] = clamp(state["score_bias"] + 0.16, -0.9, 0.9)
        elif outcome == "corrected":
            state["use_count_delta"] += 1
            state["evidence_count_delta"] += 2
            state["importance_delta"] = clamp(state["importance_delta"] + 0.07, -0.45, 0.45)
            state["score_bias"] = clamp(state["score_bias"] + 0.22, -0.9, 0.9)
        else:
            state["importance_delta"] = clamp(state["importance_delta"] - 0.06, -0.45, 0.45)
            state["score_bias"] = clamp(state["score_bias"] - 0.20, -0.9, 0.9)

    def update_signature(self, anchor: dict[str, Any], candidate: dict[str, Any], outcome: str) -> None:
        signature = relationship_signature(anchor, candidate)
        state = self.items.setdefault(
            f"signature:{signature}",
            {
                "use_count_delta": 0,
                "evidence_count_delta": 0,
                "importance_delta": 0.0,
                "score_bias": 0.0,
                "last_outcome": "",
                "protected": False,
                "feedback": Counter(),
            },
        )
        state["feedback"][outcome] += 1
        if outcome == "helpful":
            state["score_bias"] = clamp(state["score_bias"] + 0.035, -0.35, 0.35)
        elif outcome == "corrected":
            state["score_bias"] = clamp(state["score_bias"] + 0.05, -0.35, 0.35)
        else:
            state["score_bias"] = clamp(state["score_bias"] - 0.045, -0.35, 0.35)

    def score_bias(self, row: dict[str, Any], candidate_id: str) -> float:
        candidates = {candidate["id"]: candidate for candidate in row.get("candidates", [])}
        candidate = candidates.get(candidate_id)
        if not candidate:
            return 0.0
        memory = self.items.get(candidate_id, {})
        signature = self.items.get(f"signature:{relationship_signature(row['anchor'], candidate)}", {})
        return float(memory.get("score_bias") or 0.0) + float(signature.get("score_bias") or 0.0)

    def rerank(self, row: dict[str, Any], ranked: Ranked, bias_scale: float) -> Ranked:
        adjusted = [(candidate_id, score + bias_scale * self.score_bias(row, candidate_id), text) for candidate_id, score, text in ranked]
        return sorted(adjusted, key=lambda item: (-item[1], item[0]))

    def summary(self) -> dict[str, Any]:
        feedback = Counter()
        for key, state in self.items.items():
            if key.startswith("signature:"):
                continue
            feedback.update(state["feedback"])
        return {
            "memories_touched": len([key for key in self.items if not key.startswith("signature:")]),
            "signatures_touched": len([key for key in self.items if key.startswith("signature:")]),
            "feedback": dict(sorted(feedback.items())),
        }


def preference_terms(text: str) -> set[str]:
    raw_tokens = tokens(text)
    terms: set[str] = set()
    for idx, token in enumerate(raw_tokens):
        if token in PREFERENCE_MARKERS:
            terms.update(raw_tokens[idx + 1 : idx + 5])
    return {term for term in terms if len(term) > 1}


def preference_relation(anchor: dict[str, Any], candidate: dict[str, Any]) -> str:
    anchor_terms = preference_terms(anchor.get("text", ""))
    candidate_terms = preference_terms(candidate.get("text", ""))
    if not anchor_terms or not candidate_terms:
        return "preference_unknown"
    overlap = len(anchor_terms & candidate_terms) / len(anchor_terms | candidate_terms)
    return "preference_match" if overlap >= 0.34 else "preference_conflict"


def relationship_signature(anchor: dict[str, Any], candidate: dict[str, Any]) -> str:
    anchor_cluster = str(anchor.get("cluster") or "")
    candidate_cluster = str(candidate.get("cluster") or "")
    context = "same_project" if anchor_cluster and anchor_cluster == candidate_cluster else "cross_project"
    age_days = float(candidate.get("age_days") or 0.0)
    use_count = float(candidate.get("use_count") or 0.0)
    stale = "stale" if age_days >= 180 and use_count <= 0 and not bool(candidate.get("protected")) else "fresh_or_used"
    duplicate = "duplicate_like" if jaccard(anchor.get("text", ""), candidate.get("text", "")) >= 0.65 else "distinct"
    return "|".join([context, preference_relation(anchor, candidate), stale, duplicate])


def budgeted_ids(row: dict[str, Any], ranked: Ranked, default_budget: int) -> set[str]:
    budget = int((row.get("retrieval_task") or {}).get("budget") or default_budget)
    used = 0
    included: set[str] = set()
    for candidate_id, _, text in ranked:
        cost = max(1, len(tokens(text)))
        if used + cost > budget:
            break
        included.add(candidate_id)
        used += cost
    return included


def relevant_ids(row: dict[str, Any]) -> set[str]:
    return set((row.get("retrieval_task") or {}).get("relevant_ids") or [])


def record_feedback(state: MemoryState, row: dict[str, Any], candidate_id: str, outcome: str) -> None:
    candidates = {candidate["id"]: candidate for candidate in row.get("candidates", [])}
    candidate = candidates.get(candidate_id, {})
    state.update(candidate_id, outcome, protected=bool(candidate.get("protected")))
    if candidate:
        state.update_signature(row["anchor"], candidate, outcome)


def apply_feedback(state: MemoryState, row: dict[str, Any], ranked: Ranked, default_budget: int) -> dict[str, int]:
    relevant = relevant_ids(row)
    included = budgeted_ids(row, ranked, default_budget)
    counts = Counter()

    for candidate_id in included:
        if candidate_id in relevant:
            record_feedback(state, row, candidate_id, "helpful")
            counts["helpful"] += 1
        else:
            record_feedback(state, row, candidate_id, "ignored")
            counts["ignored"] += 1

    for candidate_id in relevant - included:
        record_feedback(state, row, candidate_id, "corrected")
        counts["corrected"] += 1

    return dict(counts)


def role_exposure(row: dict[str, Any], ranked: Ranked, default_budget: int) -> dict[str, dict[str, int]]:
    included = budgeted_ids(row, ranked, default_budget)
    relevant = relevant_ids(row)
    roles: dict[str, dict[str, int]] = {}
    for candidate in row.get("candidates", []):
        role = str(candidate.get("synthetic_role") or "history")
        candidate_id = candidate["id"]
        roles.setdefault(role, {"total": 0, "relevant": 0, "budget": 0})
        roles[role]["total"] += 1
        if candidate_id in relevant:
            roles[role]["relevant"] += 1
        if candidate_id in included:
            roles[role]["budget"] += 1
    return roles


def merge_role_counts(total: dict[str, dict[str, int]], item: dict[str, dict[str, int]]) -> None:
    for role, counts in item.items():
        total.setdefault(role, {"total": 0, "relevant": 0, "budget": 0})
        for key, value in counts.items():
            total[role][key] += value


def summarize_role_counts(counts: dict[str, dict[str, int]]) -> dict[str, dict[str, float | int]]:
    summary = {}
    for role, values in sorted(counts.items()):
        total = max(1, values["total"])
        summary[role] = {
            "total": values["total"],
            "relevant": values["relevant"],
            "budget": values["budget"],
            "relevant_rate": values["relevant"] / total,
            "budget_rate": values["budget"] / total,
        }
    return summary


def summarize_windows(metrics: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    if not metrics:
        return {"overall": {}, "first_half": {}, "second_half": {}}
    midpoint = max(1, len(metrics) // 2)
    return {
        "overall": average_metrics(metrics),
        "first_half": average_metrics(metrics[:midpoint]),
        "second_half": average_metrics(metrics[midpoint:]),
    }


def metric_delta(evolved: dict[str, float], static: dict[str, float]) -> dict[str, float]:
    keys = sorted(set(evolved) | set(static))
    return {key: evolved.get(key, 0.0) - static.get(key, 0.0) for key in keys}


def confidence_features(ranked: Ranked, top_k: int) -> dict[str, float]:
    if not ranked:
        return {
            "top_score": 0.0,
            "top_margin": 0.0,
            "cutoff_margin": 0.0,
            "score_spread": 0.0,
            "cutoff_crowding": 0.0,
        }
    scores = [score for _, score, _ in ranked]
    top_margin = scores[0] - scores[1] if len(scores) > 1 else abs(scores[0])
    cutoff_index = min(max(0, top_k - 1), len(scores) - 1)
    cutoff_score = scores[cutoff_index]
    next_score = scores[cutoff_index + 1] if cutoff_index + 1 < len(scores) else scores[-1]
    cutoff_crowding = sum(1 for score in scores if abs(score - cutoff_score) <= 0.03)
    return {
        "top_score": float(scores[0]),
        "top_margin": float(top_margin),
        "cutoff_margin": float(cutoff_score - next_score),
        "score_spread": float(scores[0] - scores[-1]),
        "cutoff_crowding": float(cutoff_crowding),
    }


def policy_bias_scale(
    policy: str,
    bias_scale: float,
    confidence: dict[str, float],
    gate_margin: float,
    low_confidence_score: float,
    low_spread: float,
) -> float:
    if policy == "off" or bias_scale <= 0.0:
        return 0.0
    if policy == "always":
        return bias_scale
    if policy == "uncertainty_gated":
        return bias_scale if confidence["cutoff_margin"] <= gate_margin or confidence["cutoff_crowding"] >= 4 else 0.0
    if policy == "low_confidence_only":
        return bias_scale if confidence["top_score"] <= low_confidence_score or confidence["score_spread"] <= low_spread else 0.0
    raise ValueError(f"unknown evolution policy: {policy}")


def evaluate_variant(
    rows: list[dict[str, Any]],
    scorer: Scorer,
    top_k: int,
    budget: int,
    policy: str,
    bias_scale: float,
    gate_margin: float,
    low_confidence_score: float,
    low_spread: float,
    allow_bias: bool,
) -> dict[str, Any]:
    if policy == "off":
        metrics = []
        roles: dict[str, dict[str, int]] = {}
        confidence_items = []
        for row in rows:
            ranked = scorer(row)
            metrics.append(evaluate_ranked(row, ranked, top_k, budget))
            merge_role_counts(roles, role_exposure(row, ranked, budget))
            confidence_items.append(confidence_features(ranked, top_k))
        return {
            "policy": policy,
            "bias_scale": bias_scale,
            "bias_enabled": False,
            "applied_rate": 0.0,
            "evolved": summarize_windows(metrics),
            "feedback": {},
            "state": {"memories_touched": 0, "signatures_touched": 0, "feedback": {}},
            "confidence": average_metrics(confidence_items),
            "evolved_role_metrics": summarize_role_counts(roles),
        }

    state = MemoryState()
    evolved_metrics = []
    evolved_roles: dict[str, dict[str, int]] = {}
    feedback = Counter()
    confidence_items = []
    applied_count = 0
    for row in rows:
        evolved_row = state.apply(row)
        base_ranked = scorer(evolved_row)
        confidence = confidence_features(base_ranked, top_k)
        effective_scale = (
            policy_bias_scale(policy, bias_scale, confidence, gate_margin, low_confidence_score, low_spread)
            if allow_bias
            else 0.0
        )
        evolved_ranked = state.rerank(evolved_row, base_ranked, effective_scale)
        evolved_metrics.append(evaluate_ranked(row, evolved_ranked, top_k, budget))
        merge_role_counts(evolved_roles, role_exposure(row, evolved_ranked, budget))
        feedback.update(apply_feedback(state, evolved_row, evolved_ranked, budget))
        confidence_items.append(confidence)
        if effective_scale > 0.0:
            applied_count += 1

    return {
        "policy": policy,
        "bias_scale": bias_scale,
        "bias_enabled": allow_bias,
        "applied_rate": applied_count / max(1, len(rows)),
        "evolved": summarize_windows(evolved_metrics),
        "feedback": dict(sorted(feedback.items())),
        "state": state.summary(),
        "confidence": average_metrics(confidence_items),
        "evolved_role_metrics": summarize_role_counts(evolved_roles),
    }


def evaluate_online(
    rows: list[dict[str, Any]],
    scorers: dict[str, Scorer],
    top_k: int,
    budget: int,
    policies: list[str],
    bias_scales: list[float],
    gate_margin: float,
    low_confidence_score: float,
    low_spread: float,
    selector_post_rank_bias: bool,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for name, scorer in scorers.items():
        static_metrics = []
        static_roles: dict[str, dict[str, int]] = {}
        for row in rows:
            static_ranked = scorer(row)
            static_metrics.append(evaluate_ranked(row, static_ranked, top_k, budget))
            merge_role_counts(static_roles, role_exposure(row, static_ranked, budget))

        static_summary = summarize_windows(static_metrics)
        variants = {}
        for policy in policies:
            for bias_scale in bias_scales:
                key = f"{policy}@{bias_scale:g}"
                allow_bias = name != "context_selector" or selector_post_rank_bias
                variant = evaluate_variant(
                    rows,
                    scorer,
                    top_k,
                    budget,
                    policy,
                    bias_scale,
                    gate_margin,
                    low_confidence_score,
                    low_spread,
                    allow_bias,
                )
                variant["delta"] = {
                    "overall": metric_delta(variant["evolved"]["overall"], static_summary["overall"]),
                    "second_half": metric_delta(variant["evolved"]["second_half"], static_summary["second_half"]),
                }
                variants[key] = variant
        first_variant = next(iter(variants.values()))
        results[name] = {
            "static": static_summary,
            "static_role_metrics": summarize_role_counts(static_roles),
            "variants": variants,
            "evolved": first_variant["evolved"],
            "delta": first_variant["delta"],
            "feedback": first_variant["feedback"],
            "state": first_variant["state"],
            "evolved_role_metrics": first_variant["evolved_role_metrics"],
        }
    return results


def load_or_generate_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.dataset:
        return load_rows(Path(args.dataset), args.limit)
    rows = build_cases(args.seed, args.cases, args.candidates, args.scenario)
    if args.output_dataset:
        path = Path(args.output_dataset)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return rows[: args.limit] if args.limit > 0 else rows


def build_scorers(args: argparse.Namespace) -> dict[str, Scorer]:
    scorers: dict[str, Scorer] = {
        "vector_only": vector_scores,
        "heuristic_graph": heuristic_scores,
    }
    if args.checkpoint:
        import torch

        from python.selector.benchmark_selector import selector_scores
        from python.selector.model import load_selector

        device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
        model = load_selector(args.checkpoint, device=device)
        scorers["context_selector"] = lambda row: selector_scores(model, row, args.budget)
    return scorers


def parse_bias_scales(value: str, fallback: float) -> list[float]:
    if not value.strip():
        return [fallback]
    scales = [float(item.strip()) for item in value.split(",") if item.strip()]
    return scales or [fallback]


def parse_policies(value: str) -> list[str]:
    policies = [item.strip() for item in value.split(",") if item.strip()]
    if not policies:
        return ["always"]
    unknown = sorted(set(policies) - EVOLUTION_POLICIES)
    if unknown:
        raise ValueError(f"unknown evolution policies: {', '.join(unknown)}")
    return policies


def write_markdown(result: dict[str, Any], path: Path) -> None:
    lines = [
        "# Memory Evolution Benchmark",
        "",
        f"- cases: `{result['cases']}`",
        f"- scenario: `{result['scenario']}`",
        f"- top_k: `{result['top_k']}`",
        f"- budget: `{result['budget']}`",
        f"- policies: `{', '.join(result['evolution_policies'])}`",
        f"- bias scales: `{', '.join(str(item) for item in result['evolution_bias_scales'])}`",
        f"- selector post-rank bias: `{result['selector_post_rank_bias']}`",
        "",
        "| method | policy | scale | bias | applied | recall@k | context precision | context recall | noise |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, item in result["methods"].items():
        static = item["static"]["overall"]
        lines.append(
            f"| {name} | static | 0 | off | 0.0000 | {static.get('recall_at_k', 0.0):.4f} | "
            f"{static.get('context_precision', 0.0):.4f} | {static.get('context_recall', 0.0):.4f} | "
            f"{static.get('noise', 0.0):.2f} |"
        )
        for variant in item["variants"].values():
            metrics = variant["evolved"]["overall"]
            bias = "on" if variant.get("bias_enabled") else "off"
            lines.append(
                f"| {name} | {variant['policy']} | {variant['bias_scale']:g} | {bias} | {variant.get('applied_rate', 0.0):.4f} | "
                f"{metrics.get('recall_at_k', 0.0):.4f} | "
                f"{metrics.get('context_precision', 0.0):.4f} | {metrics.get('context_recall', 0.0):.4f} | "
                f"{metrics.get('noise', 0.0):.2f} |"
            )
    lines.extend(["", "## Second-Half Delta", "", "| method | policy | scale | recall@k | context precision | context recall | noise |", "| --- | --- | ---: | ---: | ---: | ---: | ---: |"])
    for name, item in result["methods"].items():
        for variant in item["variants"].values():
            delta = variant["delta"]["second_half"]
            lines.append(
                f"| {name} | {variant['policy']} | {variant['bias_scale']:g} | "
                f"{delta.get('recall_at_k', 0.0):+.4f} | {delta.get('context_precision', 0.0):+.4f} | "
                f"{delta.get('context_recall', 0.0):+.4f} | {delta.get('noise', 0.0):+.2f} |"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--cases", type=int, default=2000)
    parser.add_argument("--candidates", type=int, default=32)
    parser.add_argument("--scenario", choices=["standard", "longitudinal", "adversarial"], default="adversarial")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--budget", type=int, default=90)
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-md", default="")
    parser.add_argument("--output-dataset", default="")
    parser.add_argument("--evolution-bias-scale", type=float, default=0.5)
    parser.add_argument("--evolution-bias-scales", default="")
    parser.add_argument("--evolution-policies", default="always")
    parser.add_argument("--gate-margin", type=float, default=0.03)
    parser.add_argument("--low-confidence-score", type=float, default=0.72)
    parser.add_argument("--low-spread", type=float, default=0.18)
    parser.add_argument("--selector-post-rank-bias", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    rows = load_or_generate_rows(args)
    policies = parse_policies(args.evolution_policies)
    bias_scales = parse_bias_scales(args.evolution_bias_scales, args.evolution_bias_scale)
    result = {
        "cases": len(rows),
        "scenario": args.scenario,
        "dataset": args.dataset,
        "checkpoint": args.checkpoint,
        "top_k": args.top_k,
        "budget": args.budget,
        "evolution_bias_scale": args.evolution_bias_scale,
        "evolution_bias_scales": bias_scales,
        "evolution_policies": policies,
        "gate_margin": args.gate_margin,
        "low_confidence_score": args.low_confidence_score,
        "low_spread": args.low_spread,
        "selector_post_rank_bias": args.selector_post_rank_bias,
        "methods": evaluate_online(
            rows,
            build_scorers(args),
            args.top_k,
            args.budget,
            policies,
            bias_scales,
            args.gate_margin,
            args.low_confidence_score,
            args.low_spread,
            args.selector_post_rank_bias,
        ),
    }
    body = json.dumps(result, indent=2)
    print(body)
    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body + "\n", encoding="utf-8")
    if args.output_md:
        write_markdown(result, Path(args.output_md))


if __name__ == "__main__":
    main()
