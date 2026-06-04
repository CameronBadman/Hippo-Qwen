from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from python.benchmarks.hippocampus_retrieval import (
    CachedEmbeddingBackend,
    activation_overlap,
    associative_recall_scores,
    build_embedding_backend,
    edge_activation,
    edge_type_boost,
    ensure_backend_embeddings,
    multihop_metrics,
    sparse_basin_scores,
    state_score,
    vector_scores_with_backend,
)
from python.benchmarks.benchmark_librarian import average_metrics, evaluate_ranked
from python.librarian.features import embed_text
from python.synthetic.generate import (
    NOISE,
    PREFERENCE_CONFLICTS,
    PREFERENCES,
    PROJECTS,
    TASKS,
    generated_card,
)


def build_large_pool_case(seed: int, pool_size: int) -> dict[str, Any]:
    rng = random.Random(seed)
    project = rng.choice(PROJECTS)
    task = rng.choice(TASKS)
    preference = rng.choice(PREFERENCES)
    wrong_preference = PREFERENCE_CONFLICTS[preference]
    route_tag = f"route-{seed:04d}"
    answer_tag = f"answer-{seed:04d}"
    anchor = {
        "id": f"anchor_{seed}",
        "text": (
            f"{project}: active memory lookup references escalation {route_tag}. "
            f"Follow the connected resolution for {task}."
        ),
        "summary": "",
        "embedding": [],
        "importance": 0.7,
        "cluster": project,
        "metadata": {"project": project},
        "age_days": 0,
        "use_count": 3,
        "evidence_count": 2,
        "last_outcome": "helpful",
        "protected": False,
    }
    anchor["embedding"] = embed_text(anchor["text"])

    bridge = generated_card(
        anchor,
        f"{project}: escalation {route_tag} came from {task}. The accepted resolution is filed as {answer_tag}.",
        "large_pool_bridge_relevant",
        seed,
        0,
        age_days=2,
        use_count=13,
        evidence_count=5,
        last_outcome="helpful",
        importance=0.74,
    )
    target = generated_card(
        anchor,
        f"{answer_tag}: accepted resolution says use the durable preference: {preference}.",
        "large_pool_target_relevant",
        seed,
        0,
        age_days=5,
        use_count=5,
        evidence_count=3,
        last_outcome="helpful",
        importance=0.62,
    )
    support = generated_card(
        anchor,
        f"{answer_tag} support note: prior answer succeeded by honoring {preference} and avoiding unrelated work.",
        "large_pool_support_relevant",
        seed,
        0,
        age_days=8,
        use_count=3,
        evidence_count=2,
        last_outcome="helpful",
        importance=0.58,
    )
    candidates = [bridge, target, support]

    hard_count = min(max(50, pool_size // 8), pool_size - len(candidates))
    for slot in range(hard_count):
        role = rng.choice(
            [
                "large_pool_same_project_decoy",
                "large_pool_wrong_project_decoy",
                "large_pool_stale_decoy",
                "large_pool_wrong_preference_decoy",
            ]
        )
        wrong_project = rng.choice([item for item in PROJECTS if item != project])
        if role == "large_pool_same_project_decoy":
            text = f"{project}: escalation {route_tag} mentioned {task}, but this note is a decoy about {rng.choice(NOISE)}."
            state = {"age_days": 3, "use_count": 8, "evidence_count": 1, "last_outcome": "ignored", "importance": 0.42}
        elif role == "large_pool_wrong_project_decoy":
            text = f"{wrong_project}: escalation {route_tag} points to {answer_tag}, but belongs to another project."
            state = {
                "cluster": wrong_project,
                "project": wrong_project,
                "age_days": 2,
                "use_count": 34,
                "evidence_count": 13,
                "last_outcome": "helpful",
                "importance": 0.78,
            }
        elif role == "large_pool_stale_decoy":
            text = f"{project}: old escalation {route_tag} branch for {task}; superseded before {answer_tag} was accepted."
            state = {"age_days": 365, "use_count": 0, "evidence_count": 0, "last_outcome": "ignored", "importance": 0.2}
        else:
            text = f"{answer_tag}: wrong standalone resolution says the user {wrong_preference}; this conflicts with the path."
            state = {"age_days": 2, "use_count": 21, "evidence_count": 8, "last_outcome": "corrected", "importance": 0.74}
        candidates.append(generated_card(anchor, text, role, seed, slot, **state))

    while len(candidates) < pool_size:
        slot = len(candidates)
        noise_project = rng.choice(PROJECTS + ["archive", "personal", "ops", "reading-list"])
        noise_task = rng.choice(TASKS + NOISE)
        noise_pref = rng.choice(PREFERENCES)
        text = rng.choice(
            [
                f"{noise_project}: {noise_task}. User {noise_pref}.",
                f"Background memory {slot}: {rng.choice(NOISE)} for {noise_project}.",
                f"{noise_project} note with reference {rng.randint(10000, 99999)} and no link to current escalation.",
            ]
        )
        candidates.append(
            generated_card(
                anchor,
                text,
                "large_pool_background",
                seed,
                slot,
                cluster=noise_project,
                project=noise_project,
                age_days=rng.choice([1, 7, 30, 180, 365]),
                use_count=rng.choice([0, 0, 1, 3, 8, 21]),
                evidence_count=rng.choice([0, 1, 2, 5]),
                last_outcome=rng.choice(["", "", "helpful", "ignored"]),
                importance=rng.choice([0.2, 0.35, 0.5, 0.65]),
            )
        )

    rng.shuffle(candidates)
    return {
        "anchor": anchor,
        "candidates": candidates,
        "retrieval_task": {
            "query": f"For {project}, follow escalation {route_tag} to the connected resolution for {task}.",
            "relevant_ids": [bridge["id"], target["id"], support["id"]],
            "bridge_ids": [bridge["id"]],
            "target_ids": [target["id"], support["id"]],
            "budget": 90,
        },
        "memory_graph": {
            "max_depth": 3,
            "edges": [
                {
                    "source": bridge["id"],
                    "target": target["id"],
                    "type": "used_with",
                    "weight": 1.15,
                    "confidence": 0.92,
                    "activation_text": f"{project} {task} {preference} {route_tag} {answer_tag}",
                },
                {
                    "source": target["id"],
                    "target": support["id"],
                    "type": "same_context",
                    "weight": 0.86,
                    "confidence": 0.82,
                    "activation_text": f"{project} {preference} {answer_tag}",
                },
            ],
        },
        "scenario": "large_pool_associative",
    }


def candidate_set_metrics(row: dict[str, Any], ranked: list[tuple[str, float, str]], thresholds: list[float], top_ns: list[int]) -> dict[str, float]:
    relevant = set((row.get("retrieval_task") or {}).get("relevant_ids") or [])
    roles = {candidate["id"]: str(candidate.get("synthetic_role") or "") for candidate in row.get("candidates", [])}
    out: dict[str, float] = {}
    for threshold in thresholds:
        pulled = [candidate_id for candidate_id, score, _ in ranked if score >= threshold]
        pulled_set = set(pulled)
        key = f"threshold_{threshold:g}"
        out[f"{key}_pulled"] = float(len(pulled))
        out[f"{key}_precision"] = len(relevant & pulled_set) / max(1, len(pulled))
        out[f"{key}_recall"] = len(relevant & pulled_set) / max(1, len(relevant))
        out[f"{key}_wrong"] = float(sum(1 for candidate_id in pulled if "decoy" in roles.get(candidate_id, "")))
    for top_n in top_ns:
        pulled = [candidate_id for candidate_id, _, _ in ranked[:top_n]]
        pulled_set = set(pulled)
        key = f"top_{top_n}"
        out[f"{key}_precision"] = len(relevant & pulled_set) / max(1, len(pulled))
        out[f"{key}_recall"] = len(relevant & pulled_set) / max(1, len(relevant))
        out[f"{key}_wrong"] = float(sum(1 for candidate_id in pulled if "decoy" in roles.get(candidate_id, "")))
    return out


def context_decoy_metrics(row: dict[str, Any], ranked: list[tuple[str, float, str]], default_budget: int) -> dict[str, float]:
    budget = int((row.get("retrieval_task") or {}).get("budget") or default_budget)
    roles = {candidate["id"]: str(candidate.get("synthetic_role") or "") for candidate in row.get("candidates", [])}
    used = 0
    included = []
    for candidate_id, _, text in ranked:
        cost = max(1, len(text.split()))
        if used + cost > budget:
            break
        included.append(candidate_id)
        used += cost
    decoys = sum(1 for candidate_id in included if "decoy" in roles.get(candidate_id, ""))
    background = sum(1 for candidate_id in included if roles.get(candidate_id) == "large_pool_background")
    return {
        "included": float(len(included)),
        "decoy_exposure": float(decoys),
        "background_exposure": float(background),
    }


def strategy_label(strategy: str) -> str:
    return strategy.replace(".", "p").replace("-", "m")


def select_candidate_ids(ranked: list[tuple[str, float, str]], strategy: str) -> set[str]:
    if not ranked:
        return set()
    if strategy.startswith("top"):
        count = max(1, int(strategy[3:]))
        return {candidate_id for candidate_id, _, _ in ranked[:count]}
    scores = [score for _, score, _ in ranked]
    if strategy.startswith("pct"):
        percentile = float(strategy[3:])
        ordered = sorted(scores)
        index = min(len(ordered) - 1, max(0, int(math.floor((percentile / 100.0) * (len(ordered) - 1)))))
        cutoff = ordered[index]
        return {candidate_id for candidate_id, score, _ in ranked if score >= cutoff}
    if strategy.startswith("z"):
        z_value = float(strategy[1:])
        mean = sum(scores) / len(scores)
        variance = sum((score - mean) ** 2 for score in scores) / max(1, len(scores))
        cutoff = mean + z_value * math.sqrt(variance)
        return {candidate_id for candidate_id, score, _ in ranked if score >= cutoff}
    if strategy.startswith("margin"):
        margin = float(strategy[len("margin") :])
        cutoff = ranked[0][1] - margin
        return {candidate_id for candidate_id, score, _ in ranked if score >= cutoff}
    raise ValueError(f"unknown calibration strategy: {strategy}")


def calibrated_associative_scores(
    row: dict[str, Any],
    vector: list[tuple[str, float, str]],
    sparse: list[tuple[str, float, str]],
    strategy: str,
    seed_count: int,
) -> list[tuple[str, float, str]]:
    from python.librarian.features import activation_mask_for_text

    candidates = {candidate["id"]: dict(candidate) for candidate in row.get("candidates", [])}
    selected = select_candidate_ids(sparse, strategy)
    for candidate_id, _, _ in sparse[:seed_count]:
        selected.add(candidate_id)

    sparse_by_id = {candidate_id: (score, text) for candidate_id, score, text in sparse}
    vector_by_id = {candidate_id: (score, text) for candidate_id, score, text in vector}
    best: dict[str, float] = {}
    texts: dict[str, str] = {}
    for rank, candidate_id in enumerate([candidate_id for candidate_id, _, _ in sparse if candidate_id in selected]):
        sparse_score, text = sparse_by_id[candidate_id]
        vector_score = vector_by_id.get(candidate_id, (0.0, text))[0]
        best[candidate_id] = max(sparse_score + 0.012 * max(0, seed_count - rank), 0.72 * vector_score)
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

    frontier = [(candidate_id, score, [candidate_id]) for candidate_id, score in sorted(best.items(), key=lambda item: (-item[1], item[0]))[:seed_count]]
    for depth in range(max_depth):
        next_frontier: list[tuple[str, float, list[str]]] = []
        for current_id, current_score, path in frontier:
            for edge in outgoing.get(current_id, []):
                target_id = str(edge.get("target") or "")
                if target_id in path:
                    continue
                target = candidates[target_id]
                edge_weight = float(edge.get("weight") or 0.0)
                confidence = float(edge.get("confidence") or 0.5)
                hop_gain = (
                    edge_weight
                    * edge_type_boost(str(edge.get("type") or "used_with"))
                    * (0.70 + 0.45 * activation_overlap(query_mask, edge_activation(edge)))
                    * (0.70 + 0.30 * confidence)
                    / (1.25 + depth)
                )
                score = 0.62 * current_score + hop_gain + 0.07 * float(target.get("importance") or 0.5) + state_score(target)
                if score > best.get(target_id, -99.0):
                    best[target_id] = score
                    texts[target_id] = target.get("text", "")
                    next_frontier.append((target_id, score, path + [target_id]))
        frontier = next_frontier
    return sorted([(candidate_id, score, texts.get(candidate_id, "")) for candidate_id, score in best.items()], key=lambda item: (-item[1], item[0]))


def evaluate_case(row: dict[str, Any], backend: CachedEmbeddingBackend, args: argparse.Namespace) -> dict[str, dict[str, float]]:
    row = ensure_backend_embeddings(row, backend)
    vector = vector_scores_with_backend(row, backend)
    sparse = sparse_basin_scores(row, backend)
    assoc = associative_recall_scores(row, backend, args.seed_count)
    thresholds = [float(item) for item in args.thresholds.split(",") if item]
    top_ns = [int(item) for item in args.candidate_top_ns.split(",") if item]
    metrics = {
        "vector_retrieval": evaluate_ranked(row, vector, args.top_k, args.budget),
        "sparse_retrieval": evaluate_ranked(row, sparse, args.top_k, args.budget),
        "associative_retrieval": evaluate_ranked(row, assoc, args.top_k, args.budget),
        "vector_multihop": multihop_metrics(row, vector, args.top_k, args.budget),
        "sparse_multihop": multihop_metrics(row, sparse, args.top_k, args.budget),
        "associative_multihop": multihop_metrics(row, assoc, args.top_k, args.budget),
        "vector_context_exposure": context_decoy_metrics(row, vector, args.budget),
        "sparse_context_exposure": context_decoy_metrics(row, sparse, args.budget),
        "associative_context_exposure": context_decoy_metrics(row, assoc, args.budget),
        "vector_candidates": candidate_set_metrics(row, vector, thresholds, top_ns),
        "sparse_candidates": candidate_set_metrics(row, sparse, thresholds, top_ns),
    }
    for strategy in [item for item in args.calibration_strategies.split(",") if item]:
        started = time.perf_counter()
        calibrated = calibrated_associative_scores(row, vector, sparse, strategy, args.seed_count)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        label = f"calibrated_{strategy_label(strategy)}"
        selected = select_candidate_ids(sparse, strategy)
        retrieval = evaluate_ranked(row, calibrated, args.top_k, args.budget)
        retrieval["latency_ms"] = elapsed_ms
        retrieval["candidate_count"] = float(len(selected))
        metrics[f"{label}_retrieval"] = retrieval
        metrics[f"{label}_multihop"] = multihop_metrics(row, calibrated, args.top_k, args.budget)
        metrics[f"{label}_context_exposure"] = context_decoy_metrics(row, calibrated, args.budget)
    return metrics


def flatten(prefix: str, values: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in values.items()}


def run(args: argparse.Namespace) -> dict[str, Any]:
    backend = build_embedding_backend(args)
    started = time.time()
    rows = []
    for offset in range(args.cases):
        seed = args.seed + offset
        row = build_large_pool_case(seed, args.pool_size)
        metrics = evaluate_case(row, backend, args)
        flat = {}
        for name, values in metrics.items():
            flat.update(flatten(name, values))
        rows.append(flat)
    keys = sorted(rows[0]) if rows else []
    averages = {key: sum(row[key] for row in rows) / max(1, len(rows)) for key in keys}
    return {
        "embedding_backend": backend.name,
        "hippo_checkpoint": args.hippo_checkpoint if backend.name == "hippo" else "",
        "cases": args.cases,
        "pool_size": args.pool_size,
        "seed": args.seed,
        "seed_count": args.seed_count,
        "thresholds": [float(item) for item in args.thresholds.split(",") if item],
        "candidate_top_ns": [int(item) for item in args.candidate_top_ns.split(",") if item],
        "calibration_strategies": [item for item in args.calibration_strategies.split(",") if item],
        "elapsed_seconds": round(time.time() - started, 2),
        "averages": averages,
        "rows": rows if args.include_rows else [],
    }


def write_markdown(result: dict[str, Any], path: Path) -> None:
    avg = result["averages"]
    lines = [
        "# Large Pool Retrieval Benchmark",
        "",
        f"- backend: `{result['embedding_backend']}`",
        f"- cases: `{result['cases']}`",
        f"- pool_size: `{result['pool_size']}`",
        f"- elapsed_seconds: `{result['elapsed_seconds']}`",
        "",
        "| view | precision | target ctx | path ctx | noise | wrong ctx |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| vector | {avg.get('vector_retrieval_context_precision', 0.0):.4f} | "
            f"{avg.get('vector_multihop_target_context_recall', 0.0):.4f} | "
            f"{avg.get('vector_multihop_path_context_success', 0.0):.4f} | "
            f"{avg.get('vector_retrieval_noise', 0.0):.2f} | "
            f"{avg.get('vector_context_exposure_decoy_exposure', 0.0):.2f} |"
        ),
        (
            f"| sparse | {avg.get('sparse_retrieval_context_precision', 0.0):.4f} | "
            f"{avg.get('sparse_multihop_target_context_recall', 0.0):.4f} | "
            f"{avg.get('sparse_multihop_path_context_success', 0.0):.4f} | "
            f"{avg.get('sparse_retrieval_noise', 0.0):.2f} | "
            f"{avg.get('sparse_context_exposure_decoy_exposure', 0.0):.2f} |"
        ),
        (
            f"| associative | {avg.get('associative_retrieval_context_precision', 0.0):.4f} | "
            f"{avg.get('associative_multihop_target_context_recall', 0.0):.4f} | "
            f"{avg.get('associative_multihop_path_context_success', 0.0):.4f} | "
            f"{avg.get('associative_retrieval_noise', 0.0):.2f} | "
            f"{avg.get('associative_context_exposure_decoy_exposure', 0.0):.2f} |"
        ),
        "",
        "| calibrated strategy | precision | target ctx | path ctx | candidates | latency ms | decoys |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key in sorted(avg):
        if not key.startswith("calibrated_") or not key.endswith("_retrieval_context_precision"):
            continue
        prefix = key[: -len("_retrieval_context_precision")]
        lines.append(
            f"| {prefix} | {avg.get(prefix + '_retrieval_context_precision', 0.0):.4f} | "
            f"{avg.get(prefix + '_multihop_target_context_recall', 0.0):.4f} | "
            f"{avg.get(prefix + '_multihop_path_context_success', 0.0):.4f} | "
            f"{avg.get(prefix + '_retrieval_candidate_count', 0.0):.2f} | "
            f"{avg.get(prefix + '_retrieval_latency_ms', 0.0):.2f} | "
            f"{avg.get(prefix + '_context_exposure_decoy_exposure', 0.0):.2f} |"
        )
    lines.extend(
        [
            "",
        "| candidate view | pulled | precision | recall | wrong |",
        "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for key in sorted(avg):
        if not key.startswith(("vector_candidates_threshold_", "sparse_candidates_threshold_")) or not key.endswith("_pulled"):
            continue
        prefix = key[: -len("_pulled")]
        lines.append(
            f"| {prefix} | {avg.get(prefix + '_pulled', 0.0):.2f} | "
            f"{avg.get(prefix + '_precision', 0.0):.4f} | "
            f"{avg.get(prefix + '_recall', 0.0):.4f} | "
            f"{avg.get(prefix + '_wrong', 0.0):.2f} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=int, default=20)
    parser.add_argument("--pool-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=9000)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--budget", type=int, default=90)
    parser.add_argument("--seed-count", type=int, default=8)
    parser.add_argument("--thresholds", default="0.25,0.3,0.35,0.4")
    parser.add_argument("--candidate-top-ns", default="16,32,64,128")
    parser.add_argument("--calibration-strategies", default="top16,top32,top64,top128,pct99,pct99.5,z1.5,z2,margin0.05,margin0.1")
    parser.add_argument("--embedding-backend", choices=["hash", "hippo"], default="hash")
    parser.add_argument("--hippo-checkpoint", default="")
    parser.add_argument("--hippo-encoder-src", default="")
    parser.add_argument("--hippo-max-length", type=int, default=128)
    parser.add_argument("--hippo-batch-size", type=int, default=128)
    parser.add_argument("--device", default="")
    parser.add_argument("--include-rows", action="store_true")
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
