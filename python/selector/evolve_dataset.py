from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from python.benchmarks.benchmark_librarian import heuristic_scores, load_rows, vector_scores
from python.selector.evolution_benchmark import MemoryState, Ranked, apply_feedback, relevant_ids


def oracle_scores(row: dict[str, Any]) -> Ranked:
    relevant = relevant_ids(row)
    ranked = []
    for candidate in row.get("candidates", []):
        score = 1.0 if candidate["id"] in relevant else 0.0
        ranked.append((candidate["id"], score, candidate.get("text", "")))
    return sorted(ranked, key=lambda item: (-item[1], item[0]))


def scorer_for(name: str):
    if name == "heuristic_graph":
        return heuristic_scores
    if name == "vector_only":
        return vector_scores
    if name == "oracle":
        return oracle_scores
    raise ValueError(f"unknown feedback scorer: {name}")


def mark_evolved(row: dict[str, Any], pass_index: int, source_index: int, feedback_scorer: str) -> dict[str, Any]:
    item = copy.deepcopy(row)
    item["evolution"] = {
        "pass": pass_index,
        "source_index": source_index,
        "feedback_scorer": feedback_scorer,
    }
    item["schema_version"] = max(int(item.get("schema_version") or 0), 5)
    return item


def evolve_rows(
    rows: list[dict[str, Any]],
    passes: int,
    feedback_scorer: str,
    budget: int,
    include_original: bool,
    shuffle_seed: int,
) -> list[dict[str, Any]]:
    scorer = scorer_for(feedback_scorer)
    state = MemoryState()
    output = [copy.deepcopy(row) for row in rows] if include_original else []
    order = list(range(len(rows)))
    rng = random.Random(shuffle_seed)
    for pass_index in range(passes):
        rng.shuffle(order)
        for source_index in order:
            evolved = state.apply(rows[source_index])
            output.append(mark_evolved(evolved, pass_index + 1, source_index, feedback_scorer))
            ranked = scorer(evolved)
            apply_feedback(state, evolved, ranked, budget)
    return output


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--passes", type=int, default=1)
    parser.add_argument("--feedback-scorer", choices=["heuristic_graph", "vector_only", "oracle"], default="heuristic_graph")
    parser.add_argument("--budget", type=int, default=90)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--include-original", action="store_true")
    parser.add_argument("--no-include-original", dest="include_original", action="store_false")
    parser.set_defaults(include_original=True)
    parser.add_argument("--shuffle-seed", type=int, default=7)
    args = parser.parse_args()

    rows = load_rows(Path(args.input), args.limit)
    evolved = evolve_rows(rows, args.passes, args.feedback_scorer, args.budget, args.include_original, args.shuffle_seed)
    write_jsonl(evolved, Path(args.output))
    summary = {
        "input": args.input,
        "output": args.output,
        "input_rows": len(rows),
        "output_rows": len(evolved),
        "passes": args.passes,
        "feedback_scorer": args.feedback_scorer,
        "include_original": args.include_original,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
