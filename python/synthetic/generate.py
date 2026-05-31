from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from python.librarian.features import EDGE_TYPES, embed_text, heuristic_action


PROJECTS = ["hippograph", "resume", "printer", "research-notes", "home-network", "colab-training"]
PREFERENCES = [
    "prefers concise answers",
    "wants source links",
    "likes Go runtimes",
    "avoids large dependencies",
    "needs reproducible commands",
    "prefers visual debugging",
]
TASKS = [
    "debugged retrieval",
    "updated deployment",
    "tested memory recall",
    "refined graph traversal",
    "reviewed API contracts",
    "designed benchmark cases",
]
NOISE = [
    "sourdough hydration ratios",
    "hotel booking loyalty points",
    "film camera lens adapters",
    "garden soil nitrogen levels",
    "running shoe sizing",
    "coffee grinder burr alignment",
]


def build_history(seed: int, count: int) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    for index in range(count):
        project = rng.choice(PROJECTS)
        preference = rng.choice(PREFERENCES)
        task = rng.choice(TASKS)
        style = rng.choice(
            [
                f"{project}: {task}. Preference: {preference}.",
                f"While handling {project}, the user {task} and {preference}.",
                f"Memory for {project}. Task was to {task}; user {preference}.",
            ]
        )
        rows.append(
            {
                "id": f"synthetic_{index:04d}",
                "text": style,
                "metadata": {"project": project},
                "positive_edge_hints": [project, preference],
                "preference": preference,
                "task": task,
            }
        )
    return rows


def memory_card(row: dict, index: int) -> dict:
    metadata = row.get("metadata") or {}
    cluster = metadata.get("project", "")
    return {
        "id": row["id"],
        "text": row["text"],
        "summary": "",
        "embedding": embed_text(row["text"]),
        "importance": 0.4 + (index % 5) * 0.1,
        "cluster": cluster,
        "metadata": metadata,
    }


def build_cases(seed: int, count: int, candidates: int) -> list[dict]:
    rows = build_history(seed, max(count * 3, candidates + 64))
    cards = [memory_card(row, idx) for idx, row in enumerate(rows)]
    rng = random.Random(seed + 1000)
    cases = []
    for idx in range(count):
        anchor = cards[idx]
        same_project = [card for card in cards if card["id"] != anchor["id"] and card["cluster"] == anchor["cluster"]]
        same_preference = [
            card
            for card, row in zip(cards, rows)
            if card["id"] != anchor["id"]
            and card["cluster"] != anchor["cluster"]
            and row.get("preference") in anchor["text"]
        ]
        other = [
            card
            for card in cards
            if card["id"] != anchor["id"]
            and card["cluster"] != anchor["cluster"]
            and card not in same_preference
        ]
        positive_target = max(1, candidates // 4)
        hard_target = max(1, candidates // 8)
        chosen = rng.sample(same_project, min(len(same_project), positive_target))
        chosen.extend(rng.sample(same_preference, min(len(same_preference), hard_target)))
        remaining_other = [card for card in other if card not in chosen]
        other_target = max(0, candidates // 4 - len(chosen))
        chosen.extend(rng.sample(remaining_other, min(len(remaining_other), other_target)))
        while len(chosen) < candidates:
            noise_text = f"{rng.choice(NOISE)}. Reference {rng.randint(1000, 9999)}."
            chosen.append(
                {
                    "id": f"noise_{idx}_{len(chosen)}",
                    "text": noise_text,
                    "summary": "",
                    "embedding": embed_text(noise_text),
                    "importance": 0.2,
                    "cluster": "noise",
                    "metadata": {"project": "noise"},
                }
            )
        rng.shuffle(chosen)
        actions = [heuristic_action(anchor, candidate) for candidate in chosen]
        cases.append(
            {
                "anchor": anchor,
                "candidates": chosen,
                "labels": {
                    "attach": [action["attach"] for action in actions],
                    "edge_type": [action["edge_type_id"] for action in actions],
                    "weight": [action["weight"] for action in actions],
                    "confidence": [action["confidence"] for action in actions],
                    "decay_rate": [action["decay_rate"] for action in actions],
                    "importance_delta": [action["importance_delta"] for action in actions],
                },
                "teacher": "heuristic_v0",
                "edge_types": EDGE_TYPES,
            }
        )
    return cases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/synthetic/librarian_cases.jsonl")
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--candidates", type=int, default=32)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--format", choices=["cases", "memories"], default="cases")
    args = parser.parse_args()
    rows = build_cases(args.seed, args.count, args.candidates) if args.format == "cases" else build_history(args.seed, args.count)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
