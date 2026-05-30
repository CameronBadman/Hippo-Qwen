from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


PROJECTS = ["hippograph", "resume", "printer", "research-notes"]
PREFERENCES = ["prefers concise answers", "wants source links", "likes Go runtimes", "avoids large dependencies"]
TASKS = ["debugged retrieval", "updated deployment", "tested memory recall", "refined graph traversal"]


def build_history(seed: int, count: int) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    for index in range(count):
        project = rng.choice(PROJECTS)
        preference = rng.choice(PREFERENCES)
        task = rng.choice(TASKS)
        rows.append(
            {
                "id": f"synthetic_{index:04d}",
                "text": f"User working on {project} {task}; user {preference}.",
                "metadata": {"project": project},
                "positive_edge_hints": [project, preference],
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/synthetic/memories.jsonl")
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    rows = build_history(args.seed, args.count)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()

