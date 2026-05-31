from __future__ import annotations

import argparse
import json
import random
import sys
from collections.abc import Iterator
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from python.librarian.features import EDGE_TYPE_TO_ID, EDGE_TYPES, embed_text, heuristic_action


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
RELEVANT_ROLES = {"relevant", "cross_relevant", "longitudinal_relevant", "preference_relevant"}


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
                "age_days": rng.choice([1, 3, 7, 14, 30, 90, 180, 365]),
                "use_count": rng.choice([0, 0, 1, 2, 5, 13, 34]),
                "evidence_count": rng.choice([0, 1, 2, 3, 5, 8]),
                "last_outcome": rng.choice(["", "", "", "helpful", "ignored", "corrected"]),
                "protected": rng.random() < 0.08,
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
        "age_days": row.get("age_days", 0),
        "use_count": row.get("use_count", 0),
        "evidence_count": row.get("evidence_count", 0),
        "last_outcome": row.get("last_outcome", ""),
        "protected": bool(row.get("protected", False)),
    }


def generated_card(anchor: dict, text: str, role: str, idx: int, slot: int, **state: object) -> dict:
    card = {
        "id": f"{role}_{idx}_{slot}",
        "text": text,
        "summary": "",
        "embedding": embed_text(text),
        "importance": float(state.pop("importance", 0.35)),
        "cluster": str(state.pop("cluster", anchor.get("cluster", ""))),
        "metadata": {"project": str(state.pop("project", anchor.get("cluster", "")))},
        "age_days": int(state.pop("age_days", 0)),
        "use_count": int(state.pop("use_count", 0)),
        "evidence_count": int(state.pop("evidence_count", 0)),
        "last_outcome": str(state.pop("last_outcome", "")),
        "protected": bool(state.pop("protected", False)),
        "synthetic_role": role,
    }
    return card


def iter_cases(seed: int, count: int, candidates: int) -> Iterator[dict]:
    rows = build_history(seed, max(count * 3, candidates + 64))
    cards = [memory_card(row, idx) for idx, row in enumerate(rows)]
    by_project: dict[str, list[int]] = {}
    by_preference: dict[str, list[int]] = {}
    for card_index, (card, row) in enumerate(zip(cards, rows)):
        by_project.setdefault(card["cluster"], []).append(card_index)
        by_preference.setdefault(row["preference"], []).append(card_index)
    rng = random.Random(seed + 1000)
    for idx in range(count):
        anchor = cards[idx]
        anchor_preference = rows[idx]["preference"]
        anchor_task = rows[idx]["task"]
        excluded = {idx}
        same_project_indexes = [card_index for card_index in by_project[anchor["cluster"]] if card_index != idx]
        same_preference_indexes = [
            card_index
            for card_index in by_preference[anchor_preference]
            if card_index != idx and cards[card_index]["cluster"] != anchor["cluster"]
        ]
        positive_target = max(2, candidates // 5)
        hard_target = max(2, candidates // 8)
        chosen_roles: list[tuple[dict, str]] = []
        chosen_indexes = rng.sample(same_project_indexes, min(len(same_project_indexes), positive_target))
        excluded.update(chosen_indexes)
        preference_sample = rng.sample(same_preference_indexes, min(len(same_preference_indexes), hard_target))
        chosen_indexes.extend(preference_sample)
        excluded.update(preference_sample)
        for card_index in chosen_indexes:
            role = "cross_relevant" if card_index in preference_sample else "relevant"
            chosen_roles.append((cards[card_index], role))

        for slot in range(max(2, candidates // 8)):
            text = f"{anchor['cluster']}: {rng.choice(NOISE)}. Not useful for {anchor_task}."
            chosen_roles.append(
                (
                    generated_card(anchor, text, "same_project_hard_negative", idx, slot, age_days=7, use_count=0),
                    "same_project_hard_negative",
                )
            )
        for slot in range(max(2, candidates // 8)):
            text = f"{anchor['cluster']}: old note about {anchor_task}. Superseded and no longer useful."
            chosen_roles.append(
                (
                    generated_card(
                        anchor,
                        text,
                        "stale_negative",
                        idx,
                        slot,
                        age_days=365,
                        use_count=0,
                        evidence_count=0,
                        last_outcome="ignored",
                        importance=0.2,
                    ),
                    "stale_negative",
                )
            )
        duplicate_text = f"{anchor['text']} Duplicate wording captured again."
        chosen_roles.append(
            (
                generated_card(anchor, duplicate_text, "near_duplicate", idx, 0, age_days=1, use_count=0, importance=0.25),
                "near_duplicate",
            )
        )

        other_target = max(0, candidates // 4 - len(chosen_roles))
        for _ in range(other_target):
            for _attempt in range(32):
                card_index = rng.randrange(len(cards))
                card = cards[card_index]
                if card_index not in excluded and card["cluster"] != anchor["cluster"]:
                    chosen_roles.append((card, "other_negative"))
                    excluded.add(card_index)
                    break
        while len(chosen_roles) < candidates:
            noise_text = f"{rng.choice(NOISE)}. Reference {rng.randint(1000, 9999)}."
            chosen_roles.append(
                (
                    generated_card(
                        anchor,
                        noise_text,
                        "noise_negative",
                        idx,
                        len(chosen_roles),
                        cluster="noise",
                        project="noise",
                        age_days=rng.choice([3, 30, 180]),
                        use_count=0,
                        importance=0.2,
                    ),
                    "noise_negative",
                )
            )
        chosen_roles = chosen_roles[:candidates]
        rng.shuffle(chosen_roles)
        chosen = [item[0] for item in chosen_roles]
        roles = [item[1] for item in chosen_roles]
        actions = [action_for_role(anchor, candidate, role) for candidate, role in chosen_roles]
        relevant_ids = [candidate["id"] for candidate, role in chosen_roles if role in RELEVANT_ROLES]
        yield {
            "anchor": anchor,
            "candidates": chosen,
            "labels": {
                "attach": [action["attach"] for action in actions],
                "rank": [action["rank"] for action in actions],
                "edge_type": [action["edge_type_id"] for action in actions],
                "weight": [action["weight"] for action in actions],
                "confidence": [action["confidence"] for action in actions],
                "decay_rate": [action["decay_rate"] for action in actions],
                "importance_delta": [action["importance_delta"] for action in actions],
            },
            "retrieval_task": {
                "query": f"Find memories useful for {anchor['cluster']} when the user {anchor_task} and {anchor_preference}.",
                "relevant_ids": relevant_ids,
                "budget": 90,
            },
            "teacher": "synthetic_retrieval_v1",
            "schema_version": 2,
            "edge_types": EDGE_TYPES,
        }


def iter_longitudinal_cases(seed: int, count: int, candidates: int) -> Iterator[dict]:
    rows = build_history(seed, max(count * 4, candidates + 128))
    cards = [memory_card(row, idx) for idx, row in enumerate(rows)]
    by_project: dict[str, list[int]] = {}
    by_preference: dict[str, list[int]] = {}
    for card_index, (card, row) in enumerate(zip(cards, rows)):
        by_project.setdefault(card["cluster"], []).append(card_index)
        by_preference.setdefault(row["preference"], []).append(card_index)

    rng = random.Random(seed + 4000)
    for idx in range(count):
        anchor = cards[idx]
        anchor_preference = rows[idx]["preference"]
        anchor_task = rows[idx]["task"]
        excluded = {idx}
        chosen_roles: list[tuple[dict, str]] = []

        same_project_indexes = [card_index for card_index in by_project[anchor["cluster"]] if card_index != idx]
        same_preference_indexes = [
            card_index
            for card_index in by_preference[anchor_preference]
            if card_index != idx and cards[card_index]["cluster"] != anchor["cluster"]
        ]

        project_positive_target = max(2, candidates // 7)
        preference_positive_target = max(1, candidates // 10)
        for card_index in rng.sample(same_project_indexes, min(len(same_project_indexes), project_positive_target)):
            card = dict(cards[card_index])
            card.update(
                {
                    "age_days": rng.choice([1, 3, 7, 14]),
                    "use_count": rng.choice([8, 13, 21, 34]),
                    "evidence_count": rng.choice([3, 5, 8]),
                    "last_outcome": "helpful",
                    "importance": max(float(card.get("importance") or 0.5), 0.65),
                    "synthetic_role": "longitudinal_relevant",
                }
            )
            chosen_roles.append((card, "longitudinal_relevant"))
            excluded.add(card_index)

        for card_index in rng.sample(same_preference_indexes, min(len(same_preference_indexes), preference_positive_target)):
            card = dict(cards[card_index])
            card.update(
                {
                    "age_days": rng.choice([1, 7, 30]),
                    "use_count": rng.choice([5, 13, 21]),
                    "evidence_count": rng.choice([3, 5]),
                    "last_outcome": "helpful",
                    "importance": max(float(card.get("importance") or 0.5), 0.6),
                    "synthetic_role": "preference_relevant",
                }
            )
            chosen_roles.append((card, "preference_relevant"))
            excluded.add(card_index)

        for slot in range(max(3, candidates // 6)):
            text = longitudinal_text(anchor["cluster"], anchor_task, anchor_preference, slot)
            chosen_roles.append(
                (
                    generated_card(
                        anchor,
                        text,
                        "stale_same_context_negative",
                        idx,
                        slot,
                        age_days=365,
                        use_count=0,
                        evidence_count=0,
                        last_outcome="ignored",
                        importance=0.25,
                    ),
                    "stale_same_context_negative",
                )
            )

        for slot in range(max(3, candidates // 6)):
            wrong_project = rng.choice([project for project in PROJECTS if project != anchor["cluster"]])
            text = longitudinal_text(wrong_project, anchor_task, anchor_preference, slot)
            chosen_roles.append(
                (
                    generated_card(
                        anchor,
                        text,
                        "popular_wrong_context_negative",
                        idx,
                        slot,
                        cluster=wrong_project,
                        project=wrong_project,
                        age_days=7,
                        use_count=55,
                        evidence_count=13,
                        last_outcome="helpful",
                        importance=0.8,
                    ),
                    "popular_wrong_context_negative",
                )
            )

        for slot in range(max(2, candidates // 8)):
            wrong_preference = rng.choice([preference for preference in PREFERENCES if preference != anchor_preference])
            text = longitudinal_text(anchor["cluster"], anchor_task, wrong_preference, slot)
            chosen_roles.append(
                (
                    generated_card(
                        anchor,
                        text,
                        "same_project_wrong_preference_negative",
                        idx,
                        slot,
                        age_days=3,
                        use_count=13,
                        evidence_count=5,
                        last_outcome="helpful",
                        importance=0.7,
                    ),
                    "same_project_wrong_preference_negative",
                )
            )

        duplicate_text = f"{anchor['text']} Repeated capture with no extra evidence."
        chosen_roles.append(
            (
                generated_card(
                    anchor,
                    duplicate_text,
                    "near_duplicate",
                    idx,
                    0,
                    age_days=1,
                    use_count=0,
                    evidence_count=0,
                    importance=0.2,
                ),
                "near_duplicate",
            )
        )

        while len(chosen_roles) < candidates:
            card_index = rng.randrange(len(cards))
            if card_index in excluded:
                continue
            card = dict(cards[card_index])
            card["synthetic_role"] = "background_negative"
            chosen_roles.append((card, "background_negative"))
            excluded.add(card_index)

        chosen_roles = chosen_roles[:candidates]
        rng.shuffle(chosen_roles)
        chosen = [item[0] for item in chosen_roles]
        roles = [item[1] for item in chosen_roles]
        actions = [action_for_role(anchor, candidate, role) for candidate, role in chosen_roles]
        relevant_ids = [candidate["id"] for candidate, role in chosen_roles if role in RELEVANT_ROLES]
        yield {
            "anchor": anchor,
            "candidates": chosen,
            "labels": {
                "attach": [action["attach"] for action in actions],
                "rank": [action["rank"] for action in actions],
                "edge_type": [action["edge_type_id"] for action in actions],
                "weight": [action["weight"] for action in actions],
                "confidence": [action["confidence"] for action in actions],
                "decay_rate": [action["decay_rate"] for action in actions],
                "importance_delta": [action["importance_delta"] for action in actions],
            },
            "retrieval_task": {
                "query": "Find memories useful for the user's current work and durable preferences.",
                "relevant_ids": relevant_ids,
                "budget": 90,
            },
            "teacher": "synthetic_longitudinal_v1",
            "schema_version": 3,
            "scenario": "longitudinal",
            "edge_types": EDGE_TYPES,
        }


def longitudinal_text(project: str, task: str, preference: str, slot: int) -> str:
    variants = [
        f"{project}: {task}. Preference: {preference}.",
        f"While handling {project}, the user {task} and {preference}.",
        f"Memory for {project}. Task was to {task}; user {preference}.",
    ]
    return variants[slot % len(variants)]


def build_cases(seed: int, count: int, candidates: int, scenario: str = "standard") -> list[dict]:
    iterator = iter_longitudinal_cases(seed, count, candidates) if scenario == "longitudinal" else iter_cases(seed, count, candidates)
    return list(iterator)


def action_for_role(anchor: dict, candidate: dict, role: str) -> dict:
    action = heuristic_action(anchor, candidate)
    if role in RELEVANT_ROLES:
        action["attach"] = 1.0
        action["rank"] = 1.0
        action["connect_score"] = max(action["connect_score"], 0.68)
        action["confidence"] = max(action["confidence"], 0.68)
        action["weight"] = max(action["weight"], 0.8)
        if role in ("cross_relevant", "preference_relevant"):
            action["edge_type"] = "preference"
            action["edge_type_id"] = EDGE_TYPE_TO_ID["preference"]
            action["decay_rate"] = 0.005
        return action
    if role == "near_duplicate":
        action["attach"] = 1.0
        action["rank"] = 0.35
        action["connect_score"] = 0.42
        action["edge_type"] = "same_context"
        action["edge_type_id"] = EDGE_TYPE_TO_ID["same_context"]
        action["weight"] = 0.28
        action["confidence"] = 0.55
        action["decay_rate"] = 0.02
        action["importance_delta"] = -0.01
        return action
    action["attach"] = 0.0
    action["rank"] = 0.0
    action["connect_score"] = min(action["connect_score"], 0.18)
    action["weight"] = min(action["weight"], 0.25)
    action["confidence"] = min(action["confidence"], 0.25)
    action["importance_delta"] = min(action["importance_delta"], 0.0)
    if role in ("stale_negative", "stale_same_context_negative"):
        action["decay_rate"] = 0.04
    return action


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/synthetic/librarian_cases.jsonl")
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--candidates", type=int, default=32)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--scenario", choices=["standard", "longitudinal"], default="standard")
    parser.add_argument("--format", choices=["cases", "memories"], default="cases")
    args = parser.parse_args()
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        if args.format == "cases":
            rows = (
                iter_longitudinal_cases(args.seed, args.count, args.candidates)
                if args.scenario == "longitudinal"
                else iter_cases(args.seed, args.count, args.candidates)
            )
        else:
            rows = build_history(args.seed, args.count)
        for row in rows:
            handle.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
