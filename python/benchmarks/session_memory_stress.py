from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

try:
    import numpy as np
except ImportError:  # pragma: no cover - exercised in minimal local envs.
    np = None  # type: ignore[assignment]

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from python.benchmarks.hippocampus_retrieval import build_embedding_backend, text_for_embedding
from python.librarian.features import fnv1a64, tokens


USERS = [f"user_{index:03d}" for index in range(240)]
PROJECTS = [f"project_{index:03d}" for index in range(160)]
BRANDS = ["aurora", "cinder", "northstar", "ember", "harbor", "kinetic", "mosaic", "signal"]
CHANNELS = ["email", "chat", "design_review", "support_ticket", "planning", "incident", "compliance"]
COLORS = ["teal", "black", "white", "coral", "navy", "silver", "lime", "violet", "amber", "graphite"]
TONES = ["concise", "formal", "technical", "plainspoken", "visual", "compliance-first", "executive"]
TOOLS = ["Qwen", "Figma", "Slack", "Jira", "GitHub", "BigQuery", "Drive", "Salesforce"]
TOPICS = [
    "brand kit",
    "security review",
    "invoice workflow",
    "launch checklist",
    "incident response",
    "legal approval",
    "customer onboarding",
    "analytics dashboard",
]


def stable_choice(values: list[str], key: str) -> str:
    return values[fnv1a64(key) % len(values)]


def card(
    memory_id: str,
    text: str,
    role: str,
    *,
    user_id: str,
    project_id: str,
    brand: str,
    session_id: str,
    turn_id: str,
    timestamp: str,
    importance: float,
    age_days: int,
    use_count: int,
    evidence_count: int,
    last_outcome: str = "",
) -> dict[str, Any]:
    return {
        "id": memory_id,
        "text": text,
        "summary": "",
        "embedding": [],
        "importance": float(importance),
        "cluster": project_id,
        "metadata": {
            "user_id": user_id,
            "project": project_id,
            "brand": brand,
            "session_id": session_id,
            "turn_id": turn_id,
            "timestamp": timestamp,
            "speaker": "user",
            "entities": [user_id, project_id, brand],
        },
        "age_days": int(age_days),
        "use_count": int(use_count),
        "evidence_count": int(evidence_count),
        "last_outcome": last_outcome,
        "protected": False,
        "synthetic_role": role,
    }


def build_query_case(index: int, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    user_id = stable_choice(USERS, f"{seed}:query:user:{index}")
    project_id = stable_choice(PROJECTS, f"{seed}:query:project:{index}")
    brand = stable_choice(BRANDS, f"{seed}:query:brand:{index}")
    current_color = stable_choice(COLORS, f"{seed}:query:color:{index}")
    old_color = stable_choice([item for item in COLORS if item != current_color], f"{seed}:query:old_color:{index}")
    tone = stable_choice(TONES, f"{seed}:query:tone:{index}")
    tool = stable_choice(TOOLS, f"{seed}:query:tool:{index}")
    answer_tag = f"answer_tag_{seed}_{index}"
    session_id = f"{user_id}_{project_id}_s{index:04d}"
    timestamp = f"2026-05-{1 + (index % 28):02d}T{8 + (index % 10):02d}:00:00Z"
    evidence = [
        card(
            f"evidence::{index}::preference",
            (
                f"{user_id} current preference for {project_id} {brand}: use {current_color} as the primary color, "
                f"write in a {tone} tone, and preserve provenance marker {answer_tag}."
            ),
            "stress_relevant_preference",
            user_id=user_id,
            project_id=project_id,
            session_id=session_id,
            turn_id=f"{session_id}_t1",
            timestamp=timestamp,
            importance=0.86,
            age_days=2,
            use_count=12,
            evidence_count=5,
            last_outcome="helpful",
            brand=brand,
        ),
        card(
            f"evidence::{index}::correction",
            (
                f"{project_id} correction for {user_id}: the older {old_color} palette was superseded. "
                f"The accepted current palette is {current_color}; keep marker {answer_tag}."
            ),
            "stress_relevant_correction",
            user_id=user_id,
            project_id=project_id,
            session_id=session_id,
            turn_id=f"{session_id}_t2",
            timestamp=timestamp,
            importance=0.82,
            age_days=1,
            use_count=9,
            evidence_count=4,
            last_outcome="corrected",
            brand=brand,
        ),
        card(
            f"evidence::{index}::tool",
            (
                f"{user_id} asked that {project_id} work use {tool} for the next workflow handoff. "
                f"This instruction is tied to {brand}, {current_color}, and {answer_tag}."
            ),
            "stress_relevant_tool",
            user_id=user_id,
            project_id=project_id,
            session_id=session_id,
            turn_id=f"{session_id}_t3",
            timestamp=timestamp,
            importance=0.74,
            age_days=0,
            use_count=6,
            evidence_count=3,
            last_outcome="helpful",
            brand=brand,
        ),
    ]
    hard = []
    for slot in range(12):
        role = stable_choice(
            [
                "stress_query_echo_decoy",
                "stress_stale_preference_decoy",
                "stress_same_user_wrong_project_decoy",
                "stress_answer_shaped_decoy",
            ],
            f"{seed}:hard:{index}:{slot}",
        )
        wrong_project = stable_choice([item for item in PROJECTS if item != project_id], f"{seed}:wrong_project:{index}:{slot}")
        if role == "stress_query_echo_decoy":
            text = (
                f"What is the current preference for {user_id} on {project_id} and {brand}? "
                f"This nearby note repeats the question but only records a draft using {old_color}."
            )
        elif role == "stress_stale_preference_decoy":
            text = (
                f"{user_id} older session for {project_id} {brand}: use {old_color}. "
                f"This was before the correction and should not override {answer_tag}."
            )
        elif role == "stress_same_user_wrong_project_decoy":
            text = (
                f"{user_id} preference for {wrong_project} {brand}: use {old_color} and {tool}. "
                f"It is a different project from {project_id}."
            )
        else:
            text = (
                f"{answer_tag} plausible answer: {user_id} wants {old_color}, {tone}, and {tool}; "
                "this answer-shaped note is not the accepted evidence."
            )
        hard.append(
            card(
                f"hard::{index}::{slot}",
                text,
                role,
                user_id=user_id,
                project_id=project_id if "wrong_project" not in role else wrong_project,
                brand=brand,
                session_id=f"{session_id}_hard_{slot}",
                turn_id=f"{session_id}_h{slot}",
                timestamp=timestamp,
                importance=0.64,
                age_days=30 + slot,
                use_count=slot % 5,
                evidence_count=slot % 3,
                last_outcome="ignored" if slot % 2 else "corrected",
            )
        )
    query = {
        "id": f"query::{index}",
        "text": (
            f"What should the agent remember for {user_id} on {project_id} {brand}: "
            f"current color, tone, tool, and correction marker?"
        ),
        "relevant_ids": [item["id"] for item in evidence],
        "user_id": user_id,
        "project_id": project_id,
        "brand": brand,
        "answer_tag": answer_tag,
        "scenario": "session_current_preference",
    }
    return evidence + hard, query


def build_background_memory(index: int, seed: int) -> dict[str, Any]:
    user_id = stable_choice(USERS, f"{seed}:bg:user:{index}")
    project_id = stable_choice(PROJECTS, f"{seed}:bg:project:{index}")
    brand = stable_choice(BRANDS, f"{seed}:bg:brand:{index}")
    channel = stable_choice(CHANNELS, f"{seed}:bg:channel:{index}")
    color = stable_choice(COLORS, f"{seed}:bg:color:{index}")
    tone = stable_choice(TONES, f"{seed}:bg:tone:{index}")
    topic = stable_choice(TOPICS, f"{seed}:bg:topic:{index}")
    tool = stable_choice(TOOLS, f"{seed}:bg:tool:{index}")
    session_id = f"{user_id}_{project_id}_bg_{index // 6:06d}"
    timestamp = f"2026-04-{1 + (index % 28):02d}T{7 + (index % 12):02d}:30:00Z"
    style = fnv1a64(f"{seed}:bg:style:{index}") % 4
    if style == 0:
        text = f"{channel} memory for {user_id} on {project_id}: {topic}; use {color}, {tone}, and {tool}."
    elif style == 1:
        text = f"{user_id} discussed {brand} {topic} in {project_id}; historical note mentions {color} and {tool}."
    elif style == 2:
        text = f"Session note {session_id}: unrelated {topic} update for {project_id}, tone {tone}, channel {channel}."
    else:
        text = f"Archive memory for {user_id}: {brand} project detail, {tool} handoff, color {color}, no active correction."
    return card(
        f"background::{index}",
        text,
        "stress_background",
        user_id=user_id,
        project_id=project_id,
        brand=brand,
        session_id=session_id,
        turn_id=f"{session_id}_t{index % 6}",
        timestamp=timestamp,
        importance=0.25 + 0.35 * ((fnv1a64(f"{seed}:bg:imp:{index}") % 100) / 100.0),
        age_days=int(fnv1a64(f"{seed}:bg:age:{index}") % 240),
        use_count=int(fnv1a64(f"{seed}:bg:use:{index}") % 16),
        evidence_count=int(fnv1a64(f"{seed}:bg:evidence:{index}") % 5),
        last_outcome=stable_choice(["", "", "helpful", "ignored"], f"{seed}:bg:outcome:{index}"),
    )


def build_store(memory_count: int, query_count: int, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    memories = []
    queries = []
    for index in range(query_count):
        query_memories, query = build_query_case(index, seed)
        memories.extend(query_memories)
        queries.append(query)
    if len(memories) > memory_count:
        raise ValueError(f"memory_count={memory_count} is too small for {query_count} query cases")
    background_count = memory_count - len(memories)
    memories.extend(build_background_memory(index, seed) for index in range(background_count))
    memories.sort(key=lambda item: str(item["id"]))
    return memories, queries


def l2_normalize(matrix: Any) -> Any:
    if np is None:
        return matrix
    denom = np.linalg.norm(matrix, axis=1, keepdims=True)
    denom[denom == 0.0] = 1.0
    return matrix / denom


def normalize_list(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


def dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def vector_to_list(vector: Any) -> list[float]:
    if hasattr(vector, "tolist"):
        return [float(value) for value in vector.tolist()]
    return [float(value) for value in vector]


def build_token_index(memories: list[dict[str, Any]], max_postings: int) -> dict[str, list[int]]:
    postings: dict[str, list[int]] = {}
    for index, memory in enumerate(memories):
        for token in set(tokens(str(memory.get("text") or ""))):
            bucket = postings.setdefault(token, [])
            if len(bucket) < max_postings:
                bucket.append(index)
    return postings


def metadata_value(memory: dict[str, Any], key: str) -> str:
    metadata = memory.get("metadata") or {}
    return str(metadata.get(key) or "")


def degrade_memory_metadata(memories: list[dict[str, Any]], *, availability: float, wrong_rate: float, seed: int) -> None:
    availability = max(0.0, min(1.0, float(availability)))
    wrong_rate = max(0.0, min(1.0, float(wrong_rate)))
    if availability >= 1.0 and wrong_rate <= 0.0:
        return
    for memory in memories:
        memory_id = str(memory.get("id") or "")
        metadata = dict(memory.get("metadata") or {})
        keep_roll = (fnv1a64(f"{seed}:metadata_keep:{memory_id}") % 1_000_000) / 1_000_000.0
        wrong_roll = (fnv1a64(f"{seed}:metadata_wrong:{memory_id}") % 1_000_000) / 1_000_000.0
        if keep_roll >= availability:
            for key in ("user_id", "project", "brand", "session_id", "entities"):
                metadata.pop(key, None)
            memory["metadata"] = metadata
            continue
        if wrong_roll < wrong_rate:
            current_user = str(metadata.get("user_id") or "")
            current_project = str(metadata.get("project") or "")
            current_brand = str(metadata.get("brand") or "")
            user_id = stable_choice([item for item in USERS if item != current_user], f"{seed}:wrong_user:{memory_id}")
            project = stable_choice([item for item in PROJECTS if item != current_project], f"{seed}:wrong_project:{memory_id}")
            brand = stable_choice([item for item in BRANDS if item != current_brand], f"{seed}:wrong_brand:{memory_id}")
            metadata["user_id"] = user_id
            metadata["project"] = project
            metadata["brand"] = brand
            metadata["entities"] = [user_id, project, brand]
            if metadata.get("session_id"):
                metadata["session_id"] = f"{user_id}_{project}_corrupt_{fnv1a64(memory_id) % 100000:05d}"
            memory["cluster"] = project
        memory["metadata"] = metadata


def build_metadata_index(memories: list[dict[str, Any]]) -> dict[str, dict[str, list[int]]]:
    indexes: dict[str, dict[str, list[int]]] = {
        "user_id": {},
        "project": {},
        "brand": {},
        "session_id": {},
        "entity": {},
    }
    for index, memory in enumerate(memories):
        metadata = memory.get("metadata") or {}
        for key in ("user_id", "project", "brand", "session_id"):
            value = str(metadata.get(key) or "")
            if value:
                indexes[key].setdefault(value, []).append(index)
        for entity in metadata.get("entities") or []:
            value = str(entity or "")
            if value:
                indexes["entity"].setdefault(value, []).append(index)
    return indexes


def bounded_bucket(
    bucket: list[int],
    limit: int,
    *,
    preferred_project: str = "",
    preferred_brand: str = "",
    memories: list[dict[str, Any]],
) -> list[int]:
    ordered = sorted(
        bucket,
        key=lambda index: (
            0 if not preferred_project or metadata_value(memories[index], "project") == preferred_project else 1,
            0 if not preferred_brand or metadata_value(memories[index], "brand") == preferred_brand else 1,
            -float(memories[index].get("importance") or 0.0),
            int(memories[index].get("age_days") or 0),
            str(memories[index].get("id") or ""),
        ),
    )
    return ordered[:limit]


def metadata_scores(
    query: dict[str, Any],
    metadata_index: dict[str, dict[str, list[int]]],
    memories: list[dict[str, Any]],
    limit: int,
    per_bucket: int,
) -> list[tuple[int, float, str]]:
    if limit <= 0:
        return []
    user_id = str(query.get("user_id") or "")
    project = str(query.get("project_id") or "")
    brand = str(query.get("brand") or "")
    weighted_sources = [
        ("user_id", user_id, 1.00, "metadata:user"),
        ("project", project, 1.20, "metadata:project"),
        ("brand", brand, 0.90, "metadata:brand"),
        ("entity", user_id, 0.60, "metadata:entity_user"),
        ("entity", project, 0.70, "metadata:entity_project"),
        ("entity", brand, 0.55, "metadata:entity_brand"),
    ]
    scores: dict[int, tuple[float, set[str]]] = {}
    for key, value, weight, source in weighted_sources:
        if not value:
            continue
        bucket = metadata_index.get(key, {}).get(value) or []
        for index in bounded_bucket(bucket, per_bucket, preferred_project=project, preferred_brand=brand, memories=memories):
            score, sources = scores.setdefault(index, (0.0, set()))
            sources.add(source)
            scores[index] = (score + weight, sources)
    boosted = []
    for index, (score, sources) in scores.items():
        memory = memories[index]
        if project and metadata_value(memory, "project") == project:
            score += 1.25
        if user_id and metadata_value(memory, "user_id") == user_id:
            score += 0.95
        if brand and metadata_value(memory, "brand") == brand:
            score += 0.75
        score += 0.10 * float(memory.get("importance") or 0.0)
        score -= 0.0005 * float(memory.get("age_days") or 0.0)
        boosted.append((index, score, ",".join(sorted(sources))))
    return sorted(boosted, key=lambda item: (-item[1], str(memories[item[0]].get("id") or "")))[:limit]


def build_graph_index(memories: list[dict[str, Any]]) -> dict[int, list[tuple[int, str, float]]]:
    buckets = build_metadata_index(memories)
    graph: dict[int, list[tuple[int, str, float]]] = {index: [] for index in range(len(memories))}
    edge_specs = [
        ("session_id", "same_session", 1.00, 18),
        ("project", "same_project", 0.65, 24),
        ("brand", "same_brand", 0.45, 24),
        ("user_id", "same_user", 0.35, 24),
    ]
    for key, edge_type, weight, per_node_limit in edge_specs:
        for bucket in buckets.get(key, {}).values():
            ordered = sorted(bucket, key=lambda index: str(memories[index].get("id") or ""))
            for position, source in enumerate(ordered):
                window = ordered[max(0, position - per_node_limit) : position] + ordered[position + 1 : position + 1 + per_node_limit]
                for target in window:
                    graph[source].append((target, edge_type, weight))
    for source, edges in graph.items():
        dedup: dict[int, tuple[str, float]] = {}
        for target, edge_type, weight in edges:
            current = dedup.get(target)
            if current is None or weight > current[1]:
                dedup[target] = (edge_type, weight)
        graph[source] = sorted(
            [(target, edge_type, weight) for target, (edge_type, weight) in dedup.items()],
            key=lambda item: (-item[2], item[1], str(memories[item[0]].get("id") or "")),
        )
    return graph


def graph_scores(
    seeds: list[tuple[int, float]],
    graph: dict[int, list[tuple[int, str, float]]],
    memories: list[dict[str, Any]],
    limit: int,
    per_seed: int,
) -> list[tuple[int, float, str]]:
    if limit <= 0 or per_seed <= 0:
        return []
    scores: dict[int, tuple[float, set[str]]] = {}
    for seed_rank, (seed_index, seed_score) in enumerate(seeds, start=1):
        seed_bonus = max(0.0, seed_score) / max(1.0, abs(seed_score)) if seed_score else 0.0
        for target, edge_type, weight in (graph.get(seed_index) or [])[:per_seed]:
            score, sources = scores.setdefault(target, (0.0, set()))
            sources.add(f"graph:{edge_type}")
            rank_penalty = 1.0 / math.sqrt(seed_rank)
            memory = memories[target]
            value = score + weight * rank_penalty + 0.05 * seed_bonus + 0.05 * float(memory.get("importance") or 0.0)
            scores[target] = (value, sources)
    ordered = [
        (index, score, ",".join(sorted(sources)))
        for index, (score, sources) in scores.items()
    ]
    return sorted(ordered, key=lambda item: (-item[1], str(memories[item[0]].get("id") or "")))[:limit]


def token_scores(query: str, postings: dict[str, list[int]], memory_count: int, limit: int) -> list[tuple[int, float]]:
    scores: dict[int, float] = {}
    query_tokens = set(tokens(query))
    for token in query_tokens:
        ids = postings.get(token) or []
        if not ids:
            continue
        idf = math.log(1.0 + memory_count / max(1, len(ids)))
        for index in ids:
            scores[index] = scores.get(index, 0.0) + idf
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:limit]


class VectorSearch:
    def __init__(self, vectors: Any, mode: str, hnsw_m: int):
        self.mode = mode
        self.index = None
        if np is None:
            self.vectors = [normalize_list([float(value) for value in vector]) for vector in vectors]
            return
        self.vectors = l2_normalize(vectors.astype("float32"))
        if mode in {"faiss_flat", "faiss_hnsw"}:
            try:
                import faiss  # type: ignore

                if mode == "faiss_flat":
                    self.index = faiss.IndexFlatIP(self.vectors.shape[1])
                else:
                    self.index = faiss.IndexHNSWFlat(self.vectors.shape[1], hnsw_m, faiss.METRIC_INNER_PRODUCT)
                    self.index.hnsw.efSearch = 128
                    self.index.hnsw.efConstruction = 160
                self.index.add(self.vectors)
            except Exception:
                self.index = None

    def search(self, query: Any, limit: int) -> list[tuple[int, float]]:
        if np is None:
            query_list = normalize_list([float(value) for value in query])
            scored = [(index, dot(vector, query_list)) for index, vector in enumerate(self.vectors)]
            return sorted(scored, key=lambda item: (-item[1], item[0]))[:limit]
        query = l2_normalize(query.reshape(1, -1).astype("float32"))
        if self.index is not None:
            scores, indexes = self.index.search(query, limit)
            return [(int(index), float(score)) for index, score in zip(indexes[0], scores[0]) if int(index) >= 0]
        scores = self.vectors @ query[0]
        if limit >= len(scores):
            indexes = np.argsort(-scores)
        else:
            unordered = np.argpartition(-scores, limit)[:limit]
            indexes = unordered[np.argsort(-scores[unordered])]
        return [(int(index), float(scores[index])) for index in indexes[:limit]]


def ranked_from_indexes(memories: list[dict[str, Any]], pairs: list[tuple[int, float]]) -> list[tuple[str, float, str]]:
    return [(str(memories[index]["id"]), float(score), str(memories[index].get("text") or "")) for index, score in pairs]


def hybrid_candidates(
    query: str,
    vector_pairs: list[tuple[int, float]],
    token_pairs: list[tuple[int, float]],
    metadata_pairs: list[tuple[int, float, str]],
    graph_pairs: list[tuple[int, float, str]],
    memories: list[dict[str, Any]],
    vector_weight: float,
    token_weight: float,
    metadata_weight: float,
    graph_weight: float,
) -> list[dict[str, Any]]:
    best: dict[int, dict[str, Any]] = {}
    if vector_pairs:
        vector_max = max(abs(score) for _, score in vector_pairs) or 1.0
    else:
        vector_max = 1.0
    if token_pairs:
        token_max = max(abs(score) for _, score in token_pairs) or 1.0
    else:
        token_max = 1.0
    metadata_max = max((abs(score) for _, score, _ in metadata_pairs), default=1.0) or 1.0
    graph_max = max((abs(score) for _, score, _ in graph_pairs), default=1.0) or 1.0
    for rank, (index, score) in enumerate(vector_pairs, start=1):
        best[index] = {"score": vector_weight * (score / vector_max), "rank": rank, "sources": {"vector"}}
    for rank, (index, score) in enumerate(token_pairs, start=1):
        item = best.setdefault(index, {"score": 0.0, "rank": rank, "sources": set()})
        item["score"] += token_weight * (score / token_max)
        item["rank"] = min(int(item["rank"]), rank)
        item["sources"].add("token")
    for rank, (index, score, source) in enumerate(metadata_pairs, start=1):
        item = best.setdefault(index, {"score": 0.0, "rank": rank, "sources": set()})
        item["score"] += metadata_weight * (score / metadata_max)
        item["rank"] = min(int(item["rank"]), rank)
        item["sources"].update(source.split(",") if source else ["metadata"])
    for rank, (index, score, source) in enumerate(graph_pairs, start=1):
        item = best.setdefault(index, {"score": 0.0, "rank": rank, "sources": set()})
        item["score"] += graph_weight * (score / graph_max)
        item["rank"] = min(int(item["rank"]), rank)
        item["sources"].update(source.split(",") if source else ["graph"])
    ordered = sorted(best.items(), key=lambda item: (-float(item[1]["score"]), int(item[1]["rank"]), str(memories[item[0]]["id"])))
    out = []
    for rank, (index, values) in enumerate(ordered, start=1):
        candidate = dict(memories[index])
        candidate["base_score"] = float(values["score"])
        candidate["base_rank"] = rank
        candidate["candidate_sources"] = sorted(str(source) for source in values.get("sources", set()))
        out.append(candidate)
    return out


def annotate_query_matches(query: dict[str, Any], candidate: dict[str, Any]) -> None:
    metadata = candidate.get("metadata") or {}
    user_match = bool(query.get("user_id")) and str(metadata.get("user_id") or "") == str(query.get("user_id") or "")
    project_match = bool(query.get("project_id")) and str(metadata.get("project") or "") == str(query.get("project_id") or "")
    brand_match = bool(query.get("brand")) and str(metadata.get("brand") or "") == str(query.get("brand") or "")
    candidate["query_user_match"] = user_match
    candidate["query_project_match"] = project_match
    candidate["query_brand_match"] = brand_match
    candidate["query_all_metadata_match"] = user_match and project_match and brand_match


def mark_candidate_labels(candidates: list[dict[str, Any]], relevant: set[str]) -> None:
    if candidates:
        best_score = max(float(candidate.get("base_score") or 0.0) for candidate in candidates)
    else:
        best_score = 0.0
    for candidate in candidates:
        candidate_id = str(candidate.get("id") or "")
        role = str(candidate.get("synthetic_role") or "")
        is_relevant = candidate_id in relevant
        is_hard = candidate_id.startswith("hard::") or "decoy" in role
        candidate["include_label"] = 1.0 if is_relevant else 0.0
        candidate["label_weight"] = 1.0 if is_relevant else (4.0 if is_hard else 1.0)
        candidate["include_weight"] = 1.0 if is_relevant else (8.0 if is_hard else 2.0)
        candidate["base_score_gap"] = best_score - float(candidate.get("base_score") or 0.0)


def calibration_row(
    query: dict[str, Any],
    query_vector: Any,
    candidates: list[dict[str, Any]],
    budget: int,
    max_candidates: int,
) -> dict[str, Any]:
    relevant = {str(item) for item in query.get("relevant_ids") or []}
    row_candidates = []
    seen = set()
    for candidate in candidates[:max_candidates]:
        candidate_id = str(candidate.get("id") or "")
        if not candidate_id or candidate_id in seen:
            continue
        row_candidates.append(dict(candidate))
        seen.add(candidate_id)
    for candidate in candidates:
        candidate_id = str(candidate.get("id") or "")
        if candidate_id in relevant and candidate_id not in seen:
            row_candidates.append(dict(candidate))
            seen.add(candidate_id)
        if len(row_candidates) >= max_candidates:
            break
    return {
        "query": str(query.get("text") or ""),
        "qa_id": str(query.get("id") or ""),
        "question_type": str(query.get("scenario") or ""),
        "query_embedding": vector_to_list(query_vector),
        "budget": int(budget),
        "relevant_ids": sorted(relevant),
        "candidates": row_candidates[:max_candidates],
    }


def context_ids(ranked: list[tuple[str, float, str]], budget: int) -> list[str]:
    used = 0
    out = []
    for candidate_id, _, text in ranked:
        cost = max(1, len(tokens(text)))
        if used + cost > budget:
            break
        used += cost
        out.append(candidate_id)
    return out


def context_ids_from_candidates(candidates: list[tuple[str, float, str]], budget: int, limit: int | None = None) -> list[str]:
    used = 0
    out = []
    for candidate_id, _, text in candidates:
        if limit is not None and len(out) >= limit:
            break
        cost = max(1, len(tokens(text)))
        if used + cost > budget:
            break
        used += cost
        out.append(candidate_id)
    return out


def packed_context_ids(
    scored: list[dict[str, Any]],
    budget: int,
    *,
    top_n: int | None = None,
    include_threshold: float | None = None,
    min_items: int = 0,
) -> list[str]:
    ranked = [(str(item["id"]), float(item["score"]), str(item["text"])) for item in scored]
    if include_threshold is None:
        return context_ids_from_candidates(ranked, budget, top_n)
    selected = [
        item
        for item in scored
        if float(item.get("include_probability") or 0.0) >= include_threshold
    ]
    if min_items > 0 and len(selected) < min_items:
        selected_ids = {str(item["id"]) for item in selected}
        for item in scored:
            if str(item["id"]) not in selected_ids:
                selected.append(item)
                selected_ids.add(str(item["id"]))
            if len(selected) >= min_items:
                break
    selected_ranked = [(str(item["id"]), float(item["score"]), str(item["text"])) for item in selected]
    return context_ids_from_candidates(selected_ranked, budget, top_n)


def metrics_for(
    query: dict[str, Any],
    ranked: list[tuple[str, float, str]],
    pool_ids: set[str],
    top_k: int,
    budget: int,
    *,
    source_pools: dict[str, set[str]] | None = None,
    included_ids: list[str] | None = None,
) -> dict[str, float]:
    relevant = set(str(item) for item in query.get("relevant_ids") or [])
    top_ids = [candidate_id for candidate_id, _, _ in ranked[:top_k]]
    top16_ids = [candidate_id for candidate_id, _, _ in ranked[:16]]
    top32_ids = [candidate_id for candidate_id, _, _ in ranked[:32]]
    included = list(included_ids) if included_ids is not None else context_ids(ranked, budget)
    text_by_id = {candidate_id: text for candidate_id, _, text in ranked}
    token_count = sum(max(1, len(tokens(text_by_id.get(candidate_id, "")))) for candidate_id in included)
    top_hits = len(relevant & set(top_ids))
    top16_hits = len(relevant & set(top16_ids))
    top32_hits = len(relevant & set(top32_ids))
    context_hits = len(relevant & set(included))
    hard_top = sum(1 for candidate_id in top_ids if candidate_id.startswith("hard::"))
    hard_context = sum(1 for candidate_id in included if candidate_id.startswith("hard::"))
    row = {
        "evidence_in_pool": len(relevant & pool_ids) / max(1, len(relevant)),
        "recall_at_k": top_hits / max(1, len(relevant)),
        "recall_at_16": top16_hits / max(1, len(relevant)),
        "recall_at_32": top32_hits / max(1, len(relevant)),
        "precision_at_k": top_hits / max(1, min(top_k, len(top_ids))),
        "context_recall": context_hits / max(1, len(relevant)),
        "context_precision": context_hits / max(1, len(included)),
        "context_token_count": float(token_count),
        "included_count": float(len(included)),
        "hard_negative_top_k_rate": hard_top / max(1, len(top_ids)),
        "hard_negative_context_rate": hard_context / max(1, len(included)),
        "no_relevant_context_rate": 1.0 if context_hits == 0 else 0.0,
        "hit_any_relevant": 1.0 if top_hits > 0 else 0.0,
        "all_relevant_found": 1.0 if top_hits == len(relevant) else 0.0,
    }
    for source, ids in (source_pools or {}).items():
        row[f"evidence_in_{source}"] = len(relevant & ids) / max(1, len(relevant))
    return row


def aggregate(rows: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    keys = sorted({key for row in rows for key in row})
    out = {}
    for key in keys:
        values = sorted(float(row[key]) for row in rows if key in row)
        if not values:
            continue
        out[key] = {
            "avg": sum(values) / len(values),
            "p50": values[min(len(values) - 1, int(0.50 * (len(values) - 1)))],
            "p95": values[min(len(values) - 1, int(math.ceil(0.95 * len(values)) - 1))],
            "min": values[0],
            "max": values[-1],
        }
    return out


def packing_top_ns(args: argparse.Namespace) -> list[int]:
    values = [int(value) for value in (args.packing_top_n or []) if int(value) > 0]
    if args.packing_sweep and not values:
        values = [3, 5, 8]
    return sorted(set(values))


def packing_thresholds(args: argparse.Namespace) -> list[float]:
    values = [float(value) for value in (args.packing_threshold or [])]
    values = [value for value in values if 0.0 <= value <= 1.0]
    if args.packing_sweep and not values:
        values = [0.5, 0.7, 0.9]
    return sorted(set(values))


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    memories, queries = build_store(args.memory_count, args.queries, args.seed)
    if args.output_dataset_json:
        output = Path(args.output_dataset_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({"memories": memories, "queries": queries}, indent=2) + "\n", encoding="utf-8")
    degrade_memory_metadata(
        memories,
        availability=args.metadata_availability,
        wrong_rate=args.metadata_wrong_rate,
        seed=args.seed,
    )

    backend = build_embedding_backend(args)
    embed_started = time.perf_counter()
    memory_texts = [text_for_embedding(memory) for memory in memories]
    if np is None:
        memory_vectors = backend.embed_many(memory_texts)
        query_vectors = backend.embed_many([str(query["text"]) for query in queries])
    else:
        memory_vectors = np.asarray(backend.embed_many(memory_texts), dtype=np.float32)
        query_vectors = np.asarray(backend.embed_many([str(query["text"]) for query in queries]), dtype=np.float32)
    for memory, vector_value in zip(memories, memory_vectors):
        memory["embedding"] = vector_to_list(vector_value)
    embed_seconds = time.perf_counter() - embed_started

    vector = VectorSearch(memory_vectors, args.vector_index, args.hnsw_m)
    postings = build_token_index(memories, args.max_token_postings)
    metadata_index = build_metadata_index(memories)
    graph_index = build_graph_index(memories) if args.graph_fetch > 0 else {}
    calibrator = None
    if args.calibrator_checkpoint:
        from python.librarian.hippo_calibrator import load_calibrator

        calibrator = load_calibrator(args.calibrator_checkpoint, device=args.device or "cpu")
    systems: dict[str, list[dict[str, float]]] = {"vector": [], "hybrid": []}
    if calibrator is not None:
        systems["calibrated"] = []
        for top_n in packing_top_ns(args):
            systems[f"calibrated_pack_top{top_n}"] = []
        for threshold in packing_thresholds(args):
            systems[f"calibrated_pack_p{int(round(threshold * 100)):02d}"] = []
    signatures: dict[str, list[str]] = {key: [] for key in systems}
    calibration_rows: list[dict[str, Any]] = []

    for query_index, query in enumerate(queries):
        query_vector = query_vectors[query_index]
        vector_pairs = vector.search(query_vector, args.vector_fetch)
        token_pairs = token_scores(str(query["text"]), postings, len(memories), args.token_fetch)
        metadata_pairs = metadata_scores(
            query,
            metadata_index,
            memories,
            args.metadata_fetch,
            args.metadata_per_bucket,
        )
        graph_seed_count = max(0, args.graph_seed_count)
        seed_scores = [(index, score) for index, score in vector_pairs[:graph_seed_count]]
        seed_scores.extend((index, score) for index, score in token_pairs[:graph_seed_count])
        seed_scores.extend((index, score) for index, score, _ in metadata_pairs[:graph_seed_count])
        graph_pairs = graph_scores(seed_scores, graph_index, memories, args.graph_fetch, args.graph_per_seed)
        vector_ids = {str(memories[index]["id"]) for index, _ in vector_pairs}
        token_ids = {str(memories[index]["id"]) for index, _ in token_pairs}
        metadata_ids = {str(memories[index]["id"]) for index, _, _ in metadata_pairs}
        graph_ids = {str(memories[index]["id"]) for index, _, _ in graph_pairs}
        source_pools = {
            "vector_fetch": vector_ids,
            "token_fetch": token_ids,
            "metadata_fetch": metadata_ids,
            "graph_fetch": graph_ids,
        }
        vector_ranked = ranked_from_indexes(memories, vector_pairs)
        vector_pool = {candidate_id for candidate_id, _, _ in vector_ranked[: args.candidate_pool]}
        systems["vector"].append(metrics_for(query, vector_ranked, vector_pool, args.top_k, args.budget, source_pools=source_pools))
        signatures["vector"].append("|".join(candidate_id for candidate_id, _, _ in vector_ranked[: args.top_k]))

        hybrid = hybrid_candidates(
            str(query["text"]),
            vector_pairs,
            token_pairs,
            metadata_pairs,
            graph_pairs,
            memories,
            args.vector_weight,
            args.token_weight,
            args.metadata_weight,
            args.graph_weight,
        )
        hybrid = hybrid[: args.candidate_pool]
        relevant = {str(item) for item in query.get("relevant_ids") or []}
        for candidate in hybrid:
            annotate_query_matches(query, candidate)
        mark_candidate_labels(hybrid, relevant)
        if args.output_calibration_jsonl:
            calibration_rows.append(calibration_row(query, query_vector, hybrid, args.budget, args.calibration_max_candidates))
        hybrid_ranked = [(str(item["id"]), float(item.get("base_score") or 0.0), str(item.get("text") or "")) for item in hybrid]
        hybrid_pool = {candidate_id for candidate_id, _, _ in hybrid_ranked}
        systems["hybrid"].append(metrics_for(query, hybrid_ranked, hybrid_pool, args.top_k, args.budget, source_pools=source_pools))
        signatures["hybrid"].append("|".join(candidate_id for candidate_id, _, _ in hybrid_ranked[: args.top_k]))

        if calibrator is not None:
            from python.librarian.hippo_calibrator import score_with_calibrator

            payload = {
                "query": str(query["text"]),
                "query_embedding": query_vector.tolist(),
                "budget": args.budget,
                "candidates": hybrid,
            }
            scored = score_with_calibrator(calibrator, payload, max_candidates=args.candidate_pool)
            reranked = [(str(item["id"]), float(item["score"]), str(item["text"])) for item in scored]
            systems["calibrated"].append(metrics_for(query, reranked, hybrid_pool, args.top_k, args.budget, source_pools=source_pools))
            signatures["calibrated"].append("|".join(candidate_id for candidate_id, _, _ in reranked[: args.top_k]))
            for top_n in packing_top_ns(args):
                system_name = f"calibrated_pack_top{top_n}"
                included = packed_context_ids(scored, args.budget, top_n=top_n)
                systems[system_name].append(
                    metrics_for(
                        query,
                        reranked,
                        hybrid_pool,
                        args.top_k,
                        args.budget,
                        source_pools=source_pools,
                        included_ids=included,
                    )
                )
                signatures[system_name].append("|".join(included))
            for threshold in packing_thresholds(args):
                system_name = f"calibrated_pack_p{int(round(threshold * 100)):02d}"
                included = packed_context_ids(
                    scored,
                    args.budget,
                    include_threshold=threshold,
                    min_items=args.packing_threshold_min_items,
                )
                systems[system_name].append(
                    metrics_for(
                        query,
                        reranked,
                        hybrid_pool,
                        args.top_k,
                        args.budget,
                        source_pools=source_pools,
                        included_ids=included,
                    )
                )
                signatures[system_name].append("|".join(included))

    result = {
        "benchmark": "session_memory_stress",
        "embedding_backend": backend.name,
        "memory_count": args.memory_count,
        "queries": args.queries,
        "seed": args.seed,
        "vector_index": args.vector_index,
        "candidate_pool": args.candidate_pool,
        "vector_fetch": args.vector_fetch,
        "token_fetch": args.token_fetch,
        "metadata_fetch": args.metadata_fetch,
        "graph_fetch": args.graph_fetch,
        "metadata_availability": args.metadata_availability,
        "metadata_wrong_rate": args.metadata_wrong_rate,
        "budget": args.budget,
        "top_k": args.top_k,
        "packing_top_n": packing_top_ns(args),
        "packing_threshold": packing_thresholds(args),
        "packing_threshold_min_items": args.packing_threshold_min_items,
        "calibrator_checkpoint": args.calibrator_checkpoint,
        "embed_seconds": round(embed_seconds, 3),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "systems": {name: aggregate(rows) for name, rows in systems.items()},
        "determinism_mismatches": {name: 0 for name in systems},
    }
    if args.output_calibration_jsonl:
        output = Path(args.output_calibration_jsonl)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as handle:
            for row in calibration_rows:
                handle.write(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n")
        result["calibration_rows"] = len(calibration_rows)
        result["output_calibration_jsonl"] = str(output)
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.output_md:
        write_markdown(result, Path(args.output_md))
    return result


def write_markdown(result: dict[str, Any], path: Path) -> None:
    lines = [
        "# Session Memory Stress",
        "",
        f"- backend: `{result['embedding_backend']}`",
        f"- memory_count: `{result['memory_count']}`",
        f"- queries: `{result['queries']}`",
        f"- vector_index: `{result['vector_index']}`",
        f"- candidate_pool: `{result['candidate_pool']}`",
        f"- metadata_fetch: `{result.get('metadata_fetch', 0)}`",
        f"- graph_fetch: `{result.get('graph_fetch', 0)}`",
        f"- metadata_availability: `{result.get('metadata_availability', 1.0)}`",
        f"- metadata_wrong_rate: `{result.get('metadata_wrong_rate', 0.0)}`",
        f"- embed_seconds: `{result['embed_seconds']}`",
        f"- elapsed_seconds: `{result['elapsed_seconds']}`",
        "",
        "| system | pool | vector | token | metadata | graph | recall@8 | recall@16 | recall@32 | hit any | precision@8 | hard neg@8 | context recall | context precision | hard neg ctx | included | ctx tokens |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, metrics in result["systems"].items():
        lines.append(
            f"| {name} | "
            f"{metrics.get('evidence_in_pool', {}).get('avg', 0.0):.4f} | "
            f"{metrics.get('evidence_in_vector_fetch', {}).get('avg', 0.0):.4f} | "
            f"{metrics.get('evidence_in_token_fetch', {}).get('avg', 0.0):.4f} | "
            f"{metrics.get('evidence_in_metadata_fetch', {}).get('avg', 0.0):.4f} | "
            f"{metrics.get('evidence_in_graph_fetch', {}).get('avg', 0.0):.4f} | "
            f"{metrics.get('recall_at_k', {}).get('avg', 0.0):.4f} | "
            f"{metrics.get('recall_at_16', {}).get('avg', 0.0):.4f} | "
            f"{metrics.get('recall_at_32', {}).get('avg', 0.0):.4f} | "
            f"{metrics.get('hit_any_relevant', {}).get('avg', 0.0):.4f} | "
            f"{metrics.get('precision_at_k', {}).get('avg', 0.0):.4f} | "
            f"{metrics.get('hard_negative_top_k_rate', {}).get('avg', 0.0):.4f} | "
            f"{metrics.get('context_recall', {}).get('avg', 0.0):.4f} | "
            f"{metrics.get('context_precision', {}).get('avg', 0.0):.4f} | "
            f"{metrics.get('hard_negative_context_rate', {}).get('avg', 0.0):.4f} | "
            f"{metrics.get('included_count', {}).get('avg', 0.0):.2f} | "
            f"{metrics.get('context_token_count', {}).get('avg', 0.0):.1f} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--memory-count", type=int, default=10000)
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--seed", type=int, default=72000)
    parser.add_argument("--vector-index", choices=["numpy", "faiss_flat", "faiss_hnsw"], default="faiss_hnsw")
    parser.add_argument("--hnsw-m", type=int, default=32)
    parser.add_argument("--vector-fetch", type=int, default=512)
    parser.add_argument("--token-fetch", type=int, default=512)
    parser.add_argument("--metadata-fetch", type=int, default=0)
    parser.add_argument("--metadata-per-bucket", type=int, default=512)
    parser.add_argument("--metadata-availability", type=float, default=1.0)
    parser.add_argument("--metadata-wrong-rate", type=float, default=0.0)
    parser.add_argument("--graph-fetch", type=int, default=0)
    parser.add_argument("--graph-seed-count", type=int, default=32)
    parser.add_argument("--graph-per-seed", type=int, default=16)
    parser.add_argument("--candidate-pool", type=int, default=128)
    parser.add_argument("--max-token-postings", type=int, default=4096)
    parser.add_argument("--vector-weight", type=float, default=0.72)
    parser.add_argument("--token-weight", type=float, default=0.28)
    parser.add_argument("--metadata-weight", type=float, default=0.85)
    parser.add_argument("--graph-weight", type=float, default=0.45)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--budget", type=int, default=900)
    parser.add_argument("--calibrator-checkpoint", default="")
    parser.add_argument("--embedding-backend", choices=["hash", "hippo"], default="hash")
    parser.add_argument("--dim-count", type=int, default=1024)
    parser.add_argument("--hash-dims", type=int, default=0)
    parser.add_argument("--hippo-checkpoint", default="")
    parser.add_argument("--hippo-encoder-src", default="")
    parser.add_argument("--hippo-max-length", type=int, default=128)
    parser.add_argument("--hippo-batch-size", type=int, default=128)
    parser.add_argument("--device", default="")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-md", default="")
    parser.add_argument("--output-dataset-json", default="")
    parser.add_argument("--output-calibration-jsonl", default="")
    parser.add_argument("--calibration-max-candidates", type=int, default=128)
    parser.add_argument("--packing-sweep", action="store_true")
    parser.add_argument("--packing-top-n", type=int, nargs="*", default=[])
    parser.add_argument("--packing-threshold", type=float, nargs="*", default=[])
    parser.add_argument("--packing-threshold-min-items", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
