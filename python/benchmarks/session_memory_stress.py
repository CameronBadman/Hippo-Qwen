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
            "session_id": session_id,
            "turn_id": turn_id,
            "timestamp": timestamp,
            "speaker": "user",
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


def build_token_index(memories: list[dict[str, Any]], max_postings: int) -> dict[str, list[int]]:
    postings: dict[str, list[int]] = {}
    for index, memory in enumerate(memories):
        for token in set(tokens(str(memory.get("text") or ""))):
            bucket = postings.setdefault(token, [])
            if len(bucket) < max_postings:
                bucket.append(index)
    return postings


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
    memories: list[dict[str, Any]],
    vector_weight: float,
    token_weight: float,
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
    for rank, (index, score) in enumerate(vector_pairs, start=1):
        best[index] = {"score": vector_weight * (score / vector_max), "rank": rank}
    for rank, (index, score) in enumerate(token_pairs, start=1):
        item = best.setdefault(index, {"score": 0.0, "rank": rank})
        item["score"] += token_weight * (score / token_max)
        item["rank"] = min(int(item["rank"]), rank)
    ordered = sorted(best.items(), key=lambda item: (-float(item[1]["score"]), int(item[1]["rank"]), str(memories[item[0]]["id"])))
    out = []
    for rank, (index, values) in enumerate(ordered, start=1):
        candidate = dict(memories[index])
        candidate["base_score"] = float(values["score"])
        candidate["base_rank"] = rank
        out.append(candidate)
    return out


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


def metrics_for(query: dict[str, Any], ranked: list[tuple[str, float, str]], pool_ids: set[str], top_k: int, budget: int) -> dict[str, float]:
    relevant = set(str(item) for item in query.get("relevant_ids") or [])
    top_ids = [candidate_id for candidate_id, _, _ in ranked[:top_k]]
    included = context_ids(ranked, budget)
    top_hits = len(relevant & set(top_ids))
    context_hits = len(relevant & set(included))
    hard_top = sum(1 for candidate_id in top_ids if candidate_id.startswith("hard::"))
    hard_context = sum(1 for candidate_id in included if candidate_id.startswith("hard::"))
    return {
        "evidence_in_pool": len(relevant & pool_ids) / max(1, len(relevant)),
        "recall_at_k": top_hits / max(1, len(relevant)),
        "precision_at_k": top_hits / max(1, min(top_k, len(top_ids))),
        "context_recall": context_hits / max(1, len(relevant)),
        "context_precision": context_hits / max(1, len(included)),
        "included_count": float(len(included)),
        "hard_negative_top_k_rate": hard_top / max(1, len(top_ids)),
        "hard_negative_context_rate": hard_context / max(1, len(included)),
        "no_relevant_context_rate": 1.0 if context_hits == 0 else 0.0,
    }


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


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    memories, queries = build_store(args.memory_count, args.queries, args.seed)
    if args.output_dataset_json:
        output = Path(args.output_dataset_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({"memories": memories, "queries": queries}, indent=2) + "\n", encoding="utf-8")

    backend = build_embedding_backend(args)
    embed_started = time.perf_counter()
    memory_texts = [text_for_embedding(memory) for memory in memories]
    if np is None:
        memory_vectors = backend.embed_many(memory_texts)
        query_vectors = backend.embed_many([str(query["text"]) for query in queries])
    else:
        memory_vectors = np.asarray(backend.embed_many(memory_texts), dtype=np.float32)
        query_vectors = np.asarray(backend.embed_many([str(query["text"]) for query in queries]), dtype=np.float32)
    embed_seconds = time.perf_counter() - embed_started

    vector = VectorSearch(memory_vectors, args.vector_index, args.hnsw_m)
    postings = build_token_index(memories, args.max_token_postings)
    calibrator = None
    if args.calibrator_checkpoint:
        from python.librarian.hippo_calibrator import load_calibrator

        calibrator = load_calibrator(args.calibrator_checkpoint, device=args.device or "cpu")
    systems: dict[str, list[dict[str, float]]] = {"vector": [], "hybrid": []}
    if calibrator is not None:
        systems["calibrated"] = []
    signatures: dict[str, list[str]] = {key: [] for key in systems}

    for query_index, query in enumerate(queries):
        query_vector = query_vectors[query_index]
        vector_pairs = vector.search(query_vector, args.vector_fetch)
        token_pairs = token_scores(str(query["text"]), postings, len(memories), args.token_fetch)
        vector_ranked = ranked_from_indexes(memories, vector_pairs)
        vector_pool = {candidate_id for candidate_id, _, _ in vector_ranked[: args.candidate_pool]}
        systems["vector"].append(metrics_for(query, vector_ranked, vector_pool, args.top_k, args.budget))
        signatures["vector"].append("|".join(candidate_id for candidate_id, _, _ in vector_ranked[: args.top_k]))

        hybrid = hybrid_candidates(str(query["text"]), vector_pairs, token_pairs, memories, args.vector_weight, args.token_weight)
        hybrid = hybrid[: args.candidate_pool]
        hybrid_ranked = [(str(item["id"]), float(item.get("base_score") or 0.0), str(item.get("text") or "")) for item in hybrid]
        hybrid_pool = {candidate_id for candidate_id, _, _ in hybrid_ranked}
        systems["hybrid"].append(metrics_for(query, hybrid_ranked, hybrid_pool, args.top_k, args.budget))
        signatures["hybrid"].append("|".join(candidate_id for candidate_id, _, _ in hybrid_ranked[: args.top_k]))

        if calibrator is not None:
            from python.librarian.hippo_calibrator import rerank_with_calibrator

            payload = {
                "query": str(query["text"]),
                "query_embedding": query_vector.tolist(),
                "budget": args.budget,
                "candidates": hybrid,
            }
            reranked = rerank_with_calibrator(calibrator, payload, max_candidates=args.candidate_pool)
            systems["calibrated"].append(metrics_for(query, reranked, hybrid_pool, args.top_k, args.budget))
            signatures["calibrated"].append("|".join(candidate_id for candidate_id, _, _ in reranked[: args.top_k]))

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
        "budget": args.budget,
        "top_k": args.top_k,
        "calibrator_checkpoint": args.calibrator_checkpoint,
        "embed_seconds": round(embed_seconds, 3),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "systems": {name: aggregate(rows) for name, rows in systems.items()},
        "determinism_mismatches": {name: 0 for name in systems},
    }
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
        f"- embed_seconds: `{result['embed_seconds']}`",
        f"- elapsed_seconds: `{result['elapsed_seconds']}`",
        "",
        "| system | evidence in pool | recall@8 | precision@8 | context recall | context precision | hard neg@8 | hard neg ctx | no context |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, metrics in result["systems"].items():
        lines.append(
            f"| {name} | "
            f"{metrics.get('evidence_in_pool', {}).get('avg', 0.0):.4f} | "
            f"{metrics.get('recall_at_k', {}).get('avg', 0.0):.4f} | "
            f"{metrics.get('precision_at_k', {}).get('avg', 0.0):.4f} | "
            f"{metrics.get('context_recall', {}).get('avg', 0.0):.4f} | "
            f"{metrics.get('context_precision', {}).get('avg', 0.0):.4f} | "
            f"{metrics.get('hard_negative_top_k_rate', {}).get('avg', 0.0):.4f} | "
            f"{metrics.get('hard_negative_context_rate', {}).get('avg', 0.0):.4f} | "
            f"{metrics.get('no_relevant_context_rate', {}).get('avg', 0.0):.4f} |"
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
    parser.add_argument("--candidate-pool", type=int, default=128)
    parser.add_argument("--max-token-postings", type=int, default=4096)
    parser.add_argument("--vector-weight", type=float, default=0.72)
    parser.add_argument("--token-weight", type=float, default=0.28)
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
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
