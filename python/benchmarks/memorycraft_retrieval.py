from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from python.benchmarks.hard_memory_regression import aggregate
from python.benchmarks.hierarchical_file_ann import Ranked, ranked_signature
from python.benchmarks.hippocampus_retrieval import build_embedding_backend, ensure_backend_embeddings
from python.benchmarks.agent_memory_graph import build_agent_memory_graph, search_agent_memory_graph
from python.benchmarks.rope_delta_grid import build_rope_delta_grid, query_embedding_for, search_rope_delta_grid
from python.benchmarks.vector_db_compare import (
    build_faiss,
    build_hnswlib,
    exact_vector_search,
    faiss_search,
    hnswlib_search,
    hybrid_union_token_search,
)
from python.field_memory.token_field import build_token_field_index
from python.librarian.features import activation_mask_for_text, fnv1a64, tokens


DEFAULT_HF_REPO = "daven3/MemoryCraft"
DEFAULT_HF_FILE = "selected/sample.jsonl"


def parse_systems(value: str) -> list[str]:
    systems = [item.strip() for item in value.split(",") if item.strip()]
    valid = {
        "exact_vector",
        "faiss_flat",
        "faiss_hnsw",
        "hnswlib",
        "hippo_rope_grid",
        "hippo_calibrated",
        "hippo_calibrated_union",
        "agent_memory_graph",
    }
    unknown = sorted(set(systems) - valid)
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown systems: {','.join(unknown)}")
    return systems


def load_dataset_path(args: argparse.Namespace) -> Path:
    if args.dataset:
        return Path(args.dataset)
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("install huggingface_hub or pass --dataset with a local JSONL file") from exc
    return Path(hf_hub_download(args.hf_repo, args.hf_file, repo_type="dataset"))


def load_records(path: Path, limit: int) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            records.append(json.loads(line))
            if limit > 0 and len(records) >= limit:
                break
    return records


def turn_text(session: dict[str, Any], turn: dict[str, Any]) -> str:
    parts = []
    timestamp = str(session.get("timestamp") or "").strip()
    speaker = str(turn.get("speaker") or turn.get("role") or "").strip()
    content = str(turn.get("content") or "").strip()
    caption = str((turn.get("metadata") or {}).get("blip_caption") or "").strip()
    if timestamp:
        parts.append(f"[{timestamp}]")
    if speaker:
        parts.append(f"{speaker}:")
    if content:
        parts.append(content)
    if caption:
        parts.append(f"Image: {caption}")
    return " ".join(parts).strip()


def flatten_turns(record: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for session_index, session in enumerate(record.get("sessions") or []):
        session_id = str(session.get("session_id") or f"session_{session_index}")
        for turn_index, turn in enumerate(session.get("turns") or []):
            turn_id = str(turn.get("turn_id") or f"{session_id}_t{turn_index}")
            text = turn_text(session, turn)
            if not text:
                continue
            out.append(
                {
                    "id": turn_id,
                    "session_id": session_id,
                    "turn_id": turn_id,
                    "speaker": str(turn.get("speaker") or turn.get("role") or ""),
                    "timestamp": str(session.get("timestamp") or ""),
                    "text": text,
                    "metadata": dict(turn.get("metadata") or {}),
                }
            )
    return out


def flatten_sessions(record: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for session_index, session in enumerate(record.get("sessions") or []):
        session_id = str(session.get("session_id") or f"session_{session_index}")
        turn_texts = []
        speakers = set()
        has_answer = False
        for turn in session.get("turns") or []:
            text = turn_text(session, turn)
            if text:
                turn_texts.append(text)
            speaker = str(turn.get("speaker") or turn.get("role") or "")
            if speaker:
                speakers.add(speaker)
            if bool((turn.get("metadata") or {}).get("has_answer")):
                has_answer = True
        if not turn_texts:
            continue
        out.append(
            {
                "id": session_id,
                "session_id": session_id,
                "turn_id": "",
                "speaker": ",".join(sorted(speakers)),
                "timestamp": str(session.get("timestamp") or ""),
                "text": "\n".join(turn_texts),
                "metadata": {"has_answer": has_answer},
            }
        )
    return out


def unit_mode(record: dict[str, Any], requested: str) -> str:
    if requested != "auto":
        return requested
    evidence = {str(item) for qa in record.get("qa") or [] for item in (qa.get("evidence") or [])}
    turn_ids = {item["id"] for item in flatten_turns(record)}
    if evidence and evidence <= turn_ids:
        return "turn"
    session_ids = {str(session.get("session_id") or "") for session in record.get("sessions") or []}
    answer_session_ids = {
        str(session.get("session_id") or "")
        for session in record.get("sessions") or []
        for turn in session.get("turns") or []
        if bool((turn.get("metadata") or {}).get("has_answer"))
    }
    if evidence and evidence <= session_ids and evidence & answer_session_ids:
        return "turn"
    if evidence and evidence <= session_ids:
        return "session"
    return "turn"


def normalize_evidence(record: dict[str, Any], qa: dict[str, Any], mode: str) -> set[str]:
    raw = {str(item) for item in qa.get("evidence") or [] if str(item)}
    if not raw:
        return set()
    if mode == "session":
        return raw
    turn_ids = {item["id"] for item in flatten_turns(record)}
    if raw <= turn_ids:
        return raw
    session_to_turns: dict[str, list[str]] = {}
    session_to_answer_turns: dict[str, list[str]] = {}
    for item in flatten_turns(record):
        session_to_turns.setdefault(str(item["session_id"]), []).append(str(item["id"]))
        if bool((item.get("metadata") or {}).get("has_answer")):
            session_to_answer_turns.setdefault(str(item["session_id"]), []).append(str(item["id"]))
    expanded = set()
    for evidence_id in raw:
        expanded.update(session_to_answer_turns.get(evidence_id) or session_to_turns.get(evidence_id, []))
    return expanded or raw


def card_for_unit(record: dict[str, Any], unit: dict[str, Any], index: int) -> dict[str, Any]:
    metadata = dict(unit.get("metadata") or {})
    metadata.update(
        {
            "project": str(record.get("uid") or "memorycraft"),
            "source": str(record.get("source") or ""),
            "session_id": str(unit.get("session_id") or ""),
            "turn_id": str(unit.get("turn_id") or ""),
            "speaker": str(unit.get("speaker") or ""),
            "timestamp": str(unit.get("timestamp") or ""),
        }
    )
    text = str(unit.get("text") or "")
    return {
        "id": str(unit["id"]),
        "text": text,
        "summary": text[:240],
        "cluster": str(record.get("uid") or "memorycraft"),
        "importance": 0.5,
        "age_days": index,
        "use_count": 0,
        "metadata": metadata,
    }


def stable_pick(values: list[str], key: str) -> str:
    return values[fnv1a64(key) % len(values)]


def adversarial_cards_for_qa(
    base: dict[str, Any],
    qa: dict[str, Any],
    relevant: set[str],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    count = max(0, int(args.adversarial_negatives))
    if count <= 0:
        return []
    query = str(qa.get("question") or "")
    answer = str(qa.get("answer") or "")
    qa_id = str(qa.get("qa_id") or "qa")
    id_to_candidate = {str(candidate.get("id") or ""): candidate for candidate in base.get("candidates", [])}
    positive_text = " ".join(str(id_to_candidate[item].get("text") or "") for item in sorted(relevant) if item in id_to_candidate)[:360]
    project = str((base.get("anchor") or {}).get("cluster") or (base.get("anchor") or {}).get("id") or "memorycraft")
    templates = [
        (
            "eval_query_echo_negative",
            "{query} This memory repeats the query wording but is not the evidence-bearing turn.",
            {"age_days": 5, "use_count": 24, "evidence_count": 4, "last_outcome": "ignored", "importance": 0.70},
        ),
        (
            "eval_same_answer_wrong_source",
            "{query} Plausible answer-shaped memory: {answer}. It came from a different session and should be excluded.",
            {"age_days": 18, "use_count": 30, "evidence_count": 6, "last_outcome": "corrected", "importance": 0.72},
        ),
        (
            "eval_superseded_conflict",
            "{query} Older conflicting note near the accepted evidence: {positive} This note was superseded.",
            {"age_days": 260, "use_count": 2, "evidence_count": 0, "last_outcome": "ignored", "importance": 0.42},
        ),
        (
            "eval_non_authoritative_duplicate",
            "{positive} Duplicate-looking memory for {query}; it is not authoritative evidence.",
            {"age_days": 2, "use_count": 5, "evidence_count": 0, "last_outcome": "", "importance": 0.50},
        ),
        (
            "eval_stale_preference",
            "{query} Stale user preference from an older context. Later evidence replaced it.",
            {"age_days": 420, "use_count": 48, "evidence_count": 5, "last_outcome": "ignored", "importance": 0.58},
        ),
    ]
    out = []
    for index in range(count):
        key = f"{qa_id}:{query}:{index}:{','.join(sorted(relevant))}"
        role, template, state = templates[fnv1a64(key) % len(templates)]
        fallback_positive = stable_pick(
            [
                "a nearby conversation note with overlapping entities",
                "an old memory from the same project",
                "a similar session turn without the answer",
            ],
            key,
        )
        text = template.format(query=query, answer=answer or "unknown", positive=positive_text or fallback_positive).strip()
        out.append(
            {
                "id": f"adversarial_eval::{qa_id or 'qa'}::{index}",
                "text": text,
                "summary": "",
                "cluster": project,
                "importance": state["importance"],
                "age_days": state["age_days"] + int(fnv1a64(f"{key}:age") % 9),
                "use_count": state["use_count"],
                "evidence_count": state["evidence_count"],
                "last_outcome": state["last_outcome"],
                "synthetic_role": role,
                "metadata": {
                    "project": project,
                    "adversarial": True,
                    "adversarial_type": role,
                    "qa_id": qa_id,
                },
            }
        )
    return out


def add_adversarial_candidates(
    base: dict[str, Any],
    qa_items: list[tuple[dict[str, Any], set[str]]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    if int(args.adversarial_negatives) <= 0:
        return base
    out = dict(base)
    candidates = [dict(candidate) for candidate in base.get("candidates", [])]
    seen = {str(candidate.get("id") or "") for candidate in candidates}
    for qa, relevant in qa_items:
        for candidate in adversarial_cards_for_qa(base, qa, relevant, args):
            candidate_id = str(candidate.get("id") or "")
            if candidate_id in seen:
                continue
            candidates.append(candidate)
            seen.add(candidate_id)
    out["candidates"] = candidates
    return out


def build_edges(cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    edges = []
    by_speaker: dict[str, str] = {}
    for index, card in enumerate(cards):
        current_id = str(card["id"])
        current_text = str(card.get("text") or "")
        if index > 0:
            previous = cards[index - 1]
            edges.append(
                {
                    "source": str(previous["id"]),
                    "target": current_id,
                    "type": "chronological",
                    "weight": 0.42,
                    "confidence": 0.85,
                    "activation_mask": activation_mask_for_text(current_text),
                }
            )
            edges.append(
                {
                    "source": current_id,
                    "target": str(previous["id"]),
                    "type": "chronological",
                    "weight": 0.24,
                    "confidence": 0.75,
                    "activation_mask": activation_mask_for_text(str(previous.get("text") or "")),
                }
            )
        speaker = str((card.get("metadata") or {}).get("speaker") or "")
        if speaker and speaker in by_speaker:
            edges.append(
                {
                    "source": by_speaker[speaker],
                    "target": current_id,
                    "type": "same_speaker",
                    "weight": 0.30,
                    "confidence": 0.65,
                    "activation_mask": activation_mask_for_text(current_text),
                }
            )
        if speaker:
            by_speaker[speaker] = current_id
    return edges


def record_index_row(record: dict[str, Any], mode: str) -> dict[str, Any]:
    units = flatten_sessions(record) if mode == "session" else flatten_turns(record)
    cards = [card_for_unit(record, unit, index) for index, unit in enumerate(units)]
    anchor_text = f"{record.get('source', '')} {record.get('uid', '')}".strip() or "memorycraft"
    return {
        "id": str(record.get("uid") or "record"),
        "anchor": {
            "id": f"{record.get('uid', 'record')}::anchor",
            "text": anchor_text,
            "cluster": str(record.get("uid") or "memorycraft"),
            "metadata": {"project": str(record.get("uid") or "memorycraft")},
        },
        "candidates": cards,
        "memory_graph": {"edges": build_edges(cards)},
        "retrieval_task": {"query": anchor_text, "relevant_ids": [], "budget": 1},
    }


def query_row(base: dict[str, Any], qa: dict[str, Any], relevant: set[str], budget: int) -> dict[str, Any]:
    row = dict(base)
    row["retrieval_task"] = {
        "query": str(qa.get("question") or ""),
        "answer": str(qa.get("answer") or ""),
        "qa_id": str(qa.get("qa_id") or ""),
        "question_type": str(qa.get("question_type") or ""),
        "relevant_ids": sorted(relevant),
        "budget": int(budget),
    }
    return row


def evidence_metrics(row: dict[str, Any], ranked: Ranked, top_k: int, budget: int) -> dict[str, float]:
    relevant = {str(item) for item in (row.get("retrieval_task") or {}).get("relevant_ids") or []}
    top_ids = [item[0] for item in ranked[:top_k]]
    if not relevant:
        return {
            "recall_at_k": 0.0,
            "precision_at_k": 0.0,
            "mrr": 0.0,
            "context_precision": 0.0,
            "context_recall": 0.0,
            "context_noise": 0.0,
            "included_count": 0.0,
            "relevant_count": 0.0,
        }
    hits_at_k = len(relevant & set(top_ids))
    mrr = 0.0
    for position, (candidate_id, _, _) in enumerate(ranked, start=1):
        if candidate_id in relevant:
            mrr = 1.0 / position
            break
    included = []
    used = 0
    for candidate_id, _, text in ranked:
        cost = max(1, len(tokens(text)))
        if used + cost > budget:
            break
        included.append(candidate_id)
        used += cost
    included_set = set(included)
    context_hits = len(relevant & included_set)
    return {
        "recall_at_k": hits_at_k / len(relevant),
        "precision_at_k": hits_at_k / max(1, min(top_k, len(ranked))),
        "mrr": mrr,
        "context_precision": context_hits / max(1, len(included)),
        "context_recall": context_hits / len(relevant),
        "context_noise": float(len(included) - context_hits),
        "included_count": float(len(included)),
        "relevant_count": float(len(relevant)),
    }


def evaluate_search(row: dict[str, Any], ranked: Ranked, stats: dict[str, float], args: argparse.Namespace) -> dict[str, float]:
    out = {
        "latency_ms": float(stats.get("latency_ms") or 0.0),
        "unique_nodes_read": float(stats.get("unique_nodes_read") or 0.0),
        "payload_reads": float(stats.get("payload_reads") or 0.0),
        "node_records_read": float(stats.get("node_records_read") or 0.0),
        "edge_reads": float(stats.get("edge_reads") or 0.0),
        "edge_expansions": float(stats.get("edge_expansions") or 0.0),
        "raw_final_candidate_count": float(stats.get("raw_final_candidate_count") or 0.0),
        "final_candidate_count": float(stats.get("final_candidate_count") or len(ranked)),
        "calibrator_latency_ms": float(stats.get("calibrator_latency_ms") or 0.0),
        "search_latency_ms": float(stats.get("search_latency_ms") or 0.0),
        "vector_candidate_latency_ms": float(stats.get("vector_candidate_latency_ms") or 0.0),
        "vector_candidate_count": float(stats.get("vector_candidate_count") or 0.0),
        "token_candidate_latency_ms": float(stats.get("token_candidate_latency_ms") or 0.0),
        "token_candidate_count": float(stats.get("token_candidate_count") or 0.0),
        "union_candidate_count": float(stats.get("union_candidate_count") or 0.0),
    }
    out.update(evidence_metrics(row, ranked, args.top_k, args.budget))
    return out


def run_queries(
    query_rows: list[dict[str, Any]],
    search: Callable[[dict[str, Any]], tuple[Ranked, dict[str, float], set[str]]],
    args: argparse.Namespace,
) -> tuple[list[dict[str, float]], int]:
    rows = []
    expected_signature = None
    mismatches = 0
    for row in query_rows:
        signatures = []
        repeated_metrics = []
        for _ in range(max(1, int(args.repeat_searches))):
            ranked, stats, _ = search(row)
            signatures.append(ranked_signature(ranked))
            repeated_metrics.append(evaluate_search(row, ranked, stats, args))
        if any(signature != signatures[0] for signature in signatures[1:]):
            mismatches += 1
        if expected_signature is None:
            expected_signature = signatures[0]
        rows.append(repeated_metrics[-1])
    return rows, mismatches


def index_size(meta: dict[str, Any]) -> int:
    return int(meta.get("grid_bytes", 0)) + int(meta.get("payload_bytes", 0)) + int(meta.get("records_bytes", 0)) + int(meta.get("edges_bytes", 0))


def calibrator_payload(
    row: dict[str, Any],
    ranked: Ranked,
    id_to_candidate: dict[str, dict[str, Any]],
    backend: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    candidates = []
    for rank, (candidate_id, score, _) in enumerate(ranked[: int(args.calibrator_max_candidates)], start=1):
        candidate = dict(id_to_candidate.get(candidate_id) or {})
        if not candidate:
            continue
        candidate["base_rank"] = rank
        candidate["base_score"] = float(score)
        candidates.append(candidate)
    task = row.get("retrieval_task") or {}
    return {
        "query": str(task.get("query") or ""),
        "answer": str(task.get("answer") or ""),
        "qa_id": str(task.get("qa_id") or ""),
        "question_type": str(task.get("question_type") or ""),
        "query_embedding": query_embedding_for(row, backend),
        "budget": int(task.get("budget") or args.budget),
        "relevant_ids": list(task.get("relevant_ids") or []),
        "candidates": candidates,
    }


def calibrated_search(
    row: dict[str, Any],
    backend: Any,
    meta: dict[str, Any],
    calibrator: Any,
    id_to_candidate: dict[str, dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[Ranked, dict[str, float], set[str]]:
    from python.librarian.hippo_calibrator import rerank_with_calibrator

    started = time.perf_counter()
    raw_ranked, stats, protected = search_rope_delta_grid(row, backend, meta, args)
    ranked = rerank_with_calibrator(
        calibrator,
        calibrator_payload(row, raw_ranked, id_to_candidate, backend, args),
        relevance_weight=args.rerank_relevance_weight,
        include_weight=args.rerank_include_weight,
        base_weight=args.rerank_base_weight,
        utility_weight=args.rerank_utility_weight,
    )
    total_latency_ms = (time.perf_counter() - started) * 1000.0
    search_latency_ms = float(stats.get("latency_ms") or 0.0)
    out = dict(stats)
    out["search_latency_ms"] = search_latency_ms
    out["latency_ms"] = total_latency_ms
    out["calibrator_latency_ms"] = max(0.0, total_latency_ms - search_latency_ms)
    return ranked, out, protected


def calibrated_union_search(
    row: dict[str, Any],
    backend: Any,
    faiss_built: dict[str, Any],
    token_index: Any,
    calibrator: Any,
    id_to_candidate: dict[str, dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[Ranked, dict[str, float], set[str]]:
    from python.librarian.hippo_calibrator import rerank_with_calibrator

    started = time.perf_counter()
    raw_ranked, stats, candidate_ids = hybrid_union_token_search(row, backend, faiss_built, token_index, args)
    ranked = rerank_with_calibrator(
        calibrator,
        calibrator_payload(row, raw_ranked, id_to_candidate, backend, args),
        relevance_weight=args.rerank_relevance_weight,
        include_weight=args.rerank_include_weight,
        base_weight=args.rerank_base_weight,
        utility_weight=args.rerank_utility_weight,
    )
    total_latency_ms = (time.perf_counter() - started) * 1000.0
    search_latency_ms = float(stats.get("latency_ms") or 0.0)
    out = dict(stats)
    out["search_latency_ms"] = search_latency_ms
    out["latency_ms"] = total_latency_ms
    out["calibrator_latency_ms"] = max(0.0, total_latency_ms - search_latency_ms)
    return ranked, out, candidate_ids


def run_record(
    record: dict[str, Any],
    record_number: int,
    backend: Any,
    work_dir: Path,
    args: argparse.Namespace,
    calibrator: Any = None,
) -> dict[str, Any] | None:
    mode = unit_mode(record, args.unit)
    base = record_index_row(record, mode)
    if len(base.get("candidates") or []) < 1:
        return None
    qa_items = []
    qa_count = 0
    for qa in record.get("qa") or []:
        if bool(qa.get("abstention")) and not args.include_abstention:
            continue
        relevant = normalize_evidence(record, qa, mode)
        if not relevant:
            continue
        qa_items.append((qa, relevant))
        qa_count += 1
        if args.limit_questions > 0 and qa_count >= args.limit_questions:
            break
    if not qa_items:
        return None
    base = add_adversarial_candidates(base, qa_items, args)
    qa_rows = [query_row(base, qa, relevant, args.budget) for qa, relevant in qa_items]

    embedded_base = ensure_backend_embeddings(qa_rows[0], backend)
    for row in qa_rows:
        row["anchor"] = embedded_base["anchor"]
        row["candidates"] = embedded_base["candidates"]
    id_to_candidate = {str(candidate.get("id") or ""): dict(candidate) for candidate in embedded_base.get("candidates", [])}

    systems = []
    if "exact_vector" in args.systems:
        rows, mismatches = run_queries(qa_rows, lambda row: exact_vector_search(row, backend, args), args)
        systems.append(
            {
                "name": "exact_vector",
                "metrics": aggregate(rows),
                "_metric_rows": rows,
                "determinism_mismatches": mismatches,
                "build_latency_ms": 0.0,
                "memory_count": len(embedded_base["candidates"]),
                "total_index_bytes": 0,
            }
        )
    for system_name, kind in (("faiss_flat", "flat"), ("faiss_hnsw", "hnsw")):
        if system_name not in args.systems:
            continue
        started = time.perf_counter()
        built = build_faiss(embedded_base, args, kind)
        build_ms = (time.perf_counter() - started) * 1000.0
        rows, mismatches = run_queries(qa_rows, lambda row, built=built: faiss_search(row, backend, built, args), args)
        systems.append(
            {
                "name": system_name,
                "metrics": aggregate(rows),
                "_metric_rows": rows,
                "determinism_mismatches": mismatches,
                "build_latency_ms": build_ms,
                "memory_count": built["memory_count"],
                "total_index_bytes": int(built.get("index_bytes") or 0),
            }
        )
    if "hnswlib" in args.systems:
        started = time.perf_counter()
        built = build_hnswlib(embedded_base, args)
        build_ms = (time.perf_counter() - started) * 1000.0
        rows, mismatches = run_queries(qa_rows, lambda row: hnswlib_search(row, backend, built, args), args)
        systems.append(
            {
                "name": "hnswlib",
                "metrics": aggregate(rows),
                "_metric_rows": rows,
                "determinism_mismatches": mismatches,
                "build_latency_ms": build_ms,
                "memory_count": built["memory_count"],
                "total_index_bytes": int(built.get("index_bytes") or 0),
            }
        )
    if "hippo_calibrated_union" in args.systems:
        if calibrator is None:
            raise ValueError("--calibrator-checkpoint is required for hippo_calibrated_union")
        started = time.perf_counter()
        faiss_built = build_faiss(embedded_base, args, "hnsw")
        token_index = build_token_field_index(embedded_base, args)
        build_ms = (time.perf_counter() - started) * 1000.0
        rows, mismatches = run_queries(
            qa_rows,
            lambda row: calibrated_union_search(row, backend, faiss_built, token_index, calibrator, id_to_candidate, args),
            args,
        )
        systems.append(
            {
                "name": "hippo_calibrated_union",
                "metrics": aggregate(rows),
                "_metric_rows": rows,
                "determinism_mismatches": mismatches,
                "build_latency_ms": build_ms,
                "memory_count": len(token_index.nodes),
                "node_record_count": len(token_index.nodes),
                "cell_count": int(sum(len(values) for values in token_index.layered_inverted.values())),
                "edge_count": 0,
                "total_index_bytes": int(token_index.index_bytes) + int(faiss_built["index_bytes"]),
            }
        )
    if "hippo_rope_grid" in args.systems or "hippo_calibrated" in args.systems:
        started = time.perf_counter()
        meta = build_rope_delta_grid(embedded_base, backend, work_dir / f"record_{record_number:05d}" / "hippo_rope", args)
        build_ms = (time.perf_counter() - started) * 1000.0
        if "hippo_rope_grid" in args.systems:
            rows, mismatches = run_queries(qa_rows, lambda row: search_rope_delta_grid(row, backend, meta, args), args)
            systems.append(
                {
                    "name": "hippo_rope_grid",
                    "metrics": aggregate(rows),
                    "_metric_rows": rows,
                    "determinism_mismatches": mismatches,
                    "build_latency_ms": build_ms,
                    "memory_count": int(meta["memory_count"]),
                    "node_record_count": int(meta["node_record_count"]),
                    "cell_count": int(meta["cell_count"]),
                    "edge_count": int(meta["edge_count"]),
                    "total_index_bytes": index_size(meta),
                }
            )
        if "hippo_calibrated" in args.systems:
            if calibrator is None:
                raise ValueError("--calibrator-checkpoint is required for hippo_calibrated")
            rows, mismatches = run_queries(
                qa_rows,
                lambda row: calibrated_search(row, backend, meta, calibrator, id_to_candidate, args),
                args,
            )
            systems.append(
                {
                    "name": "hippo_calibrated",
                    "metrics": aggregate(rows),
                    "_metric_rows": rows,
                    "determinism_mismatches": mismatches,
                    "build_latency_ms": build_ms,
                    "memory_count": int(meta["memory_count"]),
                    "node_record_count": int(meta["node_record_count"]),
                    "cell_count": int(meta["cell_count"]),
                    "edge_count": int(meta["edge_count"]),
                    "total_index_bytes": index_size(meta),
                }
            )
    if "agent_memory_graph" in args.systems:
        index = build_agent_memory_graph(embedded_base, backend, work_dir / f"record_{record_number:05d}" / "agent_memory_graph", args)
        rows, mismatches = run_queries(qa_rows, lambda row: search_agent_memory_graph(row, backend, index, args), args)
        systems.append(
            {
                "name": "agent_memory_graph",
                "metrics": aggregate(rows),
                "_metric_rows": rows,
                "determinism_mismatches": mismatches,
                "build_latency_ms": float(index.build_latency_ms),
                "memory_count": len(index.nodes),
                "node_record_count": len(index.nodes),
                "cell_count": int(args.memory_graph_layers),
                "edge_count": sum(len(edges) for edges in index.truth_edges)
                + sum(len(edges) for layer in index.routing_edges for edges in layer),
                "total_index_bytes": int(index.index_bytes),
            }
        )

    return {
        "uid": str(record.get("uid") or ""),
        "source": str(record.get("source") or ""),
        "unit": mode,
        "question_count": len(qa_rows),
        "memory_count": len(embedded_base["candidates"]),
        "systems": systems,
    }


def system_rollup(records: list[dict[str, Any]], systems: list[str]) -> dict[str, dict[str, Any]]:
    out = {}
    for name in systems:
        metric_rows = []
        build_latency = []
        index_bytes = []
        mismatches = 0
        memories = []
        for record in records:
            for system in record.get("systems") or []:
                if system.get("name") == name:
                    metric_rows.extend(system.get("_metric_rows") or [])
                    build_latency.append(float(system.get("build_latency_ms") or 0.0))
                    index_bytes.append(float(system.get("total_index_bytes") or 0.0))
                    mismatches += int(system.get("determinism_mismatches") or 0)
                    memories.append(float(system.get("memory_count") or 0.0))
        rolled_metrics = aggregate(metric_rows) if metric_rows else {}
        out[name] = {
            "record_count": len(build_latency),
            "query_count": len(metric_rows),
            "metrics": rolled_metrics,
            "build_latency_ms_avg": sum(build_latency) / max(1, len(build_latency)),
            "index_mb_avg": (sum(index_bytes) / max(1, len(index_bytes))) / (1024 * 1024),
            "memory_count_avg": sum(memories) / max(1, len(memories)),
            "determinism_mismatches": mismatches,
        }
    return out


def public_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for record in records:
        item = dict(record)
        systems = []
        for system in item.get("systems") or []:
            public_system = dict(system)
            public_system.pop("_metric_rows", None)
            systems.append(public_system)
        item["systems"] = systems
        out.append(item)
    return out


def write_markdown(result: dict[str, Any], path: Path) -> None:
    lines = [
        "# MemoryCraft Retrieval Benchmark",
        "",
        f"- dataset: `{result['dataset']}`",
        f"- records: `{result['record_count']}`",
        f"- questions: `{result['question_count']}`",
        f"- unit: `{result['unit']}`",
        f"- systems: `{','.join(result['systems'])}`",
        f"- embedding_backend: `{result['embedding_backend']}`",
        f"- dim_count: `{result['dim_count']}`",
        f"- top_k: `{result['top_k']}`",
        f"- budget: `{result['budget']}`",
        "",
        "| system | records | avg memories | index MB | build ms | p95 ms | recall@k | precision@k | context recall | context precision | mrr | deterministic |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, rollup in result["rollup"].items():
        metrics = rollup["metrics"]
        deterministic = 1.0 if int(rollup["determinism_mismatches"]) == 0 else 0.0
        lines.append(
            f"| {name} | "
            f"{rollup['record_count']} | "
            f"{rollup['memory_count_avg']:.1f} | "
            f"{rollup['index_mb_avg']:.2f} | "
            f"{rollup['build_latency_ms_avg']:.2f} | "
            f"{metrics.get('latency_ms', {}).get('p95', 0.0):.2f} | "
            f"{metrics.get('recall_at_k', {}).get('avg', 0.0):.4f} | "
            f"{metrics.get('precision_at_k', {}).get('avg', 0.0):.4f} | "
            f"{metrics.get('context_recall', {}).get('avg', 0.0):.4f} | "
            f"{metrics.get('context_precision', {}).get('avg', 0.0):.4f} | "
            f"{metrics.get('mrr', {}).get('avg', 0.0):.4f} | "
            f"{deterministic:.4f} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    dataset_path = load_dataset_path(args)
    records = load_records(dataset_path, args.limit_records)
    backend = build_embedding_backend(args)
    calibrator = None
    if "hippo_calibrated" in args.systems or "hippo_calibrated_union" in args.systems:
        if not args.calibrator_checkpoint:
            raise ValueError("--calibrator-checkpoint is required when --systems includes hippo_calibrated")
        import torch

        from python.librarian.hippo_calibrator import load_calibrator

        device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
        calibrator = load_calibrator(args.calibrator_checkpoint, device=device)
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    evaluated = []
    for index, record in enumerate(records):
        result = run_record(record, index, backend, work_dir, args, calibrator)
        if result is not None:
            evaluated.append(result)
    question_count = sum(int(record["question_count"]) for record in evaluated)
    result = {
        "benchmark": "memorycraft_retrieval",
        "dataset": str(dataset_path),
        "hf_repo": args.hf_repo,
        "hf_file": args.hf_file,
        "record_count": len(evaluated),
        "question_count": question_count,
        "unit": args.unit,
        "systems": args.systems,
        "embedding_backend": backend.name,
        "calibrator_checkpoint": args.calibrator_checkpoint,
        "dim_count": args.dim_count,
        "layers": args.layers,
        "layer_schedule": args.layer_schedule,
        "cell_width": args.cell_width,
        "radius": args.radius,
        "max_cell_scan": args.max_cell_scan,
        "top_k": args.top_k,
        "budget": args.budget,
        "hybrid_candidate_fetch": int(args.hybrid_candidate_fetch),
        "hybrid_token_candidate_fetch": int(args.hybrid_token_candidate_fetch),
        "hybrid_union_vector_weight": float(args.hybrid_union_vector_weight),
        "hybrid_union_token_weight": float(args.hybrid_union_token_weight),
        "token_encoder_checkpoint": args.token_encoder_checkpoint,
        "repeat_searches": args.repeat_searches,
        "adversarial_negatives": int(args.adversarial_negatives),
        "rerank_relevance_weight": args.rerank_relevance_weight,
        "rerank_include_weight": args.rerank_include_weight,
        "rerank_base_weight": args.rerank_base_weight,
        "rerank_utility_weight": args.rerank_utility_weight,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "rollup": system_rollup(evaluated, args.systems),
        "records": public_records(evaluated),
    }
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.output_md:
        write_markdown(result, Path(args.output_md))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="")
    parser.add_argument("--hf-repo", default=DEFAULT_HF_REPO)
    parser.add_argument("--hf-file", default=DEFAULT_HF_FILE)
    parser.add_argument("--limit-records", type=int, default=5)
    parser.add_argument("--limit-questions", type=int, default=40)
    parser.add_argument("--include-abstention", action="store_true")
    parser.add_argument("--unit", choices=["auto", "turn", "session"], default="auto")
    parser.add_argument("--systems", type=parse_systems, default=["exact_vector", "faiss_flat", "faiss_hnsw", "hnswlib", "hippo_rope_grid"])
    parser.add_argument("--repeat-searches", type=int, default=1)
    parser.add_argument("--adversarial-negatives", type=int, default=0)
    parser.add_argument("--work-dir", default="artifacts/memorycraft_retrieval")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--budget", type=int, default=900)
    parser.add_argument("--layers", type=int, default=128)
    parser.add_argument("--layer-schedule", choices=["auto", "consecutive", "spread"], default="spread")
    parser.add_argument("--dim-count", type=int, default=1024)
    parser.add_argument("--cell-width", type=float, default=0.03125)
    parser.add_argument("--radius", type=int, default=0)
    parser.add_argument("--max-cell-scan", type=int, default=4096)
    parser.add_argument("--faiss-hnsw-m", type=int, default=32)
    parser.add_argument("--faiss-ef-construction", type=int, default=200)
    parser.add_argument("--faiss-ef-search", type=int, default=128)
    parser.add_argument("--hnswlib-m", type=int, default=32)
    parser.add_argument("--hnswlib-ef-construction", type=int, default=200)
    parser.add_argument("--hnswlib-ef-search", type=int, default=128)
    parser.add_argument("--min-layer-delta", type=float, default=0.0075)
    parser.add_argument("--min-query-layers", type=int, default=8)
    parser.add_argument("--max-query-layers", type=int, default=24)
    parser.add_argument("--min-node-layer-delta", type=float, default=0.0075)
    parser.add_argument("--min-node-layers", type=int, default=1)
    parser.add_argument("--max-node-layers", type=int, default=24)
    parser.add_argument("--edge-seed-count", type=int, default=48)
    parser.add_argument("--graph-depth", type=int, default=2)
    parser.add_argument("--final-fetch", type=int, default=96)
    parser.add_argument("--calibrator-checkpoint", default="")
    parser.add_argument("--calibrator-max-candidates", type=int, default=128)
    parser.add_argument("--rerank-relevance-weight", type=float, default=None)
    parser.add_argument("--rerank-include-weight", type=float, default=None)
    parser.add_argument("--rerank-base-weight", type=float, default=None)
    parser.add_argument("--rerank-utility-weight", type=float, default=None)
    parser.add_argument("--action-count", type=int, default=256)
    parser.add_argument("--query-token-count", type=int, default=40)
    parser.add_argument("--node-token-count", type=int, default=40)
    parser.add_argument("--projection-width", type=int, default=16)
    parser.add_argument("--bucket-width", type=float, default=0.055)
    parser.add_argument("--bucket-radius", type=int, default=2)
    parser.add_argument("--min-candidates", type=int, default=16)
    parser.add_argument("--max-candidates", type=int, default=512)
    parser.add_argument("--pre-filter-candidates", type=int, default=2048)
    parser.add_argument("--routing-layers", type=int, default=1)
    parser.add_argument("--promotion-probability", type=float, default=0.45)
    parser.add_argument("--promotion-bias", type=float, default=0.12)
    parser.add_argument("--routing-beam-width", type=int, default=32)
    parser.add_argument("--include-min-collision", type=float, default=1.0)
    parser.add_argument("--include-min-overlap", type=float, default=0.01)
    parser.add_argument("--token-encoder-checkpoint", default="")
    parser.add_argument("--token-encoder-device", default="")
    parser.add_argument("--hybrid-candidate-fetch", type=int, default=512)
    parser.add_argument("--hybrid-token-candidate-fetch", type=int, default=512)
    parser.add_argument("--hybrid-union-vector-weight", type=float, default=0.70)
    parser.add_argument("--hybrid-union-token-weight", type=float, default=0.30)
    parser.add_argument("--hybrid-source-weight", type=float, default=0.75)
    parser.add_argument("--hybrid-semantic-weight", type=float, default=0.15)
    parser.add_argument("--hybrid-field-weight", type=float, default=0.08)
    parser.add_argument("--hybrid-activation-weight", type=float, default=0.02)
    parser.add_argument("--memory-graph-layers", type=int, default=8)
    parser.add_argument("--memory-graph-route-degree", type=int, default=24)
    parser.add_argument("--memory-graph-projection-count", type=int, default=3)
    parser.add_argument("--memory-graph-projection-window", type=int, default=48)
    parser.add_argument("--memory-graph-promotion-threshold", type=int, default=72)
    parser.add_argument("--memory-graph-bias-promotion", action="store_true")
    parser.add_argument("--memory-graph-bridge-degree", type=int, default=6)
    parser.add_argument("--memory-graph-importance-threshold", type=float, default=0.68)
    parser.add_argument("--memory-graph-reciprocal-routes", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--memory-graph-ef", type=int, default=80)
    parser.add_argument("--memory-graph-beam", type=int, default=12)
    parser.add_argument("--memory-graph-objective-seeds", type=int, default=16)
    parser.add_argument("--memory-graph-truth-seeds", type=int, default=16)
    parser.add_argument("--memory-graph-truth-depth", type=int, default=2)
    parser.add_argument("--memory-graph-truth-fanout", type=int, default=6)
    parser.add_argument("--memory-graph-min-results", type=int, default=4)
    parser.add_argument("--memory-graph-cutoff-margin", type=float, default=0.28)
    parser.add_argument("--memory-graph-min-score", type=float, default=-0.05)
    parser.add_argument("--embedding-backend", choices=["hash", "hippo"], default="hash")
    parser.add_argument("--hippo-checkpoint", default="")
    parser.add_argument("--hippo-encoder-src", default="")
    parser.add_argument("--hippo-max-length", type=int, default=128)
    parser.add_argument("--hippo-batch-size", type=int, default=128)
    parser.add_argument("--device", default="")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-md", default="")
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
