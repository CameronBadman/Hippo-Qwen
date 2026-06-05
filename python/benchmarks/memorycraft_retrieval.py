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
from python.benchmarks.rope_delta_grid import build_rope_delta_grid, search_rope_delta_grid
from python.benchmarks.vector_db_compare import (
    build_faiss,
    build_hnswlib,
    exact_vector_search,
    faiss_search,
    hnswlib_search,
    parse_systems,
)
from python.librarian.features import activation_mask_for_text, fnv1a64, tokens


DEFAULT_HF_REPO = "daven3/MemoryCraft"
DEFAULT_HF_FILE = "selected/sample.jsonl"


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


def run_record(record: dict[str, Any], record_number: int, backend: Any, work_dir: Path, args: argparse.Namespace) -> dict[str, Any] | None:
    mode = unit_mode(record, args.unit)
    base = record_index_row(record, mode)
    if len(base.get("candidates") or []) < 1:
        return None
    qa_rows = []
    qa_count = 0
    for qa in record.get("qa") or []:
        if bool(qa.get("abstention")) and not args.include_abstention:
            continue
        relevant = normalize_evidence(record, qa, mode)
        if not relevant:
            continue
        qa_rows.append(query_row(base, qa, relevant, args.budget))
        qa_count += 1
        if args.limit_questions > 0 and qa_count >= args.limit_questions:
            break
    if not qa_rows:
        return None

    embedded_base = ensure_backend_embeddings(qa_rows[0], backend)
    for row in qa_rows:
        row["anchor"] = embedded_base["anchor"]
        row["candidates"] = embedded_base["candidates"]

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
    if "hippo_rope_grid" in args.systems:
        started = time.perf_counter()
        meta = build_rope_delta_grid(embedded_base, backend, work_dir / f"record_{record_number:05d}" / "hippo_rope", args)
        build_ms = (time.perf_counter() - started) * 1000.0
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
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    evaluated = []
    for index, record in enumerate(records):
        result = run_record(record, index, backend, work_dir, args)
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
        "dim_count": args.dim_count,
        "layers": args.layers,
        "layer_schedule": args.layer_schedule,
        "cell_width": args.cell_width,
        "radius": args.radius,
        "max_cell_scan": args.max_cell_scan,
        "top_k": args.top_k,
        "budget": args.budget,
        "repeat_searches": args.repeat_searches,
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
