from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from python.benchmarks.hippocampus_retrieval import build_embedding_backend, ensure_backend_embeddings
from python.benchmarks.memorycraft_retrieval import (
    DEFAULT_HF_FILE,
    DEFAULT_HF_REPO,
    load_dataset_path,
    load_records,
    normalize_evidence,
    query_row,
    record_index_row,
    unit_mode,
)
from python.benchmarks.rope_delta_grid import build_rope_delta_grid, query_embedding_for, search_rope_delta_grid
from python.librarian.features import fnv1a64


def stable_pick(values: list[str], key: str) -> str:
    return values[fnv1a64(key) % len(values)]


def text_for_embedding(card: dict[str, Any]) -> str:
    return f"{card.get('text', '')} {card.get('summary', '')}".strip()


def relevant_candidates(id_to_candidate: dict[str, dict[str, Any]], relevant: set[str]) -> list[dict[str, Any]]:
    return [dict(id_to_candidate[candidate_id]) for candidate_id in sorted(relevant) if candidate_id in id_to_candidate]


def synthetic_hard_negative_cards(
    row: dict[str, Any],
    id_to_candidate: dict[str, dict[str, Any]],
    relevant: set[str],
    query_embedding: list[float],
    backend: Any,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    count = max(0, int(args.synthetic_hard_negatives))
    if count <= 0:
        return []
    task = row.get("retrieval_task") or {}
    query = str(task.get("query") or "")
    answer = str(task.get("answer") or "")
    qa_id = str(task.get("qa_id") or "qa")
    positives = relevant_candidates(id_to_candidate, relevant)
    positive_text = " ".join(str(candidate.get("text") or "") for candidate in positives)[:360]
    anchor = row.get("anchor") or {}
    project = str((anchor.get("metadata") or {}).get("project") or anchor.get("cluster") or "memorycraft")
    templates = [
        (
            "semantic_query_echo_negative",
            "{query} This note repeats the question terms but is a distractor. It was not the accepted evidence.",
            {"age_days": 4, "use_count": 18, "evidence_count": 5, "last_outcome": "ignored", "importance": 0.72},
        ),
        (
            "contradicted_positive_negative",
            "{query} Older conflicting note: {positive} This record is superseded and contradicts the accepted memory.",
            {"age_days": 240, "use_count": 1, "evidence_count": 0, "last_outcome": "ignored", "importance": 0.36},
        ),
        (
            "same_context_wrong_answer_negative",
            "{query} Plausible but wrong answer candidate: {answer}. This belongs to a different conversation turn.",
            {"age_days": 12, "use_count": 26, "evidence_count": 7, "last_outcome": "corrected", "importance": 0.68},
        ),
        (
            "stale_popular_negative",
            "{query} Previously popular memory with similar wording, now obsolete. Do not use it for the current answer.",
            {"age_days": 365, "use_count": 88, "evidence_count": 10, "last_outcome": "ignored", "importance": 0.58},
        ),
        (
            "near_duplicate_non_authoritative_negative",
            "{positive} Non-authoritative duplicate for {query}; keep the original evidence instead.",
            {"age_days": 2, "use_count": 3, "evidence_count": 0, "last_outcome": "", "importance": 0.46},
        ),
    ]
    out: list[dict[str, Any]] = []
    for index in range(count):
        key = f"{qa_id}:{query}:{index}:{','.join(sorted(relevant))}"
        role, template, state = templates[fnv1a64(key) % len(templates)]
        fallback_positive = stable_pick(
            [
                "a similar looking memory from the same project",
                "an old preference that no longer applies",
                "a nearby session note with overlapping terms",
            ],
            key,
        )
        text = template.format(
            query=query,
            answer=answer or "unknown",
            positive=positive_text or fallback_positive,
        ).strip()
        candidate = {
            "id": f"synthetic_hard_negative::{qa_id or 'qa'}::{index}",
            "text": text,
            "summary": "",
            "importance": state["importance"],
            "cluster": project,
            "metadata": {
                "project": project,
                "hard_negative_type": role,
            },
            "age_days": state["age_days"] + int(fnv1a64(f"{key}:age") % 11),
            "use_count": state["use_count"],
            "evidence_count": state["evidence_count"],
            "last_outcome": state["last_outcome"],
            "synthetic_role": role,
            "hard_negative": True,
            "label_weight": float(args.synthetic_hard_negative_weight),
            "base_rank": 1 + int(fnv1a64(f"{key}:rank") % max(1, min(16, args.max_candidates))),
            "base_score": 0.65 + 0.30 * ((fnv1a64(f"{key}:score") % 1000) / 1000.0),
            "base_score_gap": 0.0,
        }
        out.append(candidate)
    if out:
        vectors = backend.embed_many([text_for_embedding(candidate) for candidate in out])
        for candidate, vector in zip(out, vectors):
            candidate["embedding"] = vector
    return out


def keep_priority_candidates(candidates: list[dict[str, Any]], max_candidates: int, forced_ids: set[str]) -> list[dict[str, Any]]:
    if len(candidates) <= max_candidates:
        return candidates
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    for candidate in candidates:
        candidate_id = str(candidate.get("id") or "")
        if candidate_id in selected_ids:
            continue
        if len(selected) < max_candidates:
            selected.append(candidate)
            selected_ids.add(candidate_id)
    for candidate in candidates:
        candidate_id = str(candidate.get("id") or "")
        if candidate_id not in forced_ids or candidate_id in selected_ids:
            continue
        replace_index = next(
            (idx for idx in range(len(selected) - 1, -1, -1) if str(selected[idx].get("id") or "") not in forced_ids),
            None,
        )
        if replace_index is None:
            break
        selected_ids.discard(str(selected[replace_index].get("id") or ""))
        selected[replace_index] = candidate
        selected_ids.add(candidate_id)
    return selected


def payload_for_ranked(
    row: dict[str, Any],
    ranked: list[tuple[str, float, str]],
    id_to_candidate: dict[str, dict[str, Any]],
    relevant: set[str],
    query_embedding: list[float],
    backend: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    seen = set()
    candidates = []
    for rank, (candidate_id, score, _) in enumerate(ranked[: args.max_candidates], start=1):
        candidate = dict(id_to_candidate.get(candidate_id) or {})
        if not candidate:
            continue
        candidate["base_rank"] = rank
        candidate["base_score"] = float(score)
        candidates.append(candidate)
        seen.add(candidate_id)
    if args.inject_missing_relevant:
        for candidate_id in sorted(relevant - seen):
            candidate = dict(id_to_candidate.get(candidate_id) or {})
            if not candidate:
                continue
            candidate["base_rank"] = args.max_candidates + 1
            candidate["base_score"] = -1.0
            candidates.append(candidate)
            if len(candidates) >= args.max_candidates:
                break
    synthetic = synthetic_hard_negative_cards(row, id_to_candidate, relevant, query_embedding, backend, args)
    candidates.extend(synthetic)
    forced_ids = set(relevant) | {str(candidate.get("id") or "") for candidate in synthetic}
    if candidates:
        best_score = max(float(candidate.get("base_score") or 0.0) for candidate in candidates)
        for candidate in candidates:
            candidate["base_score_gap"] = best_score - float(candidate.get("base_score") or 0.0)
    task = row.get("retrieval_task") or {}
    return {
        "query": str(task.get("query") or ""),
        "answer": str(task.get("answer") or ""),
        "qa_id": str(task.get("qa_id") or ""),
        "question_type": str(task.get("question_type") or ""),
        "query_embedding": query_embedding,
        "budget": int(task.get("budget") or args.budget),
        "relevant_ids": sorted(relevant),
        "candidates": keep_priority_candidates(candidates, args.max_candidates, forced_ids),
    }


def build_rows(records: list[dict[str, Any]], backend: Any, args: argparse.Namespace) -> list[dict[str, Any]]:
    output_rows = []
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    for record_index, record in enumerate(records):
        mode = unit_mode(record, args.unit)
        base = record_index_row(record, mode)
        if not base.get("candidates"):
            continue
        first = query_row(base, {"question": base["anchor"]["text"], "qa_id": "index"}, set(), args.budget)
        embedded_base = ensure_backend_embeddings(first, backend)
        id_to_candidate = {str(candidate.get("id") or ""): dict(candidate) for candidate in embedded_base.get("candidates", [])}
        meta = build_rope_delta_grid(embedded_base, backend, work_dir / f"record_{record_index:05d}" / "hippo_rope", args)
        qa_count = 0
        for qa in record.get("qa") or []:
            if bool(qa.get("abstention")) and not args.include_abstention:
                continue
            relevant = normalize_evidence(record, qa, mode)
            if not relevant:
                continue
            row = query_row(embedded_base, qa, relevant, args.budget)
            row = ensure_backend_embeddings(row, backend)
            ranked, _, _ = search_rope_delta_grid(row, backend, meta, args)
            query_embedding = query_embedding_for(row, backend)
            output_rows.append(payload_for_ranked(row, ranked, id_to_candidate, relevant, query_embedding, backend, args))
            qa_count += 1
            if args.limit_questions > 0 and qa_count >= args.limit_questions:
                break
    return output_rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    dataset_path = load_dataset_path(args)
    records = load_records(dataset_path, args.limit_records)
    backend = build_embedding_backend(args)
    rows = build_rows(records, backend, args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n")
    result = {
        "dataset": str(dataset_path),
        "output": str(output),
        "rows": len(rows),
        "records": len(records),
        "synthetic_hard_negatives": int(args.synthetic_hard_negatives),
        "embedding_backend": backend.name,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    if args.summary_json:
        summary = Path(args.summary_json)
        summary.parent.mkdir(parents=True, exist_ok=True)
        summary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="")
    parser.add_argument("--hf-repo", default=DEFAULT_HF_REPO)
    parser.add_argument("--hf-file", default=DEFAULT_HF_FILE)
    parser.add_argument("--output", default="artifacts/hippo_calibrator/memorycraft_train.jsonl")
    parser.add_argument("--summary-json", default="")
    parser.add_argument("--limit-records", type=int, default=20)
    parser.add_argument("--limit-questions", type=int, default=20)
    parser.add_argument("--include-abstention", action="store_true")
    parser.add_argument("--unit", choices=["auto", "turn", "session"], default="auto")
    parser.add_argument("--inject-missing-relevant", action="store_true")
    parser.add_argument("--synthetic-hard-negatives", type=int, default=0)
    parser.add_argument("--synthetic-hard-negative-weight", type=float, default=3.0)
    parser.add_argument("--max-candidates", type=int, default=128)
    parser.add_argument("--budget", type=int, default=900)
    parser.add_argument("--work-dir", default="artifacts/hippo_calibrator/build")
    parser.add_argument("--layers", type=int, default=128)
    parser.add_argument("--layer-schedule", choices=["auto", "consecutive", "spread"], default="spread")
    parser.add_argument("--dim-count", type=int, default=1024)
    parser.add_argument("--cell-width", type=float, default=0.03125)
    parser.add_argument("--radius", type=int, default=0)
    parser.add_argument("--max-cell-scan", type=int, default=4096)
    parser.add_argument("--min-layer-delta", type=float, default=0.0075)
    parser.add_argument("--min-query-layers", type=int, default=8)
    parser.add_argument("--max-query-layers", type=int, default=24)
    parser.add_argument("--min-node-layer-delta", type=float, default=0.0075)
    parser.add_argument("--min-node-layers", type=int, default=1)
    parser.add_argument("--max-node-layers", type=int, default=24)
    parser.add_argument("--edge-seed-count", type=int, default=48)
    parser.add_argument("--graph-depth", type=int, default=2)
    parser.add_argument("--final-fetch", type=int, default=128)
    parser.add_argument("--embedding-backend", choices=["hash", "hippo"], default="hash")
    parser.add_argument("--hippo-checkpoint", default="")
    parser.add_argument("--hippo-encoder-src", default="")
    parser.add_argument("--hippo-max-length", type=int, default=128)
    parser.add_argument("--hippo-batch-size", type=int, default=128)
    parser.add_argument("--device", default="")
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
