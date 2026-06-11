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
    filter_decoy_templates,
    load_dataset_path,
    load_records,
    normalize_evidence,
    query_row,
    record_index_row,
    real_card_state,
    stable_candidate_pick,
    unit_mode,
)
from python.benchmarks.rope_delta_grid import build_rope_delta_grid, query_embedding_for, search_rope_delta_grid
from python.benchmarks.vector_db_compare import build_faiss, hybrid_union_token_search
from python.field_memory.token_field import build_token_field_index
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
    legacy_templates = [
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
        (
            "same_entity_wrong_time_negative",
            "{query} Same people and topic, but this was from a different time window and should not answer the current question.",
            {"age_days": 180, "use_count": 12, "evidence_count": 2, "last_outcome": "ignored", "importance": 0.62},
        ),
        (
            "evidence_adjacent_but_wrong_negative",
            "{positive} Adjacent conversation context for {query}, but this line is not the evidence-bearing memory.",
            {"age_days": 6, "use_count": 20, "evidence_count": 3, "last_outcome": "ignored", "importance": 0.66},
        ),
        (
            "stale_preference_negative",
            "{query} Old user preference with similar wording. It was replaced later and should be excluded from final context.",
            {"age_days": 420, "use_count": 44, "evidence_count": 5, "last_outcome": "corrected", "importance": 0.59},
        ),
        (
            "answer_shaped_wrong_fact_negative",
            "Answer-shaped but wrong memory for {query}: {answer}. This is a plausible false positive, not the accepted evidence.",
            {"age_days": 18, "use_count": 31, "evidence_count": 8, "last_outcome": "ignored", "importance": 0.74},
        ),
    ]
    forensic_templates = [
        ("semantic_query_echo_negative", "{query} Follow-up note from the same conversation thread.", {}),
        ("contradicted_positive_negative", "Earlier note near the same topic: {positive}", {}),
        ("same_context_wrong_answer_negative", "{query} The remembered response was: {answer}.", {}),
        ("stale_popular_negative", "{query} Older preference note from a previous session.", {}),
        ("near_duplicate_non_authoritative_negative", "{positive} Related note for {query}.", {}),
        ("same_entity_wrong_time_negative", "{query} Same topic appeared in a different weekly review.", {}),
        ("evidence_adjacent_but_wrong_negative", "{positive} Adjacent conversation context for {query}.", {}),
        ("stale_preference_negative", "{query} Previous session preference note with similar wording.", {}),
        ("answer_shaped_wrong_fact_negative", "Remembered answer for {query}: {answer}.", {}),
    ]
    negative_style = str(getattr(args, "synthetic_hard_negative_style", "forensic") or "forensic")
    templates = legacy_templates if negative_style == "legacy" else forensic_templates
    templates = filter_decoy_templates(
        templates,
        include_raw=str(getattr(args, "synthetic_hard_negative_families", "") or ""),
        exclude_raw=str(getattr(args, "synthetic_hard_negative_exclude_families", "") or ""),
        label="training",
    )
    donor_pool = [candidate for candidate_id, candidate in id_to_candidate.items() if candidate_id not in relevant] or list(id_to_candidate.values())
    out: list[dict[str, Any]] = []
    for index in range(count):
        key = f"{qa_id}:{query}:{index}:{','.join(sorted(relevant))}"
        role, template, state = templates[fnv1a64(key) % len(templates)]
        donor_state = real_card_state(stable_candidate_pick([dict(candidate) for candidate in donor_pool], key), project)
        if negative_style == "legacy":
            card_state = {
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
            }
        else:
            card_state = donor_state
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
            "summary": card_state["summary"],
            "importance": card_state["importance"],
            "cluster": card_state["cluster"],
            "metadata": card_state["metadata"],
            "age_days": card_state["age_days"],
            "use_count": card_state["use_count"],
            "evidence_count": card_state["evidence_count"],
            "last_outcome": card_state["last_outcome"],
            "synthetic_role": role,
            "hard_negative": True,
            "include_label": 0.0,
            "include_weight": float(args.synthetic_include_weight),
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
        if candidate_id in relevant:
            candidate["include_label"] = 1.0
            candidate["include_weight"] = 1.0
        else:
            candidate["include_label"] = 0.0
            candidate["include_weight"] = max(1.0, float(args.near_miss_include_weight) / float(max(1, min(rank, 32))))
        candidates.append(candidate)
        seen.add(candidate_id)
    if args.inject_missing_relevant:
        for candidate_id in sorted(relevant - seen):
            candidate = dict(id_to_candidate.get(candidate_id) or {})
            if not candidate:
                continue
            candidate["base_rank"] = args.max_candidates + 1
            candidate["base_score"] = -1.0
            candidate["include_label"] = 1.0
            candidate["include_weight"] = 1.0
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
        meta = None
        faiss_built = None
        token_index = None
        if args.candidate_source in {"rope", "union"}:
            meta = build_rope_delta_grid(embedded_base, backend, work_dir / f"record_{record_index:05d}" / "hippo_rope", args)
        if args.candidate_source in {"union"}:
            faiss_built = build_faiss(embedded_base, args, "hnsw")
            token_index = build_token_field_index(embedded_base, args)
        qa_count = 0
        for qa in record.get("qa") or []:
            if bool(qa.get("abstention")) and not args.include_abstention:
                continue
            relevant = normalize_evidence(record, qa, mode)
            if not relevant:
                continue
            row = query_row(embedded_base, qa, relevant, args.budget)
            row = ensure_backend_embeddings(row, backend)
            if args.candidate_source == "union":
                if faiss_built is None or token_index is None:
                    raise ValueError("union candidate source was not initialized")
                ranked, _, _ = hybrid_union_token_search(row, backend, faiss_built, token_index, args)
            else:
                if meta is None:
                    raise ValueError("rope candidate source was not initialized")
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
        "candidate_source": str(args.candidate_source),
        "synthetic_hard_negatives": int(args.synthetic_hard_negatives),
        "synthetic_hard_negative_style": str(args.synthetic_hard_negative_style),
        "synthetic_hard_negative_families": str(args.synthetic_hard_negative_families),
        "synthetic_hard_negative_exclude_families": str(args.synthetic_hard_negative_exclude_families),
        "synthetic_hard_negative_weight": float(args.synthetic_hard_negative_weight),
        "synthetic_include_weight": float(args.synthetic_include_weight),
        "near_miss_include_weight": float(args.near_miss_include_weight),
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
    parser.add_argument("--candidate-source", choices=["rope", "union"], default="rope")
    parser.add_argument("--inject-missing-relevant", action="store_true")
    parser.add_argument("--synthetic-hard-negatives", type=int, default=0)
    parser.add_argument("--synthetic-hard-negative-style", choices=["legacy", "forensic"], default="forensic")
    parser.add_argument("--synthetic-hard-negative-families", default="")
    parser.add_argument("--synthetic-hard-negative-exclude-families", default="")
    parser.add_argument("--synthetic-hard-negative-weight", type=float, default=3.0)
    parser.add_argument("--synthetic-include-weight", type=float, default=6.0)
    parser.add_argument("--near-miss-include-weight", type=float, default=8.0)
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
    parser.add_argument("--faiss-hnsw-m", type=int, default=32)
    parser.add_argument("--faiss-ef-construction", type=int, default=200)
    parser.add_argument("--faiss-ef-search", type=int, default=128)
    parser.add_argument("--action-count", type=int, default=256)
    parser.add_argument("--query-token-count", type=int, default=40)
    parser.add_argument("--node-token-count", type=int, default=40)
    parser.add_argument("--projection-width", type=int, default=16)
    parser.add_argument("--bucket-width", type=float, default=0.055)
    parser.add_argument("--bucket-radius", type=int, default=2)
    parser.add_argument("--min-candidates", type=int, default=16)
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
