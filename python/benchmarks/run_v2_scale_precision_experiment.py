from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def run_logged(name: str, cmd: list[str], log_path: Path, cwd: Path) -> float:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    print(f"=== {name} ===", flush=True)
    print("+ " + " ".join(cmd), flush=True)
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("+ " + " ".join(cmd) + "\n")
        handle.flush()
        result = subprocess.run(cmd, cwd=str(cwd), stdout=handle, stderr=subprocess.STDOUT, text=True)
    elapsed = time.time() - started
    print(f"{name}: returncode={result.returncode} elapsed_s={elapsed:.2f} log={log_path}", flush=True)
    if result.returncode != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
        raise RuntimeError(f"{name} failed with return code {result.returncode}\n{tail}")
    return elapsed


def metric(system: dict[str, Any], name: str, field: str = "avg") -> float:
    return float(system.get(name, {}).get(field, 0.0))


def summarize_eval(path: Path, name: str, memory_count: int, checkpoint: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    v2 = data["systems"]["calibrated_v2"]
    top3 = data["systems"].get("calibrated_v2_pack_top3", {})
    top4 = data["systems"].get("calibrated_v2_pack_top4", {})
    return {
        "name": name,
        "memory_count": memory_count,
        "checkpoint": str(checkpoint),
        "latency_p50_ms": round(metric(v2, "latency_ms", "p50"), 2),
        "latency_p95_ms": round(metric(v2, "latency_ms", "p95"), 2),
        "recall8": round(metric(v2, "recall_at_k"), 4),
        "precision8": round(metric(v2, "precision_at_k"), 4),
        "hard_neg8": round(metric(v2, "hard_negative_top_k_rate"), 4),
        "top3_context_recall": round(metric(top3, "context_recall"), 4),
        "top3_context_precision": round(metric(top3, "context_precision"), 4),
        "top4_context_recall": round(metric(top4, "context_recall"), 4),
        "top4_context_precision": round(metric(top4, "context_precision"), 4),
    }


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * p))))
    return ordered[index]


def direct_latency(checkpoint: Path, train_jsonl: Path, max_candidates: int, max_edges: int, rows: int = 32) -> dict[str, float]:
    import torch

    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from python.librarian.relational_frame_calibrator import load_relational_calibrator, score_with_relational_calibrator

    payloads = []
    with train_jsonl.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                payloads.append(json.loads(line))
            if len(payloads) >= rows:
                break
    if not payloads:
        return {"direct_p50_ms": 0.0, "direct_p95_ms": 0.0}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_relational_calibrator(checkpoint, device=device)
    for row in payloads[: min(4, len(payloads))]:
        score_with_relational_calibrator(model, row, max_candidates=max_candidates, max_edges_per_candidate=max_edges)
    if device == "cuda":
        torch.cuda.synchronize()
    timings = []
    for row in payloads:
        started = time.perf_counter()
        score_with_relational_calibrator(model, row, max_candidates=max_candidates, max_edges_per_candidate=max_edges)
        if device == "cuda":
            torch.cuda.synchronize()
        timings.append((time.perf_counter() - started) * 1000.0)
    return {
        "direct_p50_ms": round(statistics.median(timings), 2),
        "direct_p95_ms": round(percentile(timings, 0.95), 2),
    }


def benchmark_base(args: argparse.Namespace, memory_count: int) -> list[str]:
    return [
        sys.executable,
        "python/benchmarks/session_memory_stress.py",
        "--memory-count",
        str(memory_count),
        "--queries",
        str(args.queries),
        "--embedding-backend",
        "hash",
        "--dim-count",
        str(args.embedding_dim),
        "--vector-fetch",
        str(args.vector_fetch),
        "--token-fetch",
        str(args.token_fetch),
        "--metadata-fetch",
        str(args.metadata_fetch),
        "--graph-fetch",
        str(args.graph_fetch),
        "--derived-metadata",
        "rules",
        "--metadata-source",
        "both",
        "--candidate-pool",
        str(args.candidate_pool),
        "--device",
        args.device,
        "--packing-top-n",
        "3",
        "4",
        "5",
        "8",
    ]


def train_cmd(args: argparse.Namespace, train_jsonl: Path, output: Path, max_candidates: int, max_edges: int) -> list[str]:
    return [
        sys.executable,
        "python/librarian/train_relational_frame_calibrator.py",
        "--dataset",
        str(train_jsonl),
        "--output",
        str(output),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--val-fraction",
        "0.2",
        "--max-candidates",
        str(max_candidates),
        "--embedding-dim",
        str(args.embedding_dim),
        "--node-feature-dim",
        "48",
        "--edge-feature-dim",
        "24",
        "--node-frame-dim",
        "128",
        "--small-edge-dim",
        "32",
        "--large-edge-dim",
        "96",
        "--d-model",
        "64",
        "--edge-layers",
        "1",
        "--candidate-layers",
        "0",
        "--heads",
        "4",
        "--dropout",
        "0.1",
        "--max-edges-per-candidate",
        str(max_edges),
        "--lr",
        "3e-4",
        "--weight-decay",
        "0.01",
        "--rank-loss-weight",
        "0.7",
        "--include-rank-loss-weight",
        "0.8",
        "--false-positive-loss-weight",
        str(args.false_positive_loss_weight),
        "--false-positive-margin",
        str(args.false_positive_margin),
        "--negative-weight",
        str(args.negative_weight),
        "--include-negative-weight",
        str(args.include_negative_weight),
        "--top-k",
        "8",
        "--selection-metric",
        "precision",
        "--seed",
        str(args.seed),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", default="/content/hippo_v2_scale")
    parser.add_argument("--repo", default=".")
    parser.add_argument("--memory-counts", type=int, nargs="+", default=[50000, 100000])
    parser.add_argument("--queries", type=int, default=48)
    parser.add_argument("--train-queries", type=int, default=32)
    parser.add_argument("--holdout-queries", type=int, default=16)
    parser.add_argument("--candidate-pool", type=int, default=512)
    parser.add_argument("--calibration-max-candidates", type=int, default=128)
    parser.add_argument("--vector-fetch", type=int, default=1024)
    parser.add_argument("--token-fetch", type=int, default=1024)
    parser.add_argument("--metadata-fetch", type=int, default=512)
    parser.add_argument("--graph-fetch", type=int, default=256)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=9501)
    parser.add_argument("--existing-checkpoint", default="")
    parser.add_argument("--false-positive-loss-weight", type=float, default=0.25)
    parser.add_argument("--false-positive-margin", type=float, default=0.25)
    parser.add_argument("--negative-weight", type=float, default=1.5)
    parser.add_argument("--include-negative-weight", type=float, default=2.0)
    parser.add_argument("--skip-existing-eval", action="store_true")
    parser.add_argument("--skip-direct-latency", action="store_true")
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    artifact_dir = Path(args.artifact_dir)
    log_dir = artifact_dir / "logs"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    summary_path = artifact_dir / "scale_precision_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    else:
        summary: list[dict[str, Any]] = []

    def append_summary(row: dict[str, Any]) -> None:
        summary[:] = [existing for existing in summary if existing.get("name") != row.get("name")]
        summary.append(row)
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        print("SUMMARY_ROW " + json.dumps(row, sort_keys=True), flush=True)
        print(f"wrote {summary_path}", flush=True)

    variants = [
        ("fast64_edge4_l1_precision", 64, 4),
        ("fast96_edge6_l1_precision", 96, 6),
    ]

    for memory_count in args.memory_counts:
        train_jsonl = artifact_dir / f"train_{memory_count}_q{args.train_queries}_cand{args.calibration_max_candidates}.jsonl"
        if not train_jsonl.exists():
            run_logged(
                f"build train rows {memory_count}",
                benchmark_base(args, memory_count)
                + [
                    "--query-start",
                    "0",
                    "--query-limit",
                    str(args.train_queries),
                    "--calibration-max-candidates",
                    str(args.calibration_max_candidates),
                    "--output-calibration-jsonl",
                    str(train_jsonl),
                    "--output-json",
                    str(artifact_dir / f"build_{memory_count}.json"),
                ],
                log_dir / f"build_{memory_count}.log",
                repo,
            )
        else:
            print(f"reuse train rows {memory_count}: {train_jsonl}", flush=True)

        if args.existing_checkpoint and not args.skip_existing_eval:
            checkpoint = Path(args.existing_checkpoint)
            output_json = artifact_dir / f"eval_{memory_count}_existing_fast64.json"
            if not output_json.exists():
                run_logged(
                    f"eval existing fast64 {memory_count}",
                    benchmark_base(args, memory_count)
                    + [
                        "--query-start",
                        str(args.train_queries),
                        "--query-limit",
                        str(args.holdout_queries),
                        "--calibrator-v2-checkpoint",
                        str(checkpoint),
                        "--calibrator-v2-max-candidates",
                        "64",
                        "--calibrator-v2-max-edges-per-candidate",
                        "4",
                        "--output-json",
                        str(output_json),
                        "--output-md",
                        str(artifact_dir / f"eval_{memory_count}_existing_fast64.md"),
                    ],
                    log_dir / f"eval_{memory_count}_existing_fast64.log",
                    repo,
                )
            else:
                print(f"reuse eval existing fast64 {memory_count}: {output_json}", flush=True)
            row = summarize_eval(output_json, f"existing_fast64_on_{memory_count}", memory_count, checkpoint)
            if not args.skip_direct_latency:
                row.update(direct_latency(checkpoint, train_jsonl, 64, 4))
            append_summary(row)

        for variant_name, max_candidates, max_edges in variants:
            checkpoint = artifact_dir / f"v2_{variant_name}_{memory_count}.pt"
            if not checkpoint.exists():
                run_logged(
                    f"train {variant_name} {memory_count}",
                    train_cmd(args, train_jsonl, checkpoint, max_candidates, max_edges),
                    log_dir / f"train_{variant_name}_{memory_count}.log",
                    repo,
                )
            else:
                print(f"reuse checkpoint {variant_name} {memory_count}: {checkpoint}", flush=True)
            output_json = artifact_dir / f"eval_{memory_count}_{variant_name}.json"
            if not output_json.exists():
                run_logged(
                    f"eval {variant_name} {memory_count}",
                    benchmark_base(args, memory_count)
                    + [
                        "--query-start",
                        str(args.train_queries),
                        "--query-limit",
                        str(args.holdout_queries),
                        "--calibrator-v2-checkpoint",
                        str(checkpoint),
                        "--calibrator-v2-max-candidates",
                        str(max_candidates),
                        "--calibrator-v2-max-edges-per-candidate",
                        str(max_edges),
                        "--output-json",
                        str(output_json),
                        "--output-md",
                        str(artifact_dir / f"eval_{memory_count}_{variant_name}.md"),
                    ],
                    log_dir / f"eval_{variant_name}_{memory_count}.log",
                    repo,
                )
            else:
                print(f"reuse eval {variant_name} {memory_count}: {output_json}", flush=True)
            row = summarize_eval(output_json, f"{variant_name}_on_{memory_count}", memory_count, checkpoint)
            if not args.skip_direct_latency:
                row.update(direct_latency(checkpoint, train_jsonl, max_candidates, max_edges))
            append_summary(row)


if __name__ == "__main__":
    main()
