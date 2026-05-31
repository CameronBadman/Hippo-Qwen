from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from python.benchmarks.benchmark_librarian import run as run_librarian_benchmark
from python.selector.benchmark_selector import run as run_selector_benchmark


SELECTOR_ABLATIONS: dict[str, dict[str, Any]] = {
    "query_only_no_state": {
        "feature_dim": 8,
        "rank_loss_weight": 0.25,
        "reason_loss_weight": 0.1,
        "auxiliary_loss_weight": 0.05,
        "query_only": True,
    },
    "multi_seed_no_state": {
        "feature_dim": 21,
        "rank_loss_weight": 0.25,
        "reason_loss_weight": 0.1,
        "auxiliary_loss_weight": 0.05,
        "query_only": False,
    },
    "multi_seed_full": {
        "feature_dim": 31,
        "rank_loss_weight": 0.25,
        "reason_loss_weight": 0.1,
        "auxiliary_loss_weight": 0.05,
        "query_only": False,
    },
    "multi_seed_full_no_explainers": {
        "feature_dim": 31,
        "rank_loss_weight": 0.25,
        "reason_loss_weight": 0.0,
        "auxiliary_loss_weight": 0.0,
        "query_only": False,
    },
    "multi_seed_no_rank": {
        "feature_dim": 31,
        "rank_loss_weight": 0.0,
        "reason_loss_weight": 0.1,
        "auxiliary_loss_weight": 0.05,
        "query_only": False,
    },
}


def run_command(cmd: list[str], cwd: Path) -> None:
    print("$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def generate_dataset(args: argparse.Namespace, repo: Path, dataset: Path) -> None:
    if dataset.exists() and not args.force:
        print(f"using existing dataset {dataset}", flush=True)
        return
    run_command(
        [
            sys.executable,
            "python/synthetic/generate.py",
            "--output",
            str(dataset),
            "--count",
            str(args.cases),
            "--candidates",
            str(args.candidates),
            "--seed",
            str(args.seed),
            "--scenario",
            args.scenario,
        ],
        repo,
    )


def train_selector(args: argparse.Namespace, repo: Path, dataset: Path, name: str, config: dict[str, Any], output_dir: Path) -> Path:
    checkpoint = output_dir / f"{name}.pt"
    if checkpoint.exists() and not args.force:
        print(f"using existing checkpoint {checkpoint}", flush=True)
        return checkpoint

    cmd = [
        sys.executable,
        "-u",
        "-m",
        "python.selector.train_selector",
        "--dataset",
        str(dataset),
        "--output",
        str(checkpoint),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--max-candidates",
        str(args.candidates),
        "--budget",
        str(args.budget),
        "--top-k",
        str(args.top_k),
        "--feature-dim",
        str(config["feature_dim"]),
        "--rank-loss-weight",
        str(config["rank_loss_weight"]),
        "--reason-loss-weight",
        str(config["reason_loss_weight"]),
        "--auxiliary-loss-weight",
        str(config["auxiliary_loss_weight"]),
        "--d-model",
        str(args.d_model),
        "--layers",
        str(args.layers),
        "--heads",
        str(args.heads),
        "--seed",
        str(args.seed),
    ]
    if config["query_only"]:
        cmd.append("--query-only")
    if args.cpu:
        cmd.append("--cpu")
    run_command(cmd, repo)
    return checkpoint


def benchmark_selector(args: argparse.Namespace, dataset: Path, checkpoint: Path, name: str, output_dir: Path) -> dict[str, Any]:
    bench_args = argparse.Namespace(
        dataset=str(dataset),
        checkpoint=str(checkpoint),
        limit=args.eval_limit,
        top_k=args.top_k,
        budget=args.budget,
        auxiliary_threshold=0.5,
        use_checkpoint_auxiliary_thresholds=True,
        tune_auxiliary_thresholds=True,
        output_json="",
        cpu=args.cpu,
    )
    result = run_selector_benchmark(bench_args)
    path = output_dir / f"{name}_benchmark.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def benchmark_baselines(args: argparse.Namespace, dataset: Path, output_dir: Path) -> dict[str, Any]:
    bench_args = argparse.Namespace(
        dataset=str(dataset),
        checkpoint="",
        limit=args.eval_limit,
        top_k=args.top_k,
        budget=args.budget,
        output_json="",
        output_md="",
        cpu=args.cpu,
    )
    result = run_librarian_benchmark(bench_args)
    path = output_dir / "retrieval_baselines.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def write_summary(summary: dict[str, Any], output_dir: Path) -> None:
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Context Selector Ablation Suite",
        "",
        f"- dataset cases: `{summary['cases']}`",
        f"- eval limit: `{summary['eval_limit']}`",
        f"- candidates: `{summary['candidates']}`",
        f"- top_k: `{summary['top_k']}`",
        f"- budget: `{summary['budget']}`",
        f"- scenario: `{summary['scenario']}`",
        "",
        "## Retrieval Baselines",
        "",
        "| method | recall@k | mrr | context precision | context recall | noise |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, metrics in summary["baselines"]["metrics"].items():
        lines.append(format_metric_row(name, metrics))

    lines.extend(
        [
            "",
            "## Selector Ablations",
            "",
            "| ablation | recall@k | mrr | context precision | context recall | noise |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for name, result in summary["selectors"].items():
        metrics = result["metrics"]["context_selector"]
        lines.append(format_metric_row(name, metrics))
    lines.extend(["", "## Reason Head", "", "| ablation | reason accuracy | macro recall | labelled candidates |", "| --- | ---: | ---: | ---: |"])
    for name, result in summary["selectors"].items():
        reason = result.get("reason_metrics") or {}
        lines.append(f"| {name} | {reason.get('accuracy', 0.0):.4f} | {reason.get('macro_recall', 0.0):.4f} | {reason.get('total', 0)} |")
    lines.extend(["", "## Auxiliary Multi-Label Head", "", "| ablation | bit accuracy | macro f1 | tuned macro f1 |", "| --- | ---: | ---: | ---: |"])
    for name, result in summary["selectors"].items():
        auxiliary = result.get("auxiliary_metrics") or {}
        tuned = auxiliary.get("tuned") or {}
        lines.append(
            f"| {name} | {auxiliary.get('bit_accuracy', 0.0):.4f} | "
            f"{auxiliary.get('macro_f1', 0.0):.4f} | {tuned.get('macro_f1', 0.0):.4f} |"
        )
    output_dir.joinpath("summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_metric_row(name: str, metrics: dict[str, float]) -> str:
    return (
        f"| {name} | {metrics.get('recall_at_k', 0.0):.4f} | {metrics.get('mrr', 0.0):.4f} | "
        f"{metrics.get('context_precision', 0.0):.4f} | {metrics.get('context_recall', 0.0):.4f} | "
        f"{metrics.get('noise', 0.0):.2f} |"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", default="artifacts/librarian/selector_ablation")
    parser.add_argument("--cases", type=int, default=5000)
    parser.add_argument("--eval-limit", type=int, default=1000)
    parser.add_argument("--candidates", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--budget", type=int, default=90)
    parser.add_argument("--scenario", choices=["standard", "longitudinal"], default="longitudinal")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[2]
    output_dir = Path(args.work_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = output_dir / "hard_cases.jsonl"

    generate_dataset(args, repo, dataset)
    baselines = benchmark_baselines(args, dataset, output_dir)
    checkpoints = {}
    selectors = {}
    for name, config in SELECTOR_ABLATIONS.items():
        checkpoint = train_selector(args, repo, dataset, name, config, output_dir)
        checkpoints[name] = str(checkpoint)
        selectors[name] = benchmark_selector(args, dataset, checkpoint, name, output_dir)

    summary = {
        "cases": args.cases,
        "eval_limit": args.eval_limit,
        "candidates": args.candidates,
        "top_k": args.top_k,
        "budget": args.budget,
        "scenario": args.scenario,
        "dataset": str(dataset),
        "checkpoints": checkpoints,
        "baselines": baselines,
        "selectors": selectors,
    }
    write_summary(summary, output_dir)
    print(json.dumps({"summary_json": str(output_dir / "summary.json"), "summary_md": str(output_dir / "summary.md")}, indent=2))


if __name__ == "__main__":
    main()
