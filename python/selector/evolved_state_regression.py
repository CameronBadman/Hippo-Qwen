from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))


TRACKED_METRICS = ("recall_at_k", "context_precision", "context_recall", "noise", "mrr")


def run_command(cmd: list[str], cwd: Path) -> None:
    print("$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def parse_seeds(value: str) -> list[int]:
    seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not seeds:
        raise ValueError("at least one seed is required")
    return seeds


def count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def split_dataset(source: Path, train_path: Path, eval_path: Path, train_fraction: float, force: bool) -> tuple[int, int]:
    if train_path.exists() and eval_path.exists() and not force:
        return count_lines(train_path), count_lines(eval_path)

    total = count_lines(source)
    split_at = max(1, min(total - 1, int(total * train_fraction)))
    train_count = 0
    eval_count = 0
    train_path.parent.mkdir(parents=True, exist_ok=True)
    with source.open("r", encoding="utf-8") as src, train_path.open("w", encoding="utf-8") as train, eval_path.open("w", encoding="utf-8") as eval_file:
        for line in src:
            if not line.strip():
                continue
            if train_count < split_at:
                train.write(line)
                train_count += 1
            else:
                eval_file.write(line)
                eval_count += 1
    return train_count, eval_count


def generate_dataset(args: argparse.Namespace, repo: Path, seed_dir: Path, seed: int) -> Path:
    base = seed_dir / "base.jsonl"
    if base.exists() and not args.force:
        print(f"using existing dataset {base}", flush=True)
        return base
    run_command(
        [
            sys.executable,
            "-m",
            "python.synthetic.generate",
            "--output",
            str(base),
            "--count",
            str(args.cases),
            "--candidates",
            str(args.candidates),
            "--scenario",
            args.scenario,
            "--seed",
            str(seed),
        ],
        repo,
    )
    return base


def evolve_training_data(args: argparse.Namespace, repo: Path, train_path: Path, evolved_path: Path, seed: int) -> None:
    if evolved_path.exists() and not args.force:
        print(f"using existing evolved dataset {evolved_path}", flush=True)
        return
    run_command(
        [
            sys.executable,
            "-m",
            "python.selector.evolve_dataset",
            "--input",
            str(train_path),
            "--output",
            str(evolved_path),
            "--passes",
            str(args.evolution_passes),
            "--feedback-scorer",
            args.feedback_scorer,
            "--budget",
            str(args.budget),
            "--shuffle-seed",
            str(seed),
        ],
        repo,
    )


def train_selector(args: argparse.Namespace, repo: Path, dataset: Path, checkpoint: Path, seed: int) -> None:
    if checkpoint.exists() and not args.force:
        print(f"using existing checkpoint {checkpoint}", flush=True)
        return
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
        str(args.feature_dim),
        "--rank-loss-weight",
        str(args.rank_loss_weight),
        "--reason-loss-weight",
        str(args.reason_loss_weight),
        "--auxiliary-loss-weight",
        str(args.auxiliary_loss_weight),
        "--d-model",
        str(args.d_model),
        "--layers",
        str(args.layers),
        "--heads",
        str(args.heads),
        "--seed",
        str(seed),
    ]
    if args.cpu:
        cmd.append("--cpu")
    run_command(cmd, repo)


def run_evolution_benchmark(args: argparse.Namespace, repo: Path, eval_path: Path, checkpoint: Path, output_path: Path) -> dict[str, Any]:
    if not output_path.exists() or args.force:
        cmd = [
            sys.executable,
            "-u",
            "-m",
            "python.selector.evolution_benchmark",
            "--dataset",
            str(eval_path),
            "--limit",
            str(args.eval_limit),
            "--checkpoint",
            str(checkpoint),
            "--evolution-policies",
            args.evolution_policies,
            "--evolution-bias-scales",
            args.evolution_bias_scales,
            "--top-k",
            str(args.top_k),
            "--budget",
            str(args.budget),
            "--output-json",
            str(output_path),
        ]
        if args.selector_post_rank_bias:
            cmd.append("--selector-post-rank-bias")
        if args.cpu:
            cmd.append("--cpu")
        run_command(cmd, repo)
    with output_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def extract_selector_metrics(result: dict[str, Any], evolved_variant: str) -> dict[str, Any]:
    selector = result["methods"]["context_selector"]
    static = selector["static"]["overall"]
    variant = selector["variants"][evolved_variant]
    evolved = variant["evolved"]["overall"]
    second_delta = variant["delta"]["second_half"]
    return {
        "static": {key: float(static.get(key, 0.0)) for key in TRACKED_METRICS},
        "evolved": {key: float(evolved.get(key, 0.0)) for key in TRACKED_METRICS},
        "second_half_delta": {key: float(second_delta.get(key, 0.0)) for key in TRACKED_METRICS},
        "applied_rate": float(variant.get("applied_rate", 0.0)),
        "bias_enabled": bool(variant.get("bias_enabled", False)),
    }


def metric_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(statistics.fmean(values)),
        "std": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
        "min": float(min(values)),
        "max": float(max(values)),
    }


def aggregate(runs: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"raw": {}, "evolved_state_trained": {}, "delta_evolved_state_minus_raw": {}}
    for training_mode, output_key in [("raw", "raw"), ("augmented", "evolved_state_trained")]:
        for phase in ["static", "evolved", "second_half_delta"]:
            summary[output_key][phase] = {}
            for metric in TRACKED_METRICS:
                values = [run["results"][training_mode][phase][metric] for run in runs]
                summary[output_key][phase][metric] = metric_summary(values)

    for phase in ["static", "evolved", "second_half_delta"]:
        summary["delta_evolved_state_minus_raw"][phase] = {}
        for metric in TRACKED_METRICS:
            values = [
                run["results"]["augmented"][phase][metric] - run["results"]["raw"][phase][metric]
                for run in runs
            ]
            summary["delta_evolved_state_minus_raw"][phase][metric] = metric_summary(values)
    return summary


def write_markdown(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# Evolved-State Selector Regression",
        "",
        f"- seeds: `{', '.join(str(seed) for seed in summary['seeds'])}`",
        f"- scenario: `{summary['scenario']}`",
        f"- cases: `{summary['cases']}`",
        f"- train fraction: `{summary['train_fraction']}`",
        f"- eval limit: `{summary['eval_limit']}`",
        f"- evolved variant: `{summary['evolved_variant']}`",
        f"- selector post-rank bias: `{summary['selector_post_rank_bias']}`",
        "",
        "## Aggregate",
        "",
        "| training | phase | recall@k | context precision | context recall | noise | mrr |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for training_key in ["raw", "evolved_state_trained"]:
        training = summary["aggregate"][training_key]
        for phase in ["static", "evolved", "second_half_delta"]:
            metrics = training[phase]
            label = "evolved-state" if training_key == "evolved_state_trained" else "raw"
            lines.append(
                f"| {label} | {phase} | "
                f"{metrics['recall_at_k']['mean']:.4f} +/- {metrics['recall_at_k']['std']:.4f} | "
                f"{metrics['context_precision']['mean']:.4f} +/- {metrics['context_precision']['std']:.4f} | "
                f"{metrics['context_recall']['mean']:.4f} +/- {metrics['context_recall']['std']:.4f} | "
                f"{metrics['noise']['mean']:.3f} +/- {metrics['noise']['std']:.3f} | "
                f"{metrics['mrr']['mean']:.4f} +/- {metrics['mrr']['std']:.4f} |"
            )

    lines.extend(
        [
            "",
            "## Evolved-State Minus Raw",
            "",
            "| phase | recall@k | context precision | context recall | noise | mrr |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for phase, metrics in summary["aggregate"]["delta_evolved_state_minus_raw"].items():
        lines.append(
            f"| {phase} | "
            f"{metrics['recall_at_k']['mean']:+.4f} +/- {metrics['recall_at_k']['std']:.4f} | "
            f"{metrics['context_precision']['mean']:+.4f} +/- {metrics['context_precision']['std']:.4f} | "
            f"{metrics['context_recall']['mean']:+.4f} +/- {metrics['context_recall']['std']:.4f} | "
            f"{metrics['noise']['mean']:+.3f} +/- {metrics['noise']['std']:.3f} | "
            f"{metrics['mrr']['mean']:+.4f} +/- {metrics['mrr']['std']:.4f} |"
        )

    lines.extend(["", "## Per Seed", "", "| seed | training | static recall | static precision | static noise | evolved recall | evolved precision | evolved noise |", "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for run in summary["runs"]:
        for training_mode, label in [("raw", "raw"), ("augmented", "evolved-state")]:
            result = run["results"][training_mode]
            lines.append(
                f"| {run['seed']} | {label} | "
                f"{result['static']['recall_at_k']:.4f} | {result['static']['context_precision']:.4f} | {result['static']['noise']:.3f} | "
                f"{result['evolved']['recall_at_k']:.4f} | {result['evolved']['context_precision']:.4f} | {result['evolved']['noise']:.3f} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_seed(args: argparse.Namespace, repo: Path, seed: int) -> dict[str, Any]:
    seed_dir = Path(args.work_dir) / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    base = generate_dataset(args, repo, seed_dir, seed)
    train_path = seed_dir / "train.jsonl"
    eval_path = seed_dir / "eval.jsonl"
    train_cases, eval_cases = split_dataset(base, train_path, eval_path, args.train_fraction, args.force)
    evolved_train = seed_dir / "train_evolved.jsonl"
    evolve_training_data(args, repo, train_path, evolved_train, seed)

    raw_checkpoint = seed_dir / "selector_raw.pt"
    augmented_checkpoint = seed_dir / "selector_evolved_state.pt"
    train_selector(args, repo, train_path, raw_checkpoint, seed)
    train_selector(args, repo, evolved_train, augmented_checkpoint, seed)

    raw_result = run_evolution_benchmark(args, repo, eval_path, raw_checkpoint, seed_dir / "raw_evolution.json")
    augmented_result = run_evolution_benchmark(args, repo, eval_path, augmented_checkpoint, seed_dir / "evolved_state_evolution.json")
    return {
        "seed": seed,
        "train_cases": train_cases,
        "eval_cases": eval_cases,
        "paths": {
            "base": str(base),
            "train": str(train_path),
            "eval": str(eval_path),
            "evolved_train": str(evolved_train),
            "raw_checkpoint": str(raw_checkpoint),
            "evolved_state_checkpoint": str(augmented_checkpoint),
        },
        "results": {
            "raw": extract_selector_metrics(raw_result, args.evolved_variant),
            "augmented": extract_selector_metrics(augmented_result, args.evolved_variant),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", default="artifacts/librarian/evolved_state_regression")
    parser.add_argument("--seeds", default="51,53,55")
    parser.add_argument("--cases", type=int, default=5000)
    parser.add_argument("--train-fraction", type=float, default=0.75)
    parser.add_argument("--eval-limit", type=int, default=1000)
    parser.add_argument("--candidates", type=int, default=32)
    parser.add_argument("--scenario", choices=["standard", "longitudinal", "adversarial"], default="adversarial")
    parser.add_argument("--evolution-passes", type=int, default=2)
    parser.add_argument("--feedback-scorer", choices=["heuristic_graph", "vector_only", "oracle"], default="heuristic_graph")
    parser.add_argument("--evolution-policies", default="off,always")
    parser.add_argument("--evolution-bias-scales", default="0")
    parser.add_argument("--evolved-variant", default="always@0")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--feature-dim", type=int, default=31)
    parser.add_argument("--rank-loss-weight", type=float, default=0.25)
    parser.add_argument("--reason-loss-weight", type=float, default=0.1)
    parser.add_argument("--auxiliary-loss-weight", type=float, default=0.05)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--budget", type=int, default=90)
    parser.add_argument("--selector-post-rank-bias", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[2]
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    seeds = parse_seeds(args.seeds)
    runs = [run_seed(args, repo, seed) for seed in seeds]
    summary = {
        "seeds": seeds,
        "cases": args.cases,
        "train_fraction": args.train_fraction,
        "eval_limit": args.eval_limit,
        "candidates": args.candidates,
        "scenario": args.scenario,
        "evolution_passes": args.evolution_passes,
        "feedback_scorer": args.feedback_scorer,
        "evolution_policies": args.evolution_policies,
        "evolution_bias_scales": args.evolution_bias_scales,
        "evolved_variant": args.evolved_variant,
        "selector_post_rank_bias": args.selector_post_rank_bias,
        "runs": runs,
        "aggregate": aggregate(runs),
    }
    summary_path = work_dir / "summary.json"
    markdown_path = work_dir / "summary.md"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_markdown(summary, markdown_path)
    print(json.dumps({"summary_json": str(summary_path), "summary_md": str(markdown_path)}, indent=2))


if __name__ == "__main__":
    main()
