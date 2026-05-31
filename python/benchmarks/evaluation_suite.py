from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from python.benchmarks.benchmark_librarian import run as run_benchmark


ABLATIONS = {
    "full": {"feature_dim": 16, "rank_loss_weight": 0.25},
    "no_state_features": {"feature_dim": 8, "rank_loss_weight": 0.25},
    "no_ranking_loss": {"feature_dim": 16, "rank_loss_weight": 0.0},
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
        ],
        repo,
    )


def train_ablation(args: argparse.Namespace, repo: Path, dataset: Path, name: str, config: dict[str, Any], output_dir: Path) -> Path:
    checkpoint = output_dir / f"{name}.pt"
    if checkpoint.exists() and not args.force:
        print(f"using existing checkpoint {checkpoint}", flush=True)
        return checkpoint
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "python.training.train_librarian",
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
        "--feature-dim",
        str(config["feature_dim"]),
        "--rank-loss-weight",
        str(config["rank_loss_weight"]),
        "--d-model",
        str(args.d_model),
        "--layers",
        str(args.layers),
        "--heads",
        str(args.heads),
        "--seed",
        str(args.seed),
    ]
    if args.cpu:
        cmd.append("--cpu")
    run_command(cmd, repo)
    return checkpoint


def benchmark_checkpoint(args: argparse.Namespace, dataset: Path, checkpoint: Path, name: str, output_dir: Path) -> dict[str, Any]:
    bench_args = argparse.Namespace(
        dataset=str(dataset),
        checkpoint=str(checkpoint),
        limit=args.eval_limit,
        top_k=args.top_k,
        budget=args.budget,
        output_json="",
        output_md="",
        cpu=args.cpu,
    )
    result = run_benchmark(bench_args)
    path = output_dir / f"{name}_benchmark.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def threshold_sweep(args: argparse.Namespace, repo: Path, dataset: Path, checkpoint: Path, name: str, output_dir: Path) -> dict[str, Any]:
    code = f"""
import json, pathlib, sys, torch
sys.path.insert(0, {str(repo)!r})
from torch.utils.data import DataLoader
from python.training.dataset import NeighborhoodDataset
from python.librarian.model import load_checkpoint

dataset_path = pathlib.Path({str(dataset)!r})
checkpoint_path = pathlib.Path({str(checkpoint)!r})
thresholds = {args.thresholds!r}
limit = {args.eval_limit!r}
device = torch.device('cuda' if torch.cuda.is_available() and not {args.cpu!r} else 'cpu')
model = load_checkpoint(checkpoint_path, device=device)
dataset = NeighborhoodDataset(dataset_path, model.config.max_candidates, model.config.feature_dim)
if limit and len(dataset) > limit:
    dataset.rows = dataset.rows[:limit]
loader = DataLoader(dataset, batch_size=512)
stats = {{str(t): {{'tp': 0, 'fp': 0, 'fn': 0, 'pred': 0, 'label': 0, 'total': 0}} for t in thresholds}}
with torch.no_grad():
    for batch in loader:
        batch = {{k: v.to(device) for k, v in batch.items()}}
        out = model(batch['anchor'], batch['candidates'], batch['pair_features'], batch['mask'])
        probs = torch.sigmoid(out['attach_logits'])
        labels = (batch['attach'] > 0.5) & batch['mask']
        for threshold in thresholds:
            pred = (probs >= threshold) & batch['mask']
            key = str(threshold)
            stats[key]['tp'] += int((pred & labels).sum().item())
            stats[key]['fp'] += int((pred & ~labels).sum().item())
            stats[key]['fn'] += int((~pred & labels & batch['mask']).sum().item())
            stats[key]['pred'] += int(pred.sum().item())
            stats[key]['label'] += int(labels.sum().item())
            stats[key]['total'] += int(batch['mask'].sum().item())
metrics = {{}}
for key, item in stats.items():
    precision = item['tp'] / max(1, item['tp'] + item['fp'])
    recall = item['tp'] / max(1, item['tp'] + item['fn'])
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    metrics[key] = {{
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'predicted_positive_rate': item['pred'] / max(1, item['total']),
        'label_positive_rate': item['label'] / max(1, item['total']),
    }}
print(json.dumps(metrics, indent=2))
"""
    proc = subprocess.run([sys.executable, "-c", code], cwd=repo, check=True, capture_output=True, text=True)
    result = json.loads(proc.stdout)
    path = output_dir / f"{name}_thresholds.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def write_summary(summary: dict[str, Any], output_dir: Path) -> None:
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# HippoGraph Librarian Evaluation Suite",
        "",
        f"- dataset cases: `{summary['cases']}`",
        f"- eval limit: `{summary['eval_limit']}`",
        f"- candidates: `{summary['candidates']}`",
        "",
        "## Retrieval Benchmark",
        "",
        "| ablation | recall@k | mrr | context precision | context recall | noise |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, result in summary["benchmarks"].items():
        metrics = result["metrics"]["transformer_graph"]
        lines.append(
            f"| {name} | {metrics['recall_at_k']:.4f} | {metrics['mrr']:.4f} | "
            f"{metrics['context_precision']:.4f} | {metrics['context_recall']:.4f} | {metrics['noise']:.2f} |"
        )
    lines.extend(["", "## Threshold Sweep", ""])
    for name, thresholds in summary["thresholds"].items():
        lines.append(f"### {name}")
        lines.append("")
        lines.append("| threshold | precision | recall | f1 | predicted positive rate |")
        lines.append("| ---: | ---: | ---: | ---: | ---: |")
        for threshold, metrics in thresholds.items():
            lines.append(
                f"| {threshold} | {metrics['precision']:.4f} | {metrics['recall']:.4f} | "
                f"{metrics['f1']:.4f} | {metrics['predicted_positive_rate']:.4f} |"
            )
        lines.append("")
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def parse_thresholds(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", default="artifacts/librarian/eval_suite")
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
    parser.add_argument("--thresholds", type=parse_thresholds, default=parse_thresholds("0.3,0.4,0.5,0.6,0.7"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[2]
    output_dir = Path(args.work_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = output_dir / "hard_cases.jsonl"

    generate_dataset(args, repo, dataset)
    benchmarks = {}
    thresholds = {}
    checkpoints = {}
    for name, config in ABLATIONS.items():
        checkpoint = train_ablation(args, repo, dataset, name, config, output_dir)
        checkpoints[name] = str(checkpoint)
        benchmarks[name] = benchmark_checkpoint(args, dataset, checkpoint, name, output_dir)
        thresholds[name] = threshold_sweep(args, repo, dataset, checkpoint, name, output_dir)

    summary = {
        "cases": args.cases,
        "eval_limit": args.eval_limit,
        "candidates": args.candidates,
        "dataset": str(dataset),
        "checkpoints": checkpoints,
        "benchmarks": benchmarks,
        "thresholds": thresholds,
    }
    write_summary(summary, output_dir)
    print(json.dumps({"summary_json": str(output_dir / "summary.json"), "summary_md": str(output_dir / "summary.md")}, indent=2))


if __name__ == "__main__":
    main()
