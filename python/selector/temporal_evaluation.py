from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from python.selector.ablation_suite import (
    SELECTOR_ABLATIONS,
    benchmark_baselines,
    benchmark_selector,
    format_metric_row,
    generate_dataset,
    train_selector,
)
from python.selector.error_analysis import run as run_error_analysis


def run_command(cmd: list[str], cwd: Path) -> None:
    print("$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def split_dataset(source: Path, train_path: Path, eval_path: Path, train_fraction: float, force: bool) -> tuple[int, int]:
    if train_path.exists() and eval_path.exists() and not force:
        return count_lines(train_path), count_lines(eval_path)

    total = count_lines(source)
    split_at = max(1, min(total - 1, int(total * train_fraction)))
    train_count = 0
    eval_count = 0
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


def count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def write_summary(summary: dict[str, Any], output_dir: Path) -> None:
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Temporal Selector Evaluation",
        "",
        f"- scenario: `{summary['scenario']}`",
        f"- generated cases: `{summary['cases']}`",
        f"- train cases: `{summary['train_cases']}`",
        f"- eval cases: `{summary['eval_cases']}`",
        f"- eval limit: `{summary['eval_limit']}`",
        f"- candidates: `{summary['candidates']}`",
        f"- top_k: `{summary['top_k']}`",
        f"- budget: `{summary['budget']}`",
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
        lines.append(format_metric_row(name, result["metrics"]["context_selector"]))

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

    lines.extend(["", "## Role Exposure", "", "| ablation | role | relevant rate | top-k rate | budget rate | total |", "| --- | --- | ---: | ---: | ---: | ---: |"])
    for name, result in summary["selectors"].items():
        roles = result.get("role_metrics") or {}
        for role, metrics in sorted(roles.items(), key=lambda item: (-item[1].get("top_k_rate", 0.0), item[0])):
            lines.append(
                f"| {name} | {role} | {metrics.get('relevant_rate', 0.0):.4f} | "
                f"{metrics.get('top_k_rate', 0.0):.4f} | {metrics.get('budget_rate', 0.0):.4f} | {metrics.get('total', 0)} |"
            )

    lines.extend(["", "## Error Highlights", ""])
    for name, result in summary["error_analysis"].items():
        lines.append(f"### {name}")
        lines.append("")
        lines.append("| false context role | count |")
        lines.append("| --- | ---: |")
        for role, count in sorted(result["false_context_roles"].items(), key=lambda item: (-item[1], item[0]))[:8]:
            lines.append(f"| {role} | {count} |")
        lines.append("")
        lines.append("| missed relevant role | count |")
        lines.append("| --- | ---: |")
        for role, count in sorted(result["missed_context_roles"].items(), key=lambda item: (-item[1], item[0]))[:8]:
            lines.append(f"| {role} | {count} |")
        lines.append("")

    output_dir.joinpath("summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", default="artifacts/librarian/temporal_eval")
    parser.add_argument("--cases", type=int, default=8000)
    parser.add_argument("--train-fraction", type=float, default=0.75)
    parser.add_argument("--eval-limit", type=int, default=1000)
    parser.add_argument("--candidates", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--budget", type=int, default=90)
    parser.add_argument("--scenario", choices=["standard", "longitudinal", "adversarial"], default="longitudinal")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--error-cases", type=int, default=3)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[2]
    output_dir = Path(args.work_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = output_dir / "all_cases.jsonl"
    train_path = output_dir / "train_cases.jsonl"
    eval_path = output_dir / "eval_cases.jsonl"

    generate_dataset(args, repo, dataset)
    train_cases, eval_cases = split_dataset(dataset, train_path, eval_path, args.train_fraction, args.force)
    baselines = benchmark_baselines(args, eval_path, output_dir)

    checkpoints = {}
    selectors = {}
    error_analysis = {}
    for name, config in SELECTOR_ABLATIONS.items():
        checkpoint = train_selector(args, repo, train_path, name, config, output_dir)
        checkpoints[name] = str(checkpoint)
        selectors[name] = benchmark_selector(args, eval_path, checkpoint, name, output_dir)
        error_args = argparse.Namespace(
            dataset=str(eval_path),
            checkpoint=str(checkpoint),
            limit=args.eval_limit,
            top_k=args.top_k,
            budget=args.budget,
            worst_cases=args.error_cases,
            case_items=8,
            output_json="",
            output_md="",
            cpu=args.cpu,
        )
        analysis = run_error_analysis(error_args)
        error_analysis[name] = {
            "average_metrics": analysis["average_metrics"],
            "false_context_roles": analysis["false_context_roles"],
            "missed_context_roles": analysis["missed_context_roles"],
            "worst_cases": analysis["worst_cases"],
        }
        (output_dir / f"{name}_errors.json").write_text(json.dumps(analysis, indent=2) + "\n", encoding="utf-8")

    summary = {
        "cases": args.cases,
        "train_cases": train_cases,
        "eval_cases": eval_cases,
        "train_fraction": args.train_fraction,
        "eval_limit": args.eval_limit,
        "candidates": args.candidates,
        "top_k": args.top_k,
        "budget": args.budget,
        "scenario": args.scenario,
        "dataset": str(dataset),
        "train_dataset": str(train_path),
        "eval_dataset": str(eval_path),
        "checkpoints": checkpoints,
        "baselines": baselines,
        "selectors": selectors,
        "error_analysis": error_analysis,
    }
    write_summary(summary, output_dir)
    print(json.dumps({"summary_json": str(output_dir / "summary.json"), "summary_md": str(output_dir / "summary.md")}, indent=2))


if __name__ == "__main__":
    main()
