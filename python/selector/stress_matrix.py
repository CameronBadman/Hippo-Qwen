from __future__ import annotations

import argparse
import json
import selectors
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))


METRICS = ("recall_at_k", "context_precision", "context_recall", "noise", "mrr")
PHASES = ("static", "evolved", "second_half_delta")


def parse_csv(value: str) -> list[str]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError("expected at least one comma-separated value")
    return items


def write_progress(args: argparse.Namespace, event: dict[str, Any]) -> None:
    progress_file = getattr(args, "progress_file", "")
    if not progress_file:
        return
    path = Path(progress_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **event}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def run_command(cmd: list[str], cwd: Path, timeout_seconds: int = 0) -> None:
    print("$ " + " ".join(cmd), flush=True)
    started = time.monotonic()
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    selector = selectors.DefaultSelector()
    selector.register(proc.stdout, selectors.EVENT_READ)
    while proc.poll() is None:
        for key, _mask in selector.select(timeout=1.0):
            line = key.fileobj.readline()
            if line:
                print(line, end="", flush=True)
        if timeout_seconds > 0 and time.monotonic() - started > timeout_seconds:
            proc.kill()
            proc.wait()
            selector.close()
            raise TimeoutError(f"command exceeded {timeout_seconds}s: {' '.join(cmd)}")
    for line in proc.stdout:
        print(line, end="", flush=True)
    selector.close()
    return_code = proc.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, cmd)


def scenario_dir_name(scenario: str, corruption: str) -> str:
    return f"{scenario}_state_{corruption}"


def run_regression(args: argparse.Namespace, repo: Path, scenario: str, corruption: str) -> Path:
    run_dir = Path(args.work_dir) / scenario_dir_name(scenario, corruption)
    summary_path = run_dir / "summary.json"
    if summary_path.exists() and not args.force:
        print(f"using existing regression summary {summary_path}", flush=True)
        write_progress(
            args,
            {
                "event": "condition_skipped_existing",
                "scenario": scenario,
                "eval_state_corruption": corruption,
                "summary_json": str(summary_path),
            },
        )
        return summary_path

    print(f"starting condition scenario={scenario} eval_state_corruption={corruption}", flush=True)
    write_progress(
        args,
        {
            "event": "condition_started",
            "scenario": scenario,
            "eval_state_corruption": corruption,
            "work_dir": str(run_dir),
        },
    )
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "python.selector.evolved_state_regression",
        "--work-dir",
        str(run_dir),
        "--scenario",
        scenario,
        "--eval-state-corruption",
        corruption,
        "--state-corruption-rate",
        str(args.state_corruption_rate),
        "--seeds",
        args.seeds,
        "--cases",
        str(args.cases),
        "--train-fraction",
        str(args.train_fraction),
        "--eval-limit",
        str(args.eval_limit),
        "--candidates",
        str(args.candidates),
        "--evolution-passes",
        str(args.evolution_passes),
        "--feedback-scorer",
        args.feedback_scorer,
        "--evolution-policies",
        args.evolution_policies,
        "--evolution-bias-scales",
        args.evolution_bias_scales,
        "--evolved-variant",
        args.evolved_variant,
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--d-model",
        str(args.d_model),
        "--layers",
        str(args.layers),
        "--heads",
        str(args.heads),
        "--feature-dim",
        str(args.feature_dim),
        "--rank-loss-weight",
        str(args.rank_loss_weight),
        "--reason-loss-weight",
        str(args.reason_loss_weight),
        "--auxiliary-loss-weight",
        str(args.auxiliary_loss_weight),
        "--top-k",
        str(args.top_k),
        "--budget",
        str(args.budget),
    ]
    if args.selector_post_rank_bias:
        cmd.append("--selector-post-rank-bias")
    if args.force:
        cmd.append("--force")
    if args.cpu:
        cmd.append("--cpu")
    started = time.monotonic()
    run_command(cmd, repo, args.condition_timeout_seconds)
    elapsed_seconds = time.monotonic() - started
    write_progress(
        args,
        {
            "event": "condition_finished",
            "scenario": scenario,
            "eval_state_corruption": corruption,
            "elapsed_seconds": round(elapsed_seconds, 3),
            "summary_json": str(summary_path),
        },
    )
    return summary_path


def compact_summary(summary_path: Path) -> dict[str, Any]:
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)

    raw_evolved = summary["aggregate"]["raw"]["evolved"]
    trained_evolved = summary["aggregate"]["evolved_state_trained"]["evolved"]
    delta_evolved = summary["aggregate"]["delta_evolved_state_minus_raw"]["evolved"]
    delta_second_half = summary["aggregate"]["delta_evolved_state_minus_raw"]["second_half_delta"]
    return {
        "scenario": summary["scenario"],
        "eval_state_corruption": summary["eval_state_corruption"],
        "state_corruption_rate": summary["state_corruption_rate"],
        "seeds": summary["seeds"],
        "cases": summary["cases"],
        "eval_limit": summary["eval_limit"],
        "summary_json": str(summary_path),
        "raw_evolved": {metric: raw_evolved[metric] for metric in METRICS},
        "evolved_state_trained": {metric: trained_evolved[metric] for metric in METRICS},
        "delta_evolved_state_minus_raw": {metric: delta_evolved[metric] for metric in METRICS},
        "second_half_delta_gain": {metric: delta_second_half[metric] for metric in METRICS},
    }


def score_run(row: dict[str, Any]) -> float:
    delta = row["delta_evolved_state_minus_raw"]
    second_half = row["second_half_delta_gain"]
    return (
        float(delta["context_precision"]["mean"])
        + float(delta["recall_at_k"]["mean"])
        - float(delta["noise"]["mean"])
        + float(second_half["context_precision"]["mean"])
        + float(second_half["recall_at_k"]["mean"])
        - float(second_half["noise"]["mean"])
    )


def write_markdown(report: dict[str, Any], output: Path) -> None:
    rows = report["runs"]
    lines = [
        "# Selector Stress Matrix",
        "",
        f"- scenarios: `{', '.join(report['scenarios'])}`",
        f"- eval state corruption: `{', '.join(report['eval_state_corruptions'])}`",
        f"- seeds: `{report['seeds']}`",
        f"- cases per seed: `{report['cases']}`",
        f"- eval limit per seed: `{report['eval_limit']}`",
        f"- evolved variant: `{report['evolved_variant']}`",
        "",
        "## Aggregate",
        "",
        "| scenario | corruption | raw evolved recall | trained evolved recall | precision delta | noise delta | second-half precision gain | second-half noise gain | score |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        delta = row["delta_evolved_state_minus_raw"]
        second_half = row["second_half_delta_gain"]
        lines.append(
            f"| {row['scenario']} | {row['eval_state_corruption']} | "
            f"{row['raw_evolved']['recall_at_k']['mean']:.4f} +/- {row['raw_evolved']['recall_at_k']['std']:.4f} | "
            f"{row['evolved_state_trained']['recall_at_k']['mean']:.4f} +/- {row['evolved_state_trained']['recall_at_k']['std']:.4f} | "
            f"{delta['context_precision']['mean']:+.4f} +/- {delta['context_precision']['std']:.4f} | "
            f"{delta['noise']['mean']:+.3f} +/- {delta['noise']['std']:.3f} | "
            f"{second_half['context_precision']['mean']:+.4f} +/- {second_half['context_precision']['std']:.4f} | "
            f"{second_half['noise']['mean']:+.3f} +/- {second_half['noise']['std']:.3f} | "
            f"{row['score']:+.4f} |"
        )

    best = report["best_by_score"]
    weakest = report["weakest_by_precision_delta"]
    lines.extend(
        [
            "",
            "## Readout",
            "",
            f"- best score: `{best['scenario']}` / `{best['eval_state_corruption']}` at `{best['score']:+.4f}`",
            f"- weakest precision delta: `{weakest['scenario']}` / `{weakest['eval_state_corruption']}` at `{weakest['delta_evolved_state_minus_raw']['context_precision']['mean']:+.4f}`",
            "",
            "Positive precision and recall deltas mean evolved-state training improved the selector over raw training. Negative noise delta means it selected fewer irrelevant memories.",
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", default="artifacts/librarian/stress_matrix")
    parser.add_argument("--scenarios", default="adversarial,preference_shift")
    parser.add_argument("--eval-state-corruptions", default="none,mild")
    parser.add_argument("--state-corruption-rate", type=float, default=0.35)
    parser.add_argument("--seeds", default="51,53,55")
    parser.add_argument("--cases", type=int, default=5000)
    parser.add_argument("--train-fraction", type=float, default=0.75)
    parser.add_argument("--eval-limit", type=int, default=1000)
    parser.add_argument("--candidates", type=int, default=32)
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
    parser.add_argument("--condition-timeout-seconds", type=int, default=0)
    parser.add_argument("--progress-file", default="")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    scenarios = parse_csv(args.scenarios)
    corruptions = parse_csv(args.eval_state_corruptions)
    repo = Path(__file__).resolve().parents[2]
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for scenario in scenarios:
        for corruption in corruptions:
            summary_path = run_regression(args, repo, scenario, corruption)
            row = compact_summary(summary_path)
            row["score"] = score_run(row)
            rows.append(row)

    best_by_score = max(rows, key=lambda row: row["score"])
    weakest_by_precision_delta = min(
        rows,
        key=lambda row: float(row["delta_evolved_state_minus_raw"]["context_precision"]["mean"]),
    )
    report = {
        "scenarios": scenarios,
        "eval_state_corruptions": corruptions,
        "state_corruption_rate": args.state_corruption_rate,
        "seeds": args.seeds,
        "cases": args.cases,
        "train_fraction": args.train_fraction,
        "eval_limit": args.eval_limit,
        "candidates": args.candidates,
        "evolved_variant": args.evolved_variant,
        "selector_post_rank_bias": args.selector_post_rank_bias,
        "runs": rows,
        "best_by_score": best_by_score,
        "weakest_by_precision_delta": weakest_by_precision_delta,
    }
    summary_path = work_dir / "stress_matrix_summary.json"
    markdown_path = work_dir / "stress_matrix_summary.md"
    summary_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_markdown(report, markdown_path)
    print(json.dumps({"summary_json": str(summary_path), "summary_md": str(markdown_path)}, indent=2))


if __name__ == "__main__":
    main()
