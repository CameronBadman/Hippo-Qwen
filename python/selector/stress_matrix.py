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
DEFAULT_QUALITY_GATES = (
    {
        "name": "adversarial_recall_gain",
        "scenario": "adversarial",
        "corruption": "*",
        "metric_group": "delta_evolved_state_minus_raw",
        "metric": "recall_at_k",
        "op": ">=",
        "threshold": 0.01,
        "description": "Evolved-state training should recover more relevant memories in adversarial retrieval.",
    },
    {
        "name": "adversarial_precision_non_negative",
        "scenario": "adversarial",
        "corruption": "*",
        "metric_group": "delta_evolved_state_minus_raw",
        "metric": "context_precision",
        "op": ">=",
        "threshold": 0.0,
        "description": "Adversarial gains should not come from lower precision.",
    },
    {
        "name": "adversarial_noise_reduction",
        "scenario": "adversarial",
        "corruption": "*",
        "metric_group": "delta_evolved_state_minus_raw",
        "metric": "noise",
        "op": "<=",
        "threshold": 0.0,
        "description": "Adversarial gains should not add irrelevant memories.",
    },
    {
        "name": "adversarial_strong_noise_reduction",
        "scenario": "adversarial",
        "corruption": "strong",
        "metric_group": "delta_evolved_state_minus_raw",
        "metric": "noise",
        "op": "<=",
        "threshold": -0.2,
        "description": "Strong corruption should show a material noise reduction.",
    },
    {
        "name": "preference_clean_precision_floor",
        "scenario": "preference_shift",
        "corruption": "none",
        "metric_group": "delta_evolved_state_minus_raw",
        "metric": "context_precision",
        "op": ">=",
        "threshold": -0.02,
        "description": "Clean preference-shift should not lose meaningful precision when baseline is near ceiling.",
    },
    {
        "name": "preference_clean_noise_ceiling",
        "scenario": "preference_shift",
        "corruption": "none",
        "metric_group": "delta_evolved_state_minus_raw",
        "metric": "noise",
        "op": "<=",
        "threshold": 0.05,
        "description": "Clean preference-shift should not add much noise when baseline is already strong.",
    },
    {
        "name": "preference_mild_precision_floor",
        "scenario": "preference_shift",
        "corruption": "mild",
        "metric_group": "delta_evolved_state_minus_raw",
        "metric": "context_precision",
        "op": ">=",
        "threshold": -0.02,
        "description": "Mild preference-shift corruption should not lose meaningful precision.",
    },
    {
        "name": "preference_mild_noise_ceiling",
        "scenario": "preference_shift",
        "corruption": "mild",
        "metric_group": "delta_evolved_state_minus_raw",
        "metric": "noise",
        "op": "<=",
        "threshold": 0.05,
        "description": "Mild preference-shift corruption should not add much noise.",
    },
    {
        "name": "preference_strong_recall_gain",
        "scenario": "preference_shift",
        "corruption": "strong",
        "metric_group": "delta_evolved_state_minus_raw",
        "metric": "recall_at_k",
        "op": ">=",
        "threshold": 0.02,
        "description": "Strong preference-shift corruption should regain recall.",
    },
    {
        "name": "preference_strong_noise_reduction",
        "scenario": "preference_shift",
        "corruption": "strong",
        "metric_group": "delta_evolved_state_minus_raw",
        "metric": "noise",
        "op": "<=",
        "threshold": 0.0,
        "description": "Strong preference-shift corruption should reduce noise.",
    },
)


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
        "--gate-margin",
        str(args.gate_margin),
        "--low-confidence-score",
        str(args.low_confidence_score),
        "--low-spread",
        str(args.low_spread),
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


def gate_matches(row: dict[str, Any], gate: dict[str, Any]) -> bool:
    scenario = gate["scenario"]
    corruption = gate["corruption"]
    return (
        (scenario == "*" or scenario == row["scenario"])
        and (corruption == "*" or corruption == row["eval_state_corruption"])
    )


def gate_value(row: dict[str, Any], gate: dict[str, Any]) -> float:
    metric_group = gate["metric_group"]
    metric = gate["metric"]
    return float(row[metric_group][metric]["mean"])


def gate_passes(value: float, gate: dict[str, Any]) -> bool:
    threshold = float(gate["threshold"])
    op = gate["op"]
    if op == ">=":
        return value >= threshold
    if op == "<=":
        return value <= threshold
    raise ValueError(f"unsupported gate operator: {op}")


def evaluate_quality_gates(row: dict[str, Any], gates: tuple[dict[str, Any], ...]) -> dict[str, Any]:
    checks = []
    for gate in gates:
        if not gate_matches(row, gate):
            continue
        value = gate_value(row, gate)
        checks.append(
            {
                "name": gate["name"],
                "metric_group": gate["metric_group"],
                "metric": gate["metric"],
                "op": gate["op"],
                "threshold": gate["threshold"],
                "value": value,
                "passed": gate_passes(value, gate),
                "description": gate["description"],
            }
        )
    failed = [check for check in checks if not check["passed"]]
    return {
        "passed": not failed,
        "checks": checks,
        "failed": failed,
    }


def quality_gate_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total_checks = sum(len(row["quality_gates"]["checks"]) for row in rows)
    failed_checks = [
        {
            "scenario": row["scenario"],
            "eval_state_corruption": row["eval_state_corruption"],
            **check,
        }
        for row in rows
        for check in row["quality_gates"]["failed"]
    ]
    failed_runs = [
        {
            "scenario": row["scenario"],
            "eval_state_corruption": row["eval_state_corruption"],
            "failed_checks": [check["name"] for check in row["quality_gates"]["failed"]],
        }
        for row in rows
        if not row["quality_gates"]["passed"]
    ]
    return {
        "passed": not failed_checks,
        "total_checks": total_checks,
        "passed_checks": total_checks - len(failed_checks),
        "failed_checks": failed_checks,
        "failed_runs": failed_runs,
    }


def quality_status(row: dict[str, Any]) -> str:
    gates = row.get("quality_gates", {})
    if not gates.get("checks"):
        return "n/a"
    return "pass" if gates.get("passed") else "fail"


def write_markdown(report: dict[str, Any], output: Path) -> None:
    rows = report["runs"]
    quality = report["quality_gate_summary"]
    lines = [
        "# Selector Stress Matrix",
        "",
        f"- scenarios: `{', '.join(report['scenarios'])}`",
        f"- eval state corruption: `{', '.join(report['eval_state_corruptions'])}`",
        f"- seeds: `{report['seeds']}`",
        f"- cases per seed: `{report['cases']}`",
        f"- eval limit per seed: `{report['eval_limit']}`",
        f"- evolved variant: `{report['evolved_variant']}`",
        f"- quality gates: `{quality['passed_checks']}/{quality['total_checks']} passed`",
        "",
        "## Aggregate",
        "",
        "| scenario | corruption | gate | raw evolved recall | trained evolved recall | precision delta | noise delta | second-half precision gain | second-half noise gain | score |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        delta = row["delta_evolved_state_minus_raw"]
        second_half = row["second_half_delta_gain"]
        lines.append(
            f"| {row['scenario']} | {row['eval_state_corruption']} | {quality_status(row)} | "
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
            "## Quality Gates",
            "",
            f"- overall: `{'pass' if quality['passed'] else 'fail'}`",
            f"- checks: `{quality['passed_checks']}/{quality['total_checks']}` passed",
        ]
    )
    if quality["failed_checks"]:
        lines.extend(
            [
                "",
                "| scenario | corruption | gate | value | required |",
                "| --- | --- | --- | ---: | ---: |",
            ]
        )
        for failure in quality["failed_checks"]:
            lines.append(
                f"| {failure['scenario']} | {failure['eval_state_corruption']} | "
                f"{failure['name']} | {failure['value']:+.4f} | "
                f"{failure['op']} {failure['threshold']:+.4f} |"
            )
    lines.extend(
        [
            "",
            "## Readout",
            "",
            f"- best score: `{best['scenario']}` / `{best['eval_state_corruption']}` at `{best['score']:+.4f}`",
            f"- weakest precision delta: `{weakest['scenario']}` / `{weakest['eval_state_corruption']}` at `{weakest['delta_evolved_state_minus_raw']['context_precision']['mean']:+.4f}`",
            "",
            "Positive precision and recall deltas mean evolved-state training improved the selector over raw training. Negative noise delta means it selected fewer irrelevant memories.",
            "Quality gates turn those metrics into pass/fail criteria for promotion decisions.",
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
    parser.add_argument("--gate-margin", type=float, default=0.03)
    parser.add_argument("--low-confidence-score", type=float, default=0.72)
    parser.add_argument("--low-spread", type=float, default=0.18)
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
            row["quality_gates"] = evaluate_quality_gates(row, DEFAULT_QUALITY_GATES)
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
        "gate_margin": args.gate_margin,
        "low_confidence_score": args.low_confidence_score,
        "low_spread": args.low_spread,
        "selector_post_rank_bias": args.selector_post_rank_bias,
        "quality_gates": list(DEFAULT_QUALITY_GATES),
        "quality_gate_summary": quality_gate_summary(rows),
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
