"""Paper-style metrics for trajectory-memory experiments."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .schema import TrajectoryStep, read_jsonl


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def summarize_trajectories(trajectory_path: str | Path) -> dict[str, Any]:
    steps = read_jsonl(trajectory_path)
    by_trajectory: dict[str, list[TrajectoryStep]] = defaultdict(list)
    for step in steps:
        by_trajectory[step.trajectory_id].append(step)
    trajectories = [sorted(items, key=lambda step: step.step) for items in by_trajectory.values()]

    terminal_rewards = []
    total_rewards = []
    success_flags = []
    validity_rates = []
    recovery_flags = []
    repeated_failure_rates = []
    path_efficiencies = []
    final_qed_improvements = []
    final_logp_improvements = []

    for trajectory in trajectories:
        if not trajectory:
            continue
        rewards = [float(step.reward) for step in trajectory]
        valid_flags = [1.0 if step.validity else 0.0 for step in trajectory]
        terminal_rewards.append(rewards[-1])
        total_rewards.append(sum(rewards[1:]))
        success_flags.append(1.0 if any(reward > 0.0 for reward in rewards[1:]) else 0.0)
        validity_rates.append(_mean(valid_flags))

        recovered = False
        repeated_failures = 0
        failure_pairs = 0
        first_success_step = None
        for index in range(1, len(rewards)):
            if rewards[index] > 0.0 and first_success_step is None:
                first_success_step = index
            if index >= 2 and rewards[index - 1] < 0.0:
                failure_pairs += 1
                if rewards[index] > 0.0:
                    recovered = True
                if rewards[index] < 0.0:
                    repeated_failures += 1
        recovery_flags.append(1.0 if recovered else 0.0)
        repeated_failure_rates.append(float(repeated_failures / failure_pairs) if failure_pairs else 0.0)
        path_efficiencies.append(0.0 if first_success_step is None else float(1.0 / first_success_step))

        first_props = trajectory[0].properties
        last_props = trajectory[-1].properties
        final_qed_improvements.append(float(last_props.get("qed", 0.0) - first_props.get("qed", 0.0)))
        final_logp_improvements.append(float(last_props.get("logp", 0.0) - first_props.get("logp", 0.0)))

    return {
        "trajectory_path": str(trajectory_path),
        "trajectory_count": len(trajectories),
        "step_count": len(steps),
        "success_rate": _mean(success_flags),
        "validity": _mean(validity_rates),
        "final_reward": _mean(terminal_rewards),
        "total_reward": _mean(total_rewards),
        "final_qed_improvement": _mean(final_qed_improvements),
        "final_logp_improvement": _mean(final_logp_improvements),
        "recovery_after_bad_edit": _mean(recovery_flags),
        "repeated_failure_rate": _mean(repeated_failure_rates),
        "path_efficiency": _mean(path_efficiencies),
    }


def summarize_runs(run_glob: str | Path) -> list[dict[str, Any]]:
    paths = sorted(Path().glob(str(run_glob))) if not Path(str(run_glob)).is_absolute() else sorted(Path("/").glob(str(run_glob).lstrip("/")))
    rows: list[dict[str, Any]] = []
    for path in paths:
        metrics_path = path / "metrics.json"
        history_path = path / "train_history.json"
        config_path = path / "run_config.json"
        if not metrics_path.exists():
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        history = json.loads(history_path.read_text(encoding="utf-8")) if history_path.exists() else []
        config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
        losses = [float(row["loss"]) for row in history if "loss" in row]
        rows.append(
            {
                "run": str(path),
                "phase": metrics.get("phase", ""),
                "model_kind": metrics.get("model_kind", config.get("model_kind", "")),
                "examples": metrics.get("examples", 0),
                "epochs": metrics.get("epochs", 0),
                "initial_loss": metrics.get("initial_loss"),
                "final_loss": metrics.get("final_loss"),
                "best_loss": min(losses) if losses else metrics.get("final_loss"),
                "loss_decreased": metrics.get("loss_decreased", False),
            }
        )
    return rows


def write_summary(output_json: str | Path, output_csv: str | Path, summary: dict[str, Any]) -> None:
    output_json = Path(output_json)
    output_csv = Path(output_csv)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    rows = summary.get("runs", [])
    if rows:
        with output_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize trajectory-memory paper metrics.")
    parser.add_argument("--trajectory-path", default="outputs/trajectories/sketchmol_opt_bootstrap.jsonl")
    parser.add_argument("--run-glob", default="outputs/runs/sketchmol_trajectory_suite_seed7_*")
    parser.add_argument("--output-json", default="outputs/metrics/paper_metrics_summary.json")
    parser.add_argument("--output-csv", default="outputs/metrics/paper_run_summary.csv")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    summary = {
        "trajectory": summarize_trajectories(args.trajectory_path) if Path(args.trajectory_path).exists() else {},
        "runs": summarize_runs(args.run_glob),
    }
    write_summary(args.output_json, args.output_csv, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

