#!/usr/bin/env python3
"""Select a Joint v2 checkpoint using validation only and enforce forgetting gate."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from statistics import mean
from typing import Sequence


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", required=True, type=Path)
    parser.add_argument("--candidate-root", required=True, type=Path)
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--selected-checkpoint", required=True, type=Path)
    parser.add_argument("--max-denovo-drop", type=float, default=0.02)
    return parser.parse_args(argv)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def denovo_macro(path: Path) -> float:
    values = []
    for row in read_rows(path):
        try:
            group = int(str(row.get("property_count", "")))
            value = float(row["strict_success_rate"])
        except (KeyError, TypeError, ValueError):
            continue
        if 2 <= group <= 7:
            values.append(value)
    if len(values) != 6:
        raise ValueError(f"Expected 2p-7p validation groups in {path}, found {len(values)}")
    return mean(values)


def table1_mean_acc(path: Path) -> float:
    values = []
    for row in read_rows(path):
        raw = row.get("Acc_all(0.15)", "")
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        task = str(row.get("task_key", row.get("Task", row.get("task", "")))).strip().lower()
        if task not in {"", "all", "mean", "macro", "overall"}:
            values.append(value)
    if not values:
        # Some evaluator versions only emit the ten task rows without a task_key,
        # so accept all numeric rows as a compatibility fallback.
        values = [
            float(row["Acc_all(0.15)"])
            for row in read_rows(path)
            if str(row.get("Acc_all(0.15)", "")).strip()
        ]
    if not values:
        raise ValueError(f"No numeric Acc_all(0.15) values in {path}")
    return mean(values)


def metric_paths(root: Path) -> tuple[Path, Path]:
    return (
        root / "2p7p" / "denovo_2p7p" / "n20" / "finalizer" / "benchmark_summary.csv",
        root
        / "table1"
        / "moledit_table1"
        / "n20"
        / "finalizer"
        / "metrics"
        / "moledit_table_summary.csv",
    )


def validation_seed_roots(root: Path) -> list[Path]:
    seeds = sorted(path for path in root.glob("eval_seed_*") if path.is_dir())
    return seeds or [root]


def validation_metrics(root: Path) -> tuple[float, float, list[str], list[str]]:
    denovo_values = []
    table_values = []
    denovo_paths = []
    table_paths = []
    for seed_root in validation_seed_roots(root):
        denovo_path, table_path = metric_paths(seed_root)
        if not denovo_path.is_file() or not table_path.is_file():
            raise FileNotFoundError(f"Missing validation metrics under {seed_root}")
        denovo_values.append(denovo_macro(denovo_path))
        table_values.append(table1_mean_acc(table_path))
        denovo_paths.append(str(denovo_path))
        table_paths.append(str(table_path))
    return mean(denovo_values), mean(table_values), denovo_paths, table_paths


def checkpoint_epoch(path: Path) -> int:
    try:
        return int(path.stem.rsplit("_", 1)[-1])
    except ValueError:
        return 10**9


def select_checkpoint(args: argparse.Namespace) -> dict[str, object]:
    baseline_denovo, _, baseline_denovo_paths, _ = validation_metrics(args.baseline_root)
    records: list[dict[str, object]] = []
    for checkpoint in sorted(args.checkpoint_dir.glob("checkpoint_epoch_*.pt"), key=checkpoint_epoch):
        validation_root = args.candidate_root / checkpoint.stem
        try:
            denovo, table1, denovo_paths, table1_paths = validation_metrics(validation_root)
        except FileNotFoundError:
            denovo_path, table1_path = metric_paths(validation_root)
            records.append(
                {
                    "checkpoint": str(checkpoint),
                    "epoch": checkpoint_epoch(checkpoint),
                    "status": "missing_validation",
                    "denovo_summary": str(denovo_path),
                    "table1_summary": str(table1_path),
                }
            )
            continue
        passes = denovo >= baseline_denovo - float(args.max_denovo_drop)
        records.append(
            {
                "checkpoint": str(checkpoint),
                "epoch": checkpoint_epoch(checkpoint),
                "status": "eligible" if passes else "forgetting_gate_failed",
                "passes_forgetting_gate": passes,
                "denovo_macro_strict_at20": denovo,
                "denovo_delta_vs_u0": denovo - baseline_denovo,
                "table1_mean_acc_0_15_at20": table1,
                "denovo_summaries": denovo_paths,
                "table1_summaries": table1_paths,
            }
        )
    eligible = [row for row in records if row.get("passes_forgetting_gate")]
    selected = max(
        eligible,
        key=lambda row: (
            float(row["table1_mean_acc_0_15_at20"]),
            float(row["denovo_macro_strict_at20"]),
            -int(row["epoch"]),
        ),
        default=None,
    )
    return {
        "protocol": "unified_joint_fair_v2_validation_at20",
        "status": "selected" if selected else "forgetting_failure",
        "baseline_denovo_macro_strict_at20": baseline_denovo,
        "baseline_denovo_summaries": baseline_denovo_paths,
        "max_denovo_drop": float(args.max_denovo_drop),
        "selected_checkpoint": selected["checkpoint"] if selected else None,
        "selected_epoch": selected["epoch"] if selected else None,
        "selected_table1_mean_acc_0_15_at20": (
            selected["table1_mean_acc_0_15_at20"] if selected else None
        ),
        "checkpoints": records,
    }


def write_selected_symlink(link: Path, target: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(os.path.relpath(target.resolve(), start=link.parent.resolve()))


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = select_checkpoint(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if result["status"] != "selected":
        if args.selected_checkpoint.is_symlink():
            args.selected_checkpoint.unlink()
        print(json.dumps(result, indent=2, sort_keys=True))
        return 3
    write_selected_symlink(args.selected_checkpoint, Path(str(result["selected_checkpoint"])))
    result["selected_checkpoint_link"] = str(args.selected_checkpoint)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
