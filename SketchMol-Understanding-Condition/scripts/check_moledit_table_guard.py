#!/usr/bin/env python3
"""Fail a Slurm chain when MolEdit Table1 mean drops below a guardrail."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-csv", required=True, type=Path)
    parser.add_argument("--metric", default="Acc_all(0.65)")
    parser.add_argument("--min-mean", type=float, default=0.894)
    parser.add_argument("--require-tasks", type=int, default=10)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.summary_csv.exists():
        raise FileNotFoundError(f"Missing table summary CSV: {args.summary_csv}")

    rows = []
    with args.summary_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            value = _parse_float(row.get(args.metric, ""))
            n_value = _parse_float(row.get("n", ""))
            if value is None:
                continue
            if n_value is not None and n_value <= 0:
                continue
            rows.append({"task": row.get("task", ""), args.metric: value, "n": n_value})

    if args.require_tasks and len(rows) != args.require_tasks:
        status = "fail"
        reason = f"expected {args.require_tasks} tasks, found {len(rows)}"
        mean_value = 0.0
    else:
        mean_value = sum(float(row[args.metric]) for row in rows) / max(len(rows), 1)
        status = "pass" if mean_value >= args.min_mean else "fail"
        reason = "" if status == "pass" else f"{args.metric} mean {mean_value:.6f} < {args.min_mean:.6f}"

    payload = {
        "status": status,
        "reason": reason,
        "summary_csv": str(args.summary_csv),
        "metric": args.metric,
        "mean": mean_value,
        "min_mean": args.min_mean,
        "tasks": rows,
        "task_count": len(rows),
        "require_tasks": args.require_tasks,
    }
    output_json = args.output_json or args.summary_csv.with_name("moledit_table_guard.json")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if status != "pass":
        raise SystemExit(1)


def _parse_float(value: object) -> float | None:
    text = str(value or "").strip()
    if not text or text.lower() in {"nan", "none", "null", "missing-reference"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


if __name__ == "__main__":
    main()
