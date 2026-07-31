#!/usr/bin/env python3
"""Collect UMTP v1 benchmark metrics into long-form and aggregate paper tables."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Mapping, Sequence


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", required=True, type=Path)
    parser.add_argument("--output-prefix", required=True, type=Path)
    return parser.parse_args(argv)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def path_metadata(path: Path, root: Path) -> dict[str, str]:
    parts = path.relative_to(root).parts
    if len(parts) < 4 or not parts[0].startswith("train_seed_") or not parts[1].startswith("eval_seed_"):
        return {}
    budget = ""
    selection = ""
    for index, part in enumerate(parts):
        match = re.fullmatch(r"n(\d+)", part)
        if match:
            budget = match.group(1)
            selection = parts[index + 1] if index + 1 < len(parts) else ""
            break
    return {
        "train_seed": parts[0].removeprefix("train_seed_"),
        "eval_seed": parts[1].removeprefix("eval_seed_"),
        "benchmark": parts[2],
        "budget": budget,
        "selection": selection,
    }


def numeric_items(row: Mapping[str, str], ignored: set[str]):
    for key, raw in row.items():
        if key in ignored or raw is None or not str(raw).strip():
            continue
        try:
            value = float(raw)
        except ValueError:
            continue
        if math.isfinite(value):
            yield key, value


def collect(root: Path) -> list[dict[str, object]]:
    paths = sorted(root.glob("**/benchmark_summary.csv")) + sorted(root.glob("**/moledit_table_summary.csv"))
    output = []
    for path in paths:
        meta = path_metadata(path, root)
        if not meta:
            continue
        is_table1 = path.name == "moledit_table_summary.csv"
        for row in read_rows(path):
            group = str(row.get("task_key", row.get("task", ""))) if is_table1 else str(
                row.get("property_count", row.get("benchmark_label", ""))
            )
            ignored = {"model", "task", "task_key", "status"} if is_table1 else {
                "method", "benchmark_label", "property_count"
            }
            for metric, value in numeric_items(row, ignored):
                output.append(
                    {
                        **meta,
                        "group": group,
                        "metric": metric,
                        "value": value,
                        "source_summary": str(path),
                    }
                )
    return output


def aggregate(rows: list[Mapping[str, object]]) -> list[dict[str, object]]:
    keys = ("benchmark", "budget", "selection", "group", "metric")
    grouped: dict[tuple[str, ...], list[float]] = defaultdict(list)
    for row in rows:
        grouped[tuple(str(row.get(key, "")) for key in keys)].append(float(row["value"]))
    return [
        {
            **dict(zip(keys, key)),
            "mean": mean(values),
            "std": stdev(values) if len(values) > 1 else 0.0,
            "n": len(values),
        }
        for key, values in sorted(grouped.items())
    ]


def write_rows(path: Path, rows: list[Mapping[str, object]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    runs = collect(args.eval_root)
    if not runs:
        raise SystemExit(f"No UMTP benchmark summaries found under {args.eval_root}")
    aggregates = aggregate(runs)
    runs_path = args.output_prefix.with_name(args.output_prefix.name + "_runs.csv")
    aggregate_path = args.output_prefix.with_name(args.output_prefix.name + "_aggregate.csv")
    summary_path = args.output_prefix.with_name(args.output_prefix.name + "_summary.json")
    write_rows(runs_path, runs)
    write_rows(aggregate_path, aggregates)
    summary = {
        "protocol": "unified_molecular_transformation_policy_v1",
        "runs": str(runs_path),
        "aggregate": str(aggregate_path),
        "run_metric_rows": len(runs),
        "aggregate_rows": len(aggregates),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
