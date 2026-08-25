#!/usr/bin/env python3
"""Compare completed protocol-sweep summaries with MolEditRL Table 1."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable


METRICS = ("Validity", "Acc_all(0.65)", "Acc_all(0.15)")

MOLEDITRL = {
    "GSK3B:increase": (0.952, 0.342, 0.514),
    "RB:decrease": (0.984, 0.634, 0.830),
    "MW:increase": (0.960, 0.404, 0.856),
    "SA:decrease": (0.988, 0.628, 0.828),
    "HBA:decrease+SA:decrease": (0.972, 0.346, 0.510),
    "QED:increase+SA:decrease": (0.974, 0.632, 0.788),
    "HBA:decrease+LogP:increase": (0.946, 0.316, 0.800),
    "HBA:decrease+MW:decrease": (0.942, 0.252, 0.660),
    "DRD2:decrease+MW:decrease+SA:decrease": (0.986, 0.518, 0.724),
    "HBA:increase+MW:increase+QED:decrease": (0.958, 0.430, 0.756),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        metavar="LABEL=SUMMARY_JSON",
        help="Repeat for every K/model summary to compare.",
    )
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-markdown", required=True, type=Path)
    return parser.parse_args()


def mean(values: Iterable[float]) -> float:
    items = list(values)
    return sum(items) / len(items)


def rmse(values: Iterable[float]) -> float:
    return math.sqrt(mean(value * value for value in values))


def target_macro() -> dict[str, float]:
    return {
        metric: mean(values[index] for values in MOLEDITRL.values())
        for index, metric in enumerate(METRICS)
    }


def measured_tasks(block: dict[str, object]) -> dict[str, dict[str, object]]:
    rows = block.get("per_task", [])
    return {
        str(row["task_key"]): row
        for row in rows
        if isinstance(row, dict) and str(row.get("task_key", "")) in MOLEDITRL
    }


def compare(label: str, path: Path, aggregation: str, block: dict[str, object]) -> dict[str, object]:
    macro = block["macro"]
    target = target_macro()
    macro_delta = {metric: float(macro[metric]) - target[metric] for metric in METRICS}
    tasks = measured_tasks(block)
    if set(tasks) != set(MOLEDITRL):
        missing = sorted(set(MOLEDITRL) - set(tasks))
        raise ValueError(f"{path}: incomplete Table1 task vector; missing={missing}")

    per_metric_rmse: dict[str, float] = {}
    all_task_deltas: list[float] = []
    for metric_index, metric in enumerate(METRICS):
        deltas = [
            float(tasks[key][metric]) - MOLEDITRL[key][metric_index]
            for key in MOLEDITRL
        ]
        per_metric_rmse[metric] = rmse(deltas)
        all_task_deltas.extend(deltas)

    return {
        "label": label,
        "path": str(path),
        "aggregation": aggregation,
        "candidate_limit": int(json.loads(path.read_text(encoding="utf-8"))["candidate_limit"]),
        "macro": {metric: float(macro[metric]) for metric in METRICS},
        "macro_delta": macro_delta,
        "macro_l2": math.sqrt(sum(value * value for value in macro_delta.values())),
        "macro_mae": mean(abs(value) for value in macro_delta.values()),
        "task_rmse": per_metric_rmse,
        "task_vector_rmse": rmse(all_task_deltas),
        "protocol_compatible": aggregation == "candidate_level",
    }


def main() -> int:
    args = parse_args()
    comparisons: list[dict[str, object]] = []
    for spec in args.input:
        if "=" not in spec:
            raise SystemExit(f"invalid --input {spec!r}; expected LABEL=PATH")
        label, raw_path = spec.split("=", 1)
        path = Path(raw_path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        comparisons.append(compare(label, path, "candidate_level", payload["candidate_level"]))
        comparisons.append(compare(label, path, "any_at_k", payload["any_at_k"]))

    comparisons.sort(key=lambda row: (float(row["macro_l2"]), float(row["task_vector_rmse"])))
    result = {
        "protocol": "p22_moledit_table1_aggregation_audit_v1",
        "external": {"method": "MolEditRL", "macro": target_macro(), "per_task": MOLEDITRL},
        "comparisons": comparisons,
        "closest_numeric": comparisons[0],
        "closest_protocol_compatible": min(
            (row for row in comparisons if row["protocol_compatible"]),
            key=lambda row: (float(row["macro_l2"]), float(row["task_vector_rmse"])),
        ),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# P22 MolEdit Table1 protocol sweep",
        "",
        "Candidate-level rows retain every raw output in the denominator. Any@K is a",
        "condition-level budget diagnostic and is not protocol-compatible with MolEditRL `Acc_all`.",
        "",
        "| Model/K | Aggregation | Validity | Strict .65 | Relaxed .15 | Macro L2 | Task RMSE | Compatible |",
        "|---|---|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in comparisons:
        macro = row["macro"]
        lines.append(
            f"| {row['label']} | {row['aggregation']} | {macro['Validity']:.4f} | "
            f"{macro['Acc_all(0.65)']:.4f} | {macro['Acc_all(0.15)']:.4f} | "
            f"{row['macro_l2']:.4f} | {row['task_vector_rmse']:.4f} | "
            f"{'yes' if row['protocol_compatible'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            f"Numerically closest: **{result['closest_numeric']['label']} / "
            f"{result['closest_numeric']['aggregation']}**.",
            f"Closest protocol-compatible row: **{result['closest_protocol_compatible']['label']} / "
            f"{result['closest_protocol_compatible']['aggregation']}**.",
            "",
        ]
    )
    args.output_markdown.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"rows": len(comparisons), "output": str(args.output_json)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
