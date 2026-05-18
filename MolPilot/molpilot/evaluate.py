"""Aggregate MolPilot candidate CSV files."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from .artifacts import save_json


BOOL_COLUMNS = (
    "valid",
    "hard_verifiable",
    "goal_success",
    "constraint_success",
    "overall_success",
)


def main() -> None:
    args = parse_args()
    rows = _read_rows(args.candidates)
    metrics = {"rows": float(len(rows))}
    for column in BOOL_COLUMNS:
        values = [_as_bool(row.get(column, "")) for row in rows]
        metrics[column] = float(np.mean(values)) if values else 0.0
    hard_rows = [row for row in rows if _as_bool(row.get("hard_verifiable", ""))]
    metrics["hard_rows"] = float(len(hard_rows))
    metrics["hard_overall_success"] = float(np.mean([_as_bool(row.get("overall_success", "")) for row in hard_rows])) if hard_rows else 0.0
    save_json(metrics, args.out)
    print("MolPilot evaluation summary")
    for key, value in metrics.items():
        print(f"{key}={value:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate MolPilot candidates CSV.")
    parser.add_argument("--candidates", default="outputs/stages/default/stage4_samples/tables/candidates.csv")
    parser.add_argument("--out", default="outputs/stages/default/eval_metrics.json")
    return parser.parse_args()


def _read_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _as_bool(value: str) -> float:
    return 1.0 if str(value).strip().lower() in {"1", "1.0", "true", "yes"} else 0.0


if __name__ == "__main__":
    main()

