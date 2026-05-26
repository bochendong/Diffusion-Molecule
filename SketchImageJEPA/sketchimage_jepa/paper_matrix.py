"""Summarize paper-track ablation matrices across variants and seeds."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable


DEFAULT_METRICS = (
    "top1_target_tanimoto",
    "mean_best_tanimoto",
    "topk_target_hit",
    "top1_scaffold_match",
    "top1_property_success",
    "topk_property_success",
    "top1_property_delta_mae",
    "mean_best_property_delta_mae",
    "top1_property_delta_success",
    "topk_property_delta_success",
)

DEFAULT_TASK_TYPES = ("overall", "de_novo", "edit", "fragment_grow", "inpaint")


def summarize_matrix(
    matrix_name: str,
    variants: Iterable[str],
    seeds: Iterable[int | str],
    run_root: str | Path = "outputs/runs",
    metrics: Iterable[str] = DEFAULT_METRICS,
    task_types: Iterable[str] = DEFAULT_TASK_TYPES,
) -> tuple[list[dict[str, object]], list[str]]:
    """Return mean/std rows for run directories named matrix_variant_seedN."""

    run_root = Path(run_root)
    variants = [str(variant) for variant in variants]
    seeds = [str(seed) for seed in seeds]
    metrics = [str(metric) for metric in metrics]
    task_type_set = {str(task_type) for task_type in task_types}

    values: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    seen_seeds: dict[tuple[str, str], set[str]] = defaultdict(set)
    missing: list[str] = []

    for variant in variants:
        for seed in seeds:
            summary_path = run_root / f"{matrix_name}_{variant}_seed{seed}" / "task_type_summary.csv"
            if not summary_path.exists():
                missing.append(str(summary_path))
                continue
            for row in _read_rows(summary_path):
                task_type = row.get("task_type", "")
                if task_type not in task_type_set:
                    continue
                seen_seeds[(variant, task_type)].add(seed)
                for metric in metrics:
                    if metric in row and row[metric] != "":
                        values[(variant, task_type, metric)].append(_float(row[metric]))

    rows: list[dict[str, object]] = []
    for variant in variants:
        for task_type in DEFAULT_TASK_TYPES:
            if task_type not in task_type_set:
                continue
            seed_set = seen_seeds.get((variant, task_type), set())
            if not seed_set:
                continue
            record: dict[str, object] = {
                "variant": variant,
                "task_type": task_type,
                "seeds_done": len(seed_set),
                "missing_seeds": " ".join(seed for seed in seeds if seed not in seed_set),
            }
            for metric in metrics:
                metric_values = values.get((variant, task_type, metric), [])
                if metric_values:
                    record[f"{metric}_mean"] = _mean(metric_values)
                    record[f"{metric}_std"] = _std(metric_values)
            rows.append(record)

    return rows, missing


def write_matrix_summary(rows: list[dict[str, object]], out_csv: str | Path, out_json: str | Path | None = None) -> None:
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = _fieldnames(rows)
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    if out_json is not None:
        out_json = Path(out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize SketchImage-JEPA paper ablation matrices.")
    parser.add_argument("--matrix-name", required=True)
    parser.add_argument("--variants", required=True, help="Space-separated variant names.")
    parser.add_argument("--seeds", required=True, help="Space-separated seed values.")
    parser.add_argument("--run-root", default="outputs/runs")
    parser.add_argument("--out-csv", default=None)
    parser.add_argument("--out-json", default=None)
    args = parser.parse_args()

    out_csv = args.out_csv or f"outputs/paper/{args.matrix_name}_summary.csv"
    out_json = args.out_json or f"outputs/paper/{args.matrix_name}_summary.json"
    rows, missing = summarize_matrix(
        matrix_name=args.matrix_name,
        variants=args.variants.split(),
        seeds=args.seeds.split(),
        run_root=args.run_root,
    )
    write_matrix_summary(rows, out_csv=out_csv, out_json=out_json)
    print(json.dumps({"rows": len(rows), "missing": missing, "out_csv": str(out_csv), "out_json": str(out_json)}, indent=2, sort_keys=True))


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _fieldnames(rows: list[dict[str, object]]) -> list[str]:
    base = ["variant", "task_type", "seeds_done", "missing_seeds"]
    extras: list[str] = []
    for metric in DEFAULT_METRICS:
        extras.extend([f"{metric}_mean", f"{metric}_std"])
    for row in rows:
        for key in row:
            if key not in base and key not in extras:
                extras.append(key)
    return base + extras


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    return float(math.sqrt(sum((value - mean) ** 2 for value in values) / len(values)))


def _float(value: object) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


if __name__ == "__main__":
    main()
