"""Reporting helpers for SketchImage-JEPA prediction CSVs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable


def summarize_predictions_csv(
    predictions_csv: str | Path,
    out_dir: str | Path | None = None,
    hit_threshold: float = 0.65,
) -> list[dict[str, object]]:
    predictions_csv = Path(predictions_csv)
    rows = _read_rows(predictions_csv)
    summary = summarize_prediction_rows(rows, hit_threshold=hit_threshold)
    if out_dir is None:
        out_dir = predictions_csv.parent
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_summary_csv(out_dir / "task_type_summary.csv", summary)
    (out_dir / "task_type_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def summarize_prediction_rows(rows: Iterable[dict[str, str]], hit_threshold: float = 0.65) -> list[dict[str, object]]:
    by_task: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_task[row["task_id"]].append(row)

    task_summaries: list[dict[str, object]] = []
    for task_id, task_rows in by_task.items():
        task_rows.sort(key=lambda row: int(float(row.get("rank", 999999))))
        top1 = task_rows[0]
        best = max(task_rows, key=lambda row: _float(row.get("target_tanimoto")))
        task_summaries.append(
            {
                "task_id": task_id,
                "task_type": top1.get("task_type", ""),
                "candidate_count": len(task_rows),
                "top1_valid": _bool(top1.get("valid")),
                "top1_target_tanimoto": _float(top1.get("target_tanimoto")),
                "top1_scaffold_match": _bool(top1.get("scaffold_match")),
                "best_target_tanimoto": _float(best.get("target_tanimoto")),
                "topk_hit": _float(best.get("target_tanimoto")) >= hit_threshold,
            }
        )

    return _aggregate_task_summaries(task_summaries)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize SketchImage-JEPA predictions by task type.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--hit-threshold", type=float, default=0.65)
    args = parser.parse_args()
    summary = summarize_predictions_csv(args.predictions, out_dir=args.out_dir, hit_threshold=args.hit_threshold)
    print(json.dumps(summary, indent=2, sort_keys=True))


def _aggregate_task_summaries(task_summaries: list[dict[str, object]]) -> list[dict[str, object]]:
    by_type: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in task_summaries:
        by_type[str(row["task_type"])].append(row)
        by_type["__overall__"].append(row)

    out = []
    for task_type in sorted(by_type):
        rows = by_type[task_type]
        n = len(rows)
        out.append(
            {
                "task_type": "overall" if task_type == "__overall__" else task_type,
                "n": n,
                "mean_candidate_count": _mean(float(row["candidate_count"]) for row in rows),
                "top1_validity": _mean(1.0 if row["top1_valid"] else 0.0 for row in rows),
                "top1_target_tanimoto": _mean(float(row["top1_target_tanimoto"]) for row in rows),
                "top1_scaffold_match": _mean(1.0 if row["top1_scaffold_match"] else 0.0 for row in rows),
                "mean_best_tanimoto": _mean(float(row["best_target_tanimoto"]) for row in rows),
                "topk_target_hit": _mean(1.0 if row["topk_hit"] else 0.0 for row in rows),
            }
        )
    out.sort(key=lambda row: (row["task_type"] != "overall", str(row["task_type"])))
    return out


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_summary_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _mean(values: Iterable[float]) -> float:
    vals = list(values)
    return float(sum(vals) / len(vals)) if vals else 0.0


def _float(value: object) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


if __name__ == "__main__":
    main()
