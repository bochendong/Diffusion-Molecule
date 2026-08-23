#!/usr/bin/env python3
"""Honest any@k MolEdit Table1 metrics from raw candidates (no ranking)."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parents[1]
UNIFIED_SCRIPTS = REPO_DIR / "SketchMol-Unified-3MDiffusion" / "scripts"
for path in (UNIFIED_SCRIPTS, REPO_DIR / "SketchMol-Unified-3MDiffusion"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from evaluate_moledit_table_metrics import (  # noqa: E402
    ID_COLUMNS,
    PREDICTION_SMILES_COLUMNS,
    TABLE1_TASK_ORDER,
    TASK_LABELS,
    Chemistry,
    configured_oracle_provenance,
    coverage_status,
    evaluate_prediction,
    first_value,
    load_references,
    normalize_id,
    summarize_task,
    task_key,
    task_sort_key,
    task_specs_for_reference,
    write_csv,
    write_markdown,
)


CANDIDATE_INDEX_COLUMNS = (
    "direct_candidate_index",
    "candidate_index",
    "sample_index",
    "rank",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--candidate-limit", type=int, default=20)
    parser.add_argument("--thresholds", default="0.65,0.15")
    parser.add_argument("--model-name", default="DirectSMILES-anyk")
    parser.add_argument("--method-filter", default=None)
    parser.add_argument("--task-filter", choices=("all", "table1"), default="table1")
    parser.add_argument("--missing-oracle-policy", choices=("fail", "skip-task"), default="fail")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    thresholds = [float(item) for item in args.thresholds.split(",") if item.strip()]
    if not thresholds:
        raise SystemExit("--thresholds must contain at least one value")
    chem = Chemistry()
    references = load_references(args.reference)
    grouped_candidates = load_candidates(
        args.candidates,
        method_filter=args.method_filter,
        candidate_limit=max(1, int(args.candidate_limit)),
    )

    rows = []
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    matched_predictions_by_task: dict[str, int] = defaultdict(int)
    skipped_missing_oracle: dict[str, int] = defaultdict(int)
    reference_counts: dict[str, int] = defaultdict(int)
    mean_best_tanimoto: dict[str, list[float]] = defaultdict(list)
    property_hit_counts: dict[str, int] = defaultdict(int)

    for ref_id, ref in references.items():
        task_specs = task_specs_for_reference(ref)
        current_task_key = task_key(task_specs)
        if args.task_filter == "table1" and current_task_key not in TASK_LABELS:
            continue
        reference_counts[current_task_key] += 1
        pool = grouped_candidates.get(ref_id, [])
        if not pool:
            continue
        matched_predictions_by_task[current_task_key] += 1
        missing_oracles = sorted(chem.missing_oracles(task_specs))
        if missing_oracles and args.missing_oracle_policy == "skip-task":
            skipped_missing_oracle[current_task_key] += 1
            continue
        if missing_oracles and args.missing_oracle_policy == "fail":
            raise SystemExit(
                f"Missing TDC oracle(s) for task {current_task_key}: {', '.join(missing_oracles)}. "
                "Install TDC or rerun with --missing-oracle-policy skip-task."
            )
        evaluated_candidates = [
            evaluate_prediction(ref, smiles, task_specs, chem=chem, thresholds=thresholds)
            for smiles in pool
        ]
        any_row = aggregate_anyk(evaluated_candidates, thresholds=thresholds)
        any_row["task_key"] = current_task_key
        grouped[current_task_key].append(any_row)
        tanimotos = [
            float(item["source_tanimoto"])
            for item in evaluated_candidates
            if item.get("source_tanimoto") is not None
        ]
        if tanimotos:
            mean_best_tanimoto[current_task_key].append(max(tanimotos))
        if any(item.get("property_success") for item in evaluated_candidates):
            property_hit_counts[current_task_key] += 1

    for key in sorted(grouped, key=task_sort_key):
        task_rows = grouped[key]
        summary = summarize_task(task_rows, thresholds=thresholds, fcd=None)
        summary.update(
            {
                "model": args.model_name,
                "task": TASK_LABELS.get(key, key),
                "task_key": key,
                "reference_n": reference_counts.get(key, 0),
                "prediction_n": matched_predictions_by_task.get(key, 0),
                "missing_prediction_rows": max(
                    reference_counts.get(key, 0) - matched_predictions_by_task.get(key, 0), 0
                ),
                "missing_oracle_skipped_rows": skipped_missing_oracle.get(key, 0),
                "property_anyk": _mean(
                    property_hit_counts.get(key, 0), matched_predictions_by_task.get(key, 0)
                ),
                "mean_best_source_tanimoto": _mean_list(mean_best_tanimoto.get(key, [])),
                "selection": f"any@{int(args.candidate_limit)}",
            }
        )
        summary["status"] = coverage_status(summary)
        rows.append(summary)

    if args.task_filter == "table1":
        present = {row["task_key"] for row in rows}
        for key in TABLE1_TASK_ORDER:
            if key in present:
                continue
            summary = {
                "n": 0,
                "valid_n": 0,
                "Validity": "",
                "FCD": "",
                "model": args.model_name,
                "task": TASK_LABELS[key],
                "task_key": key,
                "reference_n": reference_counts.get(key, 0),
                "prediction_n": matched_predictions_by_task.get(key, 0),
                "missing_prediction_rows": max(
                    reference_counts.get(key, 0) - matched_predictions_by_task.get(key, 0), 0
                ),
                "missing_oracle_skipped_rows": skipped_missing_oracle.get(key, 0),
                "selection": f"any@{int(args.candidate_limit)}",
            }
            for threshold in thresholds:
                summary[f"Acc_all({threshold:g})"] = ""
                summary[f"Acc_valid({threshold:g})"] = ""
            summary["status"] = coverage_status(summary)
            rows.append(summary)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "moledit_table_summary.csv"
    json_path = args.output_dir / "moledit_table_summary.json"
    md_path = args.output_dir / "moledit_table_summary.md"
    write_csv(csv_path, rows)
    json_path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(md_path, rows, thresholds=thresholds)
    print(json.dumps({"rows": len(rows), "csv": str(csv_path), "markdown": str(md_path)}, indent=2, sort_keys=True))
    (args.output_dir / "oracle_provenance.json").write_text(
        json.dumps(configured_oracle_provenance(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


def load_candidates(path: Path, *, method_filter: str | None, candidate_limit: int) -> dict[str, list[str]]:
    grouped: dict[str, list[tuple[int, str]]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if method_filter is not None and str(row.get("method", "") or "") != method_filter:
                continue
            row_id = normalize_id(first_value(row, ID_COLUMNS))
            smiles = first_value(
                row,
                (*PREDICTION_SMILES_COLUMNS, "direct_candidate_canonical_smiles", "direct_candidate_raw_smiles"),
            )
            if not row_id or not smiles:
                continue
            grouped[row_id].append((candidate_index(row), smiles))
    out: dict[str, list[str]] = {}
    for row_id, items in grouped.items():
        items.sort(key=lambda item: item[0])
        out[row_id] = [smiles for _, smiles in items[:candidate_limit]]
    return out


def candidate_index(row: Mapping[str, str]) -> int:
    for key in CANDIDATE_INDEX_COLUMNS:
        raw = str(row.get(key, "") or "").strip()
        if raw:
            try:
                return int(float(raw))
            except ValueError:
                continue
    return 10**9


def aggregate_anyk(candidates: Sequence[Mapping[str, object]], *, thresholds: Sequence[float]) -> dict[str, object]:
    valid = any(bool(item.get("valid")) for item in candidates)
    property_success = any(bool(item.get("property_success")) for item in candidates)
    tanimotos = [float(item["source_tanimoto"]) for item in candidates if item.get("source_tanimoto") is not None]
    row: dict[str, object] = {
        "valid": valid,
        "property_success": property_success,
        "source_tanimoto": max(tanimotos) if tanimotos else None,
    }
    for threshold in thresholds:
        key = f"success_t{threshold:g}"
        row[key] = any(bool(item.get(key)) for item in candidates)
    return row


def _mean(numerator: int, denominator: int) -> float | str:
    if denominator <= 0:
        return ""
    return float(numerator) / float(denominator)


def _mean_list(values: Sequence[float]) -> float | str:
    if not values:
        return ""
    return sum(values) / len(values)


if __name__ == "__main__":
    raise SystemExit(main())
