#!/usr/bin/env python3
"""Any@k budget curve from one oracle pass over n=20 candidates."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
REPO_DIR = PROJECT_DIR.parent
UNIFIED_SCRIPTS = REPO_DIR / "SketchMol-Unified-3MDiffusion" / "scripts"
for path in (UNIFIED_SCRIPTS, PROJECT_DIR / "scripts", PROJECT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from evaluate_moledit_table1_anyk import aggregate_anyk, candidate_index  # noqa: E402
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

REAL_TASK_KEYS = (
    "DRD2:decrease+MW:decrease+SA:decrease",
    "GSK3B:increase",
    "MW:increase",
    "RB:decrease",
    "SA:decrease",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--ks", default="1,2,5,10,20")
    parser.add_argument("--thresholds", default="0.65,0.15")
    parser.add_argument("--model-name", default="anyk-budget")
    parser.add_argument("--task-filter", choices=("all", "table1"), default="table1")
    parser.add_argument("--missing-oracle-policy", choices=("fail", "skip-task"), default="fail")
    parser.add_argument("--max-candidates", type=int, default=20)
    return parser.parse_args()


def load_candidate_pools(path: Path, *, limit: int) -> dict[str, list[str]]:
    grouped: dict[str, list[tuple[int, str]]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
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
        out[row_id] = [smiles for _, smiles in items[: max(1, int(limit))]]
    return out


def mean_metric(rows: list[dict[str, object]], key: str) -> float | None:
    values = []
    for row in rows:
        raw = row.get(key)
        if raw in ("", None):
            continue
        values.append(float(raw))
    if not values:
        return None
    return sum(values) / len(values)


def trapezoid_auc(ks: list[int], values: list[float | None]) -> float | None:
    xs = []
    ys = []
    for k, value in zip(ks, values):
        if value is None:
            continue
        xs.append(float(k))
        ys.append(float(value))
    if len(xs) < 2:
        return None
    area = 0.0
    for left, right, y_left, y_right in zip(xs, xs[1:], ys, ys[1:]):
        area += 0.5 * (y_left + y_right) * (right - left)
    width = xs[-1] - xs[0]
    if width <= 0:
        return None
    return area / width


def unique_smiles_mean(pools: dict[str, list[str]]) -> float:
    if not pools:
        return 0.0
    sizes = [len({item for item in smiles if item}) for smiles in pools.values()]
    return float(sum(sizes) / len(sizes))


def main() -> int:
    args = parse_args()
    ks = sorted({int(item) for item in args.ks.split(",") if item.strip()})
    thresholds = [float(item) for item in args.thresholds.split(",") if item.strip()]
    if not ks or not thresholds:
        raise SystemExit("ks and thresholds must be non-empty")
    max_k = max(ks)
    chem = Chemistry()
    references = load_references(args.reference)
    pools = load_candidate_pools(args.candidates, limit=max(int(args.max_candidates), max_k))

    evaluated_by_id: dict[str, list[dict[str, object]]] = {}
    skipped_missing_oracle: dict[str, int] = defaultdict(int)
    reference_counts: dict[str, int] = defaultdict(int)
    matched_by_task: dict[str, int] = defaultdict(int)
    task_of_id: dict[str, str] = {}

    for ref_id, ref in references.items():
        task_specs = task_specs_for_reference(ref)
        current_task_key = task_key(task_specs)
        if args.task_filter == "table1" and current_task_key not in TASK_LABELS:
            continue
        reference_counts[current_task_key] += 1
        pool = pools.get(ref_id, [])
        if not pool:
            continue
        matched_by_task[current_task_key] += 1
        missing_oracles = sorted(chem.missing_oracles(task_specs))
        if missing_oracles and args.missing_oracle_policy == "skip-task":
            skipped_missing_oracle[current_task_key] += 1
            continue
        if missing_oracles and args.missing_oracle_policy == "fail":
            raise SystemExit(
                f"Missing TDC oracle(s) for task {current_task_key}: {', '.join(missing_oracles)}."
            )
        evaluated_by_id[ref_id] = [
            evaluate_prediction(ref, smiles, task_specs, chem=chem, thresholds=thresholds)
            for smiles in pool
        ]
        task_of_id[ref_id] = current_task_key

    summaries_by_k: dict[int, list[dict[str, object]]] = {}
    for k in ks:
        grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
        mean_best_tanimoto: dict[str, list[float]] = defaultdict(list)
        property_hits: dict[str, int] = defaultdict(int)
        for ref_id, evaluated in evaluated_by_id.items():
            current_task_key = task_of_id[ref_id]
            prefix = evaluated[:k]
            any_row = aggregate_anyk(prefix, thresholds=thresholds)
            any_row["task_key"] = current_task_key
            grouped[current_task_key].append(any_row)
            tanimotos = [
                float(item["source_tanimoto"])
                for item in prefix
                if item.get("source_tanimoto") is not None
            ]
            if tanimotos:
                mean_best_tanimoto[current_task_key].append(max(tanimotos))
            if any(item.get("property_success") for item in prefix):
                property_hits[current_task_key] += 1
        rows: list[dict[str, object]] = []
        for key in sorted(grouped, key=task_sort_key):
            task_rows = grouped[key]
            summary = summarize_task(task_rows, thresholds=thresholds, fcd=None)
            summary.update(
                {
                    "model": args.model_name,
                    "task": TASK_LABELS.get(key, key),
                    "task_key": key,
                    "reference_n": reference_counts.get(key, 0),
                    "prediction_n": matched_by_task.get(key, 0),
                    "missing_prediction_rows": max(
                        reference_counts.get(key, 0) - matched_by_task.get(key, 0), 0
                    ),
                    "missing_oracle_skipped_rows": skipped_missing_oracle.get(key, 0),
                    "property_anyk": (
                        float(property_hits.get(key, 0)) / float(matched_by_task.get(key, 0))
                        if matched_by_task.get(key, 0)
                        else ""
                    ),
                    "mean_best_source_tanimoto": (
                        sum(mean_best_tanimoto.get(key, [])) / len(mean_best_tanimoto.get(key, []))
                        if mean_best_tanimoto.get(key, [])
                        else ""
                    ),
                    "selection": f"any@{k}",
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
                    "prediction_n": matched_by_task.get(key, 0),
                    "missing_prediction_rows": max(
                        reference_counts.get(key, 0) - matched_by_task.get(key, 0), 0
                    ),
                    "missing_oracle_skipped_rows": skipped_missing_oracle.get(key, 0),
                    "selection": f"any@{k}",
                }
                for threshold in thresholds:
                    summary[f"Acc_all({threshold:g})"] = ""
                    summary[f"Acc_valid({threshold:g})"] = ""
                summary["status"] = coverage_status(summary)
                rows.append(summary)
        summaries_by_k[k] = rows

    k20_rows = summaries_by_k[max_k]
    by_task_curve: dict[str, dict[str, object]] = {}
    for key in TABLE1_TASK_ORDER if args.task_filter == "table1" else sorted({row["task_key"] for rows in summaries_by_k.values() for row in rows}):
        curve: dict[str, object] = {"task_key": key, "task": TASK_LABELS.get(key, key)}
        for threshold in thresholds:
            metric = f"Acc_all({threshold:g})"
            curve[metric] = {
                str(k): next(
                    (row.get(metric) for row in summaries_by_k[k] if row.get("task_key") == key),
                    "",
                )
                for k in ks
            }
        first = next((row for row in k20_rows if row.get("task_key") == key), {})
        curve["n"] = first.get("n")
        curve["Validity"] = first.get("Validity")
        by_task_curve[key] = curve

    def series_for(keys: tuple[str, ...], metric: str = "Acc_all(0.65)") -> dict[str, float | None]:
        out: dict[str, float | None] = {}
        for k in ks:
            rows = [
                row
                for row in summaries_by_k[k]
                if row.get("task_key") in keys and row.get(metric) not in ("", None)
            ]
            out[str(k)] = mean_metric(rows, metric)
        return out

    real5 = series_for(REAL_TASK_KEYS)
    gsk3b = series_for(("GSK3B:increase",))
    payload = {
        "model": args.model_name,
        "ks": ks,
        "thresholds": thresholds,
        "by_task": by_task_curve,
        "real5_anyk_t0_65": real5,
        "gsk3b_anyk_t0_65": gsk3b,
        "auc_real5_t0_65": trapezoid_auc(ks, [real5[str(k)] for k in ks]),
        "auc_gsk3b_t0_65": trapezoid_auc(ks, [gsk3b[str(k)] for k in ks]),
        "mean_unique_smiles": unique_smiles_mean(pools),
        "candidate_conditions": len(pools),
        "evaluated_conditions": len(evaluated_by_id),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    curve_path = args.output_dir / "anyk_curve.json"
    curve_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    json_path = args.output_dir / "moledit_table_summary.json"
    csv_path = args.output_dir / "moledit_table_summary.csv"
    md_path = args.output_dir / "moledit_table_summary.md"
    json_path.write_text(json.dumps(k20_rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(csv_path, k20_rows)
    write_markdown(md_path, k20_rows, thresholds=thresholds)
    (args.output_dir / "oracle_provenance.json").write_text(
        json.dumps(configured_oracle_provenance(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"curve": str(curve_path), "k20": str(json_path), **{str(k): real5[str(k)] for k in ks}},
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
