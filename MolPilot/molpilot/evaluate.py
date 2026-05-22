"""Aggregate MolPilot candidate CSV files."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .artifacts import save_json


BOOL_COLUMNS = (
    "valid",
    "hard_verifiable",
    "goal_success",
    "constraint_success",
    "overall_success",
    "soft_success",
    "soft_repair_success",
)


def main() -> None:
    args = parse_args()
    rows = _read_rows(args.candidates)
    metrics = {"rows": float(len(rows))}
    for column in BOOL_COLUMNS:
        values = [_as_bool(row.get(column, "")) for row in rows if column in row]
        if values:
            metrics[column] = float(np.mean(values))
    objective_rows = [row for row in rows if "objective_quality" in row]
    repair_quality_rows = [row for row in rows if "repair_quality" in row]
    if objective_rows:
        metrics["objective_quality"] = _float_mean(objective_rows, "objective_quality")
    if repair_quality_rows:
        metrics["repair_quality"] = _float_mean(repair_quality_rows, "repair_quality")
    hard_rows = [row for row in rows if _as_bool(row.get("hard_verifiable", ""))]
    metrics["hard_rows"] = float(len(hard_rows))
    metrics["hard_overall_success"] = float(np.mean([_as_bool(row.get("overall_success", "")) for row in hard_rows])) if hard_rows else 0.0
    metrics.update(_request_topk(rows))
    metrics.update(_task_breakdown(rows))
    metrics.update(_repair_breakdown(rows))
    metrics.update(_origin_breakdown(rows))
    metrics.update(_failure_reasons(rows))
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


def _request_topk(rows: list[dict[str, str]]) -> dict[str, float]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("request_id", ""))].append(row)
    out = {}
    for metric in ("overall_success", "goal_success", "constraint_success", "soft_success"):
        if not any(metric in row or (metric == "soft_success" and "soft_repair_success" in row) for row in rows):
            continue
        short = metric.replace("_success", "")
        for k in (1, 5, 10):
            values = []
            for request_rows in grouped.values():
                ordered = sorted(request_rows, key=lambda row: int(float(row.get("rank", 0) or 0)))
                values.append(max(_as_bool(row.get(metric, row.get("soft_repair_success", ""))) for row in ordered[:k]) if ordered else 0.0)
            out[f"request_{short}_at_{k}"] = float(np.mean(values)) if values else 0.0
    for metric in ("objective_quality", "repair_quality"):
        if not any(metric in row for row in rows):
            continue
        short = metric.replace("_quality", "")
        for k in (1, 5, 10):
            values = []
            for request_rows in grouped.values():
                ordered = sorted(request_rows, key=lambda row: int(float(row.get("rank", 0) or 0)))[:k]
                values.append(max(_as_float(row.get(metric, 0.0)) for row in ordered) if ordered else 0.0)
            out[f"request_best_{short}_quality_at_{k}"] = float(np.mean(values)) if values else 0.0
    return out


def _task_breakdown(rows: list[dict[str, str]]) -> dict[str, float]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("task_type", "unknown"))].append(row)
    out = {}
    for task, task_rows in grouped.items():
        key = "".join(ch if ch.isalnum() else "_" for ch in task).strip("_") or "unknown"
        out[f"task_{key}_rows"] = float(len(task_rows))
        out[f"task_{key}_overall_success"] = float(np.mean([_as_bool(row.get("overall_success", "")) for row in task_rows]))
        if any("soft_success" in row or "soft_repair_success" in row for row in task_rows):
            out[f"task_{key}_soft_success"] = float(
                np.mean([_as_bool(row.get("soft_success", row.get("soft_repair_success", ""))) for row in task_rows])
            )
        if any("objective_quality" in row for row in task_rows):
            out[f"task_{key}_objective_quality"] = _float_mean(task_rows, "objective_quality")
        if any("repair_quality" in row for row in task_rows):
            out[f"task_{key}_repair_quality"] = _float_mean(task_rows, "repair_quality")
        out.update(_task_request_topk(task_rows, f"task_{key}"))
    task_keys = sorted("".join(ch if ch.isalnum() else "_" for ch in task).strip("_") or "unknown" for task in grouped)
    for metric in ("overall", "goal", "constraint", "soft", "best_objective_quality", "best_repair_quality"):
        for k in (1, 5, 10):
            values = [out[f"task_{task_key}_request_{metric}_at_{k}"] for task_key in task_keys if f"task_{task_key}_request_{metric}_at_{k}" in out]
            if values:
                out[f"macro_task_request_{metric}_at_{k}"] = float(np.mean(values))
    return out


def _task_request_topk(rows: list[dict[str, str]], prefix: str) -> dict[str, float]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("request_id", ""))].append(row)
    out = {f"{prefix}_requests": float(len(grouped))}
    for metric in ("overall_success", "goal_success", "constraint_success", "soft_success"):
        if not any(metric in row or (metric == "soft_success" and "soft_repair_success" in row) for row in rows):
            continue
        short = metric.replace("_success", "")
        for k in (1, 5, 10):
            values = []
            for request_rows in grouped.values():
                ordered = sorted(request_rows, key=lambda row: int(float(row.get("rank", 0) or 0)))
                values.append(max(_as_bool(row.get(metric, row.get("soft_repair_success", ""))) for row in ordered[:k]) if ordered else 0.0)
            out[f"{prefix}_request_{short}_at_{k}"] = float(np.mean(values)) if values else 0.0
    for metric in ("objective_quality", "repair_quality"):
        if not any(metric in row for row in rows):
            continue
        short = metric.replace("_quality", "")
        for k in (1, 5, 10):
            values = []
            for request_rows in grouped.values():
                ordered = sorted(request_rows, key=lambda row: int(float(row.get("rank", 0) or 0)))[:k]
                values.append(max(_as_float(row.get(metric, 0.0)) for row in ordered) if ordered else 0.0)
            out[f"{prefix}_request_best_{short}_quality_at_{k}"] = float(np.mean(values)) if values else 0.0
    return out


def _repair_breakdown(rows: list[dict[str, str]]) -> dict[str, float]:
    repair_rows = [row for row in rows if str(row.get("task_type", "")) == "repair"]
    if not repair_rows:
        return {}
    out = {"repair_rows": float(len(repair_rows))}
    for column in ("valid", "exact_recovery", "scaffold_recovery", "novel", "novel_verified_success", "soft_repair_success"):
        values = [_as_bool(row.get(column, "")) for row in repair_rows]
        out[f"repair_{column}"] = float(np.mean(values)) if values else 0.0
    out["repair_tanimoto_to_clean"] = _float_mean(repair_rows, "tanimoto_to_clean")
    out["repair_property_mae_to_clean"] = _float_mean(repair_rows, "property_mae_to_clean")
    out["repair_quality"] = _float_mean(repair_rows, "repair_quality")
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in repair_rows:
        grouped[str(row.get("request_id", ""))].append(row)
    metric_columns = {
        "repair_validity": "valid",
        "exact_recovery": "exact_recovery",
        "scaffold_recovery": "scaffold_recovery",
        "novel_repair_success": "novel_verified_success",
        "soft_repair_success": "soft_repair_success",
        "repair_overall": "overall_success",
    }
    for out_name, column in metric_columns.items():
        for k in (1, 5, 10):
            values = []
            for request_rows in grouped.values():
                ordered = sorted(request_rows, key=lambda row: int(float(row.get("rank", 0) or 0)))
                values.append(max(_as_bool(row.get(column, "")) for row in ordered[:k]) if ordered else 0.0)
            out[f"{out_name}_at_{k}"] = float(np.mean(values)) if values else 0.0
    for k in (1, 5, 10):
        best_tanimoto = []
        best_mae = []
        for request_rows in grouped.values():
            ordered = sorted(request_rows, key=lambda row: int(float(row.get("rank", 0) or 0)))[:k]
            if ordered:
                best_tanimoto.append(max(_as_float(row.get("tanimoto_to_clean", 0.0)) for row in ordered))
                best_mae.append(min(_as_float(row.get("property_mae_to_clean", "inf")) for row in ordered))
        out[f"best_tanimoto_to_clean_at_{k}"] = float(np.mean(best_tanimoto)) if best_tanimoto else 0.0
        out[f"best_property_mae_to_clean_at_{k}"] = float(np.mean(best_mae)) if best_mae else 0.0
        best_quality = []
        for request_rows in grouped.values():
            ordered = sorted(request_rows, key=lambda row: int(float(row.get("rank", 0) or 0)))[:k]
            if ordered:
                best_quality.append(max(_as_float(row.get("repair_quality", 0.0)) for row in ordered))
        out[f"best_repair_quality_at_{k}"] = float(np.mean(best_quality)) if best_quality else 0.0
    return out


def _failure_reasons(rows: list[dict[str, str]]) -> dict[str, float]:
    counts: Counter[str] = Counter()
    for row in rows:
        for reason in str(row.get("reasons", "")).split("|"):
            reason = reason.strip()
            if reason:
                counts[reason] += 1
    return {f"failure_reason_{reason}": float(count) for reason, count in counts.most_common(20)}


def _origin_breakdown(rows: list[dict[str, str]]) -> dict[str, float]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    family_grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        origins = str(row.get("candidate_origin", "") or "unknown").split("+")
        row_families = set()
        for origin in origins:
            origin = origin.strip() or "unknown"
            grouped[origin].append(row)
            row_families.add(_origin_family(origin))
        for family in row_families:
            family_grouped[family].append(row)
    out = {}
    for origin, origin_rows in grouped.items():
        key = "".join(ch if ch.isalnum() else "_" for ch in origin).strip("_") or "unknown"
        out[f"origin_{key}_rows"] = float(len(origin_rows))
        out[f"origin_{key}_overall_success"] = float(np.mean([_as_bool(row.get("overall_success", "")) for row in origin_rows]))
    for family, family_rows in family_grouped.items():
        key = "".join(ch if ch.isalnum() else "_" for ch in family).strip("_") or "unknown"
        out[f"origin_family_{key}_rows"] = float(len(family_rows))
        out[f"origin_family_{key}_overall_success"] = float(np.mean([_as_bool(row.get("overall_success", "")) for row in family_rows]))
    return out


def _origin_family(origin: str) -> str:
    if origin.startswith("source_guided_"):
        return "source_guided"
    if origin.startswith("graph_edit_"):
        return "graph_edit"
    if origin.startswith("graph_grow_"):
        return "graph_grow"
    if origin == "source_neighborhood":
        return "source_neighborhood"
    if origin == "scaffold_library":
        return "scaffold_library"
    if origin == "condition_direct":
        return "condition_direct"
    if origin.startswith("condition_blend_"):
        return "condition_blend"
    if origin == "string_repair_prior":
        return "string_repair_prior"
    if origin == "diffusion":
        return "diffusion"
    return origin or "unknown"


def _as_float(value: str | float | int) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _float_mean(rows: list[dict[str, str]], column: str) -> float:
    values = [_as_float(row.get(column, 0.0)) for row in rows]
    return float(np.mean(values)) if values else 0.0


if __name__ == "__main__":
    main()
