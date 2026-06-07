#!/usr/bin/env python
"""Export source-conditioned edit rows for understanding-conditioned diffusion training."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path


REPO_DIR = Path(__file__).resolve().parents[2]
DATASET_DIR = REPO_DIR / "SketchMol-MultiProperty-EditDataset"
UNDERSTANDING_DIR = REPO_DIR / "SketchMol-Understanding-Condition"
for path in (DATASET_DIR, UNDERSTANDING_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from sketchmol_multiproperty_dataset.common import PROPERTY_COLUMNS, SKETCHMOL_SETTING_COLUMNS
from sketchmol_understanding_condition.chem import morgan_tanimoto


BASE_FIELDNAMES = (
    "sample_id",
    "condition_id",
    "pair_id",
    "split",
    "source_smiles",
    "target_smiles",
    "source_image",
    "target_image",
    "instruction",
    "prompt",
    "condition_properties",
    "property_count",
    "source_tanimoto",
    "source_similarity_bin",
    "generation_target",
    "conditioning_mode",
)

QUALITY_FIELDNAMES = (
    "source_scaffold",
    "target_scaffold",
    "same_scaffold",
    "scaffold_relation",
    "pair_quality_tier",
    "selection_reason",
    "same_scaffold_neighbor_count",
    "source_neighbor_count_t04",
    "source_neighbor_count_t05",
    "source_neighbor_count_t06",
    "target_neighbor_rank_by_tanimoto",
    "candidate_pool_size_t04",
    "candidate_pool_size_t05",
    "candidate_pool_size_t06",
    "strict_candidate_count_t04",
    "strict_candidate_count_t05",
    "strict_candidate_count_t06",
    "oracle_candidate_smiles_t04",
    "oracle_source_tanimoto_t04",
    "oracle_strict_success_t04",
    "oracle_property_error_t04",
    "oracle_property_errors_json_t04",
    "source_identity_strict_success",
    "instruction_template_id",
    "instruction_style",
    "preservation_constraint",
    "property_constraints_json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition-rows-csv", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--min-source-tanimoto", type=float, default=0.0)
    parser.add_argument("--max-source-tanimoto", type=float, default=1.0)
    parser.add_argument("--include-splits", default="train,eval")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    include_splits = {item.strip() for item in args.include_splits.split(",") if item.strip()}
    rows = []
    skipped_similarity = 0
    skipped_split = 0
    for row in _read_rows(args.condition_rows_csv):
        if include_splits and row.get("split", "") not in include_splits:
            skipped_split += 1
            continue
        source_tanimoto = _source_tanimoto(row)
        if math.isnan(source_tanimoto) or not (args.min_source_tanimoto <= source_tanimoto <= args.max_source_tanimoto):
            skipped_similarity += 1
            continue
        rows.append(_manifest_row(row, source_tanimoto=source_tanimoto))
        if args.limit is not None and len(rows) >= args.limit:
            break

    _write_rows(args.output_csv, rows)
    summary = {
        "condition_rows_csv": str(args.condition_rows_csv),
        "output_csv": str(args.output_csv),
        "rows": len(rows),
        "include_splits": sorted(include_splits),
        "min_source_tanimoto": args.min_source_tanimoto,
        "max_source_tanimoto": args.max_source_tanimoto,
        "skipped_split": skipped_split,
        "skipped_similarity": skipped_similarity,
        "train_rows": sum(1 for row in rows if row["split"] == "train"),
        "eval_rows": sum(1 for row in rows if row["split"] == "eval"),
        "source_similarity_bins": _bin_counts(rows),
    }
    args.output_csv.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _source_tanimoto(row: dict[str, str]) -> float:
    value = _to_float(row.get("similarity"))
    if not math.isnan(value):
        return value
    try:
        computed = morgan_tanimoto(row.get("source_smiles", ""), row.get("target_smiles", ""))
    except RuntimeError:
        computed = None
    return float(computed) if computed is not None else math.nan


def _manifest_row(row: dict[str, str], *, source_tanimoto: float) -> dict[str, object]:
    out: dict[str, object] = {
        "sample_id": row.get("condition_id", ""),
        "condition_id": row.get("condition_id", ""),
        "pair_id": row.get("pair_id", ""),
        "split": row.get("split", ""),
        "source_smiles": row.get("source_smiles", ""),
        "target_smiles": row.get("target_smiles", ""),
        "source_image": row.get("source_image", ""),
        "target_image": row.get("target_image", ""),
        "instruction": row.get("instruction") or row.get("prompt", ""),
        "prompt": row.get("instruction") or row.get("prompt", ""),
        "condition_properties": row.get("condition_properties", ""),
        "property_count": row.get("property_count", ""),
        "source_tanimoto": source_tanimoto,
        "source_similarity_bin": _similarity_bin(source_tanimoto),
        "generation_target": "target_molecule_image_or_structure",
        "conditioning_mode": "source_image_plus_instruction",
    }
    for prop in PROPERTY_COLUMNS:
        out[f"source_{prop}"] = row.get(f"source_{prop}", "")
        out[f"target_{prop}"] = row.get(f"target_{prop}", "")
        out[f"delta_{prop}"] = row.get(f"delta_{prop}", "")
        out[f"{prop}_active"] = row.get(f"{prop}_active", "")
        out[f"{prop}_direction"] = row.get(f"{prop}_direction", "")
        value_col, none_col = SKETCHMOL_SETTING_COLUMNS[prop]
        out[value_col] = row.get(value_col, "")
        out[none_col] = row.get(none_col, "")
    for field in QUALITY_FIELDNAMES:
        out[field] = row.get(field, "")
    return out


def _similarity_bin(value: float) -> str:
    if value >= 0.7:
        return "easy_high_similarity"
    if value >= 0.5:
        return "medium_similarity"
    if value >= 0.4:
        return "hard_similarity"
    return "exploratory_low_similarity"


def _bin_counts(rows: list[dict[str, object]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in rows:
        key = str(row.get("source_similarity_bin", ""))
        out[key] = out.get(key, 0) + 1
    return out


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = _manifest_fieldnames()
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _manifest_fieldnames() -> list[str]:
    fieldnames = list(BASE_FIELDNAMES)
    fieldnames.extend(QUALITY_FIELDNAMES)
    for prop in PROPERTY_COLUMNS:
        fieldnames.extend(
            [
                f"source_{prop}",
                f"target_{prop}",
                f"delta_{prop}",
                f"{prop}_active",
                f"{prop}_direction",
            ]
        )
        fieldnames.extend(SKETCHMOL_SETTING_COLUMNS[prop])
    return fieldnames


def _to_float(value: object) -> float:
    try:
        return float(str(value if value is not None else "").strip())
    except ValueError:
        return math.nan


if __name__ == "__main__":
    main()
