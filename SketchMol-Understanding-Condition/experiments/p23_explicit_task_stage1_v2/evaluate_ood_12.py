#!/usr/bin/env python3
"""Score the frozen SketchMol 12-target OOD set at candidate level."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

from sketchmol_understanding_condition.chem import canonical_smiles, molecular_properties


TOLERANCE = {"LogP": 1.0, "TPSA": 20.0, "HBA": 1.0, "RB": 1.0}
VALUE_KEY = {"LogP": "LogP", "TPSA": "TPSA", "HBA": "HBA", "RB": "rotatable"}


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


@lru_cache(maxsize=10_000)
def measure(smiles: str, prop: str) -> tuple[bool, float]:
    try:
        canonical = canonical_smiles(smiles) or ""
        values = molecular_properties(canonical) if canonical else None
    except RuntimeError:
        return False, math.nan
    if not canonical or not values:
        return False, math.nan
    return True, float(values.get(VALUE_KEY[prop], math.nan))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    references = {row["condition_id"]: row for row in read(args.reference)}
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in read(args.candidates):
        grouped[row["condition_id"]].append(row)
    details: list[dict[str, object]] = []
    for condition_id, ref in references.items():
        prop = ref["condition_properties"]
        target = float(ref[f"target_{prop}"])
        rows = grouped.get(condition_id, [])
        if len(rows) != 40:
            raise AssertionError(f"{condition_id}: expected 40 candidates, found {len(rows)}")
        valid = success = 0
        for row in rows:
            is_valid, actual = measure(row.get("SMILES", ""), prop)
            passed = is_valid and not math.isnan(actual) and abs(actual - target) <= TOLERANCE[prop]
            valid += int(is_valid)
            success += int(passed)
        details.append(
            {
                "condition_id": condition_id,
                "property": prop,
                "target": target,
                "candidates": len(rows),
                "validity": valid / len(rows),
                "success_valid": success / valid if valid else 0.0,
                "strict_all": success / len(rows),
            }
        )

    def mean(rows: list[dict[str, object]], key: str) -> float:
        return statistics.fmean(float(row[key]) for row in rows)

    per_property = {}
    for prop in TOLERANCE:
        rows = [row for row in details if row["property"] == prop]
        per_property[prop] = {
            "targets": len(rows),
            "success_valid": mean(rows, "success_valid"),
            "validity": mean(rows, "validity"),
            "strict_all": mean(rows, "strict_all"),
        }
    summary = {
        "protocol": "p23_sketchmol_supp_table3_candidate_level_raw40_v1",
        "conditions": len(details),
        "candidates_per_condition": 40,
        "property_aware_selection": False,
        "tolerances": TOLERANCE,
        "per_property": per_property,
        "macro_success_valid": mean(details, "success_valid"),
        "macro_validity": mean(details, "validity"),
        "macro_strict_all": mean(details, "strict_all"),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = args.output_dir / "ood_target_metrics.csv"
    with detail_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(details[0]))
        writer.writeheader()
        writer.writerows(details)
    (args.output_dir / "ood_metrics.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# P23 SketchMol 12-target OOD raw-40 evaluation", "",
        "| Property | Success-valid | Validity | Strict-all |",
        "| --- | ---: | ---: | ---: |",
    ]
    for prop, values in per_property.items():
        lines.append(
            f"| {prop} | {values['success_valid']:.3f} | {values['validity']:.3f} | {values['strict_all']:.3f} |"
        )
    lines.append(
        f"| Macro | {summary['macro_success_valid']:.3f} | {summary['macro_validity']:.3f} | {summary['macro_strict_all']:.3f} |"
    )
    (args.output_dir / "ood_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
