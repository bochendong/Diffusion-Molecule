#!/usr/bin/env python3
"""Honest raw candidate and pass@k metrics for the P6 hard de-novo gate."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-csv", required=True, type=Path)
    parser.add_argument("--candidates-csv", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", required=True, type=Path)
    parser.add_argument("--budgets", default="1,8,20")
    args = parser.parse_args()
    budgets = [int(value) for value in args.budgets.split(",")]
    with args.eval_csv.open(newline="", encoding="utf-8") as handle:
        eval_rows = list(csv.DictReader(handle))
    with args.candidates_csv.open(newline="", encoding="utf-8") as handle:
        candidate_rows = list(csv.DictReader(handle))
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in candidate_rows:
        key = str(row.get("condition_id") or row.get("sample_id") or "")
        grouped[key].append(row)
    for values in grouped.values():
        values.sort(key=lambda row: int(float(row.get("direct_candidate_index") or 0)))
    records = []
    for stratum in ("all", "6p", "7p"):
        conditions = [
            row for row in eval_rows
            if stratum == "all" or f"{int(float(row.get('property_count') or 0))}p" == stratum
        ]
        for budget in budgets:
            raw, passed, validity, unique = [], [], [], []
            for row in conditions:
                key = str(row.get("condition_id") or row.get("sample_id") or "")
                values = grouped.get(key, [])[:budget]
                strict = [
                    bool(item.get("direct_candidate_canonical_smiles"))
                    and math.isclose(float(item.get("direct_candidate_strict_fraction") or 0), 1.0, abs_tol=1e-9)
                    for item in values
                ]
                valid = [bool(item.get("direct_candidate_canonical_smiles")) for item in values]
                canonicals = [item.get("direct_candidate_canonical_smiles", "") for item in values if item.get("direct_candidate_canonical_smiles")]
                denominator = max(budget, 1)
                raw.append(sum(strict) / denominator)
                passed.append(float(any(strict)))
                validity.append(sum(valid) / denominator)
                unique.append(len(set(canonicals)) / denominator)
            records.append(
                {
                    "stratum": stratum,
                    "k": budget,
                    "conditions": len(conditions),
                    "raw_success_fraction": mean(raw) if raw else 0.0,
                    "pass_at_k": mean(passed) if passed else 0.0,
                    "validity_fraction": mean(validity) if validity else 0.0,
                    "unique_fraction": mean(unique) if unique else 0.0,
                }
            )
    payload = {"protocol": "p6_hard_denovo_raw_gate", "records": records}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# P6 hard de-novo gate", "",
        "| stratum | k | raw success | pass@k | validity | unique |", "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in records:
        lines.append(
            f"| {row['stratum']} | {row['k']} | {row['raw_success_fraction']:.3f} | "
            f"{row['pass_at_k']:.3f} | {row['validity_fraction']:.3f} | {row['unique_fraction']:.3f} |"
        )
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
