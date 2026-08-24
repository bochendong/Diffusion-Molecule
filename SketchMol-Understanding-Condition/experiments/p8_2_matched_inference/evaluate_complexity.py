#!/usr/bin/env python3
"""Evaluate a complete raw-order 2p--7p complexity curve."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def key(row: dict[str, str]) -> str:
    return str(row.get("condition_id") or row.get("sample_id") or "").strip()


def candidate_index(row: dict[str, str]) -> int:
    return int(float(row.get("direct_candidate_index") or row.get("candidate_index") or 0))


def canonical(row: dict[str, str]) -> str:
    return str(row.get("direct_candidate_canonical_smiles") or "").strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-csv", required=True, type=Path)
    parser.add_argument("--candidates", required=True, nargs="+", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--budgets", default="1,4,8,20")
    args = parser.parse_args()
    budgets = [int(value) for value in args.budgets.split(",") if value.strip()]
    eval_rows = read(args.eval_csv)
    candidate_rows = [row for path in args.candidates for row in read(path)]
    forbidden = {"target_smiles", "target_scaffold", "target_image"}
    if any(forbidden.intersection(row) for row in candidate_rows):
        raise SystemExit("structural evaluation target leaked into an inference candidate row")

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in candidate_rows:
        grouped[key(row)].append(row)
    eval_by_id = {key(row): row for row in eval_rows}
    if set(grouped) != set(eval_by_id):
        raise SystemExit(f"candidate condition mismatch missing={len(set(eval_by_id)-set(grouped))} extra={len(set(grouped)-set(eval_by_id))}")
    for condition_id, values in grouped.items():
        values.sort(key=candidate_index)
        if len(values) != 20 or [candidate_index(row) for row in values] != list(range(20)):
            raise SystemExit(f"{condition_id}: expected ordered candidate indices 0..19")

    records = []
    for property_count in ("all", 2, 3, 4, 5, 6, 7):
        references = [row for row in eval_rows if property_count == "all" or int(float(row["property_count"])) == property_count]
        for budget in budgets:
            raw_success, passed, raw_validity, any_valid, unique = [], [], [], [], []
            for reference in references:
                prefix = grouped[key(reference)][:budget]
                valid = [bool(canonical(row)) for row in prefix]
                strict = [
                    bool(canonical(row))
                    and math.isclose(float(row.get("direct_candidate_strict_fraction") or 0), 1.0, abs_tol=1e-9)
                    for row in prefix
                ]
                canonicals = [canonical(row) for row in prefix if canonical(row)]
                raw_success.append(sum(strict) / budget)
                passed.append(float(any(strict)))
                raw_validity.append(sum(valid) / budget)
                any_valid.append(float(any(valid)))
                unique.append(len(set(canonicals)) / budget)
            records.append({
                "property_count": property_count,
                "k": budget,
                "conditions": len(references),
                "raw_success_fraction": mean(raw_success),
                "pass_at_k": mean(passed),
                "candidate_validity": mean(raw_validity),
                "any_valid_at_k": mean(any_valid),
                "unique_fraction": mean(unique),
            })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "candidates_merged.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(candidate_rows[0]))
        writer.writeheader()
        writer.writerows(candidate_rows)
    with (args.output_dir / "complexity_curve.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    payload = {"protocol": "p8_2_complete_2p7p_raw_curve_v1", "records": records}
    (args.output_dir / "complexity_curve.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    lines = ["# P8.2 complete 2p--7p raw complexity curve", "", "| property count | k | conditions | raw success | pass@k | candidate validity | any valid | unique |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in records:
        lines.append(f"| {row['property_count']} | {row['k']} | {row['conditions']} | {row['raw_success_fraction']:.4f} | {row['pass_at_k']:.4f} | {row['candidate_validity']:.4f} | {row['any_valid_at_k']:.4f} | {row['unique_fraction']:.4f} |")
    (args.output_dir / "complexity_curve.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
