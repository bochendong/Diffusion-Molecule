#!/usr/bin/env python3
"""Freeze a small deterministic 6p/7p subset for the P1 low-budget kill test."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--conditions-per-count", type=int, default=128)
    parser.add_argument("--selection-seed", type=int, default=20260823)
    return parser.parse_args()


def condition_id(row: dict[str, str]) -> str:
    return str(row.get("condition_id") or row.get("sample_id") or "").strip()


def rank(seed: int, key: str) -> str:
    return hashlib.sha256(f"{seed}:{key}".encode("utf-8")).hexdigest()


def main() -> int:
    args = parse_args()
    with args.input_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    selected: list[dict[str, str]] = []
    available = Counter(int(float(row.get("property_count") or 0)) for row in rows)
    for count in (6, 7):
        candidates = [row for row in rows if int(float(row.get("property_count") or 0)) == count]
        candidates.sort(key=lambda row: (rank(args.selection_seed, condition_id(row)), condition_id(row)))
        if len(candidates) < args.conditions_per_count:
            raise RuntimeError(f"Need {args.conditions_per_count} {count}p conditions, found {len(candidates)}")
        selected.extend(candidates[: args.conditions_per_count])
    selected.sort(key=lambda row: (int(float(row.get("property_count") or 0)), condition_id(row)))
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected)
    manifest = {
        "protocol": "p1_fast_hard_6p7p_k20_kill_test_v1",
        "claim_scope": "interim_kill_test_not_final_preregistered_p1_result",
        "selection_seed": args.selection_seed,
        "generation_seed": 7,
        "conditions_per_property_count": args.conditions_per_count,
        "selected_counts": dict(Counter(str(int(float(row["property_count"]))) for row in selected)),
        "available_counts": {str(key): value for key, value in sorted(available.items())},
        "candidate_budget": 20,
        "reported_budgets": [1, 4, 8, 20],
        "input_sha256": hashlib.sha256(args.input_csv.read_bytes()).hexdigest(),
        "condition_ids": [condition_id(row) for row in selected],
    }
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output_csv), **manifest["selected_counts"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
