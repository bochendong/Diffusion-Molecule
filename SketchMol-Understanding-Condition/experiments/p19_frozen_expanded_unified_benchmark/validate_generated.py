#!/usr/bin/env python3
"""Validate exact CSV candidate cardinality without using newline counts."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path


def validate(path: Path, conditions: int, label: str) -> None:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != conditions * 8:
        raise AssertionError(f"{path}: expected {conditions * 8} CSV records, got {len(rows)}")
    grouped: dict[str, list[int]] = defaultdict(list)
    labels = Counter()
    for row in rows:
        key = str(row.get("condition_id") or row.get("sample_id") or "")
        grouped[key].append(int(float(row["candidate_rank"])))
        labels[str(row.get("method", ""))] += 1
    if len(grouped) != conditions:
        raise AssertionError(f"{path}: expected {conditions} conditions, got {len(grouped)}")
    if any(sorted(ranks) != list(range(1, 9)) for ranks in grouped.values()):
        raise AssertionError(f"{path}: each condition must have candidate ranks 1..8")
    if set(labels) != {label}:
        raise AssertionError(f"{path}: unexpected labels {dict(labels)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated-dir", required=True, type=Path)
    parser.add_argument("--model", required=True, choices=("p17", "p18"))
    args = parser.parse_args()
    label = f"{args.model}_frozen_expanded"
    validate(args.generated_dir / "table1.raw.csv", 100, label)
    validate(args.generated_dir / "denovo.raw.csv", 40, label)
    print(f"validated {args.model}: 100x8 Table1 and 40x8 de-novo ordered records")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
