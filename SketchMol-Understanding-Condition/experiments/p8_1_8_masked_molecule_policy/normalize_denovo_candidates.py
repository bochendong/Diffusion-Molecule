#!/usr/bin/env python3
"""Backfill the legacy de-novo evaluator fields without changing candidates."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    with args.input.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    counters: dict[str, int] = {}
    for row in rows:
        key = str(row.get("condition_id", "") or row.get("sample_id", ""))
        index = counters.get(key, 0)
        counters[key] = index + 1
        smiles = str(row.get("generated_smiles", "") or row.get("candidate_smiles", ""))
        row["direct_candidate_index"] = str(index)
        row["direct_candidate_raw_smiles"] = smiles
        row["direct_candidate_canonical_smiles"] = smiles
    fields = list(dict.fromkeys(key for row in rows for key in row))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"normalized_candidates={len(rows)} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

