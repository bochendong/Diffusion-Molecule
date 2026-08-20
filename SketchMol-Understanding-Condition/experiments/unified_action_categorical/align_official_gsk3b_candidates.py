#!/usr/bin/env python3
"""Align existing n=20 candidates onto the official 40-row GSK3B pack by source SMILES."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-reference", required=True, type=Path)
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    official = list(read_rows(args.official_reference))
    by_source: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in read_rows(args.candidates):
        task = str(row.get("task", "") or "")
        if task and "GSK3B" not in task:
            continue
        source = str(row.get("source_smiles", "") or "").strip()
        if source:
            by_source[source].append(row)
    out: list[dict[str, str]] = []
    missing = 0
    for ref in official:
        source = str(ref.get("source_smiles", "") or "").strip()
        ref_id = str(ref.get("example_id") or ref.get("condition_id") or "").strip()
        pool = by_source.get(source, [])
        if not pool or not ref_id:
            missing += 1
            continue
        for item in pool:
            aligned = dict(item)
            aligned["example_id"] = ref_id
            aligned["condition_id"] = ref_id
            aligned["source_smiles"] = source
            out.append(aligned)
    if missing:
        raise SystemExit(f"official GSK3B rows without candidates: {missing}/{len(official)}")
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = list(out[0]) if out else []
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(out)
    print(
        {
            "official_rows": len(official),
            "candidate_rows": len(out),
            "output_csv": str(args.output_csv),
        }
    )
    return 0


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


if __name__ == "__main__":
    raise SystemExit(main())
