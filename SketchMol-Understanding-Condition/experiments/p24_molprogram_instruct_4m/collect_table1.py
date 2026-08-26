#!/usr/bin/env python3
"""Collect one complete frozen P24 de-novo Table 1 row."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


def cells(path: Path) -> dict[int, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {
        int(row["property_count"]): row
        for row in rows
        if row["setting"] == "best_of_40" and row["property_count"] != "all"
        and int(row["conditions"]) > 0
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", action="append", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    merged: dict[int, dict[str, str]] = {}
    for path in args.summary:
        incoming = cells(path)
        overlap = set(merged) & set(incoming)
        if overlap:
            raise AssertionError(f"duplicate P24 cells: {sorted(overlap)}")
        merged.update(incoming)
    if set(merged) != set(range(2, 8)):
        raise AssertionError(f"missing P24 cells: {sorted(set(range(2, 8)) - set(merged))}")
    counts = {f"{count}p": int(merged[count]["conditions"]) for count in range(2, 8)}
    expected = {"2p": 100, "3p": 100, "4p": 100, "5p": 100, "6p": 20, "7p": 20}
    if counts != expected:
        raise AssertionError(f"P24 condition counts {counts} != {expected}")
    values = {f"{count}p": float(merged[count]["strict_success_rate"]) for count in range(2, 8)}
    result = {
        "protocol": "p24_frozen_denovo_table1_best40_v1",
        "model": "MolProgram P24 balanced refresh",
        "train_data": "2,000,000/569,919",
        "setting": "best_of_40",
        "conditions": counts,
        "strict_success": values,
        "average_2p_7p": statistics.fmean(values.values()),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "p24_table1.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    cells_text = " | ".join(f"{100 * values[f'{count}p']:.1f}" for count in range(2, 8))
    markdown = "\n".join([
        "# P24 frozen de-novo Table 1 row", "",
        "| Model | Train data | Avg 2p-7p | 2p | 3p | 4p | 5p | 6p | 7p |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| MolProgram P24 balanced refresh | 2M/569,919 | {100 * result['average_2p_7p']:.1f} | {cells_text} |",
    ]) + "\n"
    (args.output_dir / "p24_table1.md").write_text(markdown)
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
