#!/usr/bin/env python3
"""Collect the missing de-novo cells into complete legacy/aligned Table 1 rows."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


def best40(path: Path) -> dict[int, dict[str, str]]:
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
    parser.add_argument("--legacy-2p4p", required=True, type=Path)
    parser.add_argument("--legacy-5p", required=True, type=Path)
    parser.add_argument("--legacy-6p7p", required=True, type=Path)
    parser.add_argument("--aligned-2p4p", required=True, type=Path)
    parser.add_argument("--aligned-5p", required=True, type=Path)
    parser.add_argument("--aligned-6p7p", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    sources = {
        "legacy": [args.legacy_2p4p, args.legacy_5p, args.legacy_6p7p],
        "aligned": [args.aligned_2p4p, args.aligned_5p, args.aligned_6p7p],
    }
    result: dict[str, object] = {
        "protocol": "p23_paper_table1_denovo_missing_cells_v1",
        "setting": "best_of_40",
        "models": {},
    }
    for model, paths in sources.items():
        cells: dict[int, dict[str, str]] = {}
        for path in paths:
            overlap = set(cells) & set(best40(path))
            if overlap:
                raise AssertionError(f"duplicate {model} cells: {sorted(overlap)}")
            cells.update(best40(path))
        if set(cells) != set(range(2, 8)):
            raise AssertionError(f"{model} missing cells: {sorted(set(range(2, 8)) - set(cells))}")
        values = {f"{count}p": float(cells[count]["strict_success_rate"]) for count in range(2, 8)}
        counts = {f"{count}p": int(cells[count]["conditions"]) for count in range(2, 8)}
        expected_counts = {"2p": 100, "3p": 100, "4p": 100, "5p": 100, "6p": 20, "7p": 20}
        if counts != expected_counts:
            raise AssertionError(f"{model} condition counts {counts} != {expected_counts}")
        models = result["models"]
        assert isinstance(models, dict)
        models[model] = {
            "strict_success": values,
            "conditions": counts,
            "mean_2p_4p": statistics.fmean(values[f"{count}p"] for count in range(2, 5)),
            "average_2p_7p": statistics.fmean(values.values()),
        }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "denovo_table1_fill.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# P23 paper Table 1 de-novo fill", "",
        "| Model | Train data | Avg 2p-7p | 2p | 3p | 4p | 5p | 6p | 7p |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model, label, train in (("legacy", "MolProgram legacy", "160/160"), ("aligned", "MolProgram aligned 24k", "12k/12k")):
        models = result["models"]
        assert isinstance(models, dict)
        payload = models[model]
        assert isinstance(payload, dict)
        values = payload["strict_success"]
        assert isinstance(values, dict)
        lines.append(
            f"| {label} | {train} | {100 * payload['average_2p_7p']:.1f} | "
            + " | ".join(f"{100 * values[f'{count}p']:.1f}" for count in range(2, 8)) + " |"
        )
    (args.output_dir / "denovo_table1_fill.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
