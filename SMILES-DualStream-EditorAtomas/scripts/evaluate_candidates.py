#!/usr/bin/env python3
"""Evaluate a CSV of generated SMILES against target/source SMILES."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from smiles_dual_stream.data import write_jsonl, write_summary  # noqa: E402
from smiles_dual_stream.metrics import evaluate_prediction_row, summarize_prediction_rows  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)

    with args.predictions_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    decoded = [evaluate_prediction_row(row) for row in rows]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(decoded, args.output_dir / "decoded.jsonl")
    write_summary(summarize_prediction_rows(rows), args.output_dir / "metrics.json")
    print(f"wrote evaluation to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

