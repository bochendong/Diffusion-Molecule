#!/usr/bin/env python
"""Run diagnostics for edit-pair and baseline-variant manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sketchmol_understanding_condition.diagnostics import summarize_edit_dataset, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edit-pairs-csv", required=True, type=Path)
    parser.add_argument("--baseline-variants-csv", type=Path, default=None)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--property-name", default="QED")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = summarize_edit_dataset(
        args.edit_pairs_csv,
        baseline_variants_csv=args.baseline_variants_csv,
        property_name=args.property_name,
    )
    write_json(args.output_json, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
