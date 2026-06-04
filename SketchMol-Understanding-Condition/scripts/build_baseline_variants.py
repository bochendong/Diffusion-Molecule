#!/usr/bin/env python
"""Expand edit-pair manifests into baseline condition variants."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sketchmol_understanding_condition.baselines import (
    BASELINE_VARIANTS,
    build_baseline_rows,
    read_edit_pair_rows,
    write_baseline_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edit-pairs-csv", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    edit_rows = read_edit_pair_rows(args.edit_pairs_csv)
    baseline_rows = build_baseline_rows(edit_rows)
    write_baseline_rows(args.output_csv, baseline_rows)
    summary = {
        "edit_pairs": len(edit_rows),
        "baseline_rows": len(baseline_rows),
        "variants": list(BASELINE_VARIANTS),
        "output_csv": str(args.output_csv),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
