#!/usr/bin/env python
"""Export understanding-condition predictions for SketchMolBenchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sketchmol_understanding_condition.benchmark_export import (
    BenchmarkExportConfig,
    export_ridge_benchmark_predictions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-variants-csv", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--variant", default="full")
    parser.add_argument("--fingerprint-bits", type=int, default=512)
    parser.add_argument("--text-dim", type=int, default=128)
    parser.add_argument("--random-dim", type=int, default=128)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--eval-split", default="eval")
    parser.add_argument("--condition-features-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = export_ridge_benchmark_predictions(
        baseline_variants_csv=args.baseline_variants_csv,
        output_csv=args.output_csv,
        config=BenchmarkExportConfig(
            variant=args.variant,
            fingerprint_bits=args.fingerprint_bits,
            text_dim=args.text_dim,
            random_dim=args.random_dim,
            ridge_alpha=args.ridge_alpha,
            eval_split=args.eval_split,
            condition_features_dir=args.condition_features_dir,
        ),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
