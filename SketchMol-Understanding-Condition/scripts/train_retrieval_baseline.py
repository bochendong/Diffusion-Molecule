#!/usr/bin/env python
"""Train retrieval-style ablation baselines for understanding-condition rows."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from sketchmol_understanding_condition.retrieval_baseline import (
    RetrievalConfig,
    train_and_eval_ridge_variant,
)
from sketchmol_understanding_condition.retrieval_data import VARIANTS, build_retrieval_matrices, read_variant_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-variants-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--fingerprint-bits", type=int, default=512)
    parser.add_argument("--text-dim", type=int, default=128)
    parser.add_argument("--random-dim", type=int, default=128)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_variant_rows(args.baseline_variants_csv)
    variants = [item.strip() for item in args.variants.split(",") if item.strip()]
    config = RetrievalConfig(seed=args.seed, ridge_alpha=args.ridge_alpha)
    metrics = []
    for variant in variants:
        matrices = build_retrieval_matrices(
            rows,
            variant=variant,
            fingerprint_bits=args.fingerprint_bits,
            text_dim=args.text_dim,
            random_dim=args.random_dim,
        )
        metrics.append(train_and_eval_ridge_variant(matrices, config))

    payload = {
        "baseline_variants_csv": str(args.baseline_variants_csv),
        "fingerprint_bits": args.fingerprint_bits,
        "text_dim": args.text_dim,
        "random_dim": args.random_dim,
        "ridge_alpha": args.ridge_alpha,
        "variants": variants,
        "metrics": metrics,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_metrics_csv(args.output_dir / "metrics.csv", metrics)
    print(json.dumps(payload, indent=2, sort_keys=True))


def _write_metrics_csv(path: Path, rows: list[dict]) -> None:
    keys = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                keys.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


if __name__ == "__main__":
    main()
