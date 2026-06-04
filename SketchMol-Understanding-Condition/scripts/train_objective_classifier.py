#!/usr/bin/env python
"""Train objective-direction classifiers for baseline variants."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from sketchmol_understanding_condition.baselines import BASELINE_VARIANTS
from sketchmol_understanding_condition.condition_feature_store import load_exported_features
from sketchmol_understanding_condition.objective_classifier import (
    ObjectiveClassifierConfig,
    load_rows,
    train_and_eval_objective_variant,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-variants-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--variants", default=",".join(BASELINE_VARIANTS))
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--l2", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--condition-features-dir", type=Path, default=None)
    parser.add_argument("--feature-array", choices=["pooled", "query_tokens"], default="pooled")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.baseline_variants_csv)
    variants = [item.strip() for item in args.variants.split(",") if item.strip()]
    config = ObjectiveClassifierConfig(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        l2=args.l2,
        seed=args.seed,
    )
    features_by_variant_id = None
    if args.condition_features_dir is not None:
        features_by_variant_id = load_exported_features(args.condition_features_dir, array_name=args.feature_array)
    metrics = [
        train_and_eval_objective_variant(
            rows,
            variant=variant,
            config=config,
            features_by_variant_id=features_by_variant_id,
        )
        for variant in variants
    ]
    payload = {
        "baseline_variants_csv": str(args.baseline_variants_csv),
        "condition_features_dir": str(args.condition_features_dir) if args.condition_features_dir else None,
        "feature_array": args.feature_array if args.condition_features_dir else None,
        "config": {
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "l2": args.l2,
            "seed": args.seed,
        },
        "metrics": metrics,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_flat_csv(args.output_dir / "metrics.csv", metrics)
    print(json.dumps(payload, indent=2, sort_keys=True))


def _write_flat_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = ["variant", "accuracy", "macro_f1", "train_examples", "eval_examples", "feature_dim", "final_train_loss"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


if __name__ == "__main__":
    main()
