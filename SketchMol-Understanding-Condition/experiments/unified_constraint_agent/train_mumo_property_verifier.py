#!/usr/bin/env python3
"""Train one independent train-only MuMO property verifier on CPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
PROJECT_SCRIPTS = PROJECT_DIR / "scripts"
for path in (SCRIPT_DIR, PROJECT_DIR, PROJECT_SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import export_external_multiproperty_benchmark_rows as export  # noqa: E402
import mumo_parallel_protocol as protocol  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", required=True, type=Path)
    parser.add_argument("--property", required=True, choices=protocol.PROPERTIES)
    parser.add_argument("--output-model", required=True, type=Path)
    parser.add_argument("--metrics-json", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=1711)
    parser.add_argument("--estimators", type=int, default=96)
    parser.add_argument("--min-samples-leaf", type=int, default=2)
    parser.add_argument("--max-fit-molecules", type=int, default=100000)
    parser.add_argument("--min-fit-labels", type=int, default=1000)
    parser.add_argument("--min-dev-labels", type=int, default=100)
    parser.add_argument("--jobs", type=int, default=8)
    return parser.parse_args(argv)


def stable_order(smiles: str, seed: int) -> bytes:
    return hashlib.sha256(f"{seed}:{smiles}".encode()).digest()


def safe_correlation(a: np.ndarray, b: np.ndarray, *, kind: str) -> float:
    if len(a) < 2 or float(np.std(a)) == 0.0 or float(np.std(b)) == 0.0:
        return 0.0
    if kind == "spearman":
        from scipy.stats import spearmanr

        value = spearmanr(a, b).correlation
    else:
        value = np.corrcoef(a, b)[0, 1]
    return float(value) if math.isfinite(float(value)) else 0.0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    files = sorted(args.feature_dir.glob("features_*.npz"))
    if not files:
        raise FileNotFoundError(f"No features_*.npz found in {args.feature_dir}")
    property_index = protocol.PROPERTIES.index(args.property)
    arrays: dict[str, list[np.ndarray]] = defaultdict(list)
    for path in files:
        with np.load(path, allow_pickle=False) as payload:
            names = tuple(str(value) for value in payload["property_names"])
            if names != protocol.PROPERTIES:
                raise ValueError(f"Property schema mismatch in {path}: {names}")
            for key in (
                "fingerprint",
                "descriptors",
                "labels",
                "smiles",
                "source_group",
                "task_id",
                "partition",
                "role",
                "pair_id",
            ):
                arrays[key].append(np.asarray(payload[key]))
    data = {key: np.concatenate(values, axis=0) for key, values in arrays.items()}
    labels = np.asarray(data["labels"][:, property_index], dtype=np.float32)
    feature_matrix = np.concatenate(
        [np.asarray(data["fingerprint"], dtype=np.float32), np.asarray(data["descriptors"], dtype=np.float32)],
        axis=1,
    )

    # Canonical-SMILES deduplication with median conflict resolution.  A
    # molecule observed in fit is removed from dev, so point-level validation
    # cannot benefit from an identical molecule in training.
    by_partition: dict[str, dict[str, list[int]]] = {
        "fit": defaultdict(list),
        "dev": defaultdict(list),
    }
    for index, (smiles, partition, value) in enumerate(zip(data["smiles"], data["partition"], labels)):
        part = str(partition)
        if part in by_partition and np.isfinite(value):
            by_partition[part][str(smiles)].append(index)
    fit_smiles = set(by_partition["fit"])
    overlap_smiles = fit_smiles & set(by_partition["dev"])
    for smiles in overlap_smiles:
        by_partition["dev"].pop(smiles, None)

    fit_keys = sorted(by_partition["fit"], key=lambda value: stable_order(value, int(args.seed)))
    if int(args.max_fit_molecules) > 0:
        fit_keys = fit_keys[: int(args.max_fit_molecules)]
    dev_keys = sorted(by_partition["dev"])
    if len(fit_keys) < int(args.min_fit_labels) or len(dev_keys) < int(args.min_dev_labels):
        raise ValueError(
            f"Insufficient {args.property} labels after dedup: fit={len(fit_keys)} dev={len(dev_keys)}"
        )

    def materialize(keys: list[str], partition: str) -> tuple[np.ndarray, np.ndarray]:
        indices = [by_partition[partition][key] for key in keys]
        representatives = [values[0] for values in indices]
        y = np.asarray([float(np.median(labels[values])) for values in indices], dtype=np.float32)
        return feature_matrix[representatives], y

    x_fit, y_fit = materialize(fit_keys, "fit")
    x_dev, y_dev = materialize(dev_keys, "dev")

    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    model = ExtraTreesRegressor(
        n_estimators=int(args.estimators),
        min_samples_leaf=int(args.min_samples_leaf),
        max_features=0.5,
        n_jobs=int(args.jobs),
        random_state=int(args.seed),
    )
    model.fit(x_fit, y_fit)
    dev_prediction = np.asarray(model.predict(x_dev), dtype=np.float32)

    # Pairwise threshold recall is the verifier metric that matters to the
    # planner: can it recognize a true requested improvement from source to
    # target using only predicted property values?
    dev_pair_rows: dict[str, dict[str, int]] = defaultdict(dict)
    for index, (pair_id, partition, role) in enumerate(
        zip(data["pair_id"], data["partition"], data["role"])
    ):
        if str(partition) != "dev" or not np.isfinite(labels[index]):
            continue
        dev_pair_rows[str(pair_id)][str(role)] = index
    pair_indices = [
        (roles["source"], roles["target"])
        for roles in dev_pair_rows.values()
        if "source" in roles and "target" in roles
        and str(data["smiles"][roles["source"]]) not in fit_smiles
        and str(data["smiles"][roles["target"]]) not in fit_smiles
    ]
    actual_positive: list[bool] = []
    predicted_positive: list[bool] = []
    if pair_indices:
        unique_indices = sorted({index for pair in pair_indices for index in pair})
        prediction_lookup = dict(
            zip(unique_indices, model.predict(feature_matrix[unique_indices]).astype(float))
        )
        direction = -1.0 if export.DEFAULT_DIRECTION[args.property] == "decrease" else 1.0
        threshold = float(export.MUMO_THRESHOLDS[args.property])
        for source_index, target_index in pair_indices:
            actual_delta = direction * float(labels[target_index] - labels[source_index])
            predicted_delta = direction * float(
                prediction_lookup[target_index] - prediction_lookup[source_index]
            )
            actual_positive.append(actual_delta >= threshold)
            predicted_positive.append(predicted_delta >= threshold)
    true_positive = sum(actual and predicted for actual, predicted in zip(actual_positive, predicted_positive))
    false_negative = sum(actual and not predicted for actual, predicted in zip(actual_positive, predicted_positive))
    false_positive = sum(not actual and predicted for actual, predicted in zip(actual_positive, predicted_positive))
    true_negative = sum(not actual and not predicted for actual, predicted in zip(actual_positive, predicted_positive))

    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "protocol": protocol.PROTOCOL_VERSION,
            "property": args.property,
            "property_index": property_index,
            "fingerprint_bits": int(data["fingerprint"].shape[1]),
            "descriptor_count": int(data["descriptors"].shape[1]),
            "estimator": model,
        },
        args.output_model,
        compress=3,
    )
    metrics = {
        "protocol": protocol.PROTOCOL_VERSION,
        "stage": "property_verifier",
        "data_role": "fit_train_labels_to_disjoint_train_dev",
        "evaluation_target_access": False,
        "evaluation_oracle_access": False,
        "property": args.property,
        "feature_shards": len(files),
        "raw_labeled_rows": int(np.isfinite(labels).sum()),
        "fit_unique_molecules": len(fit_keys),
        "dev_unique_molecules": len(dev_keys),
        "fit_dev_identical_smiles_removed": len(overlap_smiles),
        "mae": float(mean_absolute_error(y_dev, dev_prediction)),
        "rmse": float(mean_squared_error(y_dev, dev_prediction, squared=False)),
        "spearman": safe_correlation(y_dev, dev_prediction, kind="spearman"),
        "pearson": safe_correlation(y_dev, dev_prediction, kind="pearson"),
        "pairwise": {
            "eligible_dev_pairs": len(pair_indices),
            "actual_positive_pairs": int(sum(actual_positive)),
            "predicted_positive_pairs": int(sum(predicted_positive)),
            "true_positive": int(true_positive),
            "false_negative": int(false_negative),
            "false_positive": int(false_positive),
            "true_negative": int(true_negative),
            "threshold_recall": float(true_positive / max(true_positive + false_negative, 1)),
            "threshold_precision": float(true_positive / max(true_positive + false_positive, 1)),
            "threshold_accuracy": float((true_positive + true_negative) / max(len(pair_indices), 1)),
            "direction": export.DEFAULT_DIRECTION[args.property],
            "threshold": float(export.MUMO_THRESHOLDS[args.property]),
        },
        "model": {
            "kind": "ExtraTreesRegressor",
            "estimators": int(args.estimators),
            "min_samples_leaf": int(args.min_samples_leaf),
            "max_features": 0.5,
            "seed": int(args.seed),
        },
    }
    protocol.write_json(args.metrics_json, metrics)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
