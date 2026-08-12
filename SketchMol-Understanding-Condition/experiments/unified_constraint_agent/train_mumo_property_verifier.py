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


def paired_indices(
    data: dict[str, np.ndarray],
    labels: np.ndarray,
    *,
    partition: str,
    excluded_smiles: set[str] | None = None,
) -> list[tuple[int, int]]:
    rows: dict[str, dict[str, int]] = defaultdict(dict)
    excluded = excluded_smiles or set()
    for index, (pair_id, raw_partition, role) in enumerate(
        zip(data["pair_id"], data["partition"], data["role"])
    ):
        if str(raw_partition) != str(partition) or not np.isfinite(labels[index]):
            continue
        rows[str(pair_id)][str(role)] = index
    output = []
    for roles in rows.values():
        if "source" not in roles or "target" not in roles:
            continue
        source_index = roles["source"]
        target_index = roles["target"]
        if str(data["smiles"][source_index]) in excluded or str(data["smiles"][target_index]) in excluded:
            continue
        output.append((source_index, target_index))
    return output


def pair_feature_matrix(
    feature_matrix: np.ndarray,
    pairs: Sequence[tuple[int, int]],
) -> np.ndarray:
    if not pairs:
        return np.empty((0, int(feature_matrix.shape[1]) * 3), dtype=np.float32)
    source_indices = np.asarray([source for source, _target in pairs], dtype=np.int64)
    target_indices = np.asarray([target for _source, target in pairs], dtype=np.int64)
    source = np.asarray(feature_matrix[source_indices], dtype=np.float32)
    target = np.asarray(feature_matrix[target_indices], dtype=np.float32)
    return np.concatenate([source, target, target - source], axis=1).astype(np.float32, copy=False)


def threshold_labels(
    labels: np.ndarray,
    pairs: Sequence[tuple[int, int]],
    *,
    direction: float,
    threshold: float,
) -> np.ndarray:
    return np.asarray(
        [direction * float(labels[target] - labels[source]) >= threshold for source, target in pairs],
        dtype=bool,
    )


def pairwise_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    *,
    direction_name: str,
    threshold: float,
    decision_source: str,
) -> dict[str, object]:
    actual = np.asarray(actual, dtype=bool)
    predicted = np.asarray(predicted, dtype=bool)
    true_positive = int(np.logical_and(actual, predicted).sum())
    false_negative = int(np.logical_and(actual, np.logical_not(predicted)).sum())
    false_positive = int(np.logical_and(np.logical_not(actual), predicted).sum())
    true_negative = int(np.logical_and(np.logical_not(actual), np.logical_not(predicted)).sum())
    return {
        "eligible_dev_pairs": int(len(actual)),
        "actual_positive_pairs": int(actual.sum()),
        "predicted_positive_pairs": int(predicted.sum()),
        "true_positive": true_positive,
        "false_negative": false_negative,
        "false_positive": false_positive,
        "true_negative": true_negative,
        "threshold_recall": float(true_positive / max(true_positive + false_negative, 1)),
        "threshold_precision": float(true_positive / max(true_positive + false_positive, 1)),
        "threshold_accuracy": float((true_positive + true_negative) / max(len(actual), 1)),
        "direction": direction_name,
        "threshold": float(threshold),
        "decision_source": decision_source,
    }


def calibrated_decision_threshold(
    actual: np.ndarray,
    probability: np.ndarray,
    *,
    target_recall: float = 0.90,
    min_precision: float = 0.80,
) -> float:
    """Choose the most conservative threshold meeting fit-only calibration gates."""

    actual = np.asarray(actual, dtype=bool)
    probability = np.asarray(probability, dtype=float)
    candidates = sorted({0.0, 0.5, 1.0, *(float(value) for value in probability)})
    eligible: list[float] = []
    for threshold in candidates:
        predicted = probability >= threshold
        metrics = pairwise_metrics(
            actual,
            predicted,
            direction_name="calibration",
            threshold=threshold,
            decision_source="fit_only_calibration",
        )
        if (
            float(metrics["threshold_recall"]) >= float(target_recall)
            and float(metrics["threshold_precision"]) >= float(min_precision)
        ):
            eligible.append(float(threshold))
    return max(eligible) if eligible else 0.5


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

    from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
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

    # A value regressor is useful for numeric margins, but threshold crossing
    # is a pairwise decision. Train a second fit-only classifier directly on
    # source -> candidate edits so the planner does not inherit conservative
    # regression-to-the-mean at the MuMO boundary.
    direction_name = export.DEFAULT_DIRECTION[args.property]
    direction = -1.0 if direction_name == "decrease" else 1.0
    threshold = float(export.MUMO_THRESHOLDS[args.property])
    fit_pairs = paired_indices(data, labels, partition="fit")
    dev_pairs = paired_indices(data, labels, partition="dev", excluded_smiles=fit_smiles)
    if len(fit_pairs) < int(args.min_fit_labels) or len(dev_pairs) < int(args.min_dev_labels):
        raise ValueError(
            f"Insufficient {args.property} pairs: fit={len(fit_pairs)} dev={len(dev_pairs)}"
        )
    fit_pair_labels = threshold_labels(
        labels,
        fit_pairs,
        direction=direction,
        threshold=threshold,
    )
    dev_pair_labels = threshold_labels(
        labels,
        dev_pairs,
        direction=direction,
        threshold=threshold,
    )
    if len(np.unique(fit_pair_labels)) != 2:
        raise ValueError(f"{args.property} fit pair labels need both classes")
    pair_features_fit = pair_feature_matrix(feature_matrix, fit_pairs)
    calibration_mask = np.asarray(
        [
            stable_order(str(data["pair_id"][source]), int(args.seed) + 37)[0] < 51
            for source, _target in fit_pairs
        ],
        dtype=bool,
    )
    if int(calibration_mask.sum()) < 100 or int((~calibration_mask).sum()) < 100:
        raise ValueError(f"{args.property} fit-only calibration split is too small")
    calibration_estimator = ExtraTreesClassifier(
        n_estimators=int(args.estimators),
        min_samples_leaf=int(args.min_samples_leaf),
        max_features=0.5,
        class_weight="balanced",
        n_jobs=int(args.jobs),
        random_state=int(args.seed),
    )
    calibration_estimator.fit(pair_features_fit[~calibration_mask], fit_pair_labels[~calibration_mask])
    calibration_probabilities = calibration_estimator.predict_proba(pair_features_fit[calibration_mask])
    calibration_positive_column = list(calibration_estimator.classes_).index(True)
    decision_threshold = calibrated_decision_threshold(
        fit_pair_labels[calibration_mask],
        calibration_probabilities[:, calibration_positive_column],
        target_recall=0.90,
        min_precision=0.80,
    )

    pair_classifier = ExtraTreesClassifier(
        n_estimators=int(args.estimators),
        min_samples_leaf=int(args.min_samples_leaf),
        max_features=0.5,
        class_weight="balanced",
        n_jobs=int(args.jobs),
        random_state=int(args.seed),
    )
    pair_classifier.fit(pair_features_fit, fit_pair_labels)
    pair_probability = pair_classifier.predict_proba(pair_feature_matrix(feature_matrix, dev_pairs))
    positive_column = list(pair_classifier.classes_).index(True)
    direct_pair_prediction = np.asarray(
        pair_probability[:, positive_column] >= decision_threshold,
        dtype=bool,
    )

    absolute_pair_prediction: list[bool] = []
    if dev_pairs:
        unique_indices = sorted({index for pair in dev_pairs for index in pair})
        prediction_lookup = dict(
            zip(unique_indices, model.predict(feature_matrix[unique_indices]).astype(float))
        )
        for source_index, target_index in dev_pairs:
            predicted_delta = direction * float(
                prediction_lookup[target_index] - prediction_lookup[source_index]
            )
            absolute_pair_prediction.append(predicted_delta >= threshold)

    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "protocol": protocol.PROTOCOL_VERSION,
            "property": args.property,
            "property_index": property_index,
            "fingerprint_bits": int(data["fingerprint"].shape[1]),
            "descriptor_count": int(data["descriptors"].shape[1]),
            "estimator": model,
            "pair_classifier": pair_classifier,
            "pair_feature_schema": "source,target,target_minus_source",
            "pair_threshold": threshold,
            "pair_direction": direction_name,
            "pair_decision_threshold": decision_threshold,
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
        "rmse": float(math.sqrt(mean_squared_error(y_dev, dev_prediction))),
        "spearman": safe_correlation(y_dev, dev_prediction, kind="spearman"),
        "pearson": safe_correlation(y_dev, dev_prediction, kind="pearson"),
        "pairwise": pairwise_metrics(
            dev_pair_labels,
            direct_pair_prediction,
            direction_name=direction_name,
            threshold=threshold,
            decision_source="direct_pair_classifier",
        ),
        "absolute_value_pairwise": pairwise_metrics(
            dev_pair_labels,
            np.asarray(absolute_pair_prediction, dtype=bool),
            direction_name=direction_name,
            threshold=threshold,
            decision_source="difference_of_absolute_regressor_predictions",
        ),
        "model": {
            "kind": "ExtraTreesRegressor+ExtraTreesClassifier",
            "estimators": int(args.estimators),
            "min_samples_leaf": int(args.min_samples_leaf),
            "max_features": 0.5,
            "pair_class_weight": "balanced",
            "fit_pairs": len(fit_pairs),
            "fit_only_calibration_pairs": int(calibration_mask.sum()),
            "pair_decision_threshold": decision_threshold,
            "calibration_target_recall": 0.90,
            "calibration_min_precision": 0.80,
            "seed": int(args.seed),
        },
    }
    protocol.write_json(args.metrics_json, metrics)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
