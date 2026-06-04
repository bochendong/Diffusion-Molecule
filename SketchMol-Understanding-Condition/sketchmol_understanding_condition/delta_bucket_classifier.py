"""Structure-aware objective plus delta-magnitude classifier."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .objective_classifier import _classification_metrics
from .retrieval_data import condition_features_for_row, read_variant_rows


@dataclass(frozen=True)
class DeltaBucketConfig:
    fingerprint_bits: int = 512
    text_dim: int = 128
    random_dim: int = 128
    epochs: int = 500
    learning_rate: float = 0.1
    l2: float = 1e-3
    seed: int = 7


def train_and_eval_delta_bucket_variant(
    rows: list[dict[str, str]],
    *,
    variant: str,
    config: DeltaBucketConfig,
    features_by_variant_id: dict[str, np.ndarray] | None = None,
) -> dict[str, object]:
    selected = [row for row in rows if row.get("variant") == variant]
    train_rows = [row for row in selected if row.get("split") == "train"]
    eval_rows = [row for row in selected if row.get("split") == "eval"]
    thresholds = _fit_bucket_thresholds(train_rows)

    labels = sorted({_label(row, thresholds) for row in selected})
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    train_x, train_y = _build_xy(train_rows, variant, label_to_idx, thresholds, config, features_by_variant_id)
    eval_x, eval_y = _build_xy(eval_rows, variant, label_to_idx, thresholds, config, features_by_variant_id)
    train_x, eval_x = _standardize(train_x, eval_x)

    weights, bias, final_loss = _fit_softmax(
        train_x,
        train_y,
        num_classes=len(labels),
        config=config,
    )
    pred = (eval_x @ weights + bias).argmax(axis=1)
    metrics = _classification_metrics(eval_y, pred, labels)
    metrics.update(
        {
            "variant": variant,
            "train_examples": int(train_x.shape[0]),
            "eval_examples": int(eval_x.shape[0]),
            "feature_dim": int(train_x.shape[1]),
            "classes": labels,
            "bucket_thresholds": thresholds,
            "final_train_loss": float(final_loss),
        }
    )
    return metrics


def load_rows(path) -> list[dict[str, str]]:
    return read_variant_rows(path)


def _fit_bucket_thresholds(train_rows: list[dict[str, str]]) -> dict[str, list[float]]:
    by_objective: dict[str, list[float]] = {}
    for row in train_rows:
        key = _objective_key(row)
        by_objective.setdefault(key, []).append(abs(_safe_float(row.get("property_delta"))))
    thresholds = {}
    for key, values in by_objective.items():
        arr = np.asarray(values, dtype=np.float32)
        if arr.size < 3:
            thresholds[key] = [float(arr.min(initial=0.0)), float(arr.max(initial=0.0))]
        else:
            thresholds[key] = [float(np.quantile(arr, 1 / 3)), float(np.quantile(arr, 2 / 3))]
    return thresholds


def _build_xy(rows, variant, label_to_idx, thresholds, config, features_by_variant_id=None):
    xs = []
    ys = []
    for row in rows:
        if features_by_variant_id is None:
            xs.append(
                condition_features_for_row(
                    row,
                    variant=variant,
                    fingerprint_bits=config.fingerprint_bits,
                    text_dim=config.text_dim,
                    random_dim=config.random_dim,
                )
            )
        else:
            variant_id = row.get("variant_id", "")
            try:
                xs.append(features_by_variant_id[variant_id])
            except KeyError as exc:
                raise KeyError(f"Missing exported feature for variant_id={variant_id}") from exc
        ys.append(label_to_idx[_label(row, thresholds)])
    if not xs:
        raise ValueError(f"No rows for variant={variant}")
    return np.stack(xs).astype(np.float32), np.asarray(ys, dtype=np.int64)


def _label(row: dict[str, str], thresholds: dict[str, list[float]]) -> str:
    key = _objective_key(row)
    delta = abs(_safe_float(row.get("property_delta")))
    low, high = thresholds.get(key, [0.0, 0.0])
    if delta <= low:
        bucket = "small"
    elif delta <= high:
        bucket = "medium"
    else:
        bucket = "large"
    return f"{key}_{bucket}"


def _objective_key(row: dict[str, str]) -> str:
    objective = row.get("objective") or row.get("property_name") or "unknown"
    direction = row.get("direction") or ("increase" if _safe_float(row.get("property_delta")) >= 0 else "decrease")
    return f"{objective}_{direction}"


def _fit_softmax(x, y, *, num_classes, config):
    rng = np.random.default_rng(config.seed)
    weights = rng.normal(0.0, 0.01, size=(x.shape[1], num_classes)).astype(np.float32)
    bias = np.zeros(num_classes, dtype=np.float32)
    y_onehot = np.eye(num_classes, dtype=np.float32)[y]
    final_loss = 0.0
    for _ in range(config.epochs):
        logits = x @ weights + bias
        probs = _softmax(logits)
        ce = -np.sum(y_onehot * np.log(np.clip(probs, 1e-8, 1.0))) / x.shape[0]
        reg = 0.5 * config.l2 * float(np.sum(weights * weights))
        final_loss = float(ce + reg)
        grad = (probs - y_onehot) / x.shape[0]
        weights -= config.learning_rate * (x.T @ grad + config.l2 * weights)
        bias -= config.learning_rate * grad.sum(axis=0)
    return weights, bias, final_loss


def _standardize(train_x, eval_x):
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (train_x - mean) / std, (eval_x - mean) / std


def _softmax(logits):
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def _safe_float(value) -> float:
    try:
        return float(str(value or "0").strip())
    except ValueError:
        return 0.0
