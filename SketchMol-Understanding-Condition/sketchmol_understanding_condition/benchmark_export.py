"""Export understanding-condition predictions in SketchMolBenchmark format."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from .chem import canonical_smiles, molecular_properties, morgan_fingerprint_bits, morgan_tanimoto, scaffold_smiles
from .retrieval_data import condition_features_for_row, read_variant_rows


PROPERTY_ALIASES = {
    "MolWt": "MW",
    "MW": "MW",
    "LogP": "LogP",
    "QED": "QED",
    "TPSA": "TPSA",
    "HBD": "HBD",
    "HBA": "HBA",
    "RB": "RB",
    "rotatable": "RB",
}

PROPERTY_SETTING_COLUMNS = {
    "LogP": ("logp_setting", "logp_None"),
    "QED": ("QED_setting", "QED_None"),
    "MW": ("MolWt_setting", "MolWt_None"),
    "TPSA": ("TPSA_setting", "TPSA_None"),
    "HBD": ("HBD_setting", "HBD_None"),
    "HBA": ("HBA_setting", "HBA_None"),
    "RB": ("rotatable_setting", "rotatable_None"),
}


@dataclass(frozen=True)
class BenchmarkExportConfig:
    variant: str = "full"
    fingerprint_bits: int = 512
    text_dim: int = 128
    random_dim: int = 128
    ridge_alpha: float = 1.0
    eval_split: str = "eval"
    condition_features_dir: str | Path | None = None


def export_ridge_benchmark_predictions(
    *,
    baseline_variants_csv: str | Path,
    output_csv: str | Path,
    config: BenchmarkExportConfig = BenchmarkExportConfig(),
) -> dict[str, object]:
    """Train a ridge condition-to-fingerprint probe and export eval predictions.

    The exported CSV keeps the source/target metadata from the understanding
    rows and adds `generated_smiles`. `SketchMolBenchmark` can then evaluate it
    with the same property-success logic used for real SketchMol+OCR rows.
    """

    rows = [row for row in read_variant_rows(baseline_variants_csv) if row.get("variant") == config.variant]
    if not rows:
        raise ValueError(f"No baseline rows found for variant={config.variant!r}")

    train_rows = [row for row in rows if row.get("split") != config.eval_split]
    eval_rows = [row for row in rows if row.get("split") == config.eval_split]
    feature_lookup = _load_condition_feature_lookup(config.condition_features_dir) if config.condition_features_dir else None
    train_x, train_y, train_targets, kept_train_rows = _featurize_rows(train_rows, config, feature_lookup=feature_lookup)
    eval_x, eval_y, _eval_targets, kept_eval_rows = _featurize_rows(eval_rows, config, feature_lookup=feature_lookup)
    if train_x.size == 0:
        raise ValueError(f"No valid train rows for variant={config.variant!r}")
    if eval_x.size == 0:
        raise ValueError(f"No valid eval rows for variant={config.variant!r}")

    train_x_std, eval_x_std = _standardize_train_eval(train_x, eval_x)
    eye = np.eye(train_x_std.shape[1], dtype=np.float32)
    weights = np.linalg.solve(
        train_x_std.T @ train_x_std + float(config.ridge_alpha) * eye,
        train_x_std.T @ train_y,
    )
    predicted = _sigmoid(eval_x_std @ weights)
    similarities = _soft_tanimoto_matrix(predicted, train_y)

    out_rows = []
    for idx, row in enumerate(kept_eval_rows):
        order = np.argsort(-similarities[idx])
        best_index = int(order[0])
        generated = train_targets[best_index]
        target = row.get("target_smiles", "")
        out_rows.append(
            _benchmark_row(
                row=row,
                generated_smiles=generated,
                score=float(similarities[idx, best_index]),
                target_fp_similarity=float(_soft_tanimoto_matrix(predicted[idx : idx + 1], eval_y[idx : idx + 1])[0, 0]),
            )
        )

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_rows(output_path, out_rows)

    summary = {
        "baseline_variants_csv": str(baseline_variants_csv),
        "output_csv": str(output_path),
        "variant": config.variant,
        "fingerprint_bits": config.fingerprint_bits,
        "text_dim": config.text_dim,
        "random_dim": config.random_dim,
        "ridge_alpha": config.ridge_alpha,
        "eval_split": config.eval_split,
        "condition_features_dir": str(config.condition_features_dir) if config.condition_features_dir else None,
        "train_rows": len(kept_train_rows),
        "eval_rows": len(kept_eval_rows),
        "candidate_pool": "train_target_smiles",
        "mean_train_pool_score": _mean(row.get("train_pool_fingerprint_score") for row in out_rows),
        "mean_target_fp_similarity": _mean(row.get("target_fingerprint_similarity") for row in out_rows),
        "mean_target_tanimoto": _mean(row.get("top1_target_tanimoto") for row in out_rows),
        "exact_target_match_rate": _fraction(row.get("exact_target_match") == "True" for row in out_rows),
        "scaffold_match_rate": _fraction(row.get("scaffold_match") == "True" for row in out_rows),
    }
    output_path.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _featurize_rows(
    rows: Iterable[dict[str, str]],
    config: BenchmarkExportConfig,
    *,
    feature_lookup: Mapping[str, np.ndarray] | None = None,
):
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    targets: list[str] = []
    kept: list[dict[str, str]] = []
    for row in rows:
        target = canonical_smiles(row.get("target_smiles", "")) or row.get("target_smiles", "")
        target_bits = morgan_fingerprint_bits(target, n_bits=config.fingerprint_bits)
        if target_bits is None:
            continue
        if feature_lookup is None:
            x = condition_features_for_row(
                row,
                variant=config.variant,
                fingerprint_bits=config.fingerprint_bits,
                text_dim=config.text_dim,
                random_dim=config.random_dim,
            )
        else:
            x = feature_lookup.get(row.get("variant_id", ""))
            if x is None:
                continue
        xs.append(x)
        ys.append(np.asarray(target_bits, dtype=np.float32))
        targets.append(target)
        kept.append(row)
    if not xs:
        return (
            np.zeros((0, config.fingerprint_bits), dtype=np.float32),
            np.zeros((0, config.fingerprint_bits), dtype=np.float32),
            [],
            [],
        )
    return np.stack(xs).astype(np.float32), np.stack(ys).astype(np.float32), targets, kept


def _load_condition_feature_lookup(condition_features_dir: str | Path) -> dict[str, np.ndarray]:
    feature_dir = Path(condition_features_dir)
    pooled = np.load(feature_dir / "pooled.npy").astype(np.float32)
    with (feature_dir / "index.csv").open(newline="", encoding="utf-8") as handle:
        index_rows = list(csv.DictReader(handle))
    if pooled.shape[0] != len(index_rows):
        raise ValueError(
            f"Condition feature row mismatch: pooled.npy has {pooled.shape[0]} rows, "
            f"index.csv has {len(index_rows)} rows"
        )
    lookup: dict[str, np.ndarray] = {}
    for row, feature in zip(index_rows, pooled):
        variant_id = row.get("variant_id", "")
        if variant_id:
            lookup[variant_id] = np.asarray(feature, dtype=np.float32)
    return lookup


def _benchmark_row(
    *,
    row: Mapping[str, str],
    generated_smiles: str,
    score: float,
    target_fp_similarity: float,
) -> dict[str, object]:
    target_smiles = canonical_smiles(row.get("target_smiles", "")) or row.get("target_smiles", "")
    generated = canonical_smiles(generated_smiles) or generated_smiles
    objective = _normalize_property(row.get("objective") or row.get("property_name"))
    condition_columns = _condition_columns_for_target(target_smiles, objective)
    target_tanimoto = morgan_tanimoto(target_smiles, generated)
    source_tanimoto = morgan_tanimoto(row.get("source_smiles", ""), generated)
    generated_scaffold = scaffold_smiles(generated) or ""
    target_scaffold = scaffold_smiles(target_smiles) or ""
    out = {
        "variant_id": row.get("variant_id", ""),
        "pair_id": row.get("pair_id", ""),
        "variant": row.get("variant", ""),
        "split": row.get("split", ""),
        "condition_mode": row.get("condition_mode", ""),
        "source_smiles": row.get("source_smiles", ""),
        "target_smiles": target_smiles,
        "generated_smiles": generated,
        "image_path": row.get("source_image", ""),
        "source_image": row.get("source_image", ""),
        "target_image": row.get("target_image", ""),
        "instruction": row.get("instruction", ""),
        "prompt": row.get("prompt", ""),
        "objective": row.get("objective", ""),
        "direction": row.get("direction", ""),
        "property_name": row.get("property_name", ""),
        "property_delta": row.get("property_delta", ""),
        "train_pool_fingerprint_score": score,
        "target_fingerprint_similarity": target_fp_similarity,
        "top1_target_tanimoto": target_tanimoto if target_tanimoto is not None else "",
        "source_tanimoto": source_tanimoto if source_tanimoto is not None else "",
        "target_scaffold": target_scaffold,
        "generated_scaffold": generated_scaffold,
        "scaffold_match": bool(target_scaffold and generated_scaffold and target_scaffold == generated_scaffold),
        "exact_target_match": bool(generated and generated == target_smiles),
        "active_property": objective or "",
    }
    out.update(condition_columns)
    return out


def _condition_columns_for_target(target_smiles: str, objective: str | None) -> dict[str, object]:
    columns: dict[str, object] = {}
    for _prop, (value_col, none_col) in PROPERTY_SETTING_COLUMNS.items():
        columns[value_col] = ""
        columns[none_col] = "True"
    if not objective:
        return columns
    props = molecular_properties(target_smiles) or {}
    prop_key = "MolWt" if objective == "MW" else ("rotatable" if objective == "RB" else objective)
    if prop_key not in props:
        return columns
    value_col, none_col = PROPERTY_SETTING_COLUMNS[objective]
    columns[value_col] = props[prop_key]
    columns[none_col] = "False"
    return columns


def _normalize_property(value: str | None) -> str | None:
    if not value:
        return None
    return PROPERTY_ALIASES.get(str(value).strip())


def _standardize_train_eval(train_x: np.ndarray, eval_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (train_x - mean) / std, (eval_x - mean) / std


def _soft_tanimoto_matrix(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    a = np.clip(a.astype(np.float32), 0.0, 1.0)
    b = np.clip(b.astype(np.float32), 0.0, 1.0)
    inter = a @ b.T
    denom = a.sum(axis=1, keepdims=True) + b.sum(axis=1, keepdims=True).T - inter
    return inter / np.maximum(denom, eps)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def _mean(values: Iterable[object]) -> float:
    numeric = [float(value) for value in values if value not in (None, "")]
    return float(np.mean(numeric)) if numeric else 0.0


def _fraction(values: Iterable[bool]) -> float:
    items = list(values)
    return sum(1 for item in items if item) / len(items) if items else 0.0


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
