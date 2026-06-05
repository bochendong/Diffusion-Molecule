#!/usr/bin/env python
"""Train a lightweight connector on frozen HF VLM condition features."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np


REPO_DIR = Path(__file__).resolve().parents[2]
DATASET_DIR = REPO_DIR / "SketchMol-MultiProperty-EditDataset"
UNDERSTANDING_DIR = REPO_DIR / "SketchMol-Understanding-Condition"
for path in (DATASET_DIR, UNDERSTANDING_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from sketchmol_multiproperty_dataset.common import PROPERTY_COLUMNS
from sketchmol_understanding_condition.chem import morgan_fingerprint_bits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition-features-dir", required=True, type=Path)
    parser.add_argument("--condition-rows-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--export-batch-size", type=int, default=8192)
    parser.add_argument("--source-fingerprint-bits", type=int, default=256)
    parser.add_argument("--source-feature-weight", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    index_rows = _read_rows(args.condition_features_dir / "index.csv")
    rows_by_condition_id = {row["condition_id"]: row for row in _read_rows(args.condition_rows_csv)}
    features = np.load(args.condition_features_dir / "pooled.npy", mmap_mode="r")
    if int(features.shape[0]) != len(index_rows):
        raise ValueError("Feature/index row mismatch")

    targets = np.stack([_target_vector(rows_by_condition_id[row["condition_id"]]) for row in index_rows]).astype(
        np.float32
    )
    source_features = np.stack(
        [
            _source_vector(
                rows_by_condition_id[row["condition_id"]],
                fingerprint_bits=args.source_fingerprint_bits,
            )
            for row in index_rows
        ]
    ).astype(np.float32)
    splits = np.asarray([rows_by_condition_id[row["condition_id"]].get("split", "") for row in index_rows])
    train_indices = np.flatnonzero(splits != "eval")
    eval_indices = np.flatnonzero(splits == "eval")
    if args.train_limit is not None and args.train_limit > 0:
        rng = np.random.default_rng(args.seed)
        train_indices = np.sort(rng.choice(train_indices, size=min(args.train_limit, len(train_indices)), replace=False))

    prop_count = len(PROPERTY_COLUMNS)
    active_mask = targets[:, prop_count : 2 * prop_count]
    target_mean = np.zeros((1, prop_count), dtype=np.float32)
    target_std = np.ones((1, prop_count), dtype=np.float32)
    for prop_idx in range(prop_count):
        active_train = train_indices[active_mask[train_indices, prop_idx] > 0.5]
        if active_train.size == 0:
            continue
        values = targets[active_train, prop_idx]
        target_mean[0, prop_idx] = float(values.mean())
        std = float(values.std())
        target_std[0, prop_idx] = std if std >= 1e-6 else 1.0
    targets[:, :prop_count] = ((targets[:, :prop_count] - target_mean) / target_std) * active_mask
    source_mean = source_features[train_indices, :prop_count].mean(axis=0, keepdims=True)
    source_std = source_features[train_indices, :prop_count].std(axis=0, keepdims=True)
    source_std = np.where(source_std < 1e-6, 1.0, source_std)
    source_features[:, :prop_count] = (source_features[:, :prop_count] - source_mean) / source_std

    summary = _train_and_export(
        features=features,
        source_features=source_features,
        targets=targets,
        train_indices=train_indices,
        eval_indices=eval_indices,
        output_dir=args.output_dir,
        source_index=args.condition_features_dir / "index.csv",
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        hidden_dim=args.hidden_dim,
        export_batch_size=args.export_batch_size,
        source_feature_weight=args.source_feature_weight,
        seed=args.seed,
    )
    summary.update(
        {
            "condition_features_dir": str(args.condition_features_dir),
            "condition_rows_csv": str(args.condition_rows_csv),
            "output_dir": str(args.output_dir),
            "properties": list(PROPERTY_COLUMNS),
            "target_mean": target_mean.reshape(-1).astype(float).tolist(),
            "target_std": target_std.reshape(-1).astype(float).tolist(),
            "source_feature_dim": int(source_features.shape[1]),
            "source_feature_weight": float(args.source_feature_weight),
            "source_fingerprint_bits": int(args.source_fingerprint_bits),
            "source_mean": source_mean.reshape(-1).astype(float).tolist(),
            "source_std": source_std.reshape(-1).astype(float).tolist(),
            "train_rows": int(len(train_indices)),
            "eval_rows": int(len(eval_indices)),
        }
    )
    (args.output_dir / "metrics.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _target_vector(row: dict[str, str]) -> np.ndarray:
    selected = {prop for prop in (row.get("condition_properties") or "").split(",") if prop}
    values = []
    masks = []
    directions = []
    for prop in PROPERTY_COLUMNS:
        source = _to_float(row.get(f"source_{prop}"))
        target = _to_float(row.get(f"target_{prop}"))
        active = prop in selected
        values.append(target if active else 0.0)
        masks.append(1.0 if active else 0.0)
        if active:
            directions.append(1.0 if target - source >= 0 else -1.0)
        else:
            directions.append(0.0)
    return np.asarray([*values, *masks, *directions], dtype=np.float32)


def _source_vector(row: dict[str, str], *, fingerprint_bits: int) -> np.ndarray:
    props = [_to_float(row.get(f"source_{prop}")) for prop in PROPERTY_COLUMNS]
    bits = morgan_fingerprint_bits(row.get("source_smiles", ""), n_bits=fingerprint_bits)
    if bits is None:
        bits = [0.0] * int(fingerprint_bits)
    return np.asarray([*props, *bits], dtype=np.float32)


def _to_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _train_and_export(
    *,
    features: np.ndarray,
    source_features: np.ndarray,
    targets: np.ndarray,
    train_indices: np.ndarray,
    eval_indices: np.ndarray,
    output_dir: Path,
    source_index: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    hidden_dim: int,
    export_batch_size: int,
    source_feature_weight: float,
    seed: int,
) -> dict[str, object]:
    import torch
    import torch.nn.functional as F
    from torch import nn
    from torch.utils.data import DataLoader, Dataset

    torch.manual_seed(seed)
    input_dim = int(features.shape[1] + source_features.shape[1])
    output_dim = int(targets.shape[1])

    class FeatureDataset(Dataset):
        def __init__(self, indices: np.ndarray) -> None:
            self.indices = indices.astype(np.int64)

        def __len__(self) -> int:
            return int(self.indices.shape[0])

        def __getitem__(self, item: int):
            idx = int(self.indices[item])
            x = np.concatenate(
                [
                    np.array(features[idx], dtype=np.float32, copy=True),
                    np.array(source_features[idx], dtype=np.float32, copy=True),
                ]
            )
            return torch.from_numpy(x), torch.from_numpy(targets[idx])

    model = nn.Sequential(
        nn.LayerNorm(input_dim),
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, output_dim),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    train_loader = DataLoader(FeatureDataset(train_indices), batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(FeatureDataset(eval_indices), batch_size=batch_size, shuffle=False)

    history = []
    for _epoch in range(int(epochs)):
        model.train()
        train_loss = 0.0
        train_total = 0
        for x, y in train_loader:
            optimizer.zero_grad(set_to_none=True)
            pred = model(x.float())
            loss = F.mse_loss(pred, y.float())
            loss.backward()
            optimizer.step()
            count = int(x.shape[0])
            train_loss += float(loss.item()) * count
            train_total += count
        history.append(
            {
                "train_mse": train_loss / max(1, train_total),
                "eval_mse": _eval_mse(model, eval_loader),
            }
        )

    model.eval()
    out_path = output_dir / "pooled.npy"
    export_dim = output_dim + (int(source_features.shape[1]) if source_feature_weight > 0 else 0)
    connected = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float32, shape=(int(features.shape[0]), export_dim))
    with torch.no_grad():
        for start in range(0, int(features.shape[0]), int(export_batch_size)):
            end = min(start + int(export_batch_size), int(features.shape[0]))
            x_np = np.concatenate(
                [
                    np.asarray(features[start:end], dtype=np.float32),
                    np.asarray(source_features[start:end], dtype=np.float32),
                ],
                axis=1,
            )
            x = torch.from_numpy(x_np)
            pred = model(x.float()).cpu().numpy().astype(np.float32)
            norm = np.linalg.norm(pred, axis=1, keepdims=True)
            pred = pred / np.maximum(norm, 1e-6)
            if source_feature_weight > 0:
                source = np.asarray(source_features[start:end], dtype=np.float32)
                source_norm = np.linalg.norm(source, axis=1, keepdims=True)
                source = source / np.maximum(source_norm, 1e-6)
                pred = np.concatenate([pred, float(source_feature_weight) * source], axis=1)
                norm = np.linalg.norm(pred, axis=1, keepdims=True)
                pred = pred / np.maximum(norm, 1e-6)
            connected[start:end] = pred
    connected.flush()
    shutil.copy2(source_index, output_dir / "index.csv")
    torch.save(
        {
            "model_state": model.state_dict(),
            "input_dim": input_dim,
            "hidden_dim": int(hidden_dim),
            "output_dim": output_dim,
            "export_dim": export_dim,
            "source_feature_weight": float(source_feature_weight),
            "history": history,
        },
        output_dir / "vlm_feature_connector.pt",
    )
    return {
        "checkpoint": str(output_dir / "vlm_feature_connector.pt"),
        "pooled_npy": str(out_path),
        "index_csv": str(output_dir / "index.csv"),
        "input_dim": input_dim,
        "hidden_dim": int(hidden_dim),
        "output_dim": output_dim,
        "export_dim": export_dim,
        "source_feature_weight": float(source_feature_weight),
        "epochs": int(epochs),
        "history": history,
    }


def _eval_mse(model, loader) -> float:
    import numpy as np
    import torch
    import torch.nn.functional as F

    model.eval()
    losses = []
    weights = []
    with torch.no_grad():
        for x, y in loader:
            pred = model(x.float())
            losses.append(float(F.mse_loss(pred, y.float()).item()))
            weights.append(int(x.shape[0]))
    if not losses:
        return 0.0
    return float(np.average(losses, weights=weights))


if __name__ == "__main__":
    main()
