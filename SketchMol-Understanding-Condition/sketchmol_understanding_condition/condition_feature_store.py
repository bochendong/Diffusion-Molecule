"""Load exported condition encoder features for downstream probes."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def load_exported_features(
    feature_dir: str | Path,
    *,
    array_name: str = "pooled",
) -> dict[str, np.ndarray]:
    """Map ``variant_id`` to exported pooled or flattened query-token features."""

    feature_dir = Path(feature_dir)
    index_path = feature_dir / "index.csv"
    if array_name == "pooled":
        array_path = feature_dir / "pooled.npy"
    elif array_name in {"query_tokens", "query"}:
        array_path = feature_dir / "query_tokens.npy"
    else:
        raise ValueError(f"Unsupported feature array: {array_name}")

    with index_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    features = np.load(array_path)
    if len(rows) != int(features.shape[0]):
        raise ValueError(
            f"Feature row mismatch: index has {len(rows)} rows but {array_path.name} has {features.shape[0]}"
        )
    flat = features.reshape(features.shape[0], -1).astype(np.float32)
    out: dict[str, np.ndarray] = {}
    for row, vec in zip(rows, flat):
        variant_id = row.get("variant_id", "")
        if not variant_id:
            raise ValueError(f"Missing variant_id in {index_path}")
        out[variant_id] = vec
    return out
