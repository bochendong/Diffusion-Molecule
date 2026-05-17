"""Dataset loading for server-scale PhysTabMol experiments."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .chem import canonicalize_smiles
from .data import build_demo_dataframe
from .features import IMAGE_FEATURE_COLUMNS, descriptor_image, extract_image_features, image_array_features, table_row_from_smiles
from .progress import iter_progress
from .schema import TABLE_COLUMNS, TARGET_COLUMNS


def load_experiment_dataframe(
    data_path: str | None,
    smiles_column: str = "smiles",
    image_column: str | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    if data_path is None:
        return build_demo_dataframe()

    source = pd.read_csv(data_path)
    if limit:
        source = source.head(limit)
    if smiles_column not in source.columns:
        raise ValueError(f"Missing SMILES column '{smiles_column}' in {data_path}")

    rows = []
    for idx, raw in iter_progress(source.iterrows(), total=len(source), label="loading experiment molecules"):
        smi = canonicalize_smiles(str(raw[smiles_column]))
        if smi is None:
            continue
        try:
            table = table_row_from_smiles(smi)
        except Exception:
            continue

        img_features = _row_image_features(raw, image_column, smi)
        row = {
            "row_id": raw.get("row_id", idx),
            "smiles": smi,
        }
        row.update(table)
        row.update(img_features)
        rows.append(row)

    if not rows:
        raise ValueError("No valid molecules found after parsing the dataset.")
    return pd.DataFrame(rows)


def train_test_split_df(df: pd.DataFrame, test_fraction: float = 0.2, seed: int = 7) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(df))
    n_test = max(1, int(round(len(df) * test_fraction))) if len(df) > 1 else 0
    test_idx = perm[:n_test]
    train_idx = perm[n_test:] if n_test else perm
    if len(train_idx) == 0:
        train_idx = test_idx
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)


def arrays_from_dataframe(df: pd.DataFrame):
    image_x = df[IMAGE_FEATURE_COLUMNS].to_numpy(dtype=float)
    target_x = df[TARGET_COLUMNS].to_numpy(dtype=float)
    table_y = df[TABLE_COLUMNS].to_numpy(dtype=float)
    condition_x = pd.concat([df[IMAGE_FEATURE_COLUMNS], df[TARGET_COLUMNS]], axis=1).to_numpy(dtype=float)
    return image_x, target_x, condition_x, table_y


def _row_image_features(row, image_column: str | None, smiles: str) -> dict[str, float]:
    if image_column and image_column in row and not pd.isna(row[image_column]):
        image_path = Path(str(row[image_column]))
        if image_path.exists():
            return extract_image_features(image_path)
    return image_array_features(descriptor_image(smiles))
