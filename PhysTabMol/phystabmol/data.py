"""Small starter datasets for quick PhysTabMol experiments."""

from __future__ import annotations

import pandas as pd

from .features import IMAGE_FEATURE_COLUMNS, descriptor_image, image_array_features, table_row_from_smiles
from .schema import TABLE_COLUMNS, TARGET_COLUMNS

STARTER_SMILES = [
    "CCOc1ccccc1",
    "CC(=O)Nc1ccccc1",
    "CCN(CC)CC",
    "CC(C)NCC(O)c1ccccc1",
    "COc1ccc(CCN)cc1",
    "CCOC(=O)c1ccccc1",
    "CC(C)Oc1ccc(O)cc1",
    "O=C(NC)c1ccccc1",
    "CCS(=O)(=O)N",
    "CC(C)CC(=O)O",
    "CCN1CCCCC1",
    "c1ccncc1",
    "c1ccoc1",
    "CC(C)N",
    "CC(C)(C)O",
    "CCCl",
    "CCBr",
    "CC(F)(F)F",
    "CC(=O)OCC",
    "CCN(CC)C(=O)C",
    "CC(C)CO",
    "CCCCN",
    "CCOC",
    "NCCO",
    "CC(C)C(=O)N",
    "CC(C)c1ccccc1",
    "COc1ccncc1",
    "CC(=O)NCCO",
    "CCCOC(=O)N",
    "CC(C)S",
]


def build_demo_dataframe(smiles: list[str] | None = None) -> pd.DataFrame:
    rows = []
    for smi in smiles or STARTER_SMILES:
        table = table_row_from_smiles(smi)
        img_features = image_array_features(descriptor_image(smi))
        row = {"smiles": smi}
        row.update(table)
        row.update(img_features)
        rows.append(row)
    return pd.DataFrame(rows)


def split_arrays(df: pd.DataFrame):
    image_x = df[IMAGE_FEATURE_COLUMNS].to_numpy(dtype=float)
    target_x = df[TARGET_COLUMNS].to_numpy(dtype=float)
    table_y = df[TABLE_COLUMNS].to_numpy(dtype=float)
    condition_x = pd.concat(
        [df[IMAGE_FEATURE_COLUMNS], df[TARGET_COLUMNS]],
        axis=1,
    ).to_numpy(dtype=float)
    return image_x, target_x, condition_x, table_y

