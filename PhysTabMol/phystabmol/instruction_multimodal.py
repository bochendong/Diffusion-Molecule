"""Multimodal context features for instruction-guided molecular editing.

The first implementation uses modalities that can be built automatically from
SMILES: rendered/proxy molecule images, optional reference molecule images, and
RDKit-backed 3D conformer descriptors when RDKit is available.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .chem import canonicalize_smiles, tanimoto
from .features import IMAGE_FEATURE_COLUMNS, descriptor_image, image_array_features
from .geometry3d import molecule_3d_metrics


MULTIMODAL_CONTEXT_MODES = (
    "none",
    "source_image",
    "source_reference",
    "source_3d",
    "full",
)

THREE_D_FEATURE_COLUMNS = [
    "3d_available",
    "3d_embed_success",
    "3d_radius_gyration",
    "3d_asphericity",
    "3d_eccentricity",
    "3d_npr1",
    "3d_npr2",
    "3d_spherocity",
]


def multimodal_feature_names(mode: str) -> list[str]:
    mode = _normalize_mode(mode)
    names: list[str] = []
    if mode in {"source_image", "source_reference", "full"}:
        names.extend([f"source_img_{name}" for name in IMAGE_FEATURE_COLUMNS])
    if mode in {"source_reference", "full"}:
        names.extend([f"reference_img_{name}" for name in IMAGE_FEATURE_COLUMNS])
        names.extend([f"reference_minus_source_img_{name}" for name in IMAGE_FEATURE_COLUMNS])
        names.append("reference_source_tanimoto")
    if mode in {"source_3d", "full"}:
        names.extend([f"source_{name}" for name in THREE_D_FEATURE_COLUMNS])
    if mode == "full":
        names.extend([f"reference_{name}" for name in THREE_D_FEATURE_COLUMNS])
        names.extend([f"reference_minus_source_{name}" for name in THREE_D_FEATURE_COLUMNS])
    return names


def multimodal_context_from_row(
    row: Any,
    mode: str = "none",
    allow_target_reference: bool = False,
) -> np.ndarray:
    mode = _normalize_mode(mode)
    if mode == "none":
        return np.zeros(0, dtype=np.float32)

    source_smiles = _row_value(row, "source_smiles")
    if not source_smiles:
        raise ValueError("Missing source_smiles for multimodal context.")
    reference_smiles = _reference_smiles(row, allow_target_reference=allow_target_reference)

    values: list[float] = []
    source_img = _image_features(source_smiles)
    if mode in {"source_image", "source_reference", "full"}:
        values.extend(source_img)

    if mode in {"source_reference", "full"}:
        if not reference_smiles:
            raise ValueError(
                "multimodal_context requires reference_smiles. Rebuild the instruction dataset, "
                "or pass --allow-target-reference to use target_smiles as an explicit oracle reference."
            )
        reference_img = _image_features(reference_smiles)
        values.extend(reference_img)
        values.extend((np.asarray(reference_img) - np.asarray(source_img)).tolist())
        values.append(float(tanimoto(source_smiles, reference_smiles)))

    source_3d = _three_d_features(source_smiles)
    if mode in {"source_3d", "full"}:
        values.extend(source_3d)

    if mode == "full":
        reference_3d = _three_d_features(reference_smiles) if reference_smiles else [0.0] * len(THREE_D_FEATURE_COLUMNS)
        values.extend(reference_3d)
        values.extend((np.asarray(reference_3d) - np.asarray(source_3d)).tolist())

    return np.asarray(values, dtype=np.float32)


def _normalize_mode(mode: str) -> str:
    if mode not in MULTIMODAL_CONTEXT_MODES:
        raise ValueError(f"Unsupported multimodal context '{mode}'. Choices: {', '.join(MULTIMODAL_CONTEXT_MODES)}")
    return mode


def _image_features(smiles: str) -> list[float]:
    return [float(image_array_features(descriptor_image(smiles))[name]) for name in IMAGE_FEATURE_COLUMNS]


def _three_d_features(smiles: str | None) -> list[float]:
    if not smiles:
        return [0.0] * len(THREE_D_FEATURE_COLUMNS)
    metrics = molecule_3d_metrics(smiles)
    values = np.asarray([float(metrics.get(name, 0.0)) for name in THREE_D_FEATURE_COLUMNS], dtype=np.float32)
    return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0).tolist()


def _reference_smiles(row: Any, allow_target_reference: bool = False) -> str | None:
    reference = _row_value(row, "reference_smiles")
    if reference:
        can = canonicalize_smiles(reference)
        if can:
            return can
    if allow_target_reference:
        target = _row_value(row, "target_smiles")
        can = canonicalize_smiles(target) if target else None
        if can:
            return can
    return None


def _row_value(row: Any, key: str) -> str | None:
    try:
        value = row[key]
    except Exception:
        value = getattr(row, key, None)
    if value is None:
        return None
    try:
        if bool(np.isnan(value)):
            return None
    except Exception:
        pass
    text = str(value).strip()
    return text if text and text.lower() != "nan" else None
