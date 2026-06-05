"""Deterministic features for unified molecular training rows."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .image_features import image_patch_feature_vector
from .text_features import hashed_text_vector
from .unified_condition_dataset import PROPERTY_COLUMNS, UnifiedConditionSample


SIMILARITY_BINS = ("exploratory_low_similarity", "hard_similarity", "medium_similarity", "easy_high_similarity")


def molecule_feature(smiles: str, dim: int = 512) -> np.ndarray:
    """Morgan fingerprint when RDKit exists, hashed SMILES fallback otherwise."""

    if smiles:
        try:
            from .chem import morgan_fingerprint_bits

            bits = morgan_fingerprint_bits(smiles, n_bits=dim)
            if bits is not None:
                return np.asarray(bits, dtype=np.float32)
        except RuntimeError:
            pass
    return hashed_text_vector(smiles, dim=dim)


def text_feature(text: str, dim: int = 256) -> np.ndarray:
    return hashed_text_vector(text, dim=dim)


def image_feature(path: str, dim: int = 256) -> np.ndarray:
    if path and Path(path).exists():
        return image_patch_feature_vector(path, dim=dim)
    return np.zeros(dim, dtype=np.float32)


def hidden_sequence_for_sample(
    sample: UnifiedConditionSample,
    *,
    token_dim: int = 512,
    text_dim: int = 256,
    image_dim: int = 256,
) -> np.ndarray:
    """Build a small source/instruction hidden-state sequence for connector training."""

    source = sample.source_smiles or sample.molecule_smiles or sample.target_smiles
    source_vec = molecule_feature(source, token_dim)
    target_text = sample.instruction or sample.description or sample.prompt
    text_vec = _resize(text_feature(target_text, text_dim), token_dim)
    image_vec = _resize(image_feature(sample.source_image, image_dim), token_dim)
    meta_vec = _resize(edit_numeric_vector(sample), token_dim)
    return np.stack([source_vec, text_vec, image_vec, meta_vec]).astype(np.float32)


def edit_numeric_vector(sample: UnifiedConditionSample) -> np.ndarray:
    values = []
    for prop in PROPERTY_COLUMNS:
        values.append(float(sample.source_properties.get(prop, 0.0)))
        values.append(float(sample.target_properties.get(prop, 0.0)))
        values.append(float(sample.property_deltas.get(prop, 0.0)))
        values.append(1.0 if sample.active_properties.get(prop, False) else 0.0)
    try:
        values.append(float(sample.source_tanimoto))
    except ValueError:
        values.append(0.0)
    return np.asarray(values, dtype=np.float32)


def target_property_vector(sample: UnifiedConditionSample) -> np.ndarray:
    return np.asarray([sample.target_properties.get(prop, 0.0) for prop in PROPERTY_COLUMNS], dtype=np.float32)


def property_delta_vector(sample: UnifiedConditionSample) -> np.ndarray:
    return np.asarray([sample.property_deltas.get(prop, 0.0) for prop in PROPERTY_COLUMNS], dtype=np.float32)


def active_property_vector(sample: UnifiedConditionSample) -> np.ndarray:
    return np.asarray([1.0 if sample.active_properties.get(prop, False) else 0.0 for prop in PROPERTY_COLUMNS], dtype=np.float32)


def direction_label_vector(sample: UnifiedConditionSample) -> np.ndarray:
    labels = []
    for prop in PROPERTY_COLUMNS:
        direction = str(sample.directions.get(prop, "")).lower()
        if direction == "decrease":
            labels.append(0)
        elif direction == "increase":
            labels.append(2)
        else:
            labels.append(1)
    return np.asarray(labels, dtype=np.int64)


def similarity_bin_label(sample: UnifiedConditionSample) -> int:
    label = sample.source_similarity_bin
    if label in SIMILARITY_BINS:
        return SIMILARITY_BINS.index(label)
    try:
        value = float(sample.source_tanimoto)
    except ValueError:
        value = 0.0
    if value >= 0.7:
        return SIMILARITY_BINS.index("easy_high_similarity")
    if value >= 0.5:
        return SIMILARITY_BINS.index("medium_similarity")
    if value >= 0.4:
        return SIMILARITY_BINS.index("hard_similarity")
    return SIMILARITY_BINS.index("exploratory_low_similarity")


def target_latent_vector(sample: UnifiedConditionSample, *, fingerprint_dim: int = 512) -> np.ndarray:
    fp = molecule_feature(sample.target_smiles, fingerprint_dim)
    props = _resize(target_property_vector(sample), 32)
    deltas = _resize(property_delta_vector(sample), 32)
    active = _resize(active_property_vector(sample), 16)
    return np.concatenate([fp, props, deltas, active]).astype(np.float32)


def _resize(vec: np.ndarray, dim: int) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float32).reshape(-1)
    if vec.shape[0] == dim:
        return vec
    if vec.shape[0] > dim:
        return vec[:dim]
    out = np.zeros(dim, dtype=np.float32)
    out[: vec.shape[0]] = vec
    return out

