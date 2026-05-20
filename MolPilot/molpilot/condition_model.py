"""Condition model loading and source-latent helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .alignment import UnderstandingAlignment
from .artifacts import load_json
from .jepa import MolecularJEPAPredictor


def build_source_latents(autoencoder, pairs, latent_dim: int | None = None) -> np.ndarray:
    source_smiles = [(request.source_smiles or "") for request, _ in pairs]
    if not source_smiles:
        dim = int(latent_dim or 0)
        return np.zeros((0, dim), dtype=np.float32)
    latents = autoencoder.encode_many(source_smiles).astype(np.float32)
    if latent_dim is not None and latents.shape[1] != latent_dim:
        latents = _pad_or_trim(latents, latent_dim)
    for idx, (request, _) in enumerate(pairs):
        if not request.source_smiles:
            latents[idx] = 0.0
    return latents


def load_condition_model(out_dir: str | Path):
    out_dir = Path(out_dir)
    metadata = load_json(out_dir / "metadata.json") if (out_dir / "metadata.json").exists() else {}
    if metadata.get("model_type") == "molecular_jepa":
        return MolecularJEPAPredictor.load(out_dir)
    return UnderstandingAlignment.load(out_dir)


def predict_condition_latents(model, raw_conditions: np.ndarray, pairs, autoencoder) -> np.ndarray:
    if getattr(model, "expects_source_latents", False):
        source_latents = build_source_latents(autoencoder, pairs)
        return model.predict(raw_conditions, source_latents)
    return model.predict(raw_conditions)


def _pad_or_trim(x: np.ndarray, dim: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.shape[1] == dim:
        return x
    if x.shape[1] > dim:
        return x[:, :dim]
    out = np.zeros((x.shape[0], dim), dtype=np.float32)
    out[:, : x.shape[1]] = x
    return out
