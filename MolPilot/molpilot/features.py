"""Feature helpers for the lightweight understanding stream."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Iterable

import numpy as np

from .chem import DESCRIPTOR_KEYS, molecular_descriptors


def stable_hash_vector(text: str, dim: int, salt: str = "") -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    for token in _tokens(text):
        digest = hashlib.sha256(f"{salt}:{token}".encode("utf-8")).digest()
        idx = int.from_bytes(digest[:4], "big") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vec[idx] += sign
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 0 else vec


def molecule_feature_vector(smiles: str | None, dim: int) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    if not smiles:
        return vec
    rec = molecular_descriptors(smiles)
    if rec.valid:
        desc = [rec.descriptors.get(key, 0.0) for key in DESCRIPTOR_KEYS]
        scales = np.asarray([500.0, 5.0, 1.0, 150.0, 5.0, 10.0, 10.0], dtype=np.float32)
        raw = np.asarray(desc, dtype=np.float32) / scales
        vec[: min(len(raw), dim)] = raw[:dim]
    hashed = stable_hash_vector(rec.smiles if rec.valid else smiles, dim, salt="smiles")
    return 0.65 * vec + 0.35 * hashed


def image_feature_vector(path: str | None, dim: int) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    if not path:
        return vec
    try:
        from PIL import Image

        image = Image.open(path).convert("L").resize((64, 64))
        arr = np.asarray(image, dtype=np.float32) / 255.0
        stats = np.asarray(
            [
                arr.mean(),
                arr.std(),
                float(np.abs(np.diff(arr, axis=0)).mean()),
                float(np.abs(np.diff(arr, axis=1)).mean()),
                float((arr < 0.2).mean()),
                float((arr > 0.8).mean()),
            ],
            dtype=np.float32,
        )
        vec[: min(dim, len(stats))] = stats[:dim]
        if dim > len(stats):
            hist, _ = np.histogram(arr, bins=min(16, dim - len(stats)), range=(0.0, 1.0), density=False)
            hist = hist.astype(np.float32) / max(1.0, float(hist.sum()))
            vec[len(stats) : len(stats) + len(hist)] = hist
    except Exception:
        vec = stable_hash_vector(str(Path(path).name), dim, salt="image-path")
    return vec


def spec_feature_vector(goals: Iterable[str], constraints: Iterable[str], proxy_goals: Iterable[str], dim: int) -> np.ndarray:
    text = " ".join(list(goals) + list(constraints) + list(proxy_goals))
    return stable_hash_vector(text, dim, salt="spec")


def concat_and_project(parts: list[np.ndarray], dim: int) -> np.ndarray:
    if not parts:
        return np.zeros(dim, dtype=np.float32)
    raw = np.concatenate(parts).astype(np.float32)
    if len(raw) == dim:
        return raw
    out = np.zeros(dim, dtype=np.float32)
    for i, value in enumerate(raw):
        out[i % dim] += float(value)
    norm = math.sqrt(float(np.dot(out, out)))
    return out / norm if norm > 0 else out


def _tokens(text: str) -> list[str]:
    cleaned = []
    token = []
    for ch in str(text).lower():
        if ch.isalnum() or ch in {"_", "-"}:
            token.append(ch)
        elif token:
            cleaned.append("".join(token))
            token = []
    if token:
        cleaned.append("".join(token))
    return cleaned

