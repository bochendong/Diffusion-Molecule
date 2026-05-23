"""Feature construction for context and target molecular latents."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Iterable

import numpy as np

from .chem import DESCRIPTOR_KEYS, molecular_descriptors, morgan_fingerprint_bits
from .schema import BenchmarkExample

MOLECULE_LATENT_VERSION = "rdkit_morgan_descriptor_v2"
FINGERPRINT_BITS = 2048


def stable_hash_vector(text: str, dim: int, salt: str = "") -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    for token in _tokens(text):
        digest = hashlib.sha256(f"{salt}:{token}".encode("utf-8")).digest()
        idx = int.from_bytes(digest[:4], "big") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vec[idx] += sign
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 0 else vec


def molecule_latent(smiles: str | None, dim: int) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    rec = molecular_descriptors(smiles)
    if not smiles:
        return vec
    desc = np.asarray([rec.descriptors.get(key, 0.0) for key in DESCRIPTOR_KEYS], dtype=np.float32)
    scales = np.asarray([500.0, 5.0, 1.0, 150.0, 5.0, 10.0, 10.0], dtype=np.float32)
    scaled = desc / scales
    vec[: min(dim, len(scaled))] = scaled[:dim]
    fp = morgan_fingerprint_bits(rec.smiles, n_bits=min(FINGERPRINT_BITS, dim - len(scaled))) if dim > len(scaled) else None
    if fp:
        fp_arr = np.asarray(fp, dtype=np.float32)
        start = min(dim, len(scaled))
        stop = min(dim, start + len(fp_arr))
        vec[start:stop] = fp_arr[: max(0, stop - start)]
        return _normalize(vec)
    hashed = stable_hash_vector(rec.smiles if rec.valid else str(smiles), dim, salt="molecule")
    return _normalize(0.6 * vec + 0.4 * hashed)


def context_vector(example: BenchmarkExample, dim: int) -> np.ndarray:
    task = stable_hash_vector(example.task_type.value, dim, salt="task")
    instruction = stable_hash_vector(example.instruction, dim, salt="instruction")
    goals = stable_hash_vector(" ".join(example.goals), dim, salt="goals")
    source = molecule_latent(example.source_smiles, dim)
    mask = stable_hash_vector(example.mask_hint or "", dim, salt="mask")
    image = image_stats_vector(example.image_path, dim)
    return _normalize(0.22 * task + 0.30 * instruction + 0.12 * goals + 0.22 * source + 0.08 * mask + 0.06 * image)


def image_stats_vector(path: str | None, dim: int) -> np.ndarray:
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
    except Exception:
        return stable_hash_vector(str(Path(path).name), dim, salt="image-path")
    return vec


def matrix_from_examples(examples: Iterable[BenchmarkExample], feature_dim: int, latent_dim: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    examples = list(examples)
    conditions = np.stack([context_vector(example, feature_dim) for example in examples]).astype(np.float32)
    targets = np.stack([molecule_latent(example.target_smiles, latent_dim) for example in examples]).astype(np.float32)
    sources = np.stack([molecule_latent(example.source_smiles, latent_dim) for example in examples]).astype(np.float32)
    return conditions, targets, sources


def _tokens(text: str) -> list[str]:
    out: list[str] = []
    token: list[str] = []
    for ch in str(text).lower():
        if ch.isalnum() or ch in {"_", "-"}:
            token.append(ch)
        elif token:
            out.append("".join(token))
            token = []
    if token:
        out.append("".join(token))
    return out


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = math.sqrt(float(np.dot(vec, vec)))
    return (vec / norm).astype(np.float32) if norm > 0 else vec.astype(np.float32)
