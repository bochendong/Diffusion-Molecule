"""Dataset utilities for retrieval-style understanding-condition baselines."""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .chem import morgan_fingerprint_bits, molecular_properties


VARIANTS = ("full", "text_only", "image_only", "random_query", "caption_bottleneck")


@dataclass(frozen=True)
class RetrievalMatrices:
    variant: str
    train_x: np.ndarray
    train_y: np.ndarray
    eval_x: np.ndarray
    eval_y: np.ndarray
    eval_target_smiles: list[str]
    eval_source_smiles: list[str]
    train_target_smiles: list[str]
    feature_dim: int
    target_dim: int


def read_variant_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def build_retrieval_matrices(
    rows: list[dict[str, str]],
    *,
    variant: str,
    fingerprint_bits: int = 512,
    text_dim: int = 128,
    random_dim: int = 128,
) -> RetrievalMatrices:
    """Build train/eval matrices for one baseline variant."""

    selected = [row for row in rows if row.get("variant") == variant]
    if not selected:
        raise ValueError(f"No rows found for variant={variant!r}")

    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    kept_rows: list[dict[str, str]] = []
    for row in selected:
        target_bits = morgan_fingerprint_bits(row["target_smiles"], n_bits=fingerprint_bits)
        if target_bits is None:
            continue
        x = condition_features_for_row(
            row,
            variant=variant,
            fingerprint_bits=fingerprint_bits,
            text_dim=text_dim,
            random_dim=random_dim,
        )
        xs.append(x)
        ys.append(np.asarray(target_bits, dtype=np.float32))
        kept_rows.append(row)

    if not xs:
        raise ValueError(f"No valid rows after featurization for variant={variant!r}")

    x_arr = np.stack(xs).astype(np.float32)
    y_arr = np.stack(ys).astype(np.float32)
    train_idx = [idx for idx, row in enumerate(kept_rows) if row.get("split") == "train"]
    eval_idx = [idx for idx, row in enumerate(kept_rows) if row.get("split") == "eval"]
    if not train_idx or not eval_idx:
        raise ValueError(f"Variant {variant!r} needs both train and eval rows")

    return RetrievalMatrices(
        variant=variant,
        train_x=x_arr[train_idx],
        train_y=y_arr[train_idx],
        eval_x=x_arr[eval_idx],
        eval_y=y_arr[eval_idx],
        eval_target_smiles=[kept_rows[idx]["target_smiles"] for idx in eval_idx],
        eval_source_smiles=[kept_rows[idx]["source_smiles"] for idx in eval_idx],
        train_target_smiles=[kept_rows[idx]["target_smiles"] for idx in train_idx],
        feature_dim=x_arr.shape[1],
        target_dim=y_arr.shape[1],
    )


def _condition_features(
    row: dict[str, str],
    *,
    variant: str,
    fingerprint_bits: int,
    text_dim: int,
    random_dim: int,
) -> np.ndarray:
    return condition_features_for_row(
        row,
        variant=variant,
        fingerprint_bits=fingerprint_bits,
        text_dim=text_dim,
        random_dim=random_dim,
    )


def condition_features_for_row(
    row: dict[str, str],
    *,
    variant: str,
    fingerprint_bits: int,
    text_dim: int,
    random_dim: int,
) -> np.ndarray:
    source_fp = np.asarray(
        morgan_fingerprint_bits(row["source_smiles"], n_bits=fingerprint_bits) or [0.0] * fingerprint_bits,
        dtype=np.float32,
    )
    source_props = _property_vector(row["source_smiles"])
    prompt_vec = _hashed_text_vector(row.get("prompt", ""), text_dim)

    if variant == "full":
        return np.concatenate([source_fp, source_props, prompt_vec])
    if variant == "text_only":
        return np.concatenate([np.zeros_like(source_fp), np.zeros_like(source_props), prompt_vec])
    if variant == "image_only":
        return np.concatenate([source_fp, source_props, np.zeros_like(prompt_vec)])
    if variant == "caption_bottleneck":
        return np.concatenate([np.zeros_like(source_fp), np.zeros_like(source_props), prompt_vec])
    if variant == "random_query":
        rand = _deterministic_random_vector(row.get("variant_id", ""), random_dim)
        return np.concatenate(
            [
                np.zeros_like(source_fp),
                np.zeros_like(source_props),
                np.zeros_like(prompt_vec),
                rand,
            ]
        )
    raise ValueError(f"Unsupported variant: {variant}")


def _property_vector(smiles: str) -> np.ndarray:
    props = molecular_properties(smiles) or {}
    values = [
        props.get("MolWt", 0.0) / 600.0,
        props.get("LogP", 0.0) / 8.0,
        props.get("QED", 0.0),
        props.get("TPSA", 0.0) / 200.0,
        props.get("HBD", 0.0) / 10.0,
        props.get("HBA", 0.0) / 15.0,
        props.get("rotatable", 0.0) / 15.0,
    ]
    return np.asarray(values, dtype=np.float32)


def _hashed_text_vector(text: str, dim: int) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    tokens = str(text or "").lower().replace(".", " ").replace(",", " ").split()
    for token in tokens:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "little") % dim
        sign = 1.0 if int.from_bytes(digest[4:], "little") % 2 == 0 else -1.0
        vec[bucket] += sign
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def _deterministic_random_vector(key: str, dim: int) -> np.ndarray:
    seed = int.from_bytes(hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest(), "little")
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 1.0, size=dim).astype(np.float32)
