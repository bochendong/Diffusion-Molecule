"""Export Unified 3M latent predictions for the multi-property benchmark."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Protocol

import numpy as np

from .unified_condition_dataset import PROPERTY_COLUMNS


class BenchmarkSample(Protocol):
    sample_id: str
    split: str
    property_count: str
    source_tanimoto: str
    source_similarity_bin: str
    metadata: dict[str, str]


def write_edit_latent_benchmark_inputs(
    samples: Iterable[BenchmarkSample],
    latents: np.ndarray,
    output_dir: str | Path,
    *,
    fingerprint_dim: int,
    variant: str = "full",
) -> dict[str, object]:
    """Write files consumed by benchmark_multiproperty_retrieval.py.

    The retrieval benchmark expects compact edit-latent rows ordered as
    target properties, deltas, active mask, and directions. Unified 3M stores
    those blocks inside a larger latent whose first block is the target
    fingerprint; we export both views so the benchmark can use property/delta
    scoring and optional fingerprint-aware reranking.
    """

    sample_list = list(samples)
    latents = np.asarray(latents, dtype=np.float32)
    if latents.ndim != 2:
        raise ValueError(f"Expected a 2D latent array, got shape {latents.shape}")
    if latents.shape[0] != len(sample_list):
        raise ValueError(f"Latent rows ({latents.shape[0]}) do not match samples ({len(sample_list)})")

    prop_count = len(PROPERTY_COLUMNS)
    prop_start = int(fingerprint_dim)
    delta_start = int(fingerprint_dim) + 32
    active_start = int(fingerprint_dim) + 64
    min_dim = active_start + prop_count
    if latents.shape[1] < min_dim:
        raise ValueError(f"Latent dim {latents.shape[1]} is too small for fingerprint_dim={fingerprint_dim}")

    target_values = latents[:, prop_start : prop_start + prop_count]
    delta_values = latents[:, delta_start : delta_start + prop_count]
    active_values = latents[:, active_start : active_start + prop_count]
    directions = np.sign(delta_values).astype(np.float32)
    edit_predictions = np.concatenate([target_values, delta_values, active_values, directions], axis=1).astype(
        np.float32
    )
    fingerprints = latents[:, :fingerprint_dim].astype(np.float32)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "edit_latent_predictions.npy", edit_predictions)
    np.save(output_dir / "edit_latent_fingerprints.npy", fingerprints)

    index_rows = []
    for sample in sample_list:
        index_rows.append(
            {
                "condition_id": _condition_id(sample),
                "sample_id": sample.sample_id,
                "variant": variant,
                "split": sample.split,
                "property_count": sample.property_count,
                "source_tanimoto": sample.source_tanimoto,
                "source_similarity_bin": sample.source_similarity_bin,
            }
        )
    with (output_dir / "index.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(index_rows[0].keys()) if index_rows else ["condition_id"])
        writer.writeheader()
        writer.writerows(index_rows)

    return {
        "rows": len(sample_list),
        "fingerprint_dim": int(fingerprint_dim),
        "variant": variant,
        "index_csv": str(output_dir / "index.csv"),
        "edit_latent_predictions": str(output_dir / "edit_latent_predictions.npy"),
        "edit_latent_fingerprints": str(output_dir / "edit_latent_fingerprints.npy"),
    }


def _condition_id(sample: BenchmarkSample) -> str:
    condition_id = str(sample.metadata.get("condition_id", "") if sample.metadata else "").strip()
    if condition_id:
        return condition_id
    prefix = "edit:multiproperty_edit:"
    if sample.sample_id.startswith(prefix):
        return sample.sample_id[len(prefix) :]
    if sample.sample_id.startswith("edit:"):
        parts = sample.sample_id.split(":", 2)
        if len(parts) == 3:
            return parts[2]
    return sample.sample_id
