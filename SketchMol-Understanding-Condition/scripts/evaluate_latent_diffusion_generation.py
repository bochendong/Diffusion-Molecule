#!/usr/bin/env python3
"""Evaluate unified latent diffusion generation in latent space."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from sketchmol_understanding_condition.edit_condition_tokens import EditConditionTokenConnector  # noqa: E402
from sketchmol_understanding_condition.latent_diffusion_generation import (  # noqa: E402
    EditLatentDenoiser,
    GaussianLatentDiffusion,
)
from sketchmol_understanding_condition.unified_condition_dataset import EDIT_GENERATION, PROPERTY_COLUMNS, read_jsonl  # noqa: E402
from sketchmol_understanding_condition.unified_featurization import (  # noqa: E402
    hidden_sequence_for_sample,
    molecule_feature,
    property_delta_vector,
    target_latent_vector,
    target_property_vector,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-jsonl", required=True, type=Path)
    parser.add_argument("--condition-connector", required=True, type=Path)
    parser.add_argument("--diffusion-checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--sample-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    connector_payload = torch.load(args.condition_connector, map_location="cpu")
    diffusion_payload = torch.load(args.diffusion_checkpoint, map_location="cpu")
    connector_config = connector_payload["config"]
    diffusion_config = diffusion_payload["config"]
    fingerprint_dim = int(diffusion_config["fingerprint_dim"])

    samples = [sample for sample in read_jsonl(args.eval_jsonl) if sample.task_type == EDIT_GENERATION]
    if args.limit is not None and args.limit > 0:
        samples = samples[: args.limit]
    if not samples:
        raise ValueError(f"No edit_generation rows found in {args.eval_jsonl}")

    connector = EditConditionTokenConnector(
        input_hidden_dim=int(connector_config["token_dim"]),
        context_dim=int(connector_config["context_dim"]),
        num_queries=int(connector_config["num_queries"]),
        hidden_dim=int(connector_config["hidden_dim"]),
        fingerprint_dim=int(connector_config["fingerprint_dim"]),
    )
    connector.load_state_dict(connector_payload["model_state"])
    connector.eval()

    denoiser = EditLatentDenoiser(
        latent_dim=int(diffusion_config["latent_dim"]),
        context_dim=int(connector_config["context_dim"]),
        hidden_dim=int(diffusion_config["hidden_dim"]),
        depth=int(diffusion_config["depth"]),
    )
    diffusion = GaussianLatentDiffusion(
        denoiser,
        timesteps=int(diffusion_config["timesteps"]),
        objective="pred_noise",
    )
    diffusion.load_state_dict(diffusion_payload["diffusion_state"])
    diffusion.eval()

    latent_mean = torch.tensor(diffusion_config["latent_mean"], dtype=torch.float32)
    latent_std = torch.tensor(diffusion_config["latent_std"], dtype=torch.float32)

    rows = []
    generated_raw = []
    target_raw = []
    source_latents = []
    with torch.no_grad():
        for start in range(0, len(samples), args.batch_size):
            batch = samples[start : start + args.batch_size]
            hidden = np.stack(
                [
                    hidden_sequence_for_sample(sample, token_dim=int(connector_config["token_dim"]))
                    for sample in batch
                ]
            ).astype(np.float32)
            hidden_tensor = torch.from_numpy(hidden)
            condition = connector(hidden_tensor)
            sampled_norm = diffusion.sample(
                condition.tokens,
                condition.attention_mask,
                steps=args.sample_steps,
            )
            sampled = sampled_norm * latent_std + latent_mean
            generated_raw.append(sampled.cpu().numpy().astype(np.float32))
            target_raw.append(
                np.stack([target_latent_vector(sample, fingerprint_dim=fingerprint_dim) for sample in batch]).astype(
                    np.float32
                )
            )
            source_latents.append(
                np.stack([_source_latent_vector(sample, fingerprint_dim=fingerprint_dim) for sample in batch]).astype(
                    np.float32
                )
            )

    gen = np.concatenate(generated_raw, axis=0)
    target = np.concatenate(target_raw, axis=0)
    source = np.concatenate(source_latents, axis=0)
    per_row = _per_row_metrics(samples, gen, target, source, fingerprint_dim=fingerprint_dim)
    metrics = _summarize(per_row)
    metrics.update(
        {
            "eval_jsonl": str(args.eval_jsonl),
            "condition_connector": str(args.condition_connector),
            "diffusion_checkpoint": str(args.diffusion_checkpoint),
            "output_dir": str(args.output_dir),
            "rows": len(samples),
            "batch_size": int(args.batch_size),
            "sample_steps": int(args.sample_steps or diffusion.timesteps),
            "fingerprint_dim": fingerprint_dim,
        }
    )
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_rows(args.output_dir / "predictions.csv", per_row)
    print(json.dumps(metrics, indent=2, sort_keys=True))


def _source_latent_vector(sample, *, fingerprint_dim: int) -> np.ndarray:
    fp = molecule_feature(sample.source_smiles or sample.molecule_smiles, fingerprint_dim)
    props = _resize(np.asarray([sample.source_properties.get(prop, 0.0) for prop in PROPERTY_COLUMNS], dtype=np.float32), 32)
    deltas = _resize(property_delta_vector(sample), 32)
    active = _resize(np.asarray([1.0 if sample.active_properties.get(prop, False) else 0.0 for prop in PROPERTY_COLUMNS], dtype=np.float32), 16)
    return np.concatenate([fp, props, deltas, active]).astype(np.float32)


def _per_row_metrics(samples, gen: np.ndarray, target: np.ndarray, source: np.ndarray, *, fingerprint_dim: int) -> list[dict[str, object]]:
    rows = []
    gen_fp = gen[:, :fingerprint_dim]
    target_fp = target[:, :fingerprint_dim]
    source_fp = source[:, :fingerprint_dim]
    prop_start = fingerprint_dim
    delta_start = fingerprint_dim + 32
    for idx, sample in enumerate(samples):
        gen_props = gen[idx, prop_start : prop_start + len(PROPERTY_COLUMNS)]
        target_props = target_property_vector(sample)
        gen_deltas = gen[idx, delta_start : delta_start + len(PROPERTY_COLUMNS)]
        target_deltas = property_delta_vector(sample)
        rows.append(
            {
                "sample_id": sample.sample_id,
                "property_count": sample.property_count,
                "source_tanimoto": sample.source_tanimoto,
                "source_similarity_bin": sample.source_similarity_bin,
                "latent_mse": _mse(gen[idx], target[idx]),
                "latent_mae": _mae(gen[idx], target[idx]),
                "target_fingerprint_cosine": _cosine(gen_fp[idx], target_fp[idx]),
                "source_fingerprint_cosine": _cosine(gen_fp[idx], source_fp[idx]),
                "source_target_fingerprint_cosine": _cosine(source_fp[idx], target_fp[idx]),
                "target_property_mae": _mae(gen_props, target_props),
                "source_target_property_mae": _mae(
                    source[idx, prop_start : prop_start + len(PROPERTY_COLUMNS)],
                    target_props,
                ),
                "delta_mae": _mae(gen_deltas, target_deltas),
            }
        )
    return rows


def _summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    numeric_keys = [
        "latent_mse",
        "latent_mae",
        "target_fingerprint_cosine",
        "source_fingerprint_cosine",
        "source_target_fingerprint_cosine",
        "target_property_mae",
        "source_target_property_mae",
        "delta_mae",
    ]
    summary: dict[str, object] = {"overall": _mean_metrics(rows, numeric_keys)}
    by_count = {}
    for count in sorted({str(row.get("property_count", "")) for row in rows}):
        selected = [row for row in rows if str(row.get("property_count", "")) == count]
        by_count[count or "unknown"] = _mean_metrics(selected, numeric_keys)
    by_similarity = {}
    for label in sorted({str(row.get("source_similarity_bin", "")) for row in rows}):
        selected = [row for row in rows if str(row.get("source_similarity_bin", "")) == label]
        by_similarity[label or "unknown"] = _mean_metrics(selected, numeric_keys)
    summary["by_property_count"] = by_count
    summary["by_source_similarity_bin"] = by_similarity
    return summary


def _mean_metrics(rows: list[dict[str, object]], keys: list[str]) -> dict[str, float]:
    out = {"rows": float(len(rows))}
    for key in keys:
        values = [float(row[key]) for row in rows]
        out[key] = float(np.mean(values)) if values else 0.0
    return out


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _resize(vec: np.ndarray, dim: int) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float32).reshape(-1)
    if vec.shape[0] == dim:
        return vec
    if vec.shape[0] > dim:
        return vec[:dim]
    out = np.zeros(dim, dtype=np.float32)
    out[: vec.shape[0]] = vec
    return out


def _mse(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.mean((np.asarray(left) - np.asarray(right)) ** 2))


def _mae(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(left) - np.asarray(right))))


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float32)
    right = np.asarray(right, dtype=np.float32)
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom <= 1e-8:
        return 0.0
    return float(np.dot(left, right) / denom)


if __name__ == "__main__":
    main()
