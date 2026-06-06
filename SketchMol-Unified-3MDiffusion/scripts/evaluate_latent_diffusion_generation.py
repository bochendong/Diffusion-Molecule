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

from sketchmol_unified_3m_diffusion.edit_condition_tokens import EditConditionTokenConnector  # noqa: E402
from sketchmol_unified_3m_diffusion.latent_diffusion_generation import (  # noqa: E402
    EditLatentDenoiser,
    GaussianLatentDiffusion,
)
from sketchmol_unified_3m_diffusion.benchmark_export import write_edit_latent_benchmark_inputs  # noqa: E402
from sketchmol_unified_3m_diffusion.runtime import device_report, resolve_device  # noqa: E402
from sketchmol_unified_3m_diffusion.unified_condition_dataset import EDIT_GENERATION, PROPERTY_COLUMNS, read_jsonl  # noqa: E402
from sketchmol_unified_3m_diffusion.unified_featurization import (  # noqa: E402
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
    parser.add_argument("--sample-eta", type=float, default=0.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    print(json.dumps({"event": "device", **device_report(device)}, sort_keys=True))

    connector_payload = torch.load(args.condition_connector, map_location=device)
    diffusion_payload = torch.load(args.diffusion_checkpoint, map_location=device)
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
    ).to(device)
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
        objective=str(diffusion_config.get("diffusion_objective", "pred_noise")),
    ).to(device)
    diffusion.load_state_dict(diffusion_payload["diffusion_state"])
    diffusion.eval()

    latent_mean = torch.tensor(diffusion_config["latent_mean"], dtype=torch.float32, device=device)
    latent_std = torch.tensor(diffusion_config["latent_std"], dtype=torch.float32, device=device)
    diffusion_target = str(diffusion_config.get("diffusion_target", "target"))

    rows = []
    generated_raw = []
    target_raw = []
    prior_raw = []
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
            hidden_tensor = torch.from_numpy(hidden).to(device)
            condition = connector(hidden_tensor)
            sampled_norm = diffusion.sample(
                condition.tokens,
                condition.attention_mask,
                steps=args.sample_steps,
                eta=args.sample_eta,
            )
            prior_norm = condition_prior_latent(
                condition,
                connector_config=connector_config,
                latent_mean=latent_mean,
                latent_std=latent_std,
                fingerprint_dim=fingerprint_dim,
                latent_dim=int(diffusion_config["latent_dim"]),
            )
            if diffusion_target == "residual":
                sampled_norm = sampled_norm + prior_norm
            sampled = sampled_norm * latent_std + latent_mean
            prior = prior_norm * latent_std + latent_mean
            generated_raw.append(sampled.cpu().numpy().astype(np.float32))
            prior_raw.append(prior.cpu().numpy().astype(np.float32))
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
    prior = np.concatenate(prior_raw, axis=0)
    source = np.concatenate(source_latents, axis=0)
    np.save(args.output_dir / "generated_latents.npy", gen.astype(np.float32))
    np.save(args.output_dir / "target_latents.npy", target.astype(np.float32))
    np.save(args.output_dir / "prior_latents.npy", prior.astype(np.float32))
    benchmark_export = write_edit_latent_benchmark_inputs(
        samples,
        gen,
        args.output_dir,
        fingerprint_dim=fingerprint_dim,
    )
    per_row = _per_row_metrics(samples, gen, target, source, prior=prior, fingerprint_dim=fingerprint_dim)
    metrics = _summarize(per_row)
    metrics["latent_block_summary"] = _latent_block_summary(gen, target, source, prior, fingerprint_dim=fingerprint_dim)
    metrics.update(
        {
            "eval_jsonl": str(args.eval_jsonl),
            "condition_connector": str(args.condition_connector),
            "diffusion_checkpoint": str(args.diffusion_checkpoint),
            "output_dir": str(args.output_dir),
            "rows": len(samples),
            "batch_size": int(args.batch_size),
            "sample_steps": int(args.sample_steps or diffusion.timesteps),
            "sample_eta": float(args.sample_eta),
            "diffusion_objective": str(diffusion_config.get("diffusion_objective", "pred_noise")),
            "diffusion_target": diffusion_target,
            "fingerprint_dim": fingerprint_dim,
            "benchmark_export": benchmark_export,
            "device": device_report(device),
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


def condition_prior_latent(
    condition,
    *,
    connector_config: dict[str, object],
    latent_mean: torch.Tensor,
    latent_std: torch.Tensor,
    fingerprint_dim: int,
    latent_dim: int,
) -> torch.Tensor:
    batch = condition.target_fingerprint_logits.shape[0]
    device = condition.target_fingerprint_logits.device
    dtype = condition.target_fingerprint_logits.dtype
    num_props = len(PROPERTY_COLUMNS)
    fingerprint = torch.sigmoid(condition.target_fingerprint_logits[:, :fingerprint_dim])
    prop_mean = _config_tensor(connector_config, "property_mean", device=device, dtype=dtype)
    prop_std = _config_tensor(connector_config, "property_std", device=device, dtype=dtype)
    delta_mean = _config_tensor(connector_config, "delta_mean", device=device, dtype=dtype)
    delta_std = _config_tensor(connector_config, "delta_std", device=device, dtype=dtype)
    props = condition.target_properties[:, :num_props] * prop_std[:, :num_props] + prop_mean[:, :num_props]
    deltas = condition.property_deltas[:, :num_props] * delta_std[:, :num_props] + delta_mean[:, :num_props]
    active = torch.sigmoid(condition.active_logits[:, :num_props])
    raw = torch.zeros(batch, latent_dim, device=device, dtype=dtype)
    raw[:, :fingerprint_dim] = fingerprint
    raw[:, fingerprint_dim : fingerprint_dim + num_props] = props
    raw[:, fingerprint_dim + 32 : fingerprint_dim + 32 + num_props] = deltas
    raw[:, fingerprint_dim + 64 : fingerprint_dim + 64 + num_props] = active
    return ((raw - latent_mean) / latent_std.clamp_min(1e-6)).to(dtype=torch.float32)


def _config_tensor(config: dict[str, object], key: str, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.as_tensor(config[key], dtype=dtype, device=device)


def _per_row_metrics(
    samples,
    gen: np.ndarray,
    target: np.ndarray,
    source: np.ndarray,
    *,
    prior: np.ndarray | None = None,
    fingerprint_dim: int,
) -> list[dict[str, object]]:
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
        row = {
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
        if prior is not None:
            prior_fp = prior[idx, :fingerprint_dim]
            prior_props = prior[idx, prop_start : prop_start + len(PROPERTY_COLUMNS)]
            prior_deltas = prior[idx, delta_start : delta_start + len(PROPERTY_COLUMNS)]
            row.update(
                {
                    "prior_latent_mae": _mae(prior[idx], target[idx]),
                    "prior_target_fingerprint_cosine": _cosine(prior_fp, target_fp[idx]),
                    "prior_target_property_mae": _mae(prior_props, target_props),
                    "prior_delta_mae": _mae(prior_deltas, target_deltas),
                    "generated_minus_prior_latent_mae": _mae(gen[idx], prior[idx]),
                }
            )
        rows.append(row)
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
        "prior_latent_mae",
        "prior_target_fingerprint_cosine",
        "prior_target_property_mae",
        "prior_delta_mae",
        "generated_minus_prior_latent_mae",
    ]
    numeric_keys = [key for key in numeric_keys if rows and key in rows[0]]
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


def _latent_block_summary(
    gen: np.ndarray,
    target: np.ndarray,
    source: np.ndarray,
    prior: np.ndarray,
    *,
    fingerprint_dim: int,
) -> dict[str, dict[str, float]]:
    blocks = {
        "fingerprint": slice(0, fingerprint_dim),
        "properties": slice(fingerprint_dim, fingerprint_dim + len(PROPERTY_COLUMNS)),
        "deltas": slice(fingerprint_dim + 32, fingerprint_dim + 32 + len(PROPERTY_COLUMNS)),
        "active": slice(fingerprint_dim + 64, fingerprint_dim + 64 + len(PROPERTY_COLUMNS)),
    }
    return {
        name: {
            "generated_target_mae": _mae(gen[:, slc], target[:, slc]),
            "prior_target_mae": _mae(prior[:, slc], target[:, slc]),
            "source_target_mae": _mae(source[:, slc], target[:, slc]),
            "generated_prior_mae": _mae(gen[:, slc], prior[:, slc]),
        }
        for name, slc in blocks.items()
    }


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
