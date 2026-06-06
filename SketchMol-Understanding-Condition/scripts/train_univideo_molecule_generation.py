#!/usr/bin/env python3
"""Train a UniVideo-style understanding-conditioned molecular generator."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from sketchmol_understanding_condition.unified_condition_dataset import EDIT_GENERATION, PROPERTY_COLUMNS, read_jsonl  # noqa: E402
from sketchmol_understanding_condition.molecule_image_vae import (  # noqa: E402
    image_to_tensor,
    load_molecule_image_vae,
    tensor_to_uint8_image,
)
from sketchmol_understanding_condition.sketchmol_vae_adapter import load_sketchmol_vae_adapter  # noqa: E402
from sketchmol_understanding_condition.univideo_molecule import (  # noqa: E402
    FrozenConditionFeatureStore,
    SourceConditionedEditDenoiser,
    SourceConditionedGaussianLatentDiffusion,
    UniVideoMoleculeConnector,
    univideo_connector_alignment_loss,
    univideo_training_arrays,
)


class UniVideoMoleculeDataset(Dataset):
    """Edit rows with frozen understanding features and source/target latents."""

    def __init__(
        self,
        jsonl: Path,
        *,
        feature_store: FrozenConditionFeatureStore | None,
        fallback_token_dim: int,
        fingerprint_dim: int,
        latent_backend: str,
        image_vae: torch.nn.Module | None = None,
        image_vae_device: torch.device | None = None,
        image_size: int = 256,
        vae_batch_size: int = 16,
        limit: int | None = None,
        stats: dict[str, np.ndarray] | None = None,
    ) -> None:
        samples = [sample for sample in read_jsonl(jsonl) if sample.task_type == EDIT_GENERATION]
        if limit is not None:
            samples = samples[:limit]
        if not samples:
            raise ValueError(f"No edit_generation rows found in {jsonl}")
        if feature_store is not None:
            fallback_token_dim = feature_store.input_hidden_dim

        self.samples = samples
        self.rows = [
            univideo_training_arrays(
                sample,
                feature_store=feature_store,
                fallback_token_dim=fallback_token_dim,
                fingerprint_dim=fingerprint_dim,
            )
            for sample in samples
        ]
        self.fingerprint_dim = int(fingerprint_dim)
        self.latent_backend = latent_backend
        self.latent_shape: tuple[int, int, int] | None = None
        if _uses_image_latents(latent_backend):
            if image_vae is None:
                raise ValueError(f"image_vae is required when latent_backend={latent_backend!r}")
            source_latents, target_latents, latent_shape = _encode_image_vae_latents(
                samples,
                image_vae,
                image_size=image_size,
                device=image_vae_device or torch.device("cpu"),
                batch_size=vae_batch_size,
            )
            self.latent_shape = latent_shape
            for row, source_latent, target_latent in zip(self.rows, source_latents, target_latents):
                row["source_latent"] = source_latent
                row["target_latent"] = target_latent
        target_latents = np.stack([row["target_latent"] for row in self.rows]).astype(np.float32)
        target_properties = np.stack([row["target_properties"] for row in self.rows]).astype(np.float32)
        property_deltas = np.stack([row["property_deltas"] for row in self.rows]).astype(np.float32)
        if stats is None:
            self.stats = {
                "latent_mean": target_latents.mean(axis=0, keepdims=True),
                "latent_std": _safe_std(target_latents),
                "property_mean": target_properties.mean(axis=0, keepdims=True),
                "property_std": _safe_std(target_properties),
                "delta_mean": property_deltas.mean(axis=0, keepdims=True),
                "delta_std": _safe_std(property_deltas),
            }
        else:
            self.stats = stats

    @property
    def input_hidden_dim(self) -> int:
        return int(np.asarray(self.rows[0]["mllm_hidden"]).shape[-1])

    @property
    def latent_dim(self) -> int:
        return int(np.asarray(self.rows[0]["target_latent"]).shape[-1])

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, object]:
        row = self.rows[idx]
        sample = self.samples[idx]
        source_latent = np.asarray(row["source_latent"], dtype=np.float32)
        target_latent = np.asarray(row["target_latent"], dtype=np.float32)
        target_properties = np.asarray(row["target_properties"], dtype=np.float32)
        property_deltas = np.asarray(row["property_deltas"], dtype=np.float32)
        return {
            "mllm_hidden": torch.from_numpy(np.asarray(row["mllm_hidden"], dtype=np.float32)),
            "source_latent": torch.from_numpy(self.normalize_latent(source_latent)).float(),
            "target_latent": torch.from_numpy(self.normalize_latent(target_latent)).float(),
            "target_properties": torch.from_numpy(self.normalize_properties(target_properties)).float(),
            "property_deltas": torch.from_numpy(self.normalize_deltas(property_deltas)).float(),
            "active_mask": torch.from_numpy(np.asarray(row["active_mask"], dtype=np.float32)).float(),
            "direction_labels": torch.from_numpy(np.asarray(row["direction_labels"], dtype=np.int64)).long(),
            "similarity_bin": torch.tensor(int(row["similarity_bin"]), dtype=torch.long),
            "sample_id": sample.sample_id,
            "condition_id": sample.metadata.get("condition_id", ""),
            "source_smiles": sample.source_smiles,
            "target_smiles": sample.target_smiles,
            "property_count": sample.property_count,
        }

    def normalize_latent(self, value: np.ndarray) -> np.ndarray:
        return ((value[None, :] - self.stats["latent_mean"]) / self.stats["latent_std"])[0].astype(np.float32)

    def denormalize_latent(self, value: np.ndarray) -> np.ndarray:
        return (value * self.stats["latent_std"][0] + self.stats["latent_mean"][0]).astype(np.float32)

    def normalize_properties(self, value: np.ndarray) -> np.ndarray:
        return ((value[None, :] - self.stats["property_mean"]) / self.stats["property_std"])[0].astype(np.float32)

    def normalize_deltas(self, value: np.ndarray) -> np.ndarray:
        return ((value[None, :] - self.stats["delta_mean"]) / self.stats["delta_std"])[0].astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", required=True, type=Path)
    parser.add_argument("--eval-jsonl", type=Path, default=None)
    parser.add_argument("--condition-features-dir", type=Path, default=None)
    parser.add_argument("--condition-feature-array", choices=["query_tokens", "pooled"], default="query_tokens")
    parser.add_argument("--condition-feature-variant", default="full")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--fallback-token-dim", type=int, default=512)
    parser.add_argument("--context-dim", type=int, default=256)
    parser.add_argument("--num-queries", type=int, default=32)
    parser.add_argument("--connector-hidden-dim", type=int, default=512)
    parser.add_argument("--denoiser-hidden-dim", type=int, default=512)
    parser.add_argument("--denoiser-depth", type=int, default=4)
    parser.add_argument("--fingerprint-dim", type=int, default=512)
    parser.add_argument(
        "--latent-backend",
        choices=["fingerprint_property_vector", "image_vae", "sketchmol_vae"],
        default="fingerprint_property_vector",
    )
    parser.add_argument("--image-vae-checkpoint", type=Path, default=None)
    parser.add_argument("--sketchmol-root", type=Path, default=None)
    parser.add_argument("--sketchmol-vae-config", type=Path, default=None)
    parser.add_argument("--sketchmol-vae-checkpoint", type=Path, default=None)
    parser.add_argument("--sketchmol-scale-factor", type=float, default=1.0)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--vae-batch-size", type=int, default=16)
    parser.add_argument("--decode-eval-images", action="store_true")
    parser.add_argument("--max-decode-images", type=int, default=64)
    parser.add_argument("--timesteps", type=int, default=100)
    parser.add_argument("--diffusion-objective", choices=["pred_noise", "pred_x0"], default="pred_noise")
    parser.add_argument("--stage1-epochs", type=int, default=2)
    parser.add_argument("--stage2-epochs", type=int, default=5)
    parser.add_argument("--stage3-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--aux-loss-weight", type=float, default=0.25)
    parser.add_argument("--condition-dropout", type=float, default=0.1)
    parser.add_argument("--source-dropout", type=float, default=0.05)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--eval-limit", type=int, default=1000)
    parser.add_argument("--sample-steps", type=int, default=20)
    parser.add_argument("--sample-eta", type=float, default=0.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--export-condition-tokens", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(args.device)

    feature_store = None
    if args.condition_features_dir is not None:
        feature_store = FrozenConditionFeatureStore(
            args.condition_features_dir,
            array_name=args.condition_feature_array,
            variant=args.condition_feature_variant,
        )
    image_vae = None
    if args.latent_backend == "image_vae":
        if args.image_vae_checkpoint is None:
            raise ValueError("--image-vae-checkpoint is required when --latent-backend image_vae")
        image_vae = load_molecule_image_vae(args.image_vae_checkpoint, map_location=device).to(device)
        image_vae.eval()
        for param in image_vae.parameters():
            param.requires_grad = False
    elif args.latent_backend == "sketchmol_vae":
        missing = [
            name
            for name, value in {
                "--sketchmol-root": args.sketchmol_root,
                "--sketchmol-vae-config": args.sketchmol_vae_config,
                "--sketchmol-vae-checkpoint": args.sketchmol_vae_checkpoint,
            }.items()
            if value is None
        ]
        if missing:
            raise ValueError(f"{', '.join(missing)} are required when --latent-backend sketchmol_vae")
        image_vae = load_sketchmol_vae_adapter(
            sketchmol_root=args.sketchmol_root,
            config_path=args.sketchmol_vae_config,
            checkpoint_path=args.sketchmol_vae_checkpoint,
            map_location=device,
            scale_factor=args.sketchmol_scale_factor,
        )

    train_data = UniVideoMoleculeDataset(
        args.train_jsonl,
        feature_store=feature_store,
        fallback_token_dim=args.fallback_token_dim,
        fingerprint_dim=args.fingerprint_dim,
        latent_backend=args.latent_backend,
        image_vae=image_vae,
        image_vae_device=device,
        image_size=args.image_size,
        vae_batch_size=args.vae_batch_size,
        limit=args.limit,
    )
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=_collate)

    connector = UniVideoMoleculeConnector(
        input_hidden_dim=train_data.input_hidden_dim,
        latent_dim=train_data.latent_dim,
        context_dim=args.context_dim,
        num_queries=args.num_queries,
        hidden_dim=args.connector_hidden_dim,
    ).to(device)
    denoiser = SourceConditionedEditDenoiser(
        latent_dim=train_data.latent_dim,
        context_dim=args.context_dim,
        hidden_dim=args.denoiser_hidden_dim,
        depth=args.denoiser_depth,
    ).to(device)
    diffusion = SourceConditionedGaussianLatentDiffusion(
        denoiser,
        timesteps=args.timesteps,
        objective=args.diffusion_objective,
        condition_dropout=args.condition_dropout,
        source_dropout=args.source_dropout,
    ).to(device)

    history = []
    if args.stage1_epochs > 0:
        optimizer = torch.optim.AdamW(connector.parameters(), lr=args.lr, weight_decay=1e-4)
        history.extend(
            _train_stage(
                "stage1_connector_alignment",
                connector,
                diffusion,
                train_loader,
                optimizer,
                device=device,
                epochs=args.stage1_epochs,
                diffusion_weight=0.0,
                aux_weight=1.0,
            )
        )
    if args.stage2_epochs > 0:
        optimizer = torch.optim.AdamW([*connector.parameters(), *diffusion.parameters()], lr=args.lr, weight_decay=1e-4)
        history.extend(
            _train_stage(
                "stage2_diffusion_finetune",
                connector,
                diffusion,
                train_loader,
                optimizer,
                device=device,
                epochs=args.stage2_epochs,
                diffusion_weight=1.0,
                aux_weight=args.aux_loss_weight,
            )
        )
    if args.stage3_epochs > 0:
        optimizer = torch.optim.AdamW([*connector.parameters(), *diffusion.parameters()], lr=args.lr, weight_decay=1e-4)
        history.extend(
            _train_stage(
                "stage3_multitask_dropout",
                connector,
                diffusion,
                train_loader,
                optimizer,
                device=device,
                epochs=args.stage3_epochs,
                diffusion_weight=1.0,
                aux_weight=args.aux_loss_weight,
            )
        )

    config = {
        "train_jsonl": str(args.train_jsonl),
        "eval_jsonl": str(args.eval_jsonl) if args.eval_jsonl else None,
        "condition_features_dir": str(args.condition_features_dir) if args.condition_features_dir else None,
        "condition_feature_array": args.condition_feature_array,
        "condition_feature_variant": args.condition_feature_variant,
        "input_hidden_dim": train_data.input_hidden_dim,
        "latent_dim": train_data.latent_dim,
        "latent_backend": args.latent_backend,
        "context_dim": args.context_dim,
        "num_queries": args.num_queries,
        "connector_hidden_dim": args.connector_hidden_dim,
        "denoiser_hidden_dim": args.denoiser_hidden_dim,
        "denoiser_depth": args.denoiser_depth,
        "fingerprint_dim": args.fingerprint_dim,
        "latent_shape": list(train_data.latent_shape) if train_data.latent_shape else None,
        "image_vae_checkpoint": str(args.image_vae_checkpoint) if args.image_vae_checkpoint else None,
        "sketchmol_root": str(args.sketchmol_root) if args.sketchmol_root else None,
        "sketchmol_vae_config": str(args.sketchmol_vae_config) if args.sketchmol_vae_config else None,
        "sketchmol_vae_checkpoint": str(args.sketchmol_vae_checkpoint) if args.sketchmol_vae_checkpoint else None,
        "sketchmol_scale_factor": args.sketchmol_scale_factor,
        "timesteps": args.timesteps,
        "diffusion_objective": args.diffusion_objective,
        "sample_eta": args.sample_eta,
        "condition_dropout": args.condition_dropout,
        "source_dropout": args.source_dropout,
        "stats": {key: value.tolist() for key, value in train_data.stats.items()},
        "history": history,
    }
    checkpoint = {
        "connector_state": connector.state_dict(),
        "diffusion_state": diffusion.state_dict(),
        "config": config,
    }
    torch.save(checkpoint, args.output_dir / "univideo_molecule_generation.pt")
    (args.output_dir / "metrics.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if args.export_condition_tokens:
        _export_condition_tokens(connector, train_data, args.output_dir / "condition_tokens_train", device=device)

    if args.eval_jsonl is not None and args.eval_jsonl.exists() and args.eval_limit != 0:
        eval_data = UniVideoMoleculeDataset(
            args.eval_jsonl,
            feature_store=feature_store,
            fallback_token_dim=args.fallback_token_dim,
            fingerprint_dim=args.fingerprint_dim,
            latent_backend=args.latent_backend,
            image_vae=image_vae,
            image_vae_device=device,
            image_size=args.image_size,
            vae_batch_size=args.vae_batch_size,
            limit=args.eval_limit,
            stats=train_data.stats,
        )
        eval_metrics = _evaluate(
            connector,
            diffusion,
            eval_data,
            output_dir=args.output_dir / "eval_latent",
            batch_size=args.eval_batch_size,
            sample_steps=args.sample_steps,
            sample_eta=args.sample_eta,
            device=device,
            latent_backend=args.latent_backend,
            image_vae=image_vae if args.decode_eval_images else None,
            latent_shape=train_data.latent_shape,
            max_decode_images=args.max_decode_images,
        )
        print(json.dumps({"history": history, "eval": eval_metrics}, indent=2, sort_keys=True))
    else:
        print(json.dumps({"history": history, "checkpoint": str(args.output_dir / "univideo_molecule_generation.pt")}, indent=2, sort_keys=True))


def _train_stage(
    stage: str,
    connector: UniVideoMoleculeConnector,
    diffusion: SourceConditionedGaussianLatentDiffusion,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    epochs: int,
    diffusion_weight: float,
    aux_weight: float,
) -> list[dict[str, float | str | int]]:
    records = []
    for epoch in range(epochs):
        connector.train()
        diffusion.train()
        losses = []
        aux_losses = []
        diffusion_losses = []
        for batch in loader:
            batch = _to_device(batch, device)
            optimizer.zero_grad()
            condition = connector(batch["mllm_hidden"], batch["mllm_mask"])
            aux_loss, _ = univideo_connector_alignment_loss(
                condition,
                target_latent=batch["target_latent"],
                target_properties=batch["target_properties"],
                property_deltas=batch["property_deltas"],
                active_mask=batch["active_mask"],
                direction_labels=batch["direction_labels"],
                similarity_bin=batch["similarity_bin"],
                weights={
                    "target_latent_mse": 1.0,
                    "target_property_mse": 0.25,
                    "delta_mse": 0.25,
                    "active_bce": 0.1,
                    "direction_ce": 0.1,
                    "similarity_ce": 0.1,
                },
            )
            if diffusion_weight > 0:
                diffusion_loss = diffusion.loss(
                    batch["target_latent"],
                    batch["source_latent"],
                    condition.tokens,
                    condition.attention_mask,
                )
            else:
                diffusion_loss = torch.zeros((), device=device)
            loss = aux_weight * aux_loss + diffusion_weight * diffusion_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_([param for group in optimizer.param_groups for param in group["params"]], 1.0)
            optimizer.step()
            losses.append(float(loss.item()))
            aux_losses.append(float(aux_loss.item()))
            diffusion_losses.append(float(diffusion_loss.item()))
        record = {
            "stage": stage,
            "epoch": epoch + 1,
            "loss": float(np.mean(losses)),
            "aux_loss": float(np.mean(aux_losses)),
            "diffusion_loss": float(np.mean(diffusion_losses)),
        }
        records.append(record)
        print(json.dumps(record, sort_keys=True))
    return records


@torch.no_grad()
def _evaluate(
    connector: UniVideoMoleculeConnector,
    diffusion: SourceConditionedGaussianLatentDiffusion,
    dataset: UniVideoMoleculeDataset,
    *,
    output_dir: Path,
    batch_size: int,
    sample_steps: int,
    sample_eta: float,
    device: torch.device,
    latent_backend: str,
    image_vae: torch.nn.Module | None,
    latent_shape: tuple[int, int, int] | None,
    max_decode_images: int,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=_collate)
    connector.eval()
    diffusion.eval()
    rows = []
    generated = []
    targets = []
    sources = []
    for batch in loader:
        metadata = _metadata_from_batch(batch)
        batch = _to_device(batch, device)
        condition = connector(batch["mllm_hidden"], batch["mllm_mask"])
        sampled_norm = diffusion.sample(
            batch["source_latent"],
            condition.tokens,
            condition.attention_mask,
            steps=sample_steps,
            eta=sample_eta,
        )
        sampled = np.stack([dataset.denormalize_latent(row) for row in sampled_norm.cpu().numpy()])
        target = np.stack([dataset.denormalize_latent(row) for row in batch["target_latent"].cpu().numpy()])
        source = np.stack([dataset.denormalize_latent(row) for row in batch["source_latent"].cpu().numpy()])
        generated.append(sampled)
        targets.append(target)
        sources.append(source)
        rows.extend(
            _per_row_metrics(
                metadata,
                sampled,
                target,
                source,
                fingerprint_dim=dataset.fingerprint_dim,
                latent_backend=latent_backend,
            )
        )

    gen = np.concatenate(generated, axis=0)
    target_all = np.concatenate(targets, axis=0)
    source_all = np.concatenate(sources, axis=0)
    np.save(output_dir / "generated_latents.npy", gen.astype(np.float32))
    np.save(output_dir / "target_latents.npy", target_all.astype(np.float32))
    np.save(output_dir / "source_latents.npy", source_all.astype(np.float32))
    image_quality = {}
    if image_vae is not None and latent_shape is not None:
        image_quality["generated"] = _decode_latents_to_images(
            image_vae,
            gen,
            output_dir / "generated_images",
            latent_shape=latent_shape,
            max_images=max_decode_images,
            filename_prefix="generated",
        )
        oracle_max_images = min(max_decode_images, 128)
        image_quality["target_oracle"] = _decode_latents_to_images(
            image_vae,
            target_all,
            output_dir / "target_oracle_images",
            latent_shape=latent_shape,
            max_images=oracle_max_images,
        )
        image_quality["source_oracle"] = _decode_latents_to_images(
            image_vae,
            source_all,
            output_dir / "source_oracle_images",
            latent_shape=latent_shape,
            max_images=oracle_max_images,
        )
    _write_rows(output_dir / "predictions.csv", rows)
    metrics = _summarize(rows)
    metrics.update(
        {
            "rows": len(rows),
            "sample_steps": int(sample_steps),
            "sample_eta": float(sample_eta),
            "output_dir": str(output_dir),
        }
    )
    if image_quality:
        metrics["decoded_image_quality"] = image_quality
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metrics


@torch.no_grad()
def _export_condition_tokens(
    connector: UniVideoMoleculeConnector,
    dataset: UniVideoMoleculeDataset,
    output_dir: Path,
    *,
    device: torch.device,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    loader = DataLoader(dataset, batch_size=128, shuffle=False, collate_fn=_collate)
    connector.eval()
    tokens = []
    pooled = []
    rows = []
    for batch in loader:
        metadata = _metadata_from_batch(batch)
        batch = _to_device(batch, device)
        output = connector(batch["mllm_hidden"], batch["mllm_mask"])
        tokens.append(output.tokens.cpu().numpy().astype(np.float32))
        pooled.append(output.pooled.cpu().numpy().astype(np.float32))
        rows.extend(metadata)
    np.save(output_dir / "query_tokens.npy", np.concatenate(tokens, axis=0))
    np.save(output_dir / "pooled.npy", np.concatenate(pooled, axis=0))
    _write_rows(output_dir / "index.csv", rows)


def _collate(items: list[dict[str, object]]) -> dict[str, object]:
    max_len = max(int(item["mllm_hidden"].shape[0]) for item in items)  # type: ignore[index, union-attr]
    hidden_dim = int(items[0]["mllm_hidden"].shape[-1])  # type: ignore[index, union-attr]
    hidden = torch.zeros(len(items), max_len, hidden_dim, dtype=torch.float32)
    mask = torch.zeros(len(items), max_len, dtype=torch.bool)
    out: dict[str, object] = {"mllm_hidden": hidden, "mllm_mask": mask}
    tensor_keys = [
        "source_latent",
        "target_latent",
        "target_properties",
        "property_deltas",
        "active_mask",
        "direction_labels",
        "similarity_bin",
    ]
    for row_idx, item in enumerate(items):
        cur = item["mllm_hidden"]  # type: ignore[index]
        length = int(cur.shape[0])
        hidden[row_idx, :length] = cur
        mask[row_idx, :length] = True
    for key in tensor_keys:
        out[key] = torch.stack([item[key] for item in items])  # type: ignore[list-item, index]
    for key in ["sample_id", "condition_id", "source_smiles", "target_smiles", "property_count"]:
        out[key] = [str(item[key]) for item in items]
    return out


def _to_device(batch: dict[str, object], device: torch.device) -> dict[str, object]:
    out = {}
    for key, value in batch.items():
        out[key] = value.to(device) if torch.is_tensor(value) else value
    return out


def _metadata_from_batch(batch: dict[str, object]) -> list[dict[str, str]]:
    size = len(batch["sample_id"])  # type: ignore[arg-type]
    return [
        {
            "sample_id": batch["sample_id"][idx],  # type: ignore[index]
            "condition_id": batch["condition_id"][idx],  # type: ignore[index]
            "source_smiles": batch["source_smiles"][idx],  # type: ignore[index]
            "target_smiles": batch["target_smiles"][idx],  # type: ignore[index]
            "property_count": batch["property_count"][idx],  # type: ignore[index]
        }
        for idx in range(size)
    ]


def _per_row_metrics(
    metadata: list[dict[str, str]],
    gen: np.ndarray,
    target: np.ndarray,
    source: np.ndarray,
    *,
    fingerprint_dim: int,
    latent_backend: str,
) -> list[dict[str, object]]:
    rows = []
    if _uses_image_latents(latent_backend):
        for idx, meta in enumerate(metadata):
            rows.append(
                {
                    **meta,
                    "latent_mse": _mse(gen[idx], target[idx]),
                    "latent_mae": _mae(gen[idx], target[idx]),
                    "target_latent_cosine": _cosine(gen[idx], target[idx]),
                    "source_latent_cosine": _cosine(gen[idx], source[idx]),
                    "source_target_latent_cosine": _cosine(source[idx], target[idx]),
                }
            )
        return rows

    gen_fp = gen[:, :fingerprint_dim]
    target_fp = target[:, :fingerprint_dim]
    source_fp = source[:, :fingerprint_dim]
    prop_start = fingerprint_dim
    for idx, meta in enumerate(metadata):
        rows.append(
            {
                **meta,
                "latent_mse": _mse(gen[idx], target[idx]),
                "latent_mae": _mae(gen[idx], target[idx]),
                "target_fingerprint_cosine": _cosine(gen_fp[idx], target_fp[idx]),
                "source_fingerprint_cosine": _cosine(gen_fp[idx], source_fp[idx]),
                "source_target_fingerprint_cosine": _cosine(source_fp[idx], target_fp[idx]),
                "target_property_mae": _mae(
                    gen[idx, prop_start : prop_start + len(PROPERTY_COLUMNS)],
                    target[idx, prop_start : prop_start + len(PROPERTY_COLUMNS)],
                ),
                "source_target_property_mae": _mae(
                    source[idx, prop_start : prop_start + len(PROPERTY_COLUMNS)],
                    target[idx, prop_start : prop_start + len(PROPERTY_COLUMNS)],
                ),
            }
        )
    return rows


def _summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    keys = [
        "latent_mse",
        "latent_mae",
    ]
    optional_keys = [
        "target_fingerprint_cosine",
        "source_fingerprint_cosine",
        "source_target_fingerprint_cosine",
        "target_property_mae",
        "source_target_property_mae",
        "target_latent_cosine",
        "source_latent_cosine",
        "source_target_latent_cosine",
    ]
    keys.extend([key for key in optional_keys if rows and key in rows[0]])
    summary = {"overall": _mean_metrics(rows, keys)}
    by_count = {}
    for count in sorted({str(row.get("property_count", "")) for row in rows}):
        selected = [row for row in rows if str(row.get("property_count", "")) == count]
        by_count[count or "unknown"] = _mean_metrics(selected, keys)
    summary["by_property_count"] = by_count
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


def _uses_image_latents(latent_backend: str) -> bool:
    return latent_backend in {"image_vae", "sketchmol_vae"}


@torch.no_grad()
def _encode_image_vae_latents(
    samples,
    image_vae: torch.nn.Module,
    *,
    image_size: int,
    device: torch.device,
    batch_size: int,
) -> tuple[list[np.ndarray], list[np.ndarray], tuple[int, int, int]]:
    image_vae.eval()
    source_latents = _encode_sample_image_batches(
        image_vae,
        samples,
        role="source",
        image_size=image_size,
        device=device,
        batch_size=batch_size,
    )
    target_latents = _encode_sample_image_batches(
        image_vae,
        samples,
        role="target",
        image_size=image_size,
        device=device,
        batch_size=batch_size,
    )
    latent_shape = tuple(int(dim) for dim in source_latents[0].shape)
    return (
        [latent.reshape(-1).astype(np.float32) for latent in source_latents],
        [latent.reshape(-1).astype(np.float32) for latent in target_latents],
        latent_shape,
    )


@torch.no_grad()
def _encode_sample_image_batches(
    image_vae: torch.nn.Module,
    samples,
    *,
    role: str,
    image_size: int,
    device: torch.device,
    batch_size: int,
) -> list[np.ndarray]:
    out = []
    for start in range(0, len(samples), batch_size):
        image_tensors = []
        for sample in samples[start : start + batch_size]:
            if role == "source":
                image_tensors.append(
                    image_to_tensor(image_path=sample.source_image, smiles=sample.source_smiles, image_size=image_size)
                )
            elif role == "target":
                image_tensors.append(
                    image_to_tensor(image_path=sample.target_image, smiles=sample.target_smiles, image_size=image_size)
                )
            else:
                raise ValueError(f"Unsupported image role: {role}")
        batch = torch.stack(image_tensors).to(device)
        latents = image_vae.encode(batch, sample=False).detach().cpu().numpy().astype(np.float32)
        out.extend([latent for latent in latents])
    return out


@torch.no_grad()
def _decode_latents_to_images(
    image_vae: torch.nn.Module,
    latents: np.ndarray,
    output_dir: Path,
    *,
    latent_shape: tuple[int, int, int],
    max_images: int,
    filename_prefix: str = "decoded",
) -> dict[str, float]:
    from PIL import Image

    output_dir.mkdir(parents=True, exist_ok=True)
    device = next(image_vae.parameters()).device
    total = min(max_images, int(latents.shape[0]))
    stats = []
    for start in range(0, total, 16):
        chunk = latents[start : start + 16].reshape(-1, *latent_shape)
        tensor = torch.from_numpy(chunk.astype(np.float32)).to(device)
        decoded = image_vae.decode(tensor).detach().cpu()
        for offset, image_tensor in enumerate(decoded):
            image = tensor_to_uint8_image(image_tensor)
            stats.append(_image_quality_stats(image))
            Image.fromarray(image).save(output_dir / f"{filename_prefix}_{start + offset:05d}.png")
    return _summarize_image_quality(stats)


def _image_quality_stats(image: np.ndarray) -> dict[str, float]:
    gray = image.astype(np.float32).mean(axis=2) / 255.0
    return {
        "mean_intensity": float(gray.mean()),
        "std_intensity": float(gray.std()),
        "nonwhite_fraction": float(np.mean(gray < 0.98)),
        "dark_fraction": float(np.mean(gray < 0.75)),
    }


def _summarize_image_quality(stats: list[dict[str, float]]) -> dict[str, float]:
    if not stats:
        return {
            "images": 0.0,
            "mean_intensity": 0.0,
            "std_intensity": 0.0,
            "nonwhite_fraction": 0.0,
            "dark_fraction": 0.0,
        }
    keys = list(stats[0].keys())
    return {
        "images": float(len(stats)),
        **{key: float(np.mean([row[key] for row in stats])) for key in keys},
    }


def _safe_std(values: np.ndarray) -> np.ndarray:
    std = values.std(axis=0, keepdims=True)
    return np.where(std < 1e-6, 1.0, std).astype(np.float32)


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


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


if __name__ == "__main__":
    main()
