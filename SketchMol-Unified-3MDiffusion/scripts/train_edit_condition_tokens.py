#!/usr/bin/env python3
"""Train edit-aware condition token connector on unified edit rows."""

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

from sketchmol_unified_3m_diffusion.edit_condition_tokens import (  # noqa: E402
    EditConditionTokenConnector,
    edit_condition_loss,
    source_aware_fingerprint_losses,
)
from sketchmol_unified_3m_diffusion.runtime import (  # noqa: E402
    checkpoint_dir,
    device_report,
    latest_checkpoint_path,
    move_batch_to_device,
    resolve_device,
)
from sketchmol_unified_3m_diffusion.unified_condition_dataset import EDIT_GENERATION, read_jsonl  # noqa: E402
from sketchmol_unified_3m_diffusion.unified_featurization import (  # noqa: E402
    active_property_vector,
    direction_label_vector,
    hidden_sequence_for_sample,
    molecule_feature,
    property_delta_vector,
    similarity_bin_label,
    target_property_vector,
)


class EditConditionDataset(Dataset):
    def __init__(self, jsonl: Path, *, token_dim: int, fingerprint_dim: int, limit: int | None = None):
        samples = [sample for sample in read_jsonl(jsonl) if sample.task_type == EDIT_GENERATION]
        if limit is not None:
            samples = samples[:limit]
        if not samples:
            raise ValueError(f"No edit_generation rows found in {jsonl}")
        self.hidden = np.stack([hidden_sequence_for_sample(sample, token_dim=token_dim) for sample in samples])
        self.target_properties = np.stack([target_property_vector(sample) for sample in samples])
        self.property_deltas = np.stack([property_delta_vector(sample) for sample in samples])
        self.active_mask = np.stack([active_property_vector(sample) for sample in samples])
        self.direction_labels = np.stack([direction_label_vector(sample) for sample in samples])
        self.target_fingerprint = np.stack([molecule_feature(sample.target_smiles, fingerprint_dim) for sample in samples])
        self.source_fingerprint = np.stack(
            [molecule_feature(sample.source_smiles or sample.molecule_smiles, fingerprint_dim) for sample in samples]
        )
        self.source_tanimoto = np.asarray([_float_or_nan(sample.source_tanimoto) for sample in samples], dtype=np.float32)
        self.similarity_bin = np.asarray([similarity_bin_label(sample) for sample in samples], dtype=np.int64)
        self.index = samples

        self.property_mean = self.target_properties.mean(axis=0, keepdims=True)
        self.property_std = self.target_properties.std(axis=0, keepdims=True)
        self.property_std = np.where(self.property_std < 1e-6, 1.0, self.property_std)
        self.delta_mean = self.property_deltas.mean(axis=0, keepdims=True)
        self.delta_std = self.property_deltas.std(axis=0, keepdims=True)
        self.delta_std = np.where(self.delta_std < 1e-6, 1.0, self.delta_std)
        self.target_properties = (self.target_properties - self.property_mean) / self.property_std
        self.property_deltas = (self.property_deltas - self.delta_mean) / self.delta_std

    def __len__(self) -> int:
        return int(self.hidden.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "hidden": torch.from_numpy(self.hidden[idx]),
            "target_properties": torch.from_numpy(self.target_properties[idx]).float(),
            "property_deltas": torch.from_numpy(self.property_deltas[idx]).float(),
            "active_mask": torch.from_numpy(self.active_mask[idx]).float(),
            "direction_labels": torch.from_numpy(self.direction_labels[idx]).long(),
            "target_fingerprint": torch.from_numpy(self.target_fingerprint[idx]).float(),
            "source_fingerprint": torch.from_numpy(self.source_fingerprint[idx]).float(),
            "source_tanimoto": torch.tensor(self.source_tanimoto[idx], dtype=torch.float32),
            "similarity_bin": torch.tensor(self.similarity_bin[idx], dtype=torch.long),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--token-dim", type=int, default=512)
    parser.add_argument("--context-dim", type=int, default=256)
    parser.add_argument("--num-queries", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--fingerprint-dim", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--export-features", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--source-similarity-loss-weight", type=float, default=0.15)
    parser.add_argument("--hard-negative-loss-weight", type=float, default=0.05)
    parser.add_argument("--source-aware-temperature", type=float, default=0.07)
    parser.add_argument("--hard-negative-margin", type=float, default=0.2)
    parser.add_argument(
        "--source-aware-shared-gradient",
        action="store_true",
        help=(
            "Allow source-aware fingerprint losses to update the shared connector trunk. "
            "By default they update only the fingerprint head, protecting property/delta control."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    print(json.dumps({"event": "device", **device_report(device)}, sort_keys=True))
    dataset = EditConditionDataset(args.train_jsonl, token_dim=args.token_dim, fingerprint_dim=args.fingerprint_dim, limit=args.limit)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.num_workers > 0,
    )
    model = EditConditionTokenConnector(
        input_hidden_dim=args.token_dim,
        context_dim=args.context_dim,
        num_queries=args.num_queries,
        hidden_dim=args.hidden_dim,
        fingerprint_dim=args.fingerprint_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    history = []
    start_epoch = 0
    config = _config(args, dataset)
    if args.resume_checkpoint is not None:
        payload = torch.load(args.resume_checkpoint, map_location=device)
        model.load_state_dict(payload["model_state"])
        optimizer.load_state_dict(payload["optimizer_state"])
        history = list(payload.get("history", []))
        start_epoch = int(payload.get("epoch", 0))
        print(json.dumps({"event": "resumed", "checkpoint": str(args.resume_checkpoint), "start_epoch": start_epoch}, sort_keys=True))

    for epoch in range(start_epoch, args.epochs):
        model.train()
        losses = []
        loss_logs: dict[str, list[float]] = {}
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            optimizer.zero_grad()
            output = model(batch["hidden"].float())
            loss, logs = edit_condition_loss(
                output,
                target_properties=batch["target_properties"],
                property_deltas=batch["property_deltas"],
                active_mask=batch["active_mask"],
                direction_labels=batch["direction_labels"],
                target_fingerprint=batch["target_fingerprint"],
                similarity_bin=batch["similarity_bin"],
                source_aware_temperature=args.source_aware_temperature,
                hard_negative_margin=args.hard_negative_margin,
                weights={
                    "fingerprint_bce": 0.5,
                    "direction_ce": 0.5,
                },
            )
            source_loss_weights = {
                "source_similarity_mse": args.source_similarity_loss_weight,
                "source_aware_hard_negative": args.hard_negative_loss_weight,
            }
            if any(float(weight) != 0.0 for weight in source_loss_weights.values()):
                fingerprint_logits = (
                    output.target_fingerprint_logits
                    if args.source_aware_shared_gradient
                    else model.fingerprint_head(output.pooled.detach())
                )
                source_losses = source_aware_fingerprint_losses(
                    fingerprint_logits,
                    target_properties=batch["target_properties"],
                    property_deltas=batch["property_deltas"],
                    active_mask=batch["active_mask"],
                    target_fingerprint=batch["target_fingerprint"],
                    source_fingerprint=batch["source_fingerprint"],
                    source_tanimoto=batch["source_tanimoto"],
                    temperature=args.source_aware_temperature,
                    hard_negative_margin=args.hard_negative_margin,
                )
                source_loss = sum(float(source_loss_weights.get(name, 1.0)) * value for name, value in source_losses.items())
                loss = loss + source_loss
                logs.update({name: value.detach() for name, value in source_losses.items()})
                logs["loss"] = loss.detach()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.item()))
            for name, value in logs.items():
                loss_logs.setdefault(name, []).append(float(value.item()))
        record = {"epoch": epoch + 1, "train_loss": float(np.mean(losses))}
        for name, values in sorted(loss_logs.items()):
            record[f"train_{name}"] = float(np.mean(values))
        history.append(record)
        print(json.dumps(record, sort_keys=True))
        if args.checkpoint_every > 0 and (epoch + 1) % args.checkpoint_every == 0:
            _save_checkpoint(
                args.output_dir,
                epoch=epoch + 1,
                model=model,
                optimizer=optimizer,
                config=config,
                history=history,
            )

    torch.save({"model_state": model.state_dict(), "config": config, "history": history}, args.output_dir / "edit_condition_connector.pt")
    (args.output_dir / "metrics.json").write_text(
        json.dumps({"history": history, "config": config, "device": device_report(device)}, indent=2) + "\n",
        encoding="utf-8",
    )
    if args.export_features:
        export_features(model, dataset, args.output_dir, device=device)


def _config(args: argparse.Namespace, dataset: EditConditionDataset) -> dict[str, object]:
    return {
        "token_dim": args.token_dim,
        "context_dim": args.context_dim,
        "num_queries": args.num_queries,
        "hidden_dim": args.hidden_dim,
        "fingerprint_dim": args.fingerprint_dim,
        "property_mean": dataset.property_mean.tolist(),
        "property_std": dataset.property_std.tolist(),
        "delta_mean": dataset.delta_mean.tolist(),
        "delta_std": dataset.delta_std.tolist(),
        "device": args.device,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": bool(args.pin_memory),
        "seed": args.seed,
        "source_similarity_loss_weight": args.source_similarity_loss_weight,
        "hard_negative_loss_weight": args.hard_negative_loss_weight,
        "source_aware_temperature": args.source_aware_temperature,
        "hard_negative_margin": args.hard_negative_margin,
        "source_aware_shared_gradient": bool(args.source_aware_shared_gradient),
    }


def _float_or_nan(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _save_checkpoint(
    output_dir: Path,
    *,
    epoch: int,
    model: EditConditionTokenConnector,
    optimizer: torch.optim.Optimizer,
    config: dict[str, object],
    history: list[dict[str, float]],
) -> None:
    payload = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": config,
        "history": history,
    }
    ckpt_dir = checkpoint_dir(output_dir)
    torch.save(payload, ckpt_dir / f"epoch_{epoch:04d}.pt")
    torch.save(payload, latest_checkpoint_path(output_dir))


@torch.no_grad()
def export_features(
    model: EditConditionTokenConnector,
    dataset: EditConditionDataset,
    output_dir: Path,
    *,
    device: torch.device,
) -> None:
    model.eval()
    hidden = torch.from_numpy(dataset.hidden).float()
    tokens = []
    pooled = []
    for start in range(0, hidden.shape[0], 128):
        output = model(hidden[start : start + 128].to(device))
        tokens.append(output.tokens.cpu().numpy())
        pooled.append(output.pooled.cpu().numpy())
    np.save(output_dir / "query_tokens.npy", np.concatenate(tokens, axis=0).astype(np.float32))
    np.save(output_dir / "pooled.npy", np.concatenate(pooled, axis=0).astype(np.float32))
    with (output_dir / "index.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["variant_id", "sample_id", "condition_id", "split", "source_smiles", "target_smiles"])
        writer.writeheader()
        for sample in dataset.index:
            writer.writerow(
                {
                    "variant_id": sample.sample_id,
                    "sample_id": sample.sample_id,
                    "condition_id": sample.metadata.get("condition_id", ""),
                    "split": sample.split,
                    "source_smiles": sample.source_smiles,
                    "target_smiles": sample.target_smiles,
                }
            )


if __name__ == "__main__":
    main()
