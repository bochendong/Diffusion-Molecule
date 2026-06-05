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

from sketchmol_understanding_condition.edit_condition_tokens import (  # noqa: E402
    EditConditionTokenConnector,
    edit_condition_loss,
)
from sketchmol_understanding_condition.unified_condition_dataset import EDIT_GENERATION, read_jsonl  # noqa: E402
from sketchmol_understanding_condition.unified_featurization import (  # noqa: E402
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
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--export-features", action="store_true")
    parser.add_argument("--seed", type=int, default=11)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = EditConditionDataset(args.train_jsonl, token_dim=args.token_dim, fingerprint_dim=args.fingerprint_dim, limit=args.limit)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    model = EditConditionTokenConnector(
        input_hidden_dim=args.token_dim,
        context_dim=args.context_dim,
        num_queries=args.num_queries,
        hidden_dim=args.hidden_dim,
        fingerprint_dim=args.fingerprint_dim,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    history = []
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for batch in loader:
            optimizer.zero_grad()
            output = model(batch["hidden"].float())
            loss, _ = edit_condition_loss(
                output,
                target_properties=batch["target_properties"],
                property_deltas=batch["property_deltas"],
                active_mask=batch["active_mask"],
                direction_labels=batch["direction_labels"],
                target_fingerprint=batch["target_fingerprint"],
                similarity_bin=batch["similarity_bin"],
                weights={"fingerprint_bce": 0.5, "direction_ce": 0.5},
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.item()))
        record = {"epoch": epoch + 1, "train_loss": float(np.mean(losses))}
        history.append(record)
        print(json.dumps(record, sort_keys=True))

    config = {
        "token_dim": args.token_dim,
        "context_dim": args.context_dim,
        "num_queries": args.num_queries,
        "hidden_dim": args.hidden_dim,
        "fingerprint_dim": args.fingerprint_dim,
        "property_mean": dataset.property_mean.tolist(),
        "property_std": dataset.property_std.tolist(),
        "delta_mean": dataset.delta_mean.tolist(),
        "delta_std": dataset.delta_std.tolist(),
    }
    torch.save({"model_state": model.state_dict(), "config": config, "history": history}, args.output_dir / "edit_condition_connector.pt")
    (args.output_dir / "metrics.json").write_text(json.dumps({"history": history, "config": config}, indent=2) + "\n", encoding="utf-8")
    if args.export_features:
        export_features(model, dataset, args.output_dir)


@torch.no_grad()
def export_features(model: EditConditionTokenConnector, dataset: EditConditionDataset, output_dir: Path) -> None:
    model.eval()
    hidden = torch.from_numpy(dataset.hidden).float()
    tokens = []
    pooled = []
    for start in range(0, hidden.shape[0], 128):
        output = model(hidden[start : start + 128])
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

