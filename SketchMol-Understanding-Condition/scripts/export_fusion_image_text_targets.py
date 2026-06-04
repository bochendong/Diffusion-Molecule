#!/usr/bin/env python
"""Export image+text fusion supervision targets for edit-aware encoder."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from sketchmol_understanding_condition.chem import molecular_properties
from sketchmol_understanding_condition.delta_bucket_classifier import _fit_bucket_thresholds, _label
from sketchmol_understanding_condition.retrieval_data import read_variant_rows


PROPERTY_KEYS = ("MolWt", "LogP", "QED", "TPSA", "HBD", "HBA", "rotatable")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-variants-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = [row for row in read_variant_rows(args.baseline_variants_csv) if row.get("variant") == "full"]
    if args.limit is not None:
        rows = rows[: args.limit]
    train_rows = [row for row in rows if row.get("split") == "train"]
    thresholds = _fit_bucket_thresholds(train_rows)
    label_names = sorted({_label(row, thresholds) for row in rows})
    label_to_idx = {label: idx for idx, label in enumerate(label_names)}

    kept = []
    image_paths = []
    prompts = []
    splits = []
    labels = []
    property_deltas = []
    objective_deltas = []
    for row in rows:
        image_path = row.get("source_image", "")
        if not image_path or not Path(image_path).exists():
            continue
        source_props = molecular_properties(row["source_smiles"])
        target_props = molecular_properties(row["target_smiles"])
        if source_props is None or target_props is None:
            continue
        kept.append(row)
        image_paths.append(image_path)
        prompts.append(row.get("prompt", ""))
        splits.append(row.get("split", ""))
        labels.append(label_to_idx[_label(row, thresholds)])
        property_deltas.append(
            np.asarray([target_props.get(key, 0.0) - source_props.get(key, 0.0) for key in PROPERTY_KEYS], dtype=np.float32)
        )
        objective_deltas.append(float(row.get("property_delta", "0") or 0.0))

    targets_npz = args.output_dir / "fusion_image_text_targets.npz"
    np.savez_compressed(
        targets_npz,
        image_paths=np.asarray(image_paths, dtype=object),
        prompts=np.asarray(prompts, dtype=object),
        splits=np.asarray(splits, dtype=object),
        labels=np.asarray(labels, dtype=np.int64),
        label_names=np.asarray(label_names, dtype=object),
        property_deltas=np.stack(property_deltas).astype(np.float32),
        objective_deltas=np.asarray(objective_deltas, dtype=np.float32),
        property_keys=np.asarray(PROPERTY_KEYS, dtype=object),
        pair_ids=np.asarray([row.get("pair_id", "") for row in kept], dtype=object),
    )
    _write_index(args.output_dir / "index.csv", kept, labels, label_names)
    summary = {
        "baseline_variants_csv": str(args.baseline_variants_csv),
        "targets_npz": str(targets_npz),
        "rows": len(kept),
        "train_rows": int(sum(1 for split in splits if split == "train")),
        "eval_rows": int(sum(1 for split in splits if split == "eval")),
        "label_names": label_names,
        "bucket_thresholds": thresholds,
        "property_keys": list(PROPERTY_KEYS),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


def _write_index(path: Path, rows: list[dict[str, str]], labels: list[int], label_names: list[str]) -> None:
    fieldnames = ["pair_id", "split", "source_image", "prompt", "objective", "direction", "label", "property_delta"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row, label in zip(rows, labels):
            writer.writerow(
                {
                    "pair_id": row.get("pair_id", ""),
                    "split": row.get("split", ""),
                    "source_image": row.get("source_image", ""),
                    "prompt": row.get("prompt", ""),
                    "objective": row.get("objective", ""),
                    "direction": row.get("direction", ""),
                    "label": label_names[int(label)],
                    "property_delta": row.get("property_delta", ""),
                }
            )


if __name__ == "__main__":
    main()
