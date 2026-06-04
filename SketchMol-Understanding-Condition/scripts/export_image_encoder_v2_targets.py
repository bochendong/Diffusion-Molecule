#!/usr/bin/env python
"""Export RDKit supervision targets for trainable image encoder v2."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from sketchmol_understanding_condition.chem import morgan_fingerprint_bits, molecular_properties
from sketchmol_understanding_condition.retrieval_data import read_variant_rows


PROPERTY_KEYS = ("MolWt", "LogP", "QED", "TPSA", "HBD", "HBA", "rotatable")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-variants-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--fingerprint-bits", type=int, default=512)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = _unique_full_rows(read_variant_rows(args.baseline_variants_csv))
    if args.limit is not None:
        rows = rows[: args.limit]

    kept = []
    image_paths = []
    splits = []
    fingerprints = []
    properties = []
    for row in rows:
        fp = morgan_fingerprint_bits(row["source_smiles"], n_bits=args.fingerprint_bits)
        props = molecular_properties(row["source_smiles"])
        image_path = row.get("source_image", "")
        if fp is None or props is None or not image_path or not Path(image_path).exists():
            continue
        kept.append(row)
        image_paths.append(image_path)
        splits.append(row.get("split", ""))
        fingerprints.append(np.asarray(fp, dtype=np.float32))
        properties.append(np.asarray([props.get(key, 0.0) for key in PROPERTY_KEYS], dtype=np.float32))

    targets_npz = args.output_dir / "image_encoder_v2_targets.npz"
    np.savez_compressed(
        targets_npz,
        image_paths=np.asarray(image_paths, dtype=object),
        splits=np.asarray(splits, dtype=object),
        fingerprints=np.stack(fingerprints).astype(np.float32),
        properties=np.stack(properties).astype(np.float32),
        property_keys=np.asarray(PROPERTY_KEYS, dtype=object),
        pair_ids=np.asarray([row.get("pair_id", "") for row in kept], dtype=object),
        source_smiles=np.asarray([row.get("source_smiles", "") for row in kept], dtype=object),
    )
    _write_index(args.output_dir / "index.csv", kept)
    summary = {
        "baseline_variants_csv": str(args.baseline_variants_csv),
        "targets_npz": str(targets_npz),
        "rows": len(kept),
        "fingerprint_bits": args.fingerprint_bits,
        "property_keys": list(PROPERTY_KEYS),
        "train_rows": int(sum(1 for split in splits if split == "train")),
        "eval_rows": int(sum(1 for split in splits if split == "eval")),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


def _unique_full_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    out = []
    seen = set()
    for row in rows:
        if row.get("variant") != "full":
            continue
        pair_id = row.get("pair_id", "")
        if pair_id in seen:
            continue
        seen.add(pair_id)
        out.append(row)
    return out


def _write_index(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = ["pair_id", "split", "source_image", "source_smiles", "objective", "direction"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


if __name__ == "__main__":
    main()
