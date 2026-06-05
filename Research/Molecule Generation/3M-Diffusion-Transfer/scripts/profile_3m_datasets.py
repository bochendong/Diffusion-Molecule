#!/usr/bin/env python3
"""Profile TSV datasets shipped with the local 3M-Diffusion clone."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from statistics import mean, median


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("Research/Molecule Generation/3M-Diffusion"),
        help="Path to the local 3M-Diffusion repository.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("Research/Molecule Generation/3M-Diffusion-Transfer/outputs/3m_dataset_profile.json"),
        help="Where to write the profile JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = args.source_root / "data"
    if not data_root.exists():
        raise FileNotFoundError(f"data directory not found: {data_root}")

    profile = {
        "source_root": str(args.source_root),
        "data_root": str(data_root),
        "datasets": {},
    }
    for dataset_dir in sorted(path for path in data_root.iterdir() if path.is_dir()):
        profile["datasets"][dataset_dir.name] = profile_dataset(dataset_dir)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(profile, indent=2, sort_keys=True))


def profile_dataset(dataset_dir: Path) -> dict[str, object]:
    splits = {}
    total_rows = 0
    all_smiles = []
    for path in sorted(dataset_dir.glob("*.txt")):
        split_profile = profile_tsv(path)
        splits[path.name] = split_profile
        total_rows += int(split_profile["rows"])
        all_smiles.extend(split_profile["_smiles"])
        del split_profile["_smiles"]

    return {
        "path": str(dataset_dir),
        "total_rows": total_rows,
        "unique_smiles": len(set(all_smiles)),
        "duplicate_smiles": total_rows - len(set(all_smiles)),
        "splits": splits,
    }


def profile_tsv(path: Path) -> dict[str, object]:
    rows = []
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            rows.append(row)

    smiles = [row.get("SMILES", "") for row in rows]
    descriptions = [row.get("description", "") for row in rows]
    columns = list(rows[0].keys()) if rows else []
    desc_word_lengths = [len(text.split()) for text in descriptions if text]
    smiles_lengths = [len(item) for item in smiles if item]
    missing = Counter()
    for row in rows:
        for column in columns:
            if not str(row.get(column, "")).strip():
                missing[column] += 1

    return {
        "path": str(path),
        "rows": len(rows),
        "columns": columns,
        "unique_smiles": len(set(smiles)),
        "duplicate_smiles": len(smiles) - len(set(smiles)),
        "missing_values": dict(missing),
        "description_words": summarize_numbers(desc_word_lengths),
        "smiles_chars": summarize_numbers(smiles_lengths),
        "_smiles": smiles,
    }


def summarize_numbers(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "count": len(values),
        "mean": round(mean(values), 3),
        "median": round(median(values), 3),
        "min": min(values),
        "max": max(values),
    }


if __name__ == "__main__":
    main()

