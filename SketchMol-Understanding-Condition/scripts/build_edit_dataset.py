#!/usr/bin/env python
"""Build an instruction-guided scaffold edit dataset from a SMILES CSV."""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from sketchmol_understanding_condition.chem import molecular_properties, rdkit_version, render_molecule_image
from sketchmol_understanding_condition.pair_mining import MoleculeRecord, mine_scaffold_edit_pairs


@dataclass(frozen=True)
class EditDatasetRow:
    pair_id: str
    split: str
    source_id: str
    target_id: str
    source_smiles: str
    target_smiles: str
    source_image: str
    target_image: str
    instruction: str
    scaffold: str
    similarity: float
    property_name: str
    property_delta: str


def read_molecules(input_csv: Path, smiles_column: str, id_column: str | None, limit: int | None) -> list[MoleculeRecord]:
    records: list[MoleculeRecord] = []
    with input_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if smiles_column not in (reader.fieldnames or []):
            raise ValueError(f"Missing SMILES column {smiles_column!r} in {input_csv}")

        for idx, row in enumerate(reader):
            if limit is not None and len(records) >= limit:
                break
            smiles = (row.get(smiles_column) or "").strip()
            if not smiles:
                continue
            props = molecular_properties(smiles)
            if props is None:
                continue
            records.append(
                MoleculeRecord(
                    smiles=smiles,
                    mol_id=(row.get(id_column) if id_column else None) or f"mol_{idx:06d}",
                    properties=props,
                )
            )
    return records


def build_dataset(
    input_csv: Path,
    output_dir: Path,
    *,
    smiles_column: str = "smiles",
    id_column: str | None = None,
    property_name: str | None = None,
    direction: str = "increase",
    limit: int | None = None,
    max_pairs: int | None = None,
    min_abs_delta: float = 0.1,
    min_similarity: float = 0.2,
    max_similarity: float = 0.95,
    max_pairs_per_scaffold: int | None = None,
    image_size: int = 256,
    eval_fraction: float = 0.2,
    seed: int = 7,
) -> dict[str, float | int | str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    records = read_molecules(input_csv, smiles_column, id_column, limit)
    pairs = mine_scaffold_edit_pairs(
        records,
        property_name=property_name,
        direction=direction,  # type: ignore[arg-type]
        min_abs_delta=min_abs_delta,
        min_similarity=min_similarity,
        max_similarity=max_similarity,
        max_pairs=max_pairs,
        max_pairs_per_scaffold=max_pairs_per_scaffold,
    )

    rng = random.Random(seed)
    shuffled = list(pairs)
    rng.shuffle(shuffled)
    eval_pair_ids = _component_aware_eval_pair_ids(shuffled, eval_fraction=eval_fraction, seed=seed)

    rows: list[EditDatasetRow] = []
    for idx, pair in enumerate(shuffled):
        pair_id = f"edit_{idx:06d}"
        split = "eval" if idx in eval_pair_ids else "train"
        source_image = render_molecule_image(pair.source_smiles, image_dir / f"{pair_id}_source.png", image_size)
        target_image = render_molecule_image(pair.target_smiles, image_dir / f"{pair_id}_target.png", image_size)
        if source_image is None or target_image is None:
            continue
        rows.append(
            EditDatasetRow(
                pair_id=pair_id,
                split=split,
                source_id=pair.source_id or "",
                target_id=pair.target_id or "",
                source_smiles=pair.source_smiles,
                target_smiles=pair.target_smiles,
                source_image=source_image,
                target_image=target_image,
                instruction=pair.instruction,
                scaffold=pair.scaffold or "",
                similarity=pair.similarity,
                property_name=pair.property_name or "",
                property_delta="" if pair.property_delta is None else f"{pair.property_delta:.6f}",
            )
        )

    manifest_path = output_dir / "edit_pairs.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(EditDatasetRow.__dataclass_fields__.keys())
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))

    for split in ("train", "eval"):
        split_path = output_dir / f"{split}.csv"
        with split_path.open("w", newline="", encoding="utf-8") as handle:
            fieldnames = list(EditDatasetRow.__dataclass_fields__.keys())
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                if row.split == split:
                    writer.writerow(asdict(row))

    summary: dict[str, float | int | str] = {
        "rdkit_version": rdkit_version(),
        "input_csv": str(input_csv),
        "molecules": len(records),
        "mined_pairs": len(pairs),
        "rendered_pairs": len(rows),
        "train_pairs": sum(1 for row in rows if row.split == "train"),
        "eval_pairs": sum(1 for row in rows if row.split == "eval"),
        "property_name": property_name or "",
        "direction": direction if property_name else "",
        "manifest": str(manifest_path),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--smiles-column", default="smiles")
    parser.add_argument("--id-column", default=None)
    parser.add_argument("--property-name", default=None)
    parser.add_argument("--direction", choices=["increase", "decrease"], default="increase")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--min-abs-delta", type=float, default=0.1)
    parser.add_argument("--min-similarity", type=float, default=0.2)
    parser.add_argument("--max-similarity", type=float, default=0.95)
    parser.add_argument("--max-pairs-per-scaffold", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--eval-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def _component_aware_eval_pair_ids(pairs, *, eval_fraction: float, seed: int) -> set[int]:
    """Split by molecule connected components to avoid train/eval molecule leakage."""

    if not pairs:
        return set()
    molecule_graph: dict[str, set[str]] = {}
    molecule_to_pair_ids: dict[str, set[int]] = {}
    for idx, pair in enumerate(pairs):
        source = pair.source_smiles
        target = pair.target_smiles
        molecule_graph.setdefault(source, set()).add(target)
        molecule_graph.setdefault(target, set()).add(source)
        molecule_to_pair_ids.setdefault(source, set()).add(idx)
        molecule_to_pair_ids.setdefault(target, set()).add(idx)

    seen: set[str] = set()
    components: list[set[int]] = []
    for molecule in molecule_graph:
        if molecule in seen:
            continue
        stack = [molecule]
        component_molecules: set[str] = set()
        seen.add(molecule)
        while stack:
            current = stack.pop()
            component_molecules.add(current)
            for neighbor in molecule_graph.get(current, set()):
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        component_pair_ids: set[int] = set()
        for component_molecule in component_molecules:
            component_pair_ids.update(molecule_to_pair_ids.get(component_molecule, set()))
        components.append(component_pair_ids)

    rng = random.Random(seed)
    rng.shuffle(components)
    target_eval = max(1, int(round(len(pairs) * eval_fraction)))
    eval_pair_ids: set[int] = set()
    for component in sorted(components, key=len):
        candidate = eval_pair_ids | component
        if len(candidate) <= target_eval or not eval_pair_ids:
            eval_pair_ids = candidate
        if len(eval_pair_ids) >= target_eval:
            break

    return eval_pair_ids


def main() -> None:
    args = parse_args()
    summary = build_dataset(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        smiles_column=args.smiles_column,
        id_column=args.id_column,
        property_name=args.property_name,
        direction=args.direction,
        limit=args.limit,
        max_pairs=args.max_pairs,
        min_abs_delta=args.min_abs_delta,
        min_similarity=args.min_similarity,
        max_similarity=args.max_similarity,
        max_pairs_per_scaffold=args.max_pairs_per_scaffold,
        image_size=args.image_size,
        eval_fraction=args.eval_fraction,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
