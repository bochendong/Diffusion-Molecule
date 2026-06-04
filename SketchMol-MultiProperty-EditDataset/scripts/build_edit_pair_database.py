#!/usr/bin/env python
"""Mine large scaffold-preserving multi-property edit pairs."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path


REPO_DIR = Path(__file__).resolve().parents[2]
DATASET_DIR = REPO_DIR / "SketchMol-MultiProperty-EditDataset"
if str(DATASET_DIR) not in sys.path:
    sys.path.insert(0, str(DATASET_DIR))

from sketchmol_multiproperty_dataset.common import (
    PROPERTY_COLUMNS,
    active_property_deltas,
    direction_from_delta,
    json_dumps,
)
from sketchmol_understanding_condition.chem import canonical_smiles, morgan_tanimoto, render_molecule_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--molecule-db-csv", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--max-pairs", type=int, default=100000)
    parser.add_argument("--max-pairs-per-scaffold", type=int, default=300)
    parser.add_argument("--max-molecules-per-scaffold", type=int, default=300)
    parser.add_argument("--min-active-properties", type=int, default=2)
    parser.add_argument("--threshold-scale", type=float, default=1.0)
    parser.add_argument("--min-similarity", type=float, default=0.2)
    parser.add_argument("--max-similarity", type=float, default=0.9)
    parser.add_argument("--eval-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--image-dir", type=Path, default=None)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--render-images", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    if args.image_dir is not None:
        args.image_dir.mkdir(parents=True, exist_ok=True)

    molecules = _read_molecules(args.molecule_db_csv)
    by_scaffold: dict[str, list[dict[str, str]]] = {}
    for row in molecules:
        by_scaffold.setdefault(row["scaffold"], []).append(row)

    scaffold_items = list(by_scaffold.items())
    rng.shuffle(scaffold_items)
    pairs = []
    skipped_similarity = 0
    skipped_delta = 0

    for scaffold, group in scaffold_items:
        if len(pairs) >= args.max_pairs:
            break
        if len(group) < 2:
            continue
        group = list(group)
        rng.shuffle(group)
        if len(group) > args.max_molecules_per_scaffold:
            group = group[: args.max_molecules_per_scaffold]
        candidate_indices = [(i, j) for i in range(len(group)) for j in range(len(group)) if i != j]
        rng.shuffle(candidate_indices)
        scaffold_pairs = 0
        for source_idx, target_idx in candidate_indices:
            if scaffold_pairs >= args.max_pairs_per_scaffold or len(pairs) >= args.max_pairs:
                break
            source = group[source_idx]
            target = group[target_idx]
            similarity = morgan_tanimoto(source["canonical_smiles"], target["canonical_smiles"])
            if similarity is None or not (args.min_similarity <= similarity <= args.max_similarity):
                skipped_similarity += 1
                continue
            source_props = _props(source)
            target_props = _props(target)
            active = active_property_deltas(
                source_props,
                target_props,
                threshold_scale=args.threshold_scale,
            )
            if len(active) < args.min_active_properties:
                skipped_delta += 1
                continue
            pair_id = f"mpair_{len(pairs):08d}"
            source_image, target_image = _image_paths(
                pair_id=pair_id,
                source=source,
                target=target,
                args=args,
            )
            row = {
                "pair_id": pair_id,
                "split": "",
                "source_mol_id": source.get("mol_id", ""),
                "target_mol_id": target.get("mol_id", ""),
                "source_smiles": source["canonical_smiles"],
                "target_smiles": target["canonical_smiles"],
                "source_image": source_image,
                "target_image": target_image,
                "scaffold": scaffold,
                "similarity": similarity,
                "active_properties": ",".join(active.keys()),
                "active_property_count": len(active),
                "active_deltas_json": json_dumps(active),
                "directions_json": json_dumps({prop: direction_from_delta(delta) for prop, delta in active.items()}),
            }
            for prop in PROPERTY_COLUMNS:
                source_value = source_props[prop]
                target_value = target_props[prop]
                row[f"source_{prop}"] = source_value
                row[f"target_{prop}"] = target_value
                row[f"delta_{prop}"] = target_value - source_value
            pairs.append(row)
            scaffold_pairs += 1

    _assign_component_split(pairs, eval_fraction=args.eval_fraction, seed=args.seed)
    _write_rows(args.output_csv, pairs)

    summary = {
        "molecule_db_csv": str(args.molecule_db_csv),
        "output_csv": str(args.output_csv),
        "molecules": len(molecules),
        "unique_scaffolds": len(by_scaffold),
        "edit_pairs": len(pairs),
        "train_pairs": sum(1 for row in pairs if row["split"] == "train"),
        "eval_pairs": sum(1 for row in pairs if row["split"] == "eval"),
        "min_active_properties": args.min_active_properties,
        "threshold_scale": args.threshold_scale,
        "min_similarity": args.min_similarity,
        "max_similarity": args.max_similarity,
        "skipped_similarity": skipped_similarity,
        "skipped_delta": skipped_delta,
        "render_images": bool(args.render_images),
    }
    args.output_csv.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def _read_molecules(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    out = []
    for row in rows:
        canonical = canonical_smiles(row.get("canonical_smiles", "")) or row.get("canonical_smiles", "")
        if not canonical or not row.get("scaffold"):
            continue
        row = dict(row)
        row["canonical_smiles"] = canonical
        out.append(row)
    return out


def _props(row: dict[str, str]) -> dict[str, float]:
    return {prop: float(row[prop]) for prop in PROPERTY_COLUMNS}


def _image_paths(*, pair_id: str, source: dict[str, str], target: dict[str, str], args: argparse.Namespace) -> tuple[str, str]:
    source_existing = source.get("image_path", "")
    target_existing = target.get("image_path", "")
    if not args.render_images:
        return source_existing, target_existing
    if args.image_dir is None:
        return source_existing, target_existing
    source_path = render_molecule_image(
        source["canonical_smiles"],
        args.image_dir / f"{pair_id}_source.png",
        args.image_size,
    )
    target_path = render_molecule_image(
        target["canonical_smiles"],
        args.image_dir / f"{pair_id}_target.png",
        args.image_size,
    )
    return source_path or source_existing, target_path or target_existing


def _assign_component_split(rows: list[dict[str, object]], *, eval_fraction: float, seed: int) -> None:
    graph: dict[str, set[str]] = {}
    molecule_to_row_ids: dict[str, set[int]] = {}
    for idx, row in enumerate(rows):
        source = str(row["source_smiles"])
        target = str(row["target_smiles"])
        graph.setdefault(source, set()).add(target)
        graph.setdefault(target, set()).add(source)
        molecule_to_row_ids.setdefault(source, set()).add(idx)
        molecule_to_row_ids.setdefault(target, set()).add(idx)

    seen: set[str] = set()
    components: list[set[int]] = []
    for molecule in graph:
        if molecule in seen:
            continue
        stack = [molecule]
        seen.add(molecule)
        component_molecules = set()
        while stack:
            current = stack.pop()
            component_molecules.add(current)
            for neighbor in graph.get(current, set()):
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        component_rows = set()
        for molecule in component_molecules:
            component_rows.update(molecule_to_row_ids.get(molecule, set()))
        components.append(component_rows)

    rng = random.Random(seed)
    rng.shuffle(components)
    target_eval = max(1, int(round(len(rows) * eval_fraction))) if rows else 0
    eval_ids: set[int] = set()
    for component in sorted(components, key=len):
        candidate = eval_ids | component
        if len(candidate) <= target_eval or not eval_ids:
            eval_ids = candidate
        if len(eval_ids) >= target_eval:
            break

    for idx, row in enumerate(rows):
        row["split"] = "eval" if idx in eval_ids else "train"


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "pair_id",
        "split",
        "source_mol_id",
        "target_mol_id",
        "source_smiles",
        "target_smiles",
        "source_image",
        "target_image",
        "scaffold",
        "similarity",
        "active_properties",
        "active_property_count",
        "active_deltas_json",
        "directions_json",
    ]
    for prop in PROPERTY_COLUMNS:
        fieldnames.extend([f"source_{prop}", f"target_{prop}", f"delta_{prop}"])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
