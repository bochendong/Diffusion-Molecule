#!/usr/bin/env python
"""Build a mixed-objective instruction edit dataset."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
import random

from build_edit_dataset import build_dataset
from sketchmol_understanding_condition.chem import canonical_smiles
from sketchmol_understanding_condition.baselines import build_baseline_rows, read_edit_pair_rows, write_baseline_rows


OBJECTIVES = (
    ("QED", "increase"),
    ("LogP", "decrease"),
    ("TPSA", "increase"),
    ("TPSA", "decrease"),
    ("MolWt", "decrease"),
)


@dataclass(frozen=True)
class MixedSummary:
    objective: str
    direction: str
    edit_pairs: int
    train_pairs: int
    eval_pairs: int
    source_dir: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=8000)
    parser.add_argument("--pairs-per-objective", type=int, default=80)
    parser.add_argument("--max-pairs-per-scaffold", type=int, default=5)
    parser.add_argument("--min-abs-delta", type=float, default=0.05)
    parser.add_argument("--min-similarity", type=float, default=0.2)
    parser.add_argument("--max-similarity", type=float, default=0.9)
    parser.add_argument("--eval-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = args.output_dir / "_by_objective"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[MixedSummary] = []
    mixed_rows_by_key: dict[str, dict[str, str]] = {}
    image_dir = args.output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    for objective_idx, (property_name, direction) in enumerate(OBJECTIVES):
        objective_name = f"{property_name}_{direction}"
        objective_dir = tmp_dir / objective_name
        summary = build_dataset(
            input_csv=args.input_csv,
            output_dir=objective_dir,
            property_name=property_name,
            direction=direction,
            limit=args.limit,
            max_pairs=args.pairs_per_objective,
            max_pairs_per_scaffold=args.max_pairs_per_scaffold,
            min_abs_delta=args.min_abs_delta,
            min_similarity=args.min_similarity,
            max_similarity=args.max_similarity,
            eval_fraction=args.eval_fraction,
            seed=args.seed + objective_idx,
        )
        objective_rows = read_edit_pair_rows(objective_dir / "edit_pairs.csv")
        summaries.append(
            MixedSummary(
                objective=property_name,
                direction=direction,
                edit_pairs=len(objective_rows),
                train_pairs=sum(1 for row in objective_rows if row.get("split") == "train"),
                eval_pairs=sum(1 for row in objective_rows if row.get("split") == "eval"),
                source_dir=str(objective_dir),
            )
        )
        for row in objective_rows:
            key = f"{objective_name}:{_canonical_pair_key(row)}"
            if key in mixed_rows_by_key:
                continue
            mixed_id = f"{objective_name}_{row['pair_id']}"
            row = dict(row)
            row["pair_id"] = mixed_id
            row["objective"] = property_name
            row["direction"] = direction
            row["source_image"] = _copy_image(row["source_image"], image_dir / f"{mixed_id}_source.png")
            row["target_image"] = _copy_image(row["target_image"], image_dir / f"{mixed_id}_target.png")
            mixed_rows_by_key[key] = row

    mixed_rows = list(mixed_rows_by_key.values())
    _assign_global_component_split(mixed_rows, eval_fraction=args.eval_fraction, seed=args.seed)

    _write_rows(args.output_dir / "edit_pairs.csv", mixed_rows)
    for split in ("train", "eval"):
        _write_rows(args.output_dir / f"{split}.csv", [row for row in mixed_rows if row.get("split") == split])

    baseline_rows = build_baseline_rows(mixed_rows)
    write_baseline_rows(args.output_dir / "baseline_variants.csv", baseline_rows)

    payload = {
        "input_csv": str(args.input_csv),
        "objectives": [asdict(summary) for summary in summaries],
        "edit_pairs": len(mixed_rows),
        "train_pairs": sum(1 for row in mixed_rows if row.get("split") == "train"),
        "eval_pairs": sum(1 for row in mixed_rows if row.get("split") == "eval"),
        "baseline_rows": len(baseline_rows),
        "output_dir": str(args.output_dir),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


def _copy_image(source: str, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    return str(target)


def _canonical_pair_key(row: dict[str, str]) -> str:
    source = canonical_smiles(row.get("source_smiles", "")) or row.get("source_smiles", "")
    target = canonical_smiles(row.get("target_smiles", "")) or row.get("target_smiles", "")
    return f"{source}>>{target}"


def _assign_global_component_split(rows: list[dict[str, str]], *, eval_fraction: float, seed: int) -> None:
    graph: dict[str, set[str]] = {}
    molecule_to_row_ids: dict[str, set[int]] = {}
    for idx, row in enumerate(rows):
        source = canonical_smiles(row.get("source_smiles", "")) or row.get("source_smiles", "")
        target = canonical_smiles(row.get("target_smiles", "")) or row.get("target_smiles", "")
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
    target_eval = max(1, int(round(len(rows) * eval_fraction)))
    eval_ids: set[int] = set()
    for component in sorted(components, key=len):
        candidate = eval_ids | component
        if len(candidate) <= target_eval or not eval_ids:
            eval_ids = candidate
        if len(eval_ids) >= target_eval:
            break

    for idx, row in enumerate(rows):
        row["split"] = "eval" if idx in eval_ids else "train"


def _write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


if __name__ == "__main__":
    main()
