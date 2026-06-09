#!/usr/bin/env python3
"""Export MolEdit-Instruct eval rows as multi-property benchmark condition_rows."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


REPO_DIR = Path(__file__).resolve().parents[2]
DATASET_DIR = REPO_DIR / "SketchMol-MultiProperty-EditDataset"
UNIFIED_DIR = REPO_DIR / "SketchMol-Unified-3MDiffusion"
for path in (DATASET_DIR, UNIFIED_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from sketchmol_multiproperty_dataset.common import (  # noqa: E402
    PROPERTY_COLUMNS,
    direction_from_delta,
    sketchmol_condition_columns,
)
from sketchmol_unified_3m_diffusion.unified_condition_dataset import (  # noqa: E402
    _active_props_from_moledit,
    _directions_from_moledit,
    _parse_instruction_tasks,
    _properties_from_prefix,
    _task_specs_from_instruction,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--moledit-eval-split", required=True, type=Path)
    parser.add_argument("--edit-latent-index", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--split-name", default="eval")
    parser.add_argument("--variant", default="full")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    condition_ids = _load_condition_ids(args.edit_latent_index, variant=args.variant)
    rows = []
    with args.moledit_eval_split.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            example_id = str(raw.get("example_id", "")).strip()
            if example_id not in condition_ids:
                continue
            rows.append(_moledit_to_condition_row(raw, split=args.split_name))
    if not rows:
        raise SystemExit(f"No MolEdit rows matched {len(condition_ids)} condition IDs from {args.edit_latent_index}")
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    _write_rows(args.output_csv, rows)
    print(
        json.dumps(
            {
                "output_csv": str(args.output_csv),
                "rows": len(rows),
                "condition_ids_in_index": len(condition_ids),
            },
            indent=2,
        )
    )


def _load_condition_ids(index_csv: Path, *, variant: str) -> set[str]:
    ids: set[str] = set()
    with index_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("variant") != variant:
                continue
            condition_id = str(row.get("condition_id", "")).strip()
            if condition_id:
                ids.add(condition_id)
    if not ids:
        raise SystemExit(f"No condition IDs found in {index_csv} for variant={variant}")
    return ids


def _moledit_to_condition_row(raw: dict[str, str], *, split: str) -> dict[str, object]:
    instruction_tasks = _parse_instruction_tasks(raw.get("instruction_tasks", ""))
    task_specs = _task_specs_from_instruction(raw, instruction_tasks)
    source_props = _properties_from_prefix(raw, "source")
    target_props = _properties_from_prefix(raw, "target")
    deltas = _properties_from_prefix(raw, "delta")
    active_props = _active_props_from_moledit(raw, task_specs, deltas)
    directions = _directions_from_moledit(raw, task_specs, deltas)
    selected = [prop for prop in PROPERTY_COLUMNS if active_props.get(prop)]
    if not selected:
        selected = [spec.get("property", "") for spec in task_specs if spec.get("property") in PROPERTY_COLUMNS]
    selected = [prop for prop in selected if prop in PROPERTY_COLUMNS]
    scaffold = raw.get("source_scaffold_smiles", "") or raw.get("target_scaffold_smiles", "")
    row: dict[str, object] = {
        "condition_id": raw.get("example_id", ""),
        "pair_id": raw.get("pair_hash", "") or raw.get("example_id", ""),
        "split": split,
        "source_smiles": raw.get("source_smiles", ""),
        "target_smiles": raw.get("target_smiles", ""),
        "source_image": "",
        "target_image": "",
        "scaffold": scaffold,
        "similarity": raw.get("source_target_tanimoto", ""),
        "source_tanimoto": raw.get("source_target_tanimoto", ""),
        "source_similarity_bin": raw.get("difficulty_bucket", ""),
        "source_scaffold": scaffold,
        "target_scaffold": raw.get("target_scaffold_smiles", "") or scaffold,
        "same_scaffold": raw.get("scaffold_match", ""),
        "scaffold_relation": "",
        "pair_quality_tier": raw.get("pair_quality", ""),
        "selection_reason": "moledit_instruct",
        "condition_properties": ",".join(selected),
        "property_count": len(selected),
        "instruction": raw.get("instruction", ""),
        "preservation_constraint": "source_tanimoto",
    }
    for prop in PROPERTY_COLUMNS:
        row[f"source_{prop}"] = source_props.get(prop, "")
        row[f"target_{prop}"] = target_props.get(prop, "")
        delta = deltas.get(prop, target_props.get(prop, 0.0) - source_props.get(prop, 0.0))
        row[f"delta_{prop}"] = delta
        row[f"{prop}_active"] = "True" if prop in selected else "False"
        row[f"{prop}_direction"] = directions.get(prop, "") or direction_from_delta(float(delta or 0.0))
    row.update(sketchmol_condition_columns(target_props, selected))
    return row


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
