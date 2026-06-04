#!/usr/bin/env python
"""Generate multi-property instruction rows from edit pairs."""

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
    direction_from_delta,
    json_dumps,
    render_instruction,
    sketchmol_condition_columns,
)


BASELINE_VARIANTS = ("full", "text_only", "image_only", "random_query", "caption_bottleneck")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edit-pairs-csv", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--baseline-variants-csv", required=True, type=Path)
    parser.add_argument("--conditions-per-pair", type=int, default=3)
    parser.add_argument("--min-properties", type=int, default=2)
    parser.add_argument("--max-properties", type=int, default=7)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.baseline_variants_csv.parent.mkdir(parents=True, exist_ok=True)

    pairs = _read_rows(args.edit_pairs_csv)
    condition_rows = []
    for pair in pairs:
        active = [prop for prop in (pair.get("active_properties") or "").split(",") if prop]
        active = [prop for prop in active if prop in PROPERTY_COLUMNS]
        if len(active) < args.min_properties:
            continue
        for sample_idx in range(args.conditions_per_pair):
            max_count = min(args.max_properties, len(active))
            min_count = min(args.min_properties, max_count)
            property_count = rng.randint(min_count, max_count)
            selected = sorted(rng.sample(active, property_count), key=PROPERTY_COLUMNS.index)
            condition_rows.append(_condition_row(pair, selected, sample_idx))

    _write_rows(args.output_csv, condition_rows)
    baseline_rows = _baseline_rows(condition_rows)
    _write_rows(args.baseline_variants_csv, baseline_rows)
    summary = {
        "edit_pairs_csv": str(args.edit_pairs_csv),
        "output_csv": str(args.output_csv),
        "baseline_variants_csv": str(args.baseline_variants_csv),
        "input_pairs": len(pairs),
        "condition_rows": len(condition_rows),
        "baseline_rows": len(baseline_rows),
        "conditions_per_pair": args.conditions_per_pair,
        "min_properties": args.min_properties,
        "max_properties": args.max_properties,
        "train_rows": sum(1 for row in condition_rows if row["split"] == "train"),
        "eval_rows": sum(1 for row in condition_rows if row["split"] == "eval"),
    }
    args.output_csv.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _condition_row(pair: dict[str, str], selected: list[str], sample_idx: int) -> dict[str, object]:
    source_props = {prop: float(pair[f"source_{prop}"]) for prop in PROPERTY_COLUMNS}
    target_props = {prop: float(pair[f"target_{prop}"]) for prop in PROPERTY_COLUMNS}
    deltas = {prop: float(pair[f"delta_{prop}"]) for prop in selected}
    directions = {prop: direction_from_delta(delta) for prop, delta in deltas.items()}
    condition_id = f"{pair['pair_id']}_cond_{sample_idx:02d}_{len(selected)}p"
    row: dict[str, object] = {
        "condition_id": condition_id,
        "pair_id": pair["pair_id"],
        "split": pair.get("split", ""),
        "source_smiles": pair.get("source_smiles", ""),
        "target_smiles": pair.get("target_smiles", ""),
        "source_image": pair.get("source_image", ""),
        "target_image": pair.get("target_image", ""),
        "scaffold": pair.get("scaffold", ""),
        "similarity": pair.get("similarity", ""),
        "condition_properties": ",".join(selected),
        "property_count": len(selected),
        "target_values_json": json_dumps({prop: target_props[prop] for prop in selected}),
        "source_values_json": json_dumps({prop: source_props[prop] for prop in selected}),
        "deltas_json": json_dumps(deltas),
        "directions_json": json_dumps(directions),
        "instruction": render_instruction(
            selected_props=selected,
            source_props=source_props,
            target_props=target_props,
            deltas=deltas,
        ),
    }
    for prop in PROPERTY_COLUMNS:
        row[f"source_{prop}"] = source_props[prop]
        row[f"target_{prop}"] = target_props[prop]
        row[f"delta_{prop}"] = target_props[prop] - source_props[prop]
        row[f"{prop}_active"] = "True" if prop in selected else "False"
        row[f"{prop}_direction"] = directions.get(prop, "")
    row.update(sketchmol_condition_columns(target_props, selected))
    return row


def _baseline_rows(condition_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    for row in condition_rows:
        for variant in BASELINE_VARIANTS:
            out = dict(row)
            out["variant_id"] = f"{row['condition_id']}:{variant}"
            out["variant"] = variant
            out["condition_mode"] = _condition_mode(variant)
            out["use_source_image"] = "True" if variant in {"full", "image_only"} else "False"
            out["use_instruction"] = "True" if variant in {"full", "text_only", "caption_bottleneck"} else "False"
            if variant == "image_only":
                out["prompt"] = "Preserve the visible molecular scaffold and make a valid local edit."
            elif variant == "random_query":
                out["prompt"] = ""
            elif variant == "caption_bottleneck":
                out["prompt"] = _caption_prompt(row)
            else:
                out["prompt"] = row["instruction"]
            rows.append(out)
    return rows


def _condition_mode(variant: str) -> str:
    return {
        "full": "mllm_image_text",
        "text_only": "mllm_text_only",
        "image_only": "mllm_image_only",
        "random_query": "random_query_tokens",
        "caption_bottleneck": "caption_bottleneck",
    }[variant]


def _caption_prompt(row: dict[str, object]) -> str:
    return (
        f"Source molecule SMILES: {row.get('source_smiles', '')}. "
        f"Shared scaffold: {row.get('scaffold', '') or 'unknown'}. "
        f"Requested multi-property edit: {row.get('instruction', '')} "
        f"Constrained properties: {row.get('condition_properties', '')}."
    )


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = []
    seen = set()
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
