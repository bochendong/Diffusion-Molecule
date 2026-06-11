#!/usr/bin/env python3
"""Convert de novo 2p-7p benchmark CSV rows to UniVideo eval JSONL."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Sequence


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from sketchmol_understanding_condition.unified_condition_dataset import (  # noqa: E402
    EDIT_GENERATION,
    PROPERTY_COLUMNS,
    UnifiedConditionSample,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", required=True, type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--split", default=None, help="Override split for every exported row.")
    parser.add_argument("--limit", type=int, default=0, help="0 exports all rows.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    rows = read_rows(args.input_csv)
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]
    samples = [sample_from_row(row, split_override=args.split) for row in rows]
    write_jsonl(args.output_jsonl, samples)
    summary = {
        "input_csv": str(args.input_csv),
        "output_jsonl": str(args.output_jsonl),
        "rows": len(samples),
        "split": args.split or "from_csv",
        "task_type": EDIT_GENERATION,
        "source_condition_mode": "zero",
    }
    args.output_jsonl.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sample_from_row(row: dict[str, str], *, split_override: str | None) -> UnifiedConditionSample:
    condition_id = value(row, "condition_id") or value(row, "sample_id")
    sample_id = value(row, "sample_id") or condition_id
    target_properties = {
        prop: parse_float(value(row, f"target_{prop}"))
        for prop in PROPERTY_COLUMNS
        if value(row, f"target_{prop}") != ""
    }
    active_properties = {
        prop: parse_bool(value(row, f"{prop}_active")) or prop in condition_properties(row)
        for prop in PROPERTY_COLUMNS
    }
    source_properties = {prop: 0.0 for prop in PROPERTY_COLUMNS}
    property_deltas = {
        prop: target_properties.get(prop, 0.0) if active_properties.get(prop, False) else 0.0
        for prop in PROPERTY_COLUMNS
    }
    directions = {
        prop: normalize_direction(value(row, f"{prop}_direction"))
        for prop in PROPERTY_COLUMNS
    }
    instruction = value(row, "instruction") or value(row, "prompt")
    metadata = {
        "condition_id": condition_id,
        "pair_id": value(row, "pair_id"),
        "benchmark_task": value(row, "benchmark_task") or "denovo_2p7p_property_design",
        "source_condition_mode": "zero",
        "denovo": "1",
        "molecule_id": value(row, "molecule_id"),
        "sketchmol_preset_str": value(row, "sketchmol_preset_str"),
        "variant": value(row, "variant") or "full",
        "variant_id": value(row, "variant_id") or condition_id,
    }
    for key, raw_value in row.items():
        if (key.startswith("ood_") or key.startswith("negative_")) and str(raw_value or "").strip():
            metadata[key] = str(raw_value).strip()
    return UnifiedConditionSample(
        sample_id=sample_id,
        task_type=EDIT_GENERATION,
        split=split_override or value(row, "split") or "eval",
        prompt=value(row, "prompt") or instruction,
        target_smiles=value(row, "target_smiles"),
        source_smiles="",
        molecule_smiles="",
        source_image=value(row, "source_image"),
        target_image=value(row, "target_image"),
        instruction=instruction,
        condition_properties=",".join(condition_properties(row)),
        property_count=value(row, "property_count") or str(len(condition_properties(row))),
        source_tanimoto="0.0",
        source_similarity_bin="exploratory_low_similarity",
        source_properties=source_properties,
        target_properties=target_properties,
        property_deltas=property_deltas,
        active_properties=active_properties,
        directions=directions,
        metadata=metadata,
    )


def condition_properties(row: dict[str, str]) -> list[str]:
    explicit = [item.strip() for item in value(row, "condition_properties").split(",") if item.strip()]
    if explicit:
        return [prop for prop in PROPERTY_COLUMNS if prop in explicit]
    return [prop for prop in PROPERTY_COLUMNS if parse_bool(value(row, f"{prop}_active"))]


def value(row: dict[str, str], key: str) -> str:
    return str(row.get(key, "") or "").strip()


def parse_float(text: str) -> float:
    try:
        return float(text)
    except ValueError:
        return 0.0


def parse_bool(text: str) -> bool:
    return text.strip().lower() in {"1", "true", "yes", "y"}


def normalize_direction(text: str) -> str:
    lowered = text.strip().lower()
    if lowered in {"increase", "up", "higher", "+", "1"}:
        return "increase"
    if lowered in {"decrease", "down", "lower", "-", "-1"}:
        return "decrease"
    return ""


def write_jsonl(path: Path, samples: list[UnifiedConditionSample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample.to_json_dict(), sort_keys=True) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
