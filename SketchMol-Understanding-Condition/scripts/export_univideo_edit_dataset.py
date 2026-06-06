#!/usr/bin/env python3
"""Export UniVideo-style edit JSONL and VLM baseline rows from condition rows."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Iterable, Mapping


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from sketchmol_understanding_condition.chem import morgan_tanimoto  # noqa: E402
from sketchmol_understanding_condition.unified_condition_dataset import (  # noqa: E402
    EDIT_GENERATION,
    PROPERTY_COLUMNS,
    UnifiedConditionSample,
    summarize_samples,
    write_jsonl,
)


BASELINE_VARIANTS = ("full", "text_only", "image_only", "caption_bottleneck", "random_query")
PROPERTY_ALIASES = {
    "MW": ("MW", "MolWt", "molecular_weight"),
    "LogP": ("LogP", "logp", "aLogP"),
    "QED": ("QED",),
    "TPSA": ("TPSA",),
    "HBD": ("HBD",),
    "HBA": ("HBA",),
    "RB": ("RB", "rotatable", "rotatable_bonds"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition-rows-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--include-splits", default="train,eval,valid,validation,test")
    parser.add_argument("--variants", default="full")
    parser.add_argument("--dataset-name", default="univideo_edit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    include_splits = {item.strip() for item in args.include_splits.split(",") if item.strip()}
    variants = tuple(item.strip() for item in args.variants.split(",") if item.strip())
    invalid = sorted(set(variants) - set(BASELINE_VARIANTS))
    if invalid:
        raise ValueError(f"Unsupported variants: {invalid}")

    condition_rows = []
    for row in _read_rows(args.condition_rows_csv):
        split = row.get("split", "train") or "train"
        if include_splits and split not in include_splits:
            continue
        if not row.get("source_smiles") or not row.get("target_smiles"):
            continue
        if not (row.get("instruction") or row.get("prompt")):
            continue
        condition_rows.append(row)
        if args.limit is not None and len(condition_rows) >= args.limit:
            break

    samples = [_sample_from_condition_row(row, dataset_name=args.dataset_name, idx=idx) for idx, row in enumerate(condition_rows)]
    train = [sample for sample in samples if sample.split not in {"eval", "valid", "validation", "test"}]
    eval_rows = [sample for sample in samples if sample.split in {"eval", "valid", "validation", "test"}]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_jsonl = args.output_dir / "univideo_edit_train.jsonl"
    eval_jsonl = args.output_dir / "univideo_edit_eval.jsonl"
    baseline_csv = args.output_dir / "baseline_variants.csv"
    write_jsonl(train_jsonl, train)
    write_jsonl(eval_jsonl, eval_rows)
    _write_rows(baseline_csv, _baseline_rows(condition_rows, variants=variants))

    summary = summarize_samples(samples, train_rows=len(train), eval_rows=len(eval_rows))
    summary.update(
        {
            "condition_rows_csv": str(args.condition_rows_csv),
            "output_dir": str(args.output_dir),
            "train_jsonl": str(train_jsonl),
            "eval_jsonl": str(eval_jsonl),
            "baseline_variants_csv": str(baseline_csv),
            "variants": list(variants),
        }
    )
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


def _sample_from_condition_row(row: Mapping[str, str], *, dataset_name: str, idx: int) -> UnifiedConditionSample:
    condition_id = row.get("condition_id") or row.get("sample_id") or f"cond_{idx:08d}"
    source_props = _properties_from_prefix(row, "source")
    target_props = _properties_from_prefix(row, "target")
    deltas = _delta_properties(row, source_props=source_props, target_props=target_props)
    selected = _selected_properties(row)
    active = {prop: _truthy(row.get(f"{prop}_active", "")) or prop in selected for prop in PROPERTY_COLUMNS}
    directions = {
        prop: row.get(f"{prop}_direction", "") or _direction_from_delta(deltas.get(prop, 0.0))
        for prop in PROPERTY_COLUMNS
    }
    source_tanimoto = _source_tanimoto(row)
    return UnifiedConditionSample(
        sample_id=f"edit:{dataset_name}:{condition_id}",
        task_type=EDIT_GENERATION,
        split=row.get("split", "train") or "train",
        prompt=row.get("instruction") or row.get("prompt", ""),
        target_smiles=row.get("target_smiles", ""),
        source_smiles=row.get("source_smiles", ""),
        molecule_smiles=row.get("source_smiles", ""),
        source_image=row.get("source_image", ""),
        target_image=row.get("target_image", ""),
        instruction=row.get("instruction") or row.get("prompt", ""),
        condition_properties=",".join(selected),
        property_count=str(row.get("property_count") or len(selected)),
        source_tanimoto="" if math.isnan(source_tanimoto) else str(source_tanimoto),
        source_similarity_bin=_similarity_bin(source_tanimoto),
        source_properties=source_props,
        target_properties=target_props,
        property_deltas=deltas,
        active_properties=active,
        directions=directions,
        metadata={
            "dataset": dataset_name,
            "condition_id": condition_id,
            "pair_id": row.get("pair_id", ""),
        },
    )


def _baseline_rows(rows: Iterable[Mapping[str, str]], *, variants: tuple[str, ...]) -> list[dict[str, str]]:
    out = []
    for idx, row in enumerate(rows):
        condition_id = row.get("condition_id") or row.get("sample_id") or f"cond_{idx:08d}"
        for variant in variants:
            item = dict(row)
            item["condition_id"] = condition_id
            item["variant_id"] = f"{condition_id}:{variant}"
            item["variant"] = variant
            item["condition_mode"] = _condition_mode(variant)
            item["prompt"] = _prompt_for_variant(row, variant)
            item["use_source_image"] = "True" if variant in {"full", "image_only"} else "False"
            item["use_instruction"] = "True" if variant in {"full", "text_only", "caption_bottleneck"} else "False"
            out.append({key: str(value) for key, value in item.items()})
    return out


def _prompt_for_variant(row: Mapping[str, str], variant: str) -> str:
    instruction = row.get("instruction") or row.get("prompt", "")
    if variant == "image_only":
        return "Preserve the visible molecular structure and generate a valid edited molecule."
    if variant == "random_query":
        return ""
    if variant == "caption_bottleneck":
        props = row.get("condition_properties", "")
        return f"Source molecule with edit target properties: {props}."
    return instruction


def _condition_mode(variant: str) -> str:
    return {
        "full": "mllm_image_text",
        "text_only": "mllm_text_only",
        "image_only": "mllm_image_only",
        "caption_bottleneck": "caption_bottleneck",
        "random_query": "random_query_tokens",
    }[variant]


def _selected_properties(row: Mapping[str, str]) -> list[str]:
    selected = [item.strip() for item in str(row.get("condition_properties", "")).split(",") if item.strip()]
    selected = [_canonical_property(item) for item in selected]
    selected = [item for item in selected if item in PROPERTY_COLUMNS]
    if selected:
        return selected
    return [prop for prop in PROPERTY_COLUMNS if _truthy(row.get(f"{prop}_active", ""))]


def _properties_from_prefix(row: Mapping[str, str], prefix: str) -> dict[str, float]:
    out = {}
    for prop in PROPERTY_COLUMNS:
        value = math.nan
        for alias in PROPERTY_ALIASES[prop]:
            value = _to_float(row.get(f"{prefix}_{alias}", ""))
            if not math.isnan(value):
                break
        if not math.isnan(value):
            out[prop] = value
    return out


def _delta_properties(
    row: Mapping[str, str],
    *,
    source_props: Mapping[str, float],
    target_props: Mapping[str, float],
) -> dict[str, float]:
    out = {}
    for prop in PROPERTY_COLUMNS:
        value = math.nan
        for alias in PROPERTY_ALIASES[prop]:
            value = _to_float(row.get(f"delta_{alias}", ""))
            if not math.isnan(value):
                break
        if math.isnan(value) and prop in source_props and prop in target_props:
            value = float(target_props[prop]) - float(source_props[prop])
        if not math.isnan(value):
            out[prop] = value
    return out


def _source_tanimoto(row: Mapping[str, str]) -> float:
    for key in ("source_tanimoto", "similarity", "tanimoto"):
        value = _to_float(row.get(key, ""))
        if not math.isnan(value):
            return value
    try:
        value = morgan_tanimoto(row.get("source_smiles", ""), row.get("target_smiles", ""))
    except RuntimeError:
        value = None
    return float(value) if value is not None else math.nan


def _similarity_bin(value: float) -> str:
    if math.isnan(value):
        return ""
    if value >= 0.7:
        return "easy_high_similarity"
    if value >= 0.5:
        return "medium_similarity"
    if value >= 0.4:
        return "hard_similarity"
    return "exploratory_low_similarity"


def _canonical_property(value: str) -> str:
    for prop, aliases in PROPERTY_ALIASES.items():
        if value in aliases:
            return prop
    return value


def _direction_from_delta(value: float) -> str:
    if value > 0:
        return "increase"
    if value < 0:
        return "decrease"
    return ""


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "active", "increase", "decrease"}


def _to_float(value: object) -> float:
    try:
        return float(str(value if value is not None else "").strip())
    except ValueError:
        return math.nan


if __name__ == "__main__":
    main()
