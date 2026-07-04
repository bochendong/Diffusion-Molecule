#!/usr/bin/env python3
"""Expand unified condition rows into feature-export variants.

The existing condition-feature exporter expects rows with `variant` and
`variant_id`.  This adapter lets the unified generator reuse that exporter for
with-image/no-image ablations without changing benchmark row schemas.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Mapping, Sequence


CONDITION_MODE = {
    "full": "mllm_image_text",
    "text_only": "mllm_text_only",
    "image_only": "mllm_image_only",
    "caption_bottleneck": "caption_bottleneck",
    "random_query": "random_query_tokens",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--variants", default="full,text_only")
    parser.add_argument("--default-split", default="")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    variants = [item.strip() for item in str(args.variants).split(",") if item.strip()]
    if not variants:
        raise SystemExit("--variants is empty")
    unsupported = sorted(set(variants) - set(CONDITION_MODE))
    if unsupported:
        raise SystemExit(f"Unsupported variants: {', '.join(unsupported)}")

    source_rows = read_rows(args.input_csv)
    out = []
    for index, row in enumerate(source_rows):
        condition_id = condition_id_for_row(row, index=index)
        for variant in variants:
            item = dict(row)
            item["condition_id"] = condition_id
            item["variant_id"] = f"{condition_id}:{variant}"
            item["variant"] = variant
            item["condition_mode"] = CONDITION_MODE[variant]
            item["prompt"] = prompt_for_variant(row, variant)
            item["use_source_image"] = "True" if variant in {"full", "image_only"} else "False"
            item["use_instruction"] = "True" if variant in {"full", "text_only", "caption_bottleneck"} else "False"
            if args.default_split and not str(item.get("split", "") or "").strip():
                item["split"] = str(args.default_split)
            out.append({key: "" if value is None else str(value) for key, value in item.items()})
    write_rows(args.output_csv, out)
    print(
        {
            "input_csv": str(args.input_csv),
            "output_csv": str(args.output_csv),
            "source_rows": len(source_rows),
            "variants": variants,
            "output_rows": len(out),
        }
    )
    return 0


def condition_id_for_row(row: Mapping[str, object], *, index: int) -> str:
    for key in ("condition_id", "sample_id", "example_id", "pair_hash", "pair_id", "variant_id"):
        value = str(row.get(key, "") or "").strip()
        if value:
            if ":" in value and key == "variant_id":
                return value.rsplit(":", 1)[0]
            return value
    return f"unified_condition_{index:08d}"


def prompt_for_variant(row: Mapping[str, object], variant: str) -> str:
    instruction = str(row.get("instruction", "") or row.get("prompt", "") or "").strip()
    if variant == "image_only":
        if task_mode_for_row(row) == "edit":
            return "Preserve the visible molecular structure and generate a valid edited molecule."
        return "Generate a valid molecule matching the visible or requested molecular condition."
    if variant == "caption_bottleneck":
        props = str(row.get("condition_properties", "") or row.get("external_task_properties", "") or "").strip()
        return f"Molecular generation/editing target properties: {props}."
    if variant == "random_query":
        return ""
    return instruction


def task_mode_for_row(row: Mapping[str, object]) -> str:
    raw = str(row.get("task_mode", "") or row.get("unified_task_mode", "") or "").strip().lower()
    normalized = raw.replace("-", "_").replace(" ", "_")
    if normalized in {"de_novo", "denovo", "generate", "generation"}:
        return "de_novo"
    if normalized in {"edit", "conditional_edit", "source_edit", "edit_generation"}:
        return "edit"
    source = str(row.get("source_smiles", "") or row.get("molecule_smiles", "") or "").strip()
    return "edit" if source else "de_novo"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_rows(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                fieldnames.append(str(key))
                seen.add(str(key))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


if __name__ == "__main__":
    raise SystemExit(main())
