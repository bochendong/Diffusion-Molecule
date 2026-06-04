"""Diagnostics for instruction-guided edit datasets and benchmark outputs."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Iterable

from .chem import canonical_smiles, is_valid_smiles, molecular_properties, morgan_tanimoto, scaffold_smiles


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def summarize_edit_dataset(
    edit_pairs_csv: str | Path,
    *,
    baseline_variants_csv: str | Path | None = None,
    property_name: str = "QED",
) -> dict[str, object]:
    """Summarize edit-pair quality before training a model."""

    rows = read_csv_rows(edit_pairs_csv)
    split_counts = Counter(row.get("split", "") for row in rows)
    scaffold_counts = Counter(row.get("scaffold", "") for row in rows)

    image_exists = []
    source_valid = []
    target_valid = []
    scaffold_matches = []
    pair_similarities = []
    property_deltas = []
    property_success = []
    source_targets_by_split: dict[str, set[str]] = defaultdict(set)
    canonical_pair_ids = set()
    duplicate_pairs = 0

    for row in rows:
        source = row.get("source_smiles", "")
        target = row.get("target_smiles", "")
        split = row.get("split", "")
        source_image = row.get("source_image", "")
        target_image = row.get("target_image", "")

        image_exists.append(bool(source_image) and Path(source_image).exists())
        image_exists.append(bool(target_image) and Path(target_image).exists())

        source_ok = is_valid_smiles(source)
        target_ok = is_valid_smiles(target)
        source_valid.append(source_ok)
        target_valid.append(target_ok)

        source_can = canonical_smiles(source) if source_ok else None
        target_can = canonical_smiles(target) if target_ok else None
        if source_can and target_can:
            key = f"{source_can}>>{target_can}"
            duplicate_pairs += int(key in canonical_pair_ids)
            canonical_pair_ids.add(key)
            source_targets_by_split[split].update([source_can, target_can])

        source_scaffold = scaffold_smiles(source) if source_ok else None
        target_scaffold = scaffold_smiles(target) if target_ok else None
        scaffold_matches.append(source_scaffold is not None and source_scaffold == target_scaffold)

        similarity = _safe_float(row.get("similarity"))
        if math.isnan(similarity) and source_ok and target_ok:
            similarity = morgan_tanimoto(source, target) or math.nan
        if not math.isnan(similarity):
            pair_similarities.append(similarity)

        delta = _safe_float(row.get("property_delta"))
        if math.isnan(delta) and source_ok and target_ok:
            source_props = molecular_properties(source) or {}
            target_props = molecular_properties(target) or {}
            if property_name in source_props and property_name in target_props:
                delta = float(target_props[property_name] - source_props[property_name])
        if not math.isnan(delta):
            property_deltas.append(delta)
            property_success.append(delta > 0.0)

    leakage = _split_leakage(source_targets_by_split)
    summary: dict[str, object] = {
        "edit_pairs": len(rows),
        "split_counts": dict(split_counts),
        "unique_scaffolds": len([key for key in scaffold_counts if key]),
        "top_scaffolds": scaffold_counts.most_common(10),
        "source_valid_fraction": _fraction(source_valid),
        "target_valid_fraction": _fraction(target_valid),
        "image_exists_fraction": _fraction(image_exists),
        "scaffold_match_fraction": _fraction(scaffold_matches),
        "duplicate_canonical_pairs": duplicate_pairs,
        "property_name": property_name,
        "property_success_fraction": _fraction(property_success),
        "split_leakage": leakage,
    }
    summary.update(_distribution("similarity", pair_similarities))
    summary.update(_distribution("property_delta", property_deltas))

    if baseline_variants_csv is not None:
        summary["baseline_variants"] = summarize_baseline_variants(baseline_variants_csv)

    return summary


def summarize_baseline_variants(path: str | Path) -> dict[str, object]:
    """Summarize baseline variant manifest coverage."""

    rows = read_csv_rows(path)
    variant_counts = Counter(row.get("variant", "") for row in rows)
    split_variant_counts = Counter(f"{row.get('split', '')}:{row.get('variant', '')}" for row in rows)
    pair_to_variants: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        pair_to_variants[row.get("pair_id", "")].add(row.get("variant", ""))
    expected = set(variant_counts.keys())
    complete_pairs = sum(1 for variants in pair_to_variants.values() if variants == expected)
    return {
        "rows": len(rows),
        "variant_counts": dict(variant_counts),
        "split_variant_counts": dict(split_variant_counts),
        "pairs": len(pair_to_variants),
        "pairs_with_all_variants": complete_pairs,
        "complete_pair_fraction": complete_pairs / len(pair_to_variants) if pair_to_variants else 0.0,
    }


def _distribution(prefix: str, values: Iterable[float]) -> dict[str, float]:
    values = [float(v) for v in values if not math.isnan(float(v))]
    if not values:
        return {
            f"{prefix}_mean": math.nan,
            f"{prefix}_median": math.nan,
            f"{prefix}_min": math.nan,
            f"{prefix}_max": math.nan,
        }
    return {
        f"{prefix}_mean": mean(values),
        f"{prefix}_median": median(values),
        f"{prefix}_min": min(values),
        f"{prefix}_max": max(values),
    }


def _fraction(flags: Iterable[bool]) -> float:
    values = list(flags)
    if not values:
        return 0.0
    return sum(1 for flag in values if flag) / len(values)


def _safe_float(value: object) -> float:
    try:
        text = str(value or "").strip()
        return float(text) if text else math.nan
    except ValueError:
        return math.nan


def _split_leakage(source_targets_by_split: dict[str, set[str]]) -> dict[str, object]:
    train = source_targets_by_split.get("train", set())
    eval_ = source_targets_by_split.get("eval", set())
    overlap = sorted(train & eval_)
    return {
        "train_eval_overlap_molecules": len(overlap),
        "train_eval_overlap_fraction_of_eval": len(overlap) / len(eval_) if eval_ else 0.0,
        "examples": overlap[:10],
    }
