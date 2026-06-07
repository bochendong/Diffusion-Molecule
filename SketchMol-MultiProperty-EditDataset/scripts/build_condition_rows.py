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
    SKETCHMOL_STRICT_TOLERANCE,
    direction_from_delta,
    json_dumps,
    normalized_property_error,
    render_instruction,
    sketchmol_condition_columns,
    strict_property_success,
)
from sketchmol_understanding_condition.chem import canonical_smiles, morgan_tanimoto


BASELINE_VARIANTS = ("full", "text_only", "image_only", "random_query", "caption_bottleneck")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edit-pairs-csv", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--baseline-variants-csv", required=True, type=Path)
    parser.add_argument("--conditions-per-pair", type=int, default=3)
    parser.add_argument("--min-properties", type=int, default=2)
    parser.add_argument("--max-properties", type=int, default=7)
    parser.add_argument(
        "--candidate-molecule-db-csv",
        type=Path,
        default=None,
        help="Optional molecule_database.csv used to add source-neighbor oracle diagnostics.",
    )
    parser.add_argument("--min-strict-candidates-t04", type=int, default=0)
    parser.add_argument(
        "--oracle-filter-splits",
        default="",
        help="Comma-separated splits where rows below --min-strict-candidates-t04 are dropped.",
    )
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.baseline_variants_csv.parent.mkdir(parents=True, exist_ok=True)

    pairs = _read_rows(args.edit_pairs_csv)
    oracle = _CandidateOracle(args.candidate_molecule_db_csv) if args.candidate_molecule_db_csv else None
    oracle_filter_splits = {item.strip() for item in args.oracle_filter_splits.split(",") if item.strip()}
    condition_rows = []
    skipped_oracle_filter = 0
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
            row = _condition_row(pair, selected, sample_idx)
            if oracle is not None:
                row.update(oracle.diagnostics(row, selected))
                if (
                    row.get("split", "") in oracle_filter_splits
                    and int(row["strict_candidate_count_t04"]) < args.min_strict_candidates_t04
                ):
                    skipped_oracle_filter += 1
                    continue
            condition_rows.append(row)

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
        "candidate_molecule_db_csv": str(args.candidate_molecule_db_csv) if args.candidate_molecule_db_csv else None,
        "min_strict_candidates_t04": args.min_strict_candidates_t04,
        "oracle_filter_splits": sorted(oracle_filter_splits),
        "skipped_oracle_filter": skipped_oracle_filter,
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
        "source_tanimoto": pair.get("source_tanimoto") or pair.get("similarity", ""),
        "source_similarity_bin": pair.get("source_similarity_bin", ""),
        "source_scaffold": pair.get("source_scaffold") or pair.get("scaffold", ""),
        "target_scaffold": pair.get("target_scaffold") or pair.get("scaffold", ""),
        "same_scaffold": pair.get("same_scaffold", ""),
        "scaffold_relation": pair.get("scaffold_relation", ""),
        "pair_quality_tier": pair.get("pair_quality_tier", ""),
        "selection_reason": pair.get("selection_reason", ""),
        "same_scaffold_neighbor_count": pair.get("same_scaffold_neighbor_count", ""),
        "source_neighbor_count_t04": pair.get("source_neighbor_count_t04", ""),
        "source_neighbor_count_t05": pair.get("source_neighbor_count_t05", ""),
        "source_neighbor_count_t06": pair.get("source_neighbor_count_t06", ""),
        "target_neighbor_rank_by_tanimoto": pair.get("target_neighbor_rank_by_tanimoto", ""),
        "condition_properties": ",".join(selected),
        "property_count": len(selected),
        "target_values_json": json_dumps({prop: target_props[prop] for prop in selected}),
        "source_values_json": json_dumps({prop: source_props[prop] for prop in selected}),
        "deltas_json": json_dumps(deltas),
        "directions_json": json_dumps(directions),
        "property_constraints_json": json_dumps(
            {
                prop: {
                    "source": source_props[prop],
                    "target": target_props[prop],
                    "delta": deltas[prop],
                    "direction": directions[prop],
                    "strict_tolerance": SKETCHMOL_STRICT_TOLERANCE[prop],
                }
                for prop in selected
            }
        ),
        "instruction_template_id": "local_edit_numeric_v1",
        "instruction_style": "mixed_numeric_direction",
        "preservation_constraint": _preservation_constraint(pair),
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


def _preservation_constraint(pair: dict[str, str]) -> str:
    if str(pair.get("same_scaffold", "")).lower() == "true":
        return "keep_same_scaffold_or_source_tanimoto_ge_0_4"
    try:
        similarity = float(pair.get("source_tanimoto") or pair.get("similarity") or "nan")
    except ValueError:
        similarity = float("nan")
    if similarity >= 0.5:
        return "keep_source_tanimoto_ge_0_5"
    return "keep_source_tanimoto_ge_0_4"


class _CandidateOracle:
    """Same-scaffold source-neighbor diagnostics for condition rows."""

    def __init__(self, molecule_db_csv: Path):
        self.rows = _read_candidate_rows(molecule_db_csv)
        self.by_scaffold: dict[str, list[dict[str, object]]] = {}
        for row in self.rows:
            self.by_scaffold.setdefault(str(row["scaffold"]), []).append(row)
        self._source_pool_cache: dict[tuple[str, str], list[dict[str, object]]] = {}

    def diagnostics(self, row: dict[str, object], selected_props: list[str]) -> dict[str, object]:
        pool = self._source_pool(row)
        target_smiles = canonical_smiles(str(row.get("target_smiles", ""))) or str(row.get("target_smiles", ""))
        target_props = {prop: float(row[f"target_{prop}"]) for prop in PROPERTY_COLUMNS}
        source_props = {prop: float(row[f"source_{prop}"]) for prop in PROPERTY_COLUMNS}
        pools = {
            "t04": [candidate for candidate in pool if float(candidate["source_tanimoto"]) >= 0.4],
            "t05": [candidate for candidate in pool if float(candidate["source_tanimoto"]) >= 0.5],
            "t06": [candidate for candidate in pool if float(candidate["source_tanimoto"]) >= 0.6],
        }
        strict_by_threshold: dict[str, list[dict[str, object]]] = {}
        for threshold, candidates in pools.items():
            strict_by_threshold[threshold] = [
                candidate
                for candidate in candidates
                if str(candidate["smiles"]) != target_smiles
                and strict_property_success(candidate["props"], target_props, selected_props)
            ]
        oracle = _best_oracle_candidate(pools["t04"], target_smiles, target_props, selected_props)
        source_identity_success = strict_property_success(source_props, target_props, selected_props)
        return {
            "candidate_pool_size_t04": len(pools["t04"]),
            "candidate_pool_size_t05": len(pools["t05"]),
            "candidate_pool_size_t06": len(pools["t06"]),
            "strict_candidate_count_t04": len(strict_by_threshold["t04"]),
            "strict_candidate_count_t05": len(strict_by_threshold["t05"]),
            "strict_candidate_count_t06": len(strict_by_threshold["t06"]),
            "oracle_candidate_smiles_t04": oracle.get("smiles", ""),
            "oracle_source_tanimoto_t04": oracle.get("source_tanimoto", ""),
            "oracle_strict_success_t04": str(bool(strict_by_threshold["t04"])),
            "oracle_property_error_t04": oracle.get("property_error", ""),
            "oracle_property_errors_json_t04": oracle.get("property_errors_json", ""),
            "source_identity_strict_success": str(bool(source_identity_success)),
        }

    def _source_pool(self, row: dict[str, object]) -> list[dict[str, object]]:
        source_smiles = canonical_smiles(str(row.get("source_smiles", ""))) or str(row.get("source_smiles", ""))
        source_scaffold = str(row.get("source_scaffold") or row.get("scaffold") or "")
        key = (source_smiles, source_scaffold)
        if key in self._source_pool_cache:
            return self._source_pool_cache[key]
        pool = []
        for candidate in self.by_scaffold.get(source_scaffold, []):
            if str(candidate["smiles"]) == source_smiles:
                continue
            similarity = morgan_tanimoto(source_smiles, str(candidate["smiles"]))
            if similarity is None:
                continue
            item = dict(candidate)
            item["source_tanimoto"] = float(similarity)
            pool.append(item)
        pool.sort(key=lambda candidate: float(candidate["source_tanimoto"]), reverse=True)
        self._source_pool_cache[key] = pool
        return pool


def _read_candidate_rows(path: Path) -> list[dict[str, object]]:
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            smiles = canonical_smiles(raw.get("canonical_smiles", "")) or raw.get("canonical_smiles", "")
            scaffold = raw.get("scaffold", "")
            if not smiles or not scaffold:
                continue
            try:
                props = {prop: float(raw[prop]) for prop in PROPERTY_COLUMNS}
            except (KeyError, ValueError):
                continue
            rows.append({"smiles": smiles, "scaffold": scaffold, "props": props})
    return rows


def _best_oracle_candidate(
    candidates: list[dict[str, object]],
    target_smiles: str,
    target_props: dict[str, float],
    selected_props: list[str],
) -> dict[str, object]:
    best: dict[str, object] | None = None
    for candidate in candidates:
        if str(candidate["smiles"]) == target_smiles:
            continue
        props = candidate["props"]
        error = normalized_property_error(props, target_props, selected_props)
        if best is None or error < float(best["property_error"]):
            best = {
                "smiles": candidate["smiles"],
                "source_tanimoto": candidate["source_tanimoto"],
                "property_error": error,
                "property_errors_json": json_dumps(
                    {
                        prop: abs(float(props[prop]) - float(target_props[prop]))
                        for prop in selected_props
                        if prop in props and prop in target_props
                    }
                ),
            }
    return best or {}


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
