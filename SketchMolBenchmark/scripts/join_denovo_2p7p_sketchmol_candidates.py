#!/usr/bin/env python3
"""Merge SketchMol shard OCR candidates and best-of-K rerank for 2p-7p eval."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

BENCH_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BENCH_ROOT.parent
PROJECT_DIR = REPO_ROOT / "SketchMol-Understanding-Condition"
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from train_direct_smiles_generator import select_generated_candidate  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-csv", type=Path, required=True)
    parser.add_argument("--shards-dir", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--candidate-budget", type=int, default=40)
    parser.add_argument("--smiles-column", default="SMILES")
    parser.add_argument("--summary-json", type=Path, default=None)
    return parser.parse_args(argv)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_shard_candidates(shards_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for shard_csv in sorted(shards_dir.glob("shard_*/shard_candidates.csv")):
        rows.extend(read_rows(shard_csv))
    return rows


def condition_key(row: Mapping[str, str]) -> str:
    return str(row.get("condition_id") or row.get("sample_id") or "").strip()


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    eval_rows = read_rows(args.eval_csv)
    candidate_rows = load_shard_candidates(args.shards_dir)

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in candidate_rows:
        key = condition_key(row)
        if key:
            grouped[key].append(row)

    predictions: list[dict[str, str]] = []
    missing_conditions: list[str] = []

    for eval_row in eval_rows:
        key = condition_key(eval_row)
        group = grouped.get(key, [])
        smiles_candidates = [
            str(item.get(args.smiles_column) or item.get("SMILES") or "").strip()
            for item in sorted(group, key=lambda item: int(item.get("candidate_index") or 0))
        ]
        smiles_candidates = [value for value in smiles_candidates if value][: args.candidate_budget]
        selected = select_generated_candidate(eval_row, smiles_candidates, property_rerank=True)

        out_row = dict(eval_row)
        out_row["generated_smiles"] = str(selected.get("generated_smiles") or "")
        out_row["method"] = f"sketchmol_real_ocr_n{args.candidate_budget}"
        out_row["sketchmol_candidate_count"] = str(selected.get("candidate_count") or 0)
        out_row["sketchmol_valid_candidate_count"] = str(selected.get("valid_candidate_count") or 0)
        out_row["sketchmol_best_candidate_rank"] = str(selected.get("best_candidate_rank") or -1)
        out_row["sketchmol_rerank_strict_fraction"] = str(selected.get("strict_fraction") or 0.0)
        out_row["sketchmol_rerank_property_distance"] = str(
            selected.get("normalized_property_distance") or float("inf")
        )

        if not group:
            missing_conditions.append(key)
        predictions.append(out_row)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(predictions[0].keys()) if predictions else list(eval_rows[0].keys())
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(predictions)

    summary = {
        "eval_rows": len(eval_rows),
        "candidate_rows": len(candidate_rows),
        "conditions_with_candidates": len(grouped),
        "missing_conditions": len(missing_conditions),
        "missing_condition_ids": missing_conditions[:20],
        "output_csv": str(args.output_csv),
        "candidate_budget": args.candidate_budget,
    }
    summary_path = args.summary_json or args.output_csv.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 1 if missing_conditions else 0


if __name__ == "__main__":
    raise SystemExit(main())
