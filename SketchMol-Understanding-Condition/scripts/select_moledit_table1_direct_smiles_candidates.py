#!/usr/bin/env python3
"""Select one direct-SMILES candidate per MolEdit Table1 input.

This is the direct-SMILES analogue of the materialized table-success rerank:
given a candidate-level CSV, choose the best candidate within the first n
samples using MolEdit Table1 edit-direction success, source similarity, and a
small source-copy penalty. The selected CSV can be passed to
evaluate_moledit_table_metrics.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[0]
REPO_DIR = PROJECT_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_direct_smiles_generator_rl import (  # noqa: E402
    _safe_canonical_smiles,
    source_copy_component,
    source_similarity_component,
    table1_edit_score_components,
)


ID_COLUMNS = ("example_id", "condition_id", "sample_id", "pair_hash")
SMILES_COLUMNS = (
    "generated_smiles",
    "direct_candidate_canonical_smiles",
    "candidate_smiles",
    "predicted_smiles",
    "smiles",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path, help="MolEdit reference CSV/JSONL.")
    parser.add_argument("--candidate-predictions", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--candidate-limit", type=int, default=20)
    parser.add_argument("--method-name", default=None)
    parser.add_argument("--valid-weight", type=float, default=1.0)
    parser.add_argument("--edit-success-weight", type=float, default=100.0)
    parser.add_argument("--distance-weight", type=float, default=5.0)
    parser.add_argument("--distance-clip", type=float, default=10.0)
    parser.add_argument("--source-similarity-weight", type=float, default=6.0)
    parser.add_argument("--source-similarity-threshold", type=float, default=0.4)
    parser.add_argument("--source-copy-penalty", type=float, default=2.0)
    parser.add_argument("--rank-penalty", type=float, default=1e-6)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    references = load_rows_by_id(args.reference)
    candidate_rows = read_rows(args.candidate_predictions)
    candidates_by_id: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in candidate_rows:
        row_id = normalized_row_id(row)
        if row_id:
            candidates_by_id[row_id].append(row)
    for rows in candidates_by_id.values():
        rows.sort(key=candidate_sort_key)

    method_name = args.method_name or f"direct_smiles_moledit_table1_rerank_n{int(args.candidate_limit)}"
    selected_rows = []
    summary = {
        "reference_rows": len(references),
        "candidate_rows": len(candidate_rows),
        "candidate_limit": int(args.candidate_limit),
        "selected_rows": 0,
        "missing_candidate_rows": 0,
        "mean_selected_edit_success_fraction": 0.0,
        "mean_selected_source_similarity_component": 0.0,
    }
    selected_successes = []
    selected_similarity = []
    for ref_id, ref in references.items():
        pool = candidates_by_id.get(ref_id, [])[: max(1, int(args.candidate_limit))]
        if not pool:
            summary["missing_candidate_rows"] += 1
            continue
        selected = max((score_candidate(ref, row, rank=rank, args=args) for rank, row in enumerate(pool)), key=lambda item: item["score"])
        out = dict(ref)
        out.update(
            {
                "generated_smiles": selected["generated_smiles"],
                "method": method_name,
                "direct_selected_candidate_rank": selected["rank"],
                "direct_selected_candidate_score": selected["score"],
                "direct_selected_edit_success_fraction": selected["edit_success_fraction"],
                "direct_selected_edit_distance": selected["edit_distance"],
                "direct_selected_source_similarity_component": selected["source_similarity_component"],
                "direct_candidate_limit": int(args.candidate_limit),
            }
        )
        selected_rows.append(out)
        selected_successes.append(float(selected["edit_success_fraction"]))
        selected_similarity.append(float(selected["source_similarity_component"]))
    summary["selected_rows"] = len(selected_rows)
    summary["mean_selected_edit_success_fraction"] = mean(selected_successes)
    summary["mean_selected_source_similarity_component"] = mean(selected_similarity)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = infer_fieldnames(
        selected_rows,
        preferred=(
            "example_id",
            "condition_id",
            "source_smiles",
            "target_smiles",
            "instruction",
            "instruction_tasks",
            "generated_smiles",
            "method",
        ),
    )
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(selected_rows)
    args.output_csv.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output_csv": str(args.output_csv), **summary}, indent=2, sort_keys=True))
    return 0


def score_candidate(ref: Mapping[str, str], row: Mapping[str, str], *, rank: int, args: argparse.Namespace) -> dict[str, float | int | str]:
    raw = first_value(row, SMILES_COLUMNS)
    canonical = _safe_canonical_smiles(raw)
    if not canonical:
        return {
            "generated_smiles": "",
            "rank": int(rank),
            "score": -1_000_000.0 - float(rank),
            "edit_success_fraction": 0.0,
            "edit_distance": math.inf,
            "source_similarity_component": 0.0,
        }
    edit_success_fraction, edit_distance = table1_edit_score_components(ref, canonical)
    edit_distance = min(float(edit_distance), float(args.distance_clip))
    source_similarity = source_similarity_component(
        ref,
        canonical,
        threshold=float(args.source_similarity_threshold),
    )
    copy_penalty = source_copy_component(ref, canonical)
    score = (
        float(args.valid_weight)
        + float(args.edit_success_weight) * float(edit_success_fraction)
        - float(args.distance_weight) * float(edit_distance)
        + float(args.source_similarity_weight) * float(source_similarity)
        - float(args.source_copy_penalty) * float(copy_penalty)
        - float(args.rank_penalty) * float(rank)
    )
    return {
        "generated_smiles": canonical,
        "rank": int(rank),
        "score": float(score),
        "edit_success_fraction": float(edit_success_fraction),
        "edit_distance": float(edit_distance),
        "source_similarity_component": float(source_similarity),
    }


def load_rows_by_id(path: Path) -> dict[str, dict[str, str]]:
    out = {}
    for row in read_rows(path):
        row_id = normalized_row_id(row)
        if row_id:
            out[row_id] = row
    return out


def read_rows(path: Path) -> list[dict[str, str]]:
    if path.suffix == ".jsonl":
        rows = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append({str(k): "" if v is None else str(v) for k, v in json.loads(line).items()})
        return rows
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def normalized_row_id(row: Mapping[str, object]) -> str:
    text = first_value(row, ID_COLUMNS)
    if text.startswith("edit:"):
        text = text.split(":")[-1]
    return text


def first_value(row: Mapping[str, object], columns: Iterable[str]) -> str:
    for column in columns:
        value = str(row.get(column, "") or "").strip()
        if value:
            return value
    return ""


def candidate_sort_key(row: Mapping[str, object]) -> tuple[int, str]:
    raw = str(row.get("direct_candidate_index", "") or "").strip()
    try:
        index = int(float(raw))
    except ValueError:
        index = 1_000_000
    return index, first_value(row, SMILES_COLUMNS)


def infer_fieldnames(rows: list[Mapping[str, object]], *, preferred: Sequence[str]) -> list[str]:
    keys = []
    seen = set()
    for key in preferred:
        if key not in seen:
            keys.append(key)
            seen.add(key)
    for row in rows:
        for key in row.keys():
            if key not in seen:
                keys.append(str(key))
                seen.add(str(key))
    return keys


def mean(values: Sequence[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return sum(finite) / len(finite) if finite else 0.0


if __name__ == "__main__":
    raise SystemExit(main())
