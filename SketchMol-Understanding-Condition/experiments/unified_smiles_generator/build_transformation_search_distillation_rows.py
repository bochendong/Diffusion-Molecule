#!/usr/bin/env python3
"""Turn verifier-scored train-pool candidates into leakage-safe policy-improvement rows."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import unified_smiles_generator as core  # noqa: E402

ID_COLUMNS = ("condition_id", "sample_id", "example_id", "pair_hash", "variant_id", "pair_id")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-rows-csv", required=True, type=Path)
    parser.add_argument("--candidate-csv", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--manifest-json", required=True, type=Path)
    parser.add_argument("--min-property-success", type=float, default=1.0)
    parser.add_argument("--min-edit-similarity", type=float, default=0.65)
    parser.add_argument("--winners-per-condition", type=int, default=1)
    parser.add_argument(
        "--source-replay-ratio",
        type=float,
        default=1.0,
        help="Number of original source rows replayed per verifier-selected row.",
    )
    parser.add_argument("--seed", type=int, default=73)
    return parser.parse_args(argv)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def row_id(row: Mapping[str, str]) -> str:
    for key in ID_COLUMNS:
        value = str(row.get(key, "") or "").strip()
        if value:
            return value
    return ""


def number(row: Mapping[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(str(row.get(key, "") or "").strip())
    except ValueError:
        return float(default)


def truthy(value: object) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def candidate_smiles(row: Mapping[str, str]) -> str:
    return str(
        row.get("generated_smiles", "")
        or row.get("predicted_smiles", "")
        or row.get("candidate_smiles", "")
        or ""
    ).strip()


def candidate_is_feasible(
    source_row: Mapping[str, str],
    candidate: Mapping[str, str],
    *,
    min_property_success: float,
    min_edit_similarity: float,
) -> tuple[bool, str]:
    smiles = candidate_smiles(candidate)
    if not smiles or not truthy(candidate.get("valid_smiles", "")):
        return False, "invalid"
    if number(candidate, "unified_property_success_fraction", -1.0) < float(min_property_success):
        return False, "property"
    if core.task_mode_for_row(source_row) == core.EDIT_MODE:
        if number(candidate, "source_tanimoto", -1.0) < float(min_edit_similarity):
            return False, "similarity"
        source = str(source_row.get("source_smiles", "") or source_row.get("molecule_smiles", "") or "").strip()
        if core.safe_canonical_smiles(source) == core.safe_canonical_smiles(smiles):
            return False, "source_copy"
    return True, "feasible"


def candidate_rank(row: Mapping[str, str]) -> tuple[float, float, float, float]:
    return (
        number(row, "unified_property_success_fraction", 0.0),
        number(row, "source_tanimoto", 0.0),
        number(row, "unified_finalizer_score", -1e9),
        -number(row, "generation_rank", number(row, "candidate_rank", 1e9)),
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_rows(path: Path, rows: list[Mapping[str, object]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_rows(args: argparse.Namespace) -> tuple[list[dict[str, object]], dict[str, object]]:
    source_rows = read_rows(args.source_rows_csv)
    source_by_id = {row_id(row): row for row in source_rows if row_id(row)}
    if len(source_by_id) != len(source_rows):
        raise ValueError("Every source row must have a unique condition/sample identifier")

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    unknown_candidate_ids = set()
    for candidate in read_rows(args.candidate_csv):
        identifier = row_id(candidate)
        if identifier not in source_by_id:
            unknown_candidate_ids.add(identifier)
            continue
        grouped[identifier].append(candidate)
    if unknown_candidate_ids:
        examples = sorted(unknown_candidate_ids)[:5]
        raise ValueError(f"Candidate rows are outside the authorized source pool: {examples}")

    selected_rows: list[dict[str, object]] = []
    rejection_counts: dict[str, int] = defaultdict(int)
    winners_by_mode: dict[str, int] = defaultdict(int)
    for identifier, source_row in source_by_id.items():
        feasible = []
        for candidate in grouped.get(identifier, []):
            accepted, reason = candidate_is_feasible(
                source_row,
                candidate,
                min_property_success=float(args.min_property_success),
                min_edit_similarity=float(args.min_edit_similarity),
            )
            if accepted:
                feasible.append(candidate)
            else:
                rejection_counts[reason] += 1
        for winner_index, candidate in enumerate(
            sorted(feasible, key=candidate_rank, reverse=True)[: max(1, int(args.winners_per_condition))],
            start=1,
        ):
            output = dict(source_row)
            output["reference_target_smiles"] = str(source_row.get("target_smiles", "") or "")
            output["target_smiles"] = candidate_smiles(candidate)
            output["distillation_origin"] = "verifier_search"
            output["distillation_winner_rank"] = winner_index
            output["distillation_property_success"] = candidate.get("unified_property_success_fraction", "")
            output["distillation_source_tanimoto"] = candidate.get("source_tanimoto", "")
            output["distillation_finalizer_score"] = candidate.get("unified_finalizer_score", "")
            selected_rows.append(output)
            winners_by_mode[core.task_mode_for_row(source_row)] += 1

    rng = random.Random(int(args.seed))
    replay_count = min(len(source_rows), max(0, round(len(selected_rows) * float(args.source_replay_ratio))))
    replay_rows = rng.sample(source_rows, replay_count) if replay_count else []
    for source_row in replay_rows:
        output = dict(source_row)
        output["distillation_origin"] = "source_replay"
        selected_rows.append(output)
    rng.shuffle(selected_rows)

    manifest = {
        "protocol": "unified_molecular_transformation_policy_search_distillation_v1",
        "source_rows_csv": str(args.source_rows_csv),
        "source_rows_sha256": sha256(args.source_rows_csv),
        "candidate_csv": str(args.candidate_csv),
        "candidate_csv_sha256": sha256(args.candidate_csv),
        "source_rows": len(source_rows),
        "conditions_with_candidates": len(grouped),
        "selected_search_rows": sum(winners_by_mode.values()),
        "source_replay_rows": len(replay_rows),
        "output_rows": len(selected_rows),
        "winners_by_mode": dict(sorted(winners_by_mode.items())),
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "min_property_success": float(args.min_property_success),
        "min_edit_similarity": float(args.min_edit_similarity),
        "winners_per_condition": int(args.winners_per_condition),
        "source_replay_ratio": float(args.source_replay_ratio),
        "seed": int(args.seed),
    }
    return selected_rows, manifest


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    rows, manifest = build_rows(args)
    if not rows:
        raise SystemExit("No feasible search-distillation rows were found")
    write_rows(args.output_csv, rows)
    args.manifest_json.parent.mkdir(parents=True, exist_ok=True)
    manifest["output_csv"] = str(args.output_csv)
    manifest["output_csv_sha256"] = sha256(args.output_csv)
    args.manifest_json.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
