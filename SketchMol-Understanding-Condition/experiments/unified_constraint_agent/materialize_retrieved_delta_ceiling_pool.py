#!/usr/bin/env python3
"""Materialize an oracle-blind top-k diagnostic prefix from an enumerated pool.

This is deliberately not a paper-facing candidate set.  It freezes the same
train-derived ordering available to the v6 planner, keeps at most ``k`` rows
per condition, and records enough provenance to distinguish a support-ceiling
diagnostic from the fixed final n=20 benchmark contract.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--enumerated-candidates-csv", required=True, type=Path)
    parser.add_argument("--source-manifest-json", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--manifest-json", required=True, type=Path)
    parser.add_argument("--candidate-limit", type=int, default=96)
    parser.add_argument("--paper-candidate-budget", type=int, default=20)
    parser.add_argument("--expected-conditions", type=int, default=50)
    return parser.parse_args(argv)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_rows(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def condition_id(row: Mapping[str, object]) -> str:
    value = str(row.get("condition_id", "") or "").strip()
    if not value:
        raise ValueError("Every enumerated candidate requires condition_id")
    return value


def candidate_rank(row: Mapping[str, object], fallback: int) -> tuple[float, int]:
    try:
        rank = float(str(row.get("candidate_rank", "") or "").strip())
    except ValueError:
        rank = float(fallback + 1)
    if not math.isfinite(rank):
        rank = float(fallback + 1)
    return rank, fallback


def validate_source_manifest(payload: Mapping[str, object], *, expected_conditions: int) -> None:
    if payload.get("evaluation_target_access") is not False:
        raise ValueError("Source candidate manifest must declare evaluation_target_access=false")
    conditions = int(payload.get("evaluation_conditions", 0) or 0)
    if conditions != int(expected_conditions):
        raise ValueError(
            f"Source manifest has {conditions} evaluation conditions; expected {expected_conditions}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if int(args.paper_candidate_budget) != 20:
        raise ValueError("The paper-facing benchmark contract is fixed at n=20")
    if int(args.candidate_limit) < int(args.paper_candidate_budget):
        raise ValueError("Diagnostic candidate limit cannot be below the final n=20 budget")

    source_manifest = json.loads(args.source_manifest_json.read_text(encoding="utf-8"))
    if not isinstance(source_manifest, dict):
        raise ValueError("Source manifest must contain one JSON object")
    validate_source_manifest(source_manifest, expected_conditions=int(args.expected_conditions))

    rows = read_rows(args.enumerated_candidates_csv)
    grouped: dict[str, list[tuple[int, dict[str, str]]]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[condition_id(row)].append((index, row))
    if len(grouped) != int(args.expected_conditions):
        raise ValueError(
            f"Enumerated pool has {len(grouped)} conditions; expected {args.expected_conditions}"
        )

    output: list[dict[str, object]] = []
    counts: list[int] = []
    for key in sorted(grouped):
        ordered = sorted(grouped[key], key=lambda item: candidate_rank(item[1], item[0]))
        prefix = ordered[: int(args.candidate_limit)]
        counts.append(len(prefix))
        for prefix_rank, (_index, row) in enumerate(prefix, start=1):
            output.append(
                {
                    **row,
                    "candidate_selected": "False",
                    "diagnostic_prefix_rank": int(prefix_rank),
                    "diagnostic_candidate_limit": int(args.candidate_limit),
                }
            )

    write_rows(args.output_csv, output)
    manifest = {
        "protocol": "retrieved_delta_oracle_blind_support_ceiling_v1",
        "data_role": "train_only_heldout_diagnostic",
        "diagnostic_only": True,
        "evaluation_target_access": False,
        "oracle_used_for_selection": False,
        "paper_facing_candidate_budget": int(args.paper_candidate_budget),
        "diagnostic_candidate_limit": int(args.candidate_limit),
        "evaluation_conditions": len(grouped),
        "output_rows": len(output),
        "min_candidates_per_condition": min(counts),
        "mean_candidates_per_condition": sum(counts) / max(len(counts), 1),
        "max_candidates_per_condition": max(counts),
        "complete_limit_conditions": sum(value == int(args.candidate_limit) for value in counts),
        "short_condition_count": sum(value < int(args.candidate_limit) for value in counts),
        "selection_key": "pre-oracle candidate_rank",
        "enumerated_candidates_csv": str(args.enumerated_candidates_csv),
        "enumerated_candidates_sha256": sha256(args.enumerated_candidates_csv),
        "source_manifest_json": str(args.source_manifest_json),
        "source_manifest_sha256": sha256(args.source_manifest_json),
    }
    args.manifest_json.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_json.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
