#!/usr/bin/env python3
"""Freeze small, stratified validation subsets for the P2 repair gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--two-p-seven-p-csv", required=True, type=Path)
    parser.add_argument("--ood-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--per-property-count", type=int, default=64)
    parser.add_argument("--per-ood-bucket", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260823)
    return parser.parse_args(argv)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def condition_key(row: Mapping[str, object]) -> str:
    return str(row.get("condition_id") or row.get("sample_id") or "")


def stable_rank(row: Mapping[str, object], seed: int) -> str:
    return hashlib.sha256(f"{seed}|{condition_key(row)}".encode("utf-8")).hexdigest()


def stratified_sample(
    rows: list[dict[str, str]],
    *,
    stratum_column: str,
    per_stratum: int,
    seed: int,
) -> tuple[list[dict[str, str]], dict[str, int]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        value = str(row.get(stratum_column) or "").strip()
        if stratum_column == "property_count" and value:
            value = str(int(float(value)))
        grouped[value or "unknown"].append(row)
    selected: list[dict[str, str]] = []
    counts: dict[str, int] = {}
    for stratum, items in sorted(grouped.items()):
        if len(items) < per_stratum:
            raise RuntimeError(f"{stratum_column}={stratum} has {len(items)} rows; need {per_stratum}")
        chosen = sorted(items, key=lambda row: stable_rank(row, seed))[:per_stratum]
        selected.extend(chosen)
        counts[stratum] = len(chosen)
    selected.sort(key=lambda row: (str(row.get(stratum_column) or ""), condition_key(row)))
    return selected, counts


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    two_p_rows, property_counts = stratified_sample(
        read_rows(args.two_p_seven_p_csv),
        stratum_column="property_count",
        per_stratum=int(args.per_property_count),
        seed=int(args.seed),
    )
    ood_rows, ood_counts = stratified_sample(
        read_rows(args.ood_csv),
        stratum_column="ood_bucket",
        per_stratum=int(args.per_ood_bucket),
        seed=int(args.seed),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    two_p_path = args.output_dir / "denovo_2p7p_eval.csv"
    ood_path = args.output_dir / "denovo_ood_eval.csv"
    write_rows(two_p_path, two_p_rows)
    write_rows(ood_path, ood_rows)
    summary = {
        "protocol": "p2_validity_edit_repair_validation_subsets_v1",
        "seed": int(args.seed),
        "two_p_to_seven_p": {"path": str(two_p_path), "rows": len(two_p_rows), "strata": property_counts},
        "ood": {"path": str(ood_path), "rows": len(ood_rows), "strata": ood_counts},
    }
    (args.output_dir / "subset_manifest.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
