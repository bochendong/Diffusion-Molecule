#!/usr/bin/env python3
"""Merge frozen ranks 1-8 with generated ranks 9-40 and emit evaluator CSV."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", required=True, type=Path)
    parser.add_argument("--extension", required=True, type=Path)
    parser.add_argument("--raw-output", required=True, type=Path)
    parser.add_argument("--eval-output", required=True, type=Path)
    parser.add_argument("--expected-conditions", required=True, type=int)
    args = parser.parse_args()

    prefix, extension = read(args.prefix), read(args.extension)
    if len(prefix) != args.expected_conditions * 8:
        raise AssertionError(f"prefix rows {len(prefix)} != {args.expected_conditions * 8}")
    if len(extension) != args.expected_conditions * 32:
        raise AssertionError(f"extension rows {len(extension)} != {args.expected_conditions * 32}")
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in [*prefix, *extension]:
        grouped[row["condition_id"]].append(row)
    if len(grouped) != args.expected_conditions:
        raise AssertionError(f"conditions {len(grouped)} != {args.expected_conditions}")
    for key, rows in grouped.items():
        indices = sorted(int(float(row["direct_candidate_index"])) for row in rows)
        if indices != list(range(40)):
            raise AssertionError(f"{key}: candidate indices are not exactly 0..39")

    merged = [
        row
        for key in sorted(grouped)
        for row in sorted(grouped[key], key=lambda item: int(float(item["direct_candidate_index"])))
    ]
    write(args.raw_output, merged, list(merged[0]))
    evaluator = [
        {
            "condition_id": row["condition_id"],
            "direct_candidate_index": int(float(row["direct_candidate_index"])),
            "SMILES": row.get("direct_candidate_canonical_smiles", ""),
        }
        for row in merged
    ]
    write(args.eval_output, evaluator, ["condition_id", "direct_candidate_index", "SMILES"])
    manifest = {
        "protocol": "p23_frozen_raw40_merge_v1",
        "conditions": len(grouped),
        "rows": len(merged),
        "candidate_indices": [0, 39],
        "prefix_sha256": hashlib.sha256(args.prefix.read_bytes()).hexdigest(),
        "extension_sha256": hashlib.sha256(args.extension.read_bytes()).hexdigest(),
        "eval_sha256": hashlib.sha256(args.eval_output.read_bytes()).hexdigest(),
    }
    args.eval_output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
