#!/usr/bin/env python3
"""Audit unique edit-pair capacity for exactly balanced 1p--7p task buckets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import build_release as release


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", required=True, type=Path)
    parser.add_argument("--heldout", type=Path, action="append", default=[])
    parser.add_argument("--target-rows", type=int, default=2_000_000)
    parser.add_argument("--seed", type=int, default=24002)
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args()
    frozen = release.heldout_hashes(args.heldout)
    counts: Counter[str] = Counter()
    assignments: Counter[int] = Counter()
    seen: set[int] = set()
    with args.input_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            counts["input"] += 1
            fields = release.edit_fields(row)
            if fields is None:
                counts["ineligible"] += 1
                continue
            source, target, active = fields
            if (
                hashlib.sha256(source.encode()).hexdigest() in frozen
                or hashlib.sha256(target.encode()).hexdigest() in frozen
            ):
                counts["heldout_overlap"] += 1
                continue
            pair = release.rank64(f"{source}\n{target}")
            if pair in seen:
                counts["duplicate_pair"] += 1
                continue
            seen.add(pair)
            counts[f"active_count_{len(active)}"] += 1
            raw_id = str(row.get("example_id", "") or row.get("sample_id", ""))
            material = f"{args.seed}:{raw_id}:{source}:{target}"
            bucket = release.assigned_bucket(
                list(range(1, min(7, len(active)) + 1)), assignments, material,
            )
            assignments[bucket] += 1
    quotas = release.balanced_quotas(args.target_rows, list(range(1, 8)))
    capacity = {f"{key}p": assignments[key] for key in range(1, 8)}
    shortfalls = {
        f"{key}p": max(0, quotas[key] - assignments[key]) for key in range(1, 8)
    }
    result = {
        "protocol": "p24_edit_balanced_capacity_audit_v1",
        "target_rows": args.target_rows,
        "target_bucket_quotas": {f"{key}p": value for key, value in quotas.items()},
        "assigned_capacity": capacity,
        "max_exact_balanced_rows": min(assignments.values()) * 7,
        "shortfall": shortfalls,
        "can_build_exact_balanced_release": not any(shortfalls.values()),
        "counts": dict(counts),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return int(not result["can_build_exact_balanced_release"])


if __name__ == "__main__":
    raise SystemExit(main())
