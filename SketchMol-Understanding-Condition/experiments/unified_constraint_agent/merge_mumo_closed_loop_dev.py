#!/usr/bin/env python3
"""Merge and audit fixed-n MuMO dev candidate shards."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-dir", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--manifest-json", required=True, type=Path)
    parser.add_argument("--shard-count", type=int, default=16)
    parser.add_argument("--expected-conditions", type=int, required=True)
    return parser.parse_args(argv)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    rows = []
    seen_conditions = set()
    for index in range(int(args.shard_count)):
        candidate_path = args.shard_dir / f"candidates_{index:03d}.csv"
        manifest_path = args.shard_dir / f"manifest_{index:03d}.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["evaluation_target_access"] is not False or int(manifest["candidate_budget"]) != 20:
            raise ValueError(f"Shard {index} contract violation")
        shard_rows = read_csv(candidate_path)
        shard_conditions = {row["condition_id"] for row in shard_rows}
        if seen_conditions & shard_conditions:
            raise ValueError(f"Duplicate conditions across shard {index}")
        seen_conditions.update(shard_conditions)
        rows.extend(shard_rows)
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["condition_id"]].append(row)
    if len(grouped) != int(args.expected_conditions):
        raise ValueError(f"conditions={len(grouped)} expected={args.expected_conditions}")
    for condition, items in grouped.items():
        ranks = sorted(int(row["candidate_rank"]) for row in items)
        if ranks != list(range(1, 21)):
            raise ValueError(f"{condition} rank contract failed: {ranks}")
    rows.sort(key=lambda row: (row["condition_id"], int(row["candidate_rank"])))
    write_csv(args.output_csv, rows)
    manifest = {
        "protocol": "mumo_fit_only_pair_verifier_closed_loop_dev_v1",
        "evaluation_target_access": False,
        "evaluation_oracle_access": False,
        "official_test_content_access": False,
        "candidate_budget": 20,
        "conditions": len(grouped),
        "candidate_rows": len(rows),
        "shard_count": int(args.shard_count),
    }
    args.manifest_json.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_json.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
