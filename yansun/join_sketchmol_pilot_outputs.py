#!/usr/bin/env python3
"""Join Yansun pilot shard outputs into one CSV with image paths and SMILES."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Sequence


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-csv", required=True, type=Path)
    parser.add_argument("--shards-dir", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--summary-json", type=Path, default=None)
    return parser.parse_args(argv)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    eval_rows = read_rows(args.eval_csv)
    eval_ids = {str(row.get("condition_id") or "") for row in eval_rows}

    joined: list[dict[str, str]] = []
    for shard_csv in sorted(args.shards_dir.glob("shard_*/shard_candidates.csv")):
        joined.extend(read_rows(shard_csv))

    present_ids = {str(row.get("condition_id") or "") for row in joined}
    missing = sorted(cid for cid in eval_ids if cid and cid not in present_ids)

    write_rows(args.output_csv, joined)
    summary = {
        "eval_rows": len(eval_rows),
        "candidate_rows": len(joined),
        "conditions_with_candidates": len(present_ids & eval_ids),
        "missing_conditions": missing,
        "output_csv": str(args.output_csv),
    }
    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2))
    if missing:
        print(f"WARNING: {len(missing)} eval rows still missing candidates.", file=__import__("sys").stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
