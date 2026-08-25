#!/usr/bin/env python3
"""Merge the locked byte-identical R1 prefix with R2 ranks 9-40."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore"); writer.writeheader(); writer.writerows(rows)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--audit", required=True, type=Path)
    p.add_argument("--model", required=True, choices=("p17", "p18"))
    p.add_argument("--prefix", required=True, type=Path)
    p.add_argument("--extension", required=True, type=Path)
    p.add_argument("--raw-output", required=True, type=Path)
    p.add_argument("--eval-output", required=True, type=Path)
    args = p.parse_args()
    audit = json.loads(args.audit.read_text())
    if hashlib.sha256(args.prefix.read_bytes()).hexdigest() != audit["locked_prefix"][args.model]["sha256"]:
        raise AssertionError("R1 prefix is not byte-identical")
    prefix, extension = read(args.prefix), read(args.extension)
    if len(prefix) != 2400 or len(extension) != 9600:
        raise AssertionError(f"unexpected rows prefix={len(prefix)} extension={len(extension)}")
    grouped = defaultdict(list)
    for row in [*prefix, *extension]:
        grouped[row["condition_id"]].append(row)
    if len(grouped) != 300:
        raise AssertionError("expected 300 conditions")
    for key, values in grouped.items():
        ranks = sorted(int(float(row["candidate_rank"])) for row in values)
        if ranks != list(range(1, 41)):
            raise AssertionError(f"{key}: ranks are not exactly 1..40")
    merged = [row for key in sorted(grouped) for row in sorted(grouped[key], key=lambda item: int(float(item["candidate_rank"])))]
    write(args.raw_output, merged, list(merged[0]))
    evaluator = [
        {"condition_id": row["condition_id"], "direct_candidate_index": int(float(row["candidate_rank"])) - 1,
         "SMILES": row.get("direct_candidate_canonical_smiles", "")}
        for row in merged
    ]
    write(args.eval_output, evaluator, ["condition_id", "direct_candidate_index", "SMILES"])
    payload = {
        "protocol": "p20_r2_merged_fair40_v1", "model": args.model,
        "conditions": len(grouped), "rows": len(merged), "ranks": [1, 40],
        "prefix_sha256_verified": audit["locked_prefix"][args.model]["sha256"],
        "raw_output_sha256": hashlib.sha256(args.raw_output.read_bytes()).hexdigest(),
        "eval_output_sha256": hashlib.sha256(args.eval_output.read_bytes()).hexdigest(),
        "rank_counts": dict(Counter(int(float(row["candidate_rank"])) for row in merged)),
    }
    args.raw_output.with_suffix(".manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: payload[k] for k in ("model", "conditions", "rows", "ranks", "prefix_sha256_verified")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
