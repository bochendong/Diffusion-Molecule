#!/usr/bin/env python3
"""Build disjoint P25.1 dev and final gates after excluding the P25 gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
P25_DIR = SCRIPT_DIR.parent / "p25_p23_joint_group_rl"
sys.path.insert(0, str(P25_DIR))
import train_p23_joint_grpo as p25  # noqa: E402


def identity(row) -> tuple[str, str]:
    return str(row.get("condition_id", "")), str(row.get("sample_id", ""))


def stable_key(row, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{identity(row)}".encode()).hexdigest()


def write_gate(path: Path, rows, name: str, seed: int, per_bucket: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    counts = defaultdict(int)
    for row in rows:
        counts[p25.target_bucket(row)] += 1
    path.with_suffix(".manifest.json").write_text(json.dumps({
        "protocol": "p25_1_disjoint_gate_v1",
        "name": name,
        "seed": seed,
        "per_bucket": per_bucket,
        "rows": len(rows),
        "bucket_counts": dict(sorted(counts.items())),
        "target_access": False,
    }, indent=2, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--denovo-5p", required=True, type=Path)
    parser.add_argument("--denovo-6p7p", required=True, type=Path)
    parser.add_argument("--edit-table2", required=True, type=Path)
    parser.add_argument("--exclude-gate", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--per-bucket", type=int, default=20)
    parser.add_argument("--seed", type=int, default=251250)
    args = parser.parse_args()
    excluded = {identity(row) for row in p25.read_jsonl(args.exclude_gate)}
    grouped = defaultdict(list)
    for source in (args.denovo_5p, args.denovo_6p7p, args.edit_table2):
        for row in p25.read_jsonl(source):
            bucket = p25.target_bucket(row)
            if bucket and identity(row) not in excluded:
                grouped[bucket].append(row)
    dev, final = [], []
    for bucket in p25.TARGET_BUCKETS:
        values = sorted(grouped[bucket], key=lambda row: stable_key(row, args.seed))
        needed = 2 * args.per_bucket
        if len(values) < needed:
            raise ValueError(f"{bucket} has {len(values)} unused rows, needs {needed}")
        dev.extend(values[: args.per_bucket])
        final.extend(values[args.per_bucket : needed])
    dev.sort(key=lambda row: stable_key(row, args.seed + 1))
    final.sort(key=lambda row: stable_key(row, args.seed + 2))
    if {identity(row) for row in dev} & {identity(row) for row in final}:
        raise AssertionError("dev and final gates overlap")
    write_gate(args.output_dir / "dev.jsonl", dev, "dev", args.seed, args.per_bucket)
    write_gate(args.output_dir / "final.jsonl", final, "final", args.seed, args.per_bucket)
    print(json.dumps({"dev_rows": len(dev), "final_rows": len(final)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
