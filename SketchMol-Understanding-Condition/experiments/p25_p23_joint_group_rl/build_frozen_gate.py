#!/usr/bin/env python3
"""Build a deterministic target-blind P25 gate from frozen P23 evaluation prompts."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import train_p23_joint_grpo as rl


def stable_key(row: dict[str, object], seed: int) -> str:
    identity = row.get("condition_id", row.get("sample_id", ""))
    return hashlib.sha256(f"{seed}:{identity}".encode()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--denovo-5p", required=True, type=Path)
    parser.add_argument("--denovo-6p7p", required=True, type=Path)
    parser.add_argument("--edit-table2", required=True, type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--per-bucket", type=int, default=20)
    parser.add_argument("--seed", type=int, default=25250)
    args = parser.parse_args()

    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for path in (args.denovo_5p, args.denovo_6p7p, args.edit_table2):
        for row in rl.read_jsonl(path):
            bucket = rl.target_bucket(row)
            if bucket:
                grouped[bucket].append(row)
    selected: list[dict[str, object]] = []
    counts = {}
    for bucket in rl.TARGET_BUCKETS:
        values = sorted(grouped[bucket], key=lambda row: stable_key(row, args.seed))
        if len(values) < args.per_bucket:
            raise ValueError(f"gate bucket {bucket} has {len(values)} rows")
        chosen = values[: args.per_bucket]
        for row in chosen:
            row["gate_bucket"] = bucket
        selected.extend(chosen)
        counts[bucket] = len(chosen)
    selected.sort(key=lambda row: stable_key(row, args.seed + 1))
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.output_jsonl.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in selected)
    )
    manifest = {
        "protocol": "p25_frozen_joint_gate_v1",
        "seed": args.seed,
        "per_bucket": args.per_bucket,
        "rows": len(selected),
        "bucket_counts": counts,
        "target_molecule_fields_present": False,
        "sources": [str(args.denovo_5p), str(args.denovo_6p7p), str(args.edit_table2)],
    }
    args.output_jsonl.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
