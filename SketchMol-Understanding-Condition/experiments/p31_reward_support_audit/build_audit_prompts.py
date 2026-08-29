#!/usr/bin/env python3
"""Select frozen training-only P31 support-audit prompts."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
P30_DIR = SCRIPT_DIR.parent / "p30_balanced_shared_policy_rl"
if str(P30_DIR) not in sys.path:
    sys.path.insert(0, str(P30_DIR))
import train_balanced_shared_rl as p30  # noqa: E402


def stable_key(row: Mapping[str, object], seed: int) -> str:
    identity = row.get("example_id", row.get("sample_id", ""))
    return hashlib.sha256(f"{seed}:{identity}".encode()).hexdigest()


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--per-bucket", type=int, default=60)
    parser.add_argument("--shards", type=int, default=4)
    parser.add_argument("--seed", type=int, default=31001)
    args = parser.parse_args(argv)

    expected = (*p30.DE_NOVO_BUCKETS, *p30.EDIT_BUCKETS)
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in read_jsonl(args.input_jsonl):
        bucket = p30.balanced_bucket(row)
        if bucket in expected:
            grouped[bucket].append(row)
    selected: dict[str, list[dict[str, object]]] = {}
    for bucket in expected:
        rows = sorted(grouped[bucket], key=lambda row: stable_key(row, args.seed))
        if len(rows) < args.per_bucket:
            raise ValueError(f"{bucket}: {len(rows)} < {args.per_bucket}")
        selected[bucket] = rows[: args.per_bucket]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    handles = [
        (args.output_dir / f"shard-{index:02d}.jsonl").open("w", encoding="utf-8")
        for index in range(args.shards)
    ]
    shard_counts = [0] * args.shards
    try:
        for bucket_index, bucket in enumerate(expected):
            shard = bucket_index % args.shards
            for row in selected[bucket]:
                item = dict(row)
                item["_audit_bucket"] = bucket
                handles[shard].write(json.dumps(item, sort_keys=True) + "\n")
                shard_counts[shard] += 1
    finally:
        for handle in handles:
            handle.close()

    total = len(expected) * args.per_bucket
    unique = {
        str(row.get("example_id", row.get("sample_id", "")))
        for rows in selected.values() for row in rows
    }
    if len(unique) != total or sum(shard_counts) != total:
        raise AssertionError("P31 prompt selection is not unique and complete")
    manifest = {
        "protocol": "p31_p24_training_prompt_selection_v1",
        "source": str(args.input_jsonl),
        "seed": args.seed,
        "frozen_evaluation_rows_used": 0,
        "prompts_per_bucket": args.per_bucket,
        "bucket_counts": {bucket: len(selected[bucket]) for bucket in expected},
        "shards": args.shards,
        "shard_counts": shard_counts,
        "total_prompts": total,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
