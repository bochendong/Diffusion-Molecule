#!/usr/bin/env python3
"""Write reproducibility metadata and candidate-pool integrity checks for one v2 run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Sequence


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--train-seed", required=True)
    parser.add_argument("--eval-seed", required=True, type=int)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--eval-csv", required=True, type=Path)
    parser.add_argument("--candidate-csv", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--candidate-budgets", required=True)
    parser.add_argument("--selection-modes", required=True)
    parser.add_argument("--max-candidates", required=True, type=int)
    parser.add_argument("--input-modality", required=True)
    parser.add_argument("--include-source-copy-candidate", action="store_true")
    return parser.parse_args(argv)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    budgets = [int(value) for value in args.candidate_budgets.split(",") if value]
    if budgets and max(budgets) > args.max_candidates:
        raise ValueError(f"Largest budget {max(budgets)} exceeds max candidate pool {args.max_candidates}")
    pools: dict[str, list[int]] = defaultdict(list)
    pool_hashes: dict[str, set[str]] = defaultdict(set)
    source_copy_rows = 0
    with args.candidate_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    with args.eval_csv.open(newline="", encoding="utf-8") as handle:
        eval_rows = sum(1 for _ in csv.DictReader(handle))
    for row in rows:
        pool_id = str(row.get("candidate_pool_id", "")).strip()
        if pool_id:
            pools[pool_id].append(int(float(row.get("generation_rank", "0") or 0)))
            pool_hashes[pool_id].add(str(row.get("candidate_pool_hash", "")).strip())
        if str(row.get("candidate_source_copy", "")).strip().lower() in {"1", "true", "yes"}:
            source_copy_rows += 1
    pool_sizes = [len(values) for values in pools.values()]
    generation_order_unique = all(len(values) == len(set(values)) for values in pools.values())
    generation_order_is_prefix = all(
        sorted(values) == list(range(1, len(values) + 1)) for values in pools.values()
    )
    one_content_hash_per_pool = all(len(values - {""}) == 1 for values in pool_hashes.values())
    payload = {
        "protocol": "unified_joint_fair_v2_single_pool",
        "stage": args.stage,
        "train_seed": args.train_seed,
        "eval_seed": args.eval_seed,
        "task": args.task,
        "input_modality": args.input_modality,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint),
        "eval_csv": str(args.eval_csv),
        "eval_csv_sha256": sha256(args.eval_csv),
        "candidate_csv": str(args.candidate_csv),
        "candidate_csv_sha256": sha256(args.candidate_csv),
        "candidate_budgets": budgets,
        "selection_modes": [value for value in args.selection_modes.split(",") if value],
        "max_candidates": args.max_candidates,
        "candidate_rows": len(rows),
        "candidate_pools": len(pools),
        "expected_candidate_pools": eval_rows,
        "min_unique_candidates_per_pool": min(pool_sizes, default=0),
        "max_unique_candidates_per_pool": max(pool_sizes, default=0),
        "generation_rank_unique_within_pool": generation_order_unique,
        "generation_rank_is_contiguous_prefix": generation_order_is_prefix,
        "one_content_hash_per_pool": one_content_hash_per_pool,
        "include_source_copy_candidate": args.include_source_copy_candidate,
        "source_copy_candidate_rows": source_copy_rows,
    }
    if not args.include_source_copy_candidate and source_copy_rows:
        raise ValueError("Formal candidate pool unexpectedly contains source-copy augmentation")
    if len(pools) != eval_rows:
        raise ValueError(f"Expected one candidate pool per eval row ({eval_rows}), found {len(pools)}")
    if not generation_order_unique or not generation_order_is_prefix or not one_content_hash_per_pool:
        raise ValueError("Candidate pool failed generation-order/hash integrity checks")
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
