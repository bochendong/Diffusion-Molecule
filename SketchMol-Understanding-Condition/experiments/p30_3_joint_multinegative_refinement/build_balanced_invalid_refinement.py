#!/usr/bin/env python3
"""Select a balanced train-only invalid-corruption refinement set."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
P25_DIR = SCRIPT_DIR.parent / "p25_p23_joint_group_rl"
if str(P25_DIR) not in sys.path:
    sys.path.insert(0, str(P25_DIR))
import train_p23_joint_grpo as p25  # noqa: E402


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def stable_key(row: Mapping[str, object], seed: int) -> str:
    return hashlib.sha256(f"{seed}:{row.get('pair_id', row.get('example_id', ''))}".encode()).hexdigest()


def bucket(row: Mapping[str, object]) -> str:
    mode = str(row.get("task_mode", ""))
    if mode == "de_novo":
        count = p25.property_count(row)
        return f"de_novo:{count}p" if 2 <= count <= 7 else ""
    if mode == "edit":
        task = str(row.get("task_key", ""))
        return f"edit:{task}" if task in p25.TARGET_EDIT_TASKS else ""
    return ""


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True, type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--denovo-per-arity", type=int, default=100)
    parser.add_argument("--edit-per-task", type=int, default=60)
    parser.add_argument("--seed", type=int, default=30301)
    args = parser.parse_args(argv)

    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in read_jsonl(args.input_jsonl):
        if str(row.get("negative_type", "")) == "invalid_corruption":
            key = bucket(row)
            if key:
                grouped[key].append(row)
    quotas = {f"de_novo:{count}p": args.denovo_per_arity for count in range(2, 8)}
    quotas.update({f"edit:{task}": args.edit_per_task for task in sorted(p25.TARGET_EDIT_TASKS)})
    selected: list[dict[str, object]] = []
    for key, quota in quotas.items():
        rows = sorted(grouped[key], key=lambda row: stable_key(row, args.seed))
        if len(rows) < quota:
            raise ValueError(f"insufficient invalid pairs for {key}: {len(rows)} < {quota}")
        for row in rows[:quota]:
            item = dict(row)
            item["chosen_ce_weight"] = 1.0
            item["negative_weight"] = 0.20
            item["margin"] = 0.15
            selected.append(item)
    selected.sort(key=lambda row: stable_key(row, args.seed + 1))
    expected = 6 * args.denovo_per_arity + len(p25.TARGET_EDIT_TASKS) * args.edit_per_task
    if len(selected) != expected or len({row["example_id"] for row in selected}) != expected:
        raise AssertionError("refinement selection is not unique and complete")
    counts = Counter(bucket(row) for row in selected)
    manifest = {
        "protocol": "p30_3_balanced_invalid_refinement_data_v1",
        "input": str(args.input_jsonl),
        "seed": args.seed,
        "pairs": len(selected),
        "bucket_counts": dict(sorted(counts.items())),
        "negative_types": dict(Counter(str(row["negative_type"]) for row in selected)),
        "chosen_ce_weight": 1.0,
        "negative_weight": 0.20,
        "margin": 0.15,
        "eval_rows_used": 0,
    }
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
