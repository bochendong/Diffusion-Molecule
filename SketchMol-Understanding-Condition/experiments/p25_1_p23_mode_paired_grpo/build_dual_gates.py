#!/usr/bin/env python3
"""Build disjoint P25.1 dev and final gates after excluding the P25 gate."""

from __future__ import annotations

import argparse
import csv
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


def sha_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def denovo_prompt(row: dict[str, str]) -> dict[str, object]:
    messages, source, mode = p25.protocol.build_prompt(row)
    if mode != "de_novo" or source != "<EMPTY>":
        raise ValueError("P25.1 de-novo gate source produced a non-de-novo prompt")
    program = p25.protocol.condition_program(row, mode)
    condition_hash = p25.protocol.condition_hash_from_program(program)
    sample_id = str(row.get("sample_id", row.get("condition_id", "")))
    return {
        "condition_hash": condition_hash,
        "condition_id": str(row.get("condition_id", sample_id)),
        "messages": messages,
        "sample_id": sample_id,
        "source_smiles": "<EMPTY>",
        "task_key": "+".join(
            f"{item['property']}:around={float(item['goal']['around'])}"
            for item in program
        ),
        "task_mode": "de_novo",
        "_target_hash": sha_text(p25.protocol.canonical_smiles(row.get("target_smiles", ""))),
    }


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
    parser.add_argument("--denovo-eval-csv", required=True, type=Path)
    parser.add_argument("--edit-table2", required=True, type=Path)
    parser.add_argument("--train-jsonl", required=True, type=Path)
    parser.add_argument("--exclude-gate", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--per-bucket", type=int, default=20)
    parser.add_argument("--seed", type=int, default=251250)
    args = parser.parse_args()
    old_gate = p25.read_jsonl(args.exclude_gate)
    excluded = {identity(row) for row in old_gate}
    excluded_conditions = {str(row.get("condition_hash", "")) for row in old_gate}
    train_rows = p25.read_jsonl(args.train_jsonl)
    train_targets = {str(row.get("target_hash", "")) for row in train_rows}
    train_conditions = {str(row.get("condition_hash", "")) for row in train_rows}
    grouped = defaultdict(list)
    with args.denovo_eval_csv.open(newline="", encoding="utf-8") as handle:
        for source_row in csv.DictReader(handle):
            if int(float(source_row.get("property_count", "0") or 0)) not in {5, 6, 7}:
                continue
            row = denovo_prompt(source_row)
            if row["condition_hash"] in excluded_conditions | train_conditions:
                continue
            if row["_target_hash"] in train_targets:
                continue
            row.pop("_target_hash")
            grouped[p25.target_bucket(row)].append(row)
    for row in p25.read_jsonl(args.edit_table2):
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
    if any("target" in json.dumps(row, sort_keys=True).lower() for row in dev + final):
        raise AssertionError("target field leaked into a P25.1 gate prompt row")
    write_gate(args.output_dir / "dev.jsonl", dev, "dev", args.seed, args.per_bucket)
    write_gate(args.output_dir / "final.jsonl", final, "final", args.seed, args.per_bucket)
    print(json.dumps({"dev_rows": len(dev), "final_rows": len(final)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
