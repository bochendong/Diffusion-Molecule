#!/usr/bin/env python3
"""Audit and lock the R1 8-candidate prefix before extending P20 to fair @40."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preregistration", required=True, type=Path)
    p.add_argument("--r1-manifest", required=True, type=Path)
    p.add_argument("--reference", required=True, type=Path)
    p.add_argument("--prompts", required=True, type=Path)
    p.add_argument("--p17-prefix", required=True, type=Path)
    p.add_argument("--p18-prefix", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    prereg = json.loads(args.preregistration.read_text())
    r1 = json.loads(args.r1_manifest.read_text())
    current = {"reference": sha(args.reference), "prompts": sha(args.prompts)}
    expected = {
        "reference": prereg["locked_subset_reference_sha256"],
        "prompts": prereg["locked_prompts_sha256"],
    }
    if current != expected or r1["locked_sha256"] != expected:
        raise AssertionError(f"locked R1 data changed: current={current}, expected={expected}")
    ref = rows(args.reference)
    ids = {row["condition_id"] for row in ref}
    prefix_audit = {}
    for model, path in (("p17", args.p17_prefix), ("p18", args.p18_prefix)):
        digest = sha(path)
        if digest != prereg["locked_r1_prefix_sha256"][model]:
            raise AssertionError(f"{model} prefix SHA changed")
        data = rows(path)
        counts = Counter(row["condition_id"] for row in data)
        ranks = {key: sorted(int(float(row["candidate_rank"])) for row in data if row["condition_id"] == key) for key in ids}
        if len(data) != 2400 or set(counts) != ids or set(counts.values()) != {8}:
            raise AssertionError(f"{model}: prefix is not 300 x 8")
        if any(value != list(range(1, 9)) for value in ranks.values()):
            raise AssertionError(f"{model}: prefix ranks changed")
        prefix_audit[model] = {"sha256": digest, "rows": len(data), "conditions": len(counts)}
    payload = {
        "protocol": prereg["protocol"],
        "prepared_before_ranks_9_to_40": True,
        "locked_data_sha256": current,
        "locked_prefix": prefix_audit,
        "conditions": len(ref),
        "distribution": dict(Counter(f"{row['property_count']}p" for row in ref)),
        "training_target_overlap": r1["training_target_overlap"],
        "generation_target_access": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
