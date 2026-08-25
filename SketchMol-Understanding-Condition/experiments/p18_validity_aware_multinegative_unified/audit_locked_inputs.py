#!/usr/bin/env python3
"""Verify every reused P17 train/dev/pilot input against preregistered hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p17-output", required=True, type=Path)
    parser.add_argument("--preregistration", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    relpaths = {
        "train": "data/train.paired.jsonl",
        "id_dev": "data/dev.id_condition_source_isolated.jsonl",
        "ood_dev": "data/dev.condition_source_ood.jsonl",
        "table1_prompts": "pilot/table1_pilot.prompts.jsonl",
        "denovo_prompts": "pilot/denovo_pilot.prompts.jsonl",
        "table1_reference": "pilot/table1_pilot.reference.csv",
        "denovo_reference": "pilot/denovo_pilot.reference.csv",
    }
    locked = json.loads(args.preregistration.read_text())["locked_p17_input_sha256"]
    records = {}
    for key, relpath in relpaths.items():
        path = args.p17_output / relpath
        actual = sha256(path)
        records[key] = {"path": str(path), "sha256": actual, "matches_preregistration": actual == locked[key]}
    failures = [key for key, record in records.items() if not record["matches_preregistration"]]
    payload = {
        "protocol": "p18_locked_input_audit_v1",
        "all_hashes_match": not failures,
        "failures": failures,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if failures:
        raise SystemExit("locked P17 input audit failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
