#!/usr/bin/env python3
"""Fail closed unless empty-source de novo sampling is functionally unchanged."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def candidate_signature(path: Path) -> tuple[str, int]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    payload = [
        [
            row.get("condition_id", ""),
            row.get("candidate_rank", ""),
            row.get("generated_smiles", ""),
            row.get("direct_candidate_raw_smiles", ""),
        ]
        for row in rows
    ]
    encoded = json.dumps(payload, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(encoded).hexdigest(), len(rows)


def metrics(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8")).get("records")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", required=True, type=Path)
    parser.add_argument("--r1-dir", required=True, type=Path)
    parser.add_argument("--r2-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    dirs = {"base": args.base_dir, "r1": args.r1_dir, "r2": args.r2_dir}
    signatures = {name: candidate_signature(path / "candidates.csv") for name, path in dirs.items()}
    metric_records = {name: metrics(path / "metrics.json") for name, path in dirs.items()}
    hashes = {name: value[0] for name, value in signatures.items()}
    counts = {name: value[1] for name, value in signatures.items()}
    checks = {
        "candidate_order_and_smiles_sha_exact": len(set(hashes.values())) == 1,
        "candidate_rows_exact": counts == {"base": 1280, "r1": 1280, "r2": 1280},
        "metrics_records_exact": metric_records["base"] == metric_records["r1"] == metric_records["r2"],
    }
    payload = {
        "protocol": "p8_2_empty_source_functional_replay_v1",
        "seed": 1982,
        "num_samples": 20,
        "candidate_sha256": hashes,
        "candidate_rows": counts,
        "checks": checks,
        "status": "pass" if all(checks.values()) else "fail",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
