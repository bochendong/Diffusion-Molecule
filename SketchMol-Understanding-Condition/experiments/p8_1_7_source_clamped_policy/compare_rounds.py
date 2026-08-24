#!/usr/bin/env python3
"""Verify the P8.1.7 single-factor comparison after R2."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def denovo_semantic_sha256(path: Path) -> str:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    payload = [
        (
            str(row.get("condition_id") or row.get("sample_id") or ""),
            str(row.get("direct_candidate_index") or ""),
            str(row.get("direct_candidate_canonical_smiles") or ""),
            str(row.get("direct_candidate_strict_fraction") or ""),
        )
        for row in rows
    ]
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--r1", required=True, type=Path)
    parser.add_argument("--r2", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    r1 = json.loads((args.r1 / "source_audit.json").read_text(encoding="utf-8"))
    r2 = json.loads((args.r2 / "source_audit.json").read_text(encoding="utf-8"))
    r1_denovo = args.r1 / "eval/denovo/candidates_normalized.csv"
    r2_denovo = args.r2 / "eval/denovo/candidates_normalized.csv"
    payload = {
        "protocol": "p8_1_7_single_factor_comparison_v1",
        "same_checkpoint": r1["checkpoint_sha256"] == r2["checkpoint_sha256"],
        "only_changed_factor": "source_clamp_scale",
        "r1_scale": r1["source_clamp_scale"],
        "r2_scale": r2["source_clamp_scale"],
        "denovo_candidates_bit_identical": denovo_semantic_sha256(r1_denovo) == denovo_semantic_sha256(r2_denovo),
        "edit_deltas": {
            "validity": r2["validity"] - r1["validity"],
            "identity_fraction": r2["identity_fraction"] - r1["identity_fraction"],
            "strict_nonidentity_fraction": r2["strict_nonidentity_fraction"] - r1["strict_nonidentity_fraction"],
            "unique_fraction": r2["unique_fraction"] - r1["unique_fraction"],
            "mean_source_tanimoto": None if r1["mean_source_tanimoto"] is None or r2["mean_source_tanimoto"] is None else r2["mean_source_tanimoto"] - r1["mean_source_tanimoto"],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["same_checkpoint"] or not payload["denovo_candidates_bit_identical"]:
        raise SystemExit("P8.1.7 causal contract failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
