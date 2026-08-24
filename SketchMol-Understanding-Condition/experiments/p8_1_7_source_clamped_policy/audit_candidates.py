#!/usr/bin/env python3
"""Audit raw source retention without allowing identity-copy inflation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--scale", required=True, type=float)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    with args.candidates.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    valid = identity = strict_nonidentity = 0
    similarities = []
    unique = set()
    for row in rows:
        generated = str(row.get("generated_smiles", "") or "").strip()
        source = str(row.get("source_smiles", "") or "").strip()
        is_valid = str(row.get("valid_smiles", "")).lower() == "true"
        if is_valid:
            valid += 1
            unique.add(generated)
            is_identity = generated == source
            identity += int(is_identity)
            strict = str(row.get("table1_strict_success", "")).lower() == "true"
            strict_nonidentity += int(strict and not is_identity)
            try:
                value = float(row.get("source_tanimoto", ""))
            except (TypeError, ValueError):
                value = math.nan
            if math.isfinite(value):
                similarities.append(value)
    payload = {
        "protocol": "p8_1_7_raw_source_audit_v1",
        "checkpoint_sha256": sha256(args.checkpoint),
        "source_clamp_scale": args.scale,
        "candidate_rows": len(rows),
        "validity": valid / max(len(rows), 1),
        "identity_fraction": identity / max(len(rows), 1),
        "identity_among_valid": identity / max(valid, 1),
        "strict_nonidentity_fraction": strict_nonidentity / max(len(rows), 1),
        "unique_fraction": len(unique) / max(len(rows), 1),
        "mean_source_tanimoto": statistics.fmean(similarities) if similarities else None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
