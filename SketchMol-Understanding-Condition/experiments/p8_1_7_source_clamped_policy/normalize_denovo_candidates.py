#!/usr/bin/env python3
"""Map common full-SMILES candidate columns to the honest P6 raw evaluator."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
UNIFIED_DIR = SCRIPT_DIR.parent / "unified_smiles_generator"
sys.path.insert(0, str(UNIFIED_DIR))
import unified_smiles_generator as core  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    with args.input.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        generated = str(row.get("generated_smiles", "") or "").strip()
        canonical = core.safe_canonical_smiles(generated)
        row["direct_candidate_index"] = str(max(0, int(float(row.get("candidate_rank") or 1)) - 1))
        row["direct_candidate_canonical_smiles"] = canonical
        row["direct_candidate_strict_fraction"] = str(row.get("unified_property_success_fraction", "0") or "0")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"normalized_denovo_candidates={len(rows)} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
