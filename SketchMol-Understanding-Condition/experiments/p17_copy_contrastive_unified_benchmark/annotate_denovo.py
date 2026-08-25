#!/usr/bin/env python3
"""Post-generation property audit for raw P17 de-novo candidates."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
UNIFIED_DIR = SCRIPT_DIR.parent / "unified_smiles_generator"
sys.path.insert(0, str(UNIFIED_DIR))
import unified_smiles_generator as unified  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--raw-candidates", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    with args.reference.open(newline="", encoding="utf-8") as handle:
        refs = {str(row.get("condition_id") or row.get("sample_id")): dict(row) for row in csv.DictReader(handle)}
    with args.raw_candidates.open(newline="", encoding="utf-8") as handle:
        candidates = list(csv.DictReader(handle))
    output = []
    for candidate in candidates:
        key = str(candidate.get("condition_id") or candidate.get("sample_id"))
        ref = refs[key]
        smiles = str(candidate.get("direct_candidate_canonical_smiles", ""))
        item = dict(ref)
        item.update(candidate)
        item.update(unified.candidate_metrics(ref, smiles, source_similarity_threshold=0.65))
        item["direct_candidate_strict_fraction"] = item["unified_property_success_fraction"]
        output.append(item)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]), extrasaction="ignore")
        writer.writeheader(); writer.writerows(output)
    print(f"annotated {len(output)} candidates without changing generation order")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
