#!/usr/bin/env python3
"""Relabel reused P17 generator output without changing candidate content/order."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    with args.csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    before = [(row["condition_id"], row["candidate_rank"], row["direct_candidate_raw_smiles"]) for row in rows]
    for row in rows:
        row["method"] = args.label
    with args.csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    after = [(row["condition_id"], row["candidate_rank"], row["direct_candidate_raw_smiles"]) for row in rows]
    if before != after:
        raise AssertionError("candidate content/order changed during relabel")
    summary_path = args.csv.parent / f"{args.csv.stem}.summary.json"
    summary = json.loads(summary_path.read_text())
    summary["protocol"] = "p19_frozen_expanded_raw_generation_v1"
    summary["model_label"] = args.label
    summary["reused_generator"] = "P17 generate_pilot.py; labels only rewritten after generation"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
