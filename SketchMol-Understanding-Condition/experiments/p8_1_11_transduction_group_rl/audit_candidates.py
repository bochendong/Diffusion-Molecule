#!/usr/bin/env python3
"""Raw candidate audit for the P8.1.11 transduction policy."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--candidates", required=True, type=Path); parser.add_argument("--output", required=True, type=Path); args = parser.parse_args()
    with args.candidates.open(newline="", encoding="utf-8") as handle: rows = list(csv.DictReader(handle))
    valid = identity = strict_nonidentity = 0; unique = set(); sims = []
    for row in rows:
        generated = str(row.get("generated_smiles", "") or ""); source = str(row.get("source_smiles", "") or "")
        if generated:
            valid += 1; unique.add(generated); same = generated == source; identity += int(same)
            strict_nonidentity += int(str(row.get("table1_strict_success", "")).lower() == "true" and not same)
            try: sims.append(float(row.get("source_tanimoto", "")))
            except (TypeError, ValueError): pass
    payload = {
        "protocol": "p8_1_11_raw_candidate_audit_v1", "rows": len(rows), "validity": valid/max(len(rows),1),
        "identity_fraction": identity/max(len(rows),1), "identity_among_valid": identity/max(valid,1),
        "strict_nonidentity_fraction": strict_nonidentity/max(len(rows),1), "unique_fraction": len(unique)/max(len(rows),1),
        "mean_source_tanimoto": statistics.fmean(sims) if sims else None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n"); print(json.dumps(payload, indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
