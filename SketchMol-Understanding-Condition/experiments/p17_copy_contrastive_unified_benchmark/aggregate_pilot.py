#!/usr/bin/env python3
"""Collect frozen P17 pilot metrics without altering evaluator output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path):
    return json.loads(path.read_text())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", required=True, type=Path)
    parser.add_argument("--pilot-manifest", required=True, type=Path)
    parser.add_argument("--table1-root", required=True, type=Path)
    parser.add_argument("--denovo", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    table1 = {}
    for budget in (1, 4, 8):
        table1[str(budget)] = load(args.table1_root / f"any{budget}" / "moledit_table_summary.json")
    payload = {
        "protocol": "p17_frozen_benchmark_pilot_estimate_v1",
        "status_label": "pilot estimate; not full Table1 or full de-novo benchmark",
        "validation_gate": load(args.gate),
        "pilot_manifest": load(args.pilot_manifest),
        "table1_anyk": table1,
        "denovo_raw_gate": load(args.denovo),
        "tuned_on_benchmark": False,
        "static_candidate_pool": False,
        "property_reranking": False,
        "candidate_order": "greedy first then seven raw samples",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "protocol": payload["protocol"], "status_label": payload["status_label"],
        "gate_passed": payload["validation_gate"]["gate_passed"],
        "table1_budgets": [1, 4, 8], "denovo_records": len(payload["denovo_raw_gate"]["records"]),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
