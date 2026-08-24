#!/usr/bin/env python3
"""Create target-structure-free 2p--7p inference shards and audit coverage."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

FORBIDDEN = ("target_smiles", "target_scaffold", "target_image")


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def key(row: dict[str, str]) -> str:
    return str(row.get("condition_id") or row.get("sample_id") or "").strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--denovo-eval", required=True, type=Path)
    parser.add_argument("--table1-eval", required=True, type=Path)
    parser.add_argument("--table1-candidates", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    denovo = read(args.denovo_eval)
    table1 = read(args.table1_eval)
    table_candidates = read(args.table1_candidates)
    counts = Counter(int(float(row.get("property_count") or 0)) for row in denovo)
    table_ids = {key(row) for row in table1}
    table_candidate_counts = Counter(key(row) for row in table_candidates)
    table_tasks = Counter(str(row.get("moledit_task_key") or row.get("external_task_key") or "") for row in table1)

    if counts != Counter({value: 1000 for value in range(2, 8)}):
        raise SystemExit(f"incomplete 2p--7p support: {dict(counts)}")
    if len(denovo) != 6000 or len({key(row) for row in denovo}) != 6000:
        raise SystemExit("2p--7p rows or IDs are not complete")
    if len(table1) != 200 or len(table_ids) != 200 or set(table_candidate_counts) != table_ids:
        raise SystemExit("Table1 reference/candidate condition mismatch")
    if set(table_candidate_counts.values()) != {20} or len(table_candidates) != 4000:
        raise SystemExit("Table1 does not contain raw20 for every condition")
    if len([name for name in table_tasks if name]) != 10:
        raise SystemExit(f"expected ten Table1 tasks, found {dict(table_tasks)}")

    for property_count in range(2, 8):
        rows = []
        for source in denovo:
            if int(float(source.get("property_count") or 0)) != property_count:
                continue
            row = {name: value for name, value in source.items() if name not in FORBIDDEN}
            row["task_mode"] = "de_novo"
            rows.append(row)
        write(args.output_dir / f"pc{property_count}_inference.csv", rows)

    payload = {
        "protocol": "p8_2_matched_inference_support_v1",
        "denovo_reference_rows": len(denovo),
        "denovo_property_count_rows": {str(k): v for k, v in sorted(counts.items())},
        "denovo_unique_condition_ids": len({key(row) for row in denovo}),
        "denovo_structural_target_columns_removed": list(FORBIDDEN),
        "table1_reference_rows": len(table1),
        "table1_tasks": len([name for name in table_tasks if name]),
        "table1_raw_candidate_rows_reused": len(table_candidates),
        "table1_raw_candidates_per_condition": 20,
        "property_reranking": False,
        "eval_target_molecule_used_at_inference": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "support_audit.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
