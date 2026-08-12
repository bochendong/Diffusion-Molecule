#!/usr/bin/env python3
"""Materialize a source-only MuMO dev view for target-blind generation."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
PROJECT_SCRIPTS = PROJECT_DIR / "scripts"
for path in (SCRIPT_DIR, PROJECT_DIR, PROJECT_SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import export_external_multiproperty_benchmark_rows as export  # noqa: E402
import mumo_parallel_protocol as protocol  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--manifest-json", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    specs = {spec.task_id: spec for spec in export.TASK_SPECS if spec.suite == "mumo"}
    output = []
    task_counts: Counter[str] = Counter()
    for path in sorted(args.data_dir.glob("train_shard_*.jsonl")):
        for raw in protocol.read_jsonl(path):
            if raw.get("_uca_partition") != "dev":
                continue
            task_id = str(raw["_uca_task_id"])
            spec = specs[task_id]
            row: dict[str, object] = {
                "_uca_task_id": task_id,
                "_uca_task_key": str(raw["_uca_task_key"]),
                "_uca_source_group": str(raw["_uca_source_group"]),
                "_uca_pair_digest": str(raw["_uca_pair_digest"]),
                "source_smiles": str(raw["source_smiles"]),
            }
            for prop in spec.properties:
                value = export.read_property_value(raw, prop, prefix="source")
                if value is not None:
                    row[f"external_source_{prop}"] = float(value)
            forbidden = [key for key in row if "target" in key.lower()]
            if forbidden:
                raise AssertionError(f"Target fields leaked into source-only row: {forbidden}")
            output.append(row)
            task_counts[task_id] += 1
    output.sort(key=lambda row: (str(row["_uca_task_id"]), str(row["_uca_source_group"])))
    protocol.write_jsonl(args.output_jsonl, output)
    manifest = {
        "protocol": "mumo_source_only_dev_view_v1",
        "data_role": "source_only_disjoint_train_dev",
        "generation_target_access": False,
        "target_fields_written": 0,
        "rows": len(output),
        "task_rows": dict(sorted(task_counts.items())),
        "source_label_role": "post_freeze_evaluator_baseline_only",
    }
    protocol.write_json(args.manifest_json, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
