#!/usr/bin/env python3
"""Freeze an equal-quota refresh set for 6 de-novo and 10 paper edit tasks."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
P23_DIR = SCRIPT_DIR.parent / "p23_explicit_task_stage1_v2"
sys.path.insert(0, str(P23_DIR))
import p23_protocol as protocol  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True, type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--summary-json", required=True, type=Path)
    parser.add_argument("--rows-per-task", type=int, default=720)
    args = parser.parse_args()

    edit_tasks = set(protocol.TABLE1_TASK_KEYS.values())
    buckets: dict[str, list[dict[str, object]]] = defaultdict(list)
    with args.input_jsonl.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            mode = str(row.get("task_mode", ""))
            if mode == "de_novo":
                count = len(row.get("condition_program", []))
                if 2 <= count <= 7:
                    buckets[f"de_novo:{count}p"].append(row)
            elif mode == "edit" and str(row.get("task_key", "")) in edit_tasks:
                buckets[f"edit:{row['task_key']}"].append(row)

    expected = {
        *{f"de_novo:{count}p" for count in range(2, 8)},
        *{f"edit:{task}" for task in edit_tasks},
    }
    missing = expected.difference(buckets)
    if missing:
        raise ValueError(f"missing refresh tasks: {sorted(missing)}")
    selected: dict[str, list[dict[str, object]]] = {}
    for key in sorted(expected):
        ordered = sorted(buckets[key], key=lambda row: str(row.get("selection_rank", row["example_id"])))
        if len(ordered) < args.rows_per_task:
            raise ValueError(f"{key}: only {len(ordered)} rows, need {args.rows_per_task}")
        selected[key] = ordered[: args.rows_per_task]

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as output:
        for position in range(args.rows_per_task):
            for key in sorted(expected):
                output.write(json.dumps(selected[key][position], sort_keys=True) + "\n")
    summary = {
        "protocol": "p24_alignment_refresh_v1",
        "source": str(args.input_jsonl),
        "rows_per_task": args.rows_per_task,
        "task_count": len(expected),
        "total_rows": args.rows_per_task * len(expected),
        "tasks": {key: len(selected[key]) for key in sorted(expected)},
        "sampling": "seeded-rank selection followed by exact round-robin interleave",
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
