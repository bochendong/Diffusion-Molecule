#!/usr/bin/env python3
"""Build the C7 GRPO train pack: real MolEdit Table1 rows with existing features.

Keeps GSK3B / MW / SA / RB / DRD2 only. Drops synthetic-table1 HBA rows.
RB and DRD2 are duplicated once so the weak real tasks get more updates.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


REAL_TASKS = {
    "DRD2:decrease+MW:decrease+SA:decrease",
    "GSK3B:increase",
    "MW:increase",
    "RB:decrease",
    "SA:decrease",
}
DEFAULT_DUPLICATE_TASKS = (
    "DRD2:decrease+MW:decrease+SA:decrease",
    "RB:decrease",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-moledit-csv", required=True, type=Path)
    parser.add_argument("--train-condition-csv", required=True, type=Path)
    parser.add_argument("--feature-index-csv", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument(
        "--duplicate-tasks",
        default=",".join(DEFAULT_DUPLICATE_TASKS),
        help="Comma-separated task keys to duplicate once.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    feature_ids = load_feature_ids(args.feature_index_csv)
    conditions = {str(row.get("condition_id") or "").strip(): row for row in read_rows(args.train_condition_csv)}
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    skipped = {"synthetic": 0, "no_feature": 0, "other_task": 0, "no_condition": 0}
    for moledit in read_rows(args.train_moledit_csv):
        example_id = str(moledit.get("example_id") or moledit.get("condition_id") or "").strip()
        if example_id.startswith("synthetic-table1"):
            skipped["synthetic"] += 1
            continue
        task = task_key(moledit)
        if task not in REAL_TASKS:
            skipped["other_task"] += 1
            continue
        if example_id not in feature_ids:
            skipped["no_feature"] += 1
            continue
        condition = conditions.get(example_id)
        if condition is None:
            skipped["no_condition"] += 1
            continue
        grouped[task].append(merge_row(condition, moledit, example_id, task))

    duplicate_tasks = {
        item.strip() for item in str(args.duplicate_tasks).split(",") if item.strip()
    }
    rows: list[dict[str, str]] = []
    counts: dict[str, dict[str, int]] = {}
    for task, items in sorted(grouped.items()):
        copies = 2 if task in duplicate_tasks else 1
        selected = items
        rows.extend(selected * copies)
        counts[task] = {"unique": len(selected), "written": len(selected) * copies}

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    write_rows(args.output_csv, rows)
    summary = {
        "train_rows": len(rows),
        "unique_conditions": sum(item["unique"] for item in counts.values()),
        "by_task": counts,
        "skipped": skipped,
        "real_tasks": sorted(REAL_TASKS),
        "weak_tasks_duplicated": sorted(duplicate_tasks),
        "output_csv": str(args.output_csv),
    }
    args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def merge_row(
    condition: dict[str, str],
    moledit: dict[str, str],
    example_id: str,
    task: str,
) -> dict[str, str]:
    row = dict(condition)
    row.update(
        {
            "condition_id": example_id,
            "example_id": example_id,
            "instruction": moledit.get("instruction", ""),
            "instruction_tasks": moledit.get("instruction_tasks", ""),
            "instruction_task_properties": moledit.get("instruction_task_properties", ""),
            "instruction_task_directions": moledit.get("instruction_task_directions", ""),
            "source_smiles": condition.get("source_smiles") or moledit.get("source_smiles", ""),
            "target_smiles": condition.get("target_smiles") or moledit.get("target_smiles", ""),
            "moledit_task_key": task,
            "task_mode": "edit",
        }
    )
    return row


def task_key(row: dict[str, str]) -> str:
    raw = str(row.get("instruction_tasks") or "").strip()
    if not raw.startswith("["):
        return ""
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return ""
    pairs = []
    if isinstance(parsed, list):
        for item in parsed:
            if not isinstance(item, dict):
                continue
            prop = str(item.get("property") or "").strip()
            direction = str(item.get("direction") or "").strip()
            if prop and direction:
                pairs.append(f"{prop}:{direction}")
    return "+".join(sorted(pairs))


def load_feature_ids(path: Path) -> set[str]:
    ids: set[str] = set()
    for row in read_rows(path):
        for key in ("condition_id", "sample_id", "pair_id", "variant_id"):
            value = str(row.get(key) or "").strip()
            if value:
                ids.add(value)
    return ids


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
