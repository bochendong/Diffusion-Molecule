#!/usr/bin/env python3
"""Build a GSK3B-only Table1 pack with instruction_tasks attached for RL."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


GSK3B_TASK_KEY = "GSK3B:increase"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-condition-csv", required=True, type=Path)
    parser.add_argument("--train-moledit-csv", required=True, type=Path)
    parser.add_argument("--eval-reference-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--eval-limit", type=int, default=40)
    parser.add_argument("--task-key", default=GSK3B_TASK_KEY)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_moledit = load_by_id(args.train_moledit_csv, ("example_id", "condition_id", "pair_hash"))
    train_rows = []
    for row in read_rows(args.train_condition_csv):
        matched = match_row(row, train_moledit)
        if matched is None:
            continue
        merged = dict(row)
        merged["instruction_tasks"] = matched.get("instruction_tasks", "")
        merged["instruction_task_properties"] = matched.get("instruction_task_properties", "")
        merged["instruction_task_directions"] = matched.get("instruction_task_directions", "")
        if task_key(merged) != args.task_key:
            continue
        if not str(merged.get("condition_id") or "").strip():
            merged["condition_id"] = str(matched.get("example_id") or "").strip()
        train_rows.append(merged)

    eval_rows = []
    eval_reference = []
    for row in read_rows(args.eval_reference_csv):
        if task_key(row) != args.task_key:
            continue
        eval_reference.append(row)
        if args.eval_limit > 0 and len(eval_rows) >= int(args.eval_limit):
            continue
        eval_rows.append(condition_row_from_reference(row))

    if not train_rows:
        raise SystemExit(f"No train rows for task {args.task_key}")
    if not eval_rows:
        raise SystemExit(f"No eval rows for task {args.task_key}")

    train_csv = args.output_dir / "table1_train_gsk3b_condition_rows.csv"
    eval_csv = args.output_dir / "table1_eval_gsk3b_condition_rows.csv"
    eval_ref_csv = args.output_dir / "table1_eval_gsk3b_moledit_rows.csv"
    write_csv(train_csv, train_rows)
    write_csv(eval_csv, eval_rows)
    write_csv(eval_ref_csv, eval_reference[: len(eval_rows)])
    summary = {
        "task_key": args.task_key,
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "eval_reference_available": len(eval_reference),
        "train_csv": str(train_csv),
        "eval_csv": str(eval_csv),
        "eval_reference_csv": str(eval_ref_csv),
    }
    (args.output_dir / "gsk3b_pilot_pack.summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def condition_row_from_reference(row: dict[str, str]) -> dict[str, str]:
    example_id = str(row.get("example_id") or row.get("condition_id") or "").strip()
    out = dict(row)
    out["condition_id"] = example_id
    out["split"] = "eval"
    if not out.get("condition_properties"):
        out["condition_properties"] = str(row.get("computed_active_properties") or "").replace("|", ",")
    return out


def task_key(row: MappingLike) -> str:
    raw = str(row.get("instruction_tasks") or "").strip()
    if raw:
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = []
        pairs = []
        if isinstance(parsed, list):
            for item in parsed:
                if not isinstance(item, dict):
                    continue
                prop = str(item.get("property") or "").strip()
                direction = str(item.get("direction") or "").strip()
                if prop and direction:
                    pairs.append(f"{prop}:{direction}")
        if pairs:
            return "+".join(sorted(pairs))
    props = str(row.get("instruction_task_properties") or "").strip()
    directions_raw = str(row.get("instruction_task_directions") or "").strip()
    if props == "GSK3B" or "GSK3" in str(row.get("instruction") or "").upper():
        if "increase" in directions_raw.lower() or "improve" in str(row.get("instruction") or "").lower():
            return GSK3B_TASK_KEY
    return props


def match_row(row: dict[str, str], lookup: dict[str, dict[str, str]]) -> dict[str, str] | None:
    for key in ("condition_id", "example_id", "pair_id", "pair_hash"):
        value = str(row.get(key) or "").strip()
        if value and value in lookup:
            return lookup[value]
    return None


def load_by_id(path: Path, keys: tuple[str, ...]) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for row in read_rows(path):
        for key in keys:
            value = str(row.get(key) or "").strip()
            if value and value not in out:
                out[value] = row
    return out


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


MappingLike = dict[str, str]


if __name__ == "__main__":
    raise SystemExit(main())
