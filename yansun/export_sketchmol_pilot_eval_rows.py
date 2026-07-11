#!/usr/bin/env python3
"""Export Yansun SketchMol target CSVs into eval rows for sampling."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


SKETCHMOL_PROPS = ("MW", "LogP", "QED", "TPSA", "HBD", "HBA", "RB")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--single-csv", type=Path, required=True)
    parser.add_argument("--multi-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--single-task-id", default="101")
    parser.add_argument("--multi-task-id", default="201")
    return parser.parse_args()


def format_value(prop: str, value: str) -> str:
    text = str(value).strip()
    if prop in {"HBD", "HBA", "RB"}:
        if "." in text:
            return str(int(round(float(text))))
        return text
    if prop == "QED":
        return f"{float(text):.4f}".rstrip("0").rstrip(".")
    if prop == "TPSA":
        return f"{float(text):.2f}".rstrip("0").rstrip(".")
    return f"{float(text):.4f}".rstrip("0").rstrip(".")


def render_preset(pairs: list[tuple[str, str]]) -> str:
    return ",".join(f"{prop}:{format_value(prop, value)}" for prop, value in pairs)


def load_rows(path: Path, task_id: str) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle) if row.get("task_id") == task_id]


def export_single_rows(rows: list[dict[str, str]], *, task_id: str) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for idx, row in enumerate(rows):
        prop = str(row.get("property") or "").strip()
        target = str(row.get("target_value") or "").strip()
        if not prop or not target:
            raise ValueError(f"single row missing property/target_value: task={task_id} idx={idx}")
        condition_id = f"yansun_s{task_id}_{idx:05d}"
        out.append(
            {
                **row,
                "source_file": "single",
                "pilot_task_id": task_id,
                "row_index": str(idx),
                "condition_id": condition_id,
                "property_count": "1",
                "sketchmol_preset_str": render_preset([(prop, target)]),
            }
        )
    return out


def export_multi_rows(rows: list[dict[str, str]], *, task_id: str) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for idx, row in enumerate(rows):
        pairs: list[tuple[str, str]] = []
        for prop in SKETCHMOL_PROPS:
            value = str(row.get(f"{prop}_target") or "").strip()
            if value:
                pairs.append((prop, value))
        if len(pairs) != 2:
            raise ValueError(f"multi row expected 2 targets, got {len(pairs)}: task={task_id} idx={idx}")
        condition_id = f"yansun_m{task_id}_{idx:05d}"
        out.append(
            {
                **row,
                "source_file": "multi",
                "pilot_task_id": task_id,
                "row_index": str(idx),
                "condition_id": condition_id,
                "property_count": "2",
                "sketchmol_preset_str": render_preset(pairs),
            }
        )
    return out


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    single_rows = load_rows(args.single_csv, args.single_task_id)
    multi_rows = load_rows(args.multi_csv, args.multi_task_id)
    if not single_rows:
        raise SystemExit(f"No rows for single task_id={args.single_task_id}")
    if not multi_rows:
        raise SystemExit(f"No rows for multi task_id={args.multi_task_id}")

    rows = export_single_rows(single_rows, task_id=args.single_task_id)
    rows.extend(export_multi_rows(multi_rows, task_id=args.multi_task_id))
    write_rows(args.output_csv, rows)

    print(f"Wrote {len(rows)} pilot eval rows to {args.output_csv}")
    print(f"  single task {args.single_task_id}: {len(single_rows)}")
    print(f"  multi task {args.multi_task_id}: {len(multi_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
