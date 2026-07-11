#!/usr/bin/env python3
"""Collect all Unified Joint v2 benchmark summaries into one comparison CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Mapping, Sequence


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", required=True, type=Path)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--output-csv", required=True, type=Path)
    return parser.parse_args(argv)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def metadata(path: Path, stage_root: Path) -> dict[str, str]:
    parts = path.relative_to(stage_root).parts
    task = parts[0] if parts else ""
    budget = parts[1].removeprefix("at") if len(parts) > 1 else ""
    selection = parts[2] if len(parts) > 2 else ""
    return {"task": task, "budget": budget, "selection": selection, "source_summary": str(path)}


def write_rows(path: Path, rows: Sequence[Mapping[str, str]]) -> None:
    fields = []
    seen = set()
    for preferred in ("stage", "task", "budget", "selection", "source_summary"):
        fields.append(preferred)
        seen.add(preferred)
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    stage_root = args.eval_root / str(args.stage)
    summaries = sorted(stage_root.glob("**/benchmark_summary.csv")) + sorted(
        stage_root.glob("**/moledit_table_summary.csv")
    )
    rows = []
    for path in summaries:
        meta = metadata(path, stage_root)
        for row in read_rows(path):
            rows.append({"stage": str(args.stage), **meta, **row})
    if not rows:
        raise SystemExit(f"No benchmark summaries found under {stage_root}")
    write_rows(args.output_csv, rows)
    summary = {
        "stage": str(args.stage),
        "eval_root": str(args.eval_root),
        "summary_files": len(summaries),
        "rows": len(rows),
        "output_csv": str(args.output_csv),
    }
    args.output_csv.with_suffix(".json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
