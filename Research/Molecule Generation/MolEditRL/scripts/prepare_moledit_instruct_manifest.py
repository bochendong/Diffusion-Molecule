#!/usr/bin/env python3
"""Convert MolEdit-Instruct txt rows into CSV/JSONL benchmark manifests."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import csv
import json
from pathlib import Path
from typing import Iterable


FIELDNAMES = ["example_id", "instruction", "source_smiles", "target_smiles"]


def iter_rows(path: Path) -> Iterable[dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 4:
                raise ValueError(
                    f"{path}:{line_number}: expected 4 tab-separated fields, got {len(parts)}"
                )
            example_id, instruction, source_smiles, target_smiles = parts
            yield {
                "example_id": example_id,
                "instruction": instruction,
                "source_smiles": source_smiles,
                "target_smiles": target_smiles,
            }


def convert(
    input_path: Path,
    output_csv: Path | None,
    output_jsonl: Path | None,
    limit: int | None,
) -> int:
    count = 0
    with ExitStack() as stack:
        csv_writer = None
        jsonl_handle = None
        if output_csv:
            output_csv.parent.mkdir(parents=True, exist_ok=True)
            csv_handle = stack.enter_context(
                output_csv.open("w", encoding="utf-8", newline="")
            )
            csv_writer = csv.DictWriter(csv_handle, fieldnames=FIELDNAMES)
            csv_writer.writeheader()
        if output_jsonl:
            output_jsonl.parent.mkdir(parents=True, exist_ok=True)
            jsonl_handle = stack.enter_context(output_jsonl.open("w", encoding="utf-8"))

        for row in iter_rows(input_path):
            if limit is not None and count >= limit:
                break
            if csv_writer:
                csv_writer.writerow(row)
            if jsonl_handle:
                jsonl_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--output-jsonl", type=Path)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be positive")
    if not args.output_csv and not args.output_jsonl:
        raise SystemExit("provide --output-csv and/or --output-jsonl")

    count = convert(args.input, args.output_csv, args.output_jsonl, args.limit)
    if args.output_csv:
        print(f"wrote {count} rows to {args.output_csv}")
    if args.output_jsonl:
        print(f"wrote {count} rows to {args.output_jsonl}")


if __name__ == "__main__":
    main()
