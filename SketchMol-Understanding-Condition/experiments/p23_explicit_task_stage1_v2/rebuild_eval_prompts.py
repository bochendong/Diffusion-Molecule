#!/usr/bin/env python3
"""Rebuild frozen-reference inference prompts with the corrected P23 contract."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Sequence

import p23_protocol as protocol


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--mode", required=True, choices=("edit", "de_novo"))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    with args.reference.open(newline="", encoding="utf-8") as handle:
        refs = list(csv.DictReader(handle))
    output = []
    tasks: Counter[str] = Counter()
    for index, row in enumerate(refs):
        messages, source, mode = protocol.build_prompt(row)
        if mode != args.mode:
            raise ValueError(f"reference mode mismatch at row {index}: {mode} != {args.mode}")
        program = protocol.condition_program(row, mode)
        key = protocol.task_key(program)
        condition_id = str(row.get("condition_id") or row.get("sample_id") or row.get("pair_id") or index)
        sample_id = str(row.get("sample_id") or row.get("condition_id") or condition_id)
        record = {
            "condition_id": condition_id, "sample_id": sample_id,
            "task_mode": mode, "source_smiles": source, "messages": messages,
            "condition_hash": protocol.condition_hash_from_program(program), "task_key": key,
        }
        user_payload = json.loads(messages[1]["content"])
        if set(user_payload) != {"conditions", "source"}:
            raise AssertionError(f"unexpected inference prompt fields at row {index}")
        output.append(record)
        tasks[key] += 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in output:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    summary = {
        "protocol": protocol.PROTOCOL, "reference": str(args.reference),
        "mode": args.mode, "rows": len(output), "task_counts": dict(sorted(tasks.items())),
        "prompt_target_fields": False,
    }
    args.output.with_suffix(args.output.suffix + ".summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
