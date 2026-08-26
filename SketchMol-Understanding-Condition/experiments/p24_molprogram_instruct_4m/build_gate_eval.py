#!/usr/bin/env python3
"""Freeze target-blind P24 gate prompts from every broad release bucket."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", required=True, type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--manifest-json", required=True, type=Path)
    parser.add_argument("--rows-per-bucket", type=int, default=10)
    args = parser.parse_args()

    expected = {
        *{f"de_novo:{count}p" for count in range(2, 8)},
        *{f"edit:{count}p" for count in range(1, 8)},
    }
    selected: dict[str, list[dict[str, object]]] = defaultdict(list)
    for mode in ("de_novo", "edit"):
        mode_expected = {key for key in expected if key.startswith(f"{mode}:")}
        for path in sorted((args.release_root / mode).glob("*.jsonl")):
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    row = json.loads(line)
                    bucket = f"{row['task_mode']}:{row['property_count']}p"
                    if len(selected[bucket]) < args.rows_per_bucket:
                        selected[bucket].append(row)
                    if all(len(selected[key]) == args.rows_per_bucket for key in mode_expected):
                        break
            if all(len(selected[key]) == args.rows_per_bucket for key in mode_expected):
                break

    counts = {key: len(selected[key]) for key in sorted(expected)}
    if any(value != args.rows_per_bucket for value in counts.values()):
        raise ValueError(f"incomplete gate buckets: {counts}")
    prompts: list[dict[str, object]] = []
    for bucket in sorted(expected):
        for index, row in enumerate(selected[bucket]):
            messages = row.get("messages")
            if not isinstance(messages, list) or len(messages) != 3:
                raise ValueError(f"invalid messages in {row.get('example_id')}")
            prompts.append({
                "condition_id": f"p24_gate:{bucket}:{index:02d}",
                "sample_id": str(row["example_id"]),
                "task_mode": str(row["task_mode"]),
                "property_count": int(row["property_count"]),
                "source_smiles": str(row["source_smiles"]),
                "messages": messages[:-1],
            })
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in prompts)
    args.output_jsonl.write_text(payload, encoding="utf-8")
    manifest = {
        "protocol": "p24_gate_validity_noncopy_v1",
        "rows_per_bucket": args.rows_per_bucket,
        "prompt_rows": len(prompts),
        "candidates_per_prompt": 8,
        "buckets": counts,
        "generation_target_access": False,
        "prompt_sha256": hashlib.sha256(payload.encode()).hexdigest(),
    }
    args.manifest_json.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_json.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
