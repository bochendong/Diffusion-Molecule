#!/usr/bin/env python3
"""Freeze a small P24 Raw@1 gate and reuse its completed baseline scores."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def property_count(row: Mapping[str, object]) -> int:
    for message in list(row.get("messages", [])):
        if str(message.get("role")) != "user":
            continue
        payload = json.loads(str(message.get("content", "{}")))
        conditions = payload.get("conditions", []) if isinstance(payload, dict) else []
        return len(conditions) if isinstance(conditions, list) else 0
    return 0


def stable_key(row: Mapping[str, object], seed: int) -> str:
    condition_id = str(row.get("condition_id", ""))
    return hashlib.sha256(f"{seed}:{condition_id}".encode()).hexdigest()


def load_baseline(paths: Sequence[Path]) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row["setting"] != "raw_at_1":
                    continue
                result[row["condition_id"]] = {
                    "strict": float(row["strict_value"]),
                    "validity": float(row["validity_value"]),
                }
    return result


def mean(values: Sequence[float]) -> float:
    return sum(values) / max(len(values), 1)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", action="append", required=True, type=Path)
    parser.add_argument("--baseline-detail", action="append", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--per-arity", type=int, default=20)
    parser.add_argument("--seed", type=int, default=30131)
    args = parser.parse_args(argv)

    grouped: dict[int, list[dict[str, object]]] = defaultdict(list)
    seen: set[str] = set()
    for path in args.prompts:
        for row in read_jsonl(path):
            condition_id = str(row.get("condition_id", ""))
            count = property_count(row)
            if condition_id and condition_id not in seen and 2 <= count <= 7:
                grouped[count].append(row)
                seen.add(condition_id)
    baseline = load_baseline(args.baseline_detail)
    selected: list[dict[str, object]] = []
    for count in range(2, 8):
        candidates = [row for row in grouped[count] if str(row["condition_id"]) in baseline]
        candidates.sort(key=lambda row: stable_key(row, args.seed))
        if len(candidates) < args.per_arity:
            raise ValueError(f"insufficient matched {count}p prompts: {len(candidates)}")
        selected.extend(candidates[: args.per_arity])
    if len(selected) != 6 * args.per_arity:
        raise AssertionError("small Raw@1 gate has the wrong size")

    buckets: dict[str, dict[str, float]] = {}
    for count in range(2, 8):
        ids = [str(row["condition_id"]) for row in selected if property_count(row) == count]
        buckets[f"{count}p"] = {
            "conditions": len(ids),
            "strict_rate": mean([baseline[value]["strict"] for value in ids]),
            "valid_rate": mean([baseline[value]["validity"] for value in ids]),
        }
    summary = {
        "protocol": "p30_1_frozen_p24_small_raw1_gate_v1",
        "seed": args.seed,
        "per_arity": args.per_arity,
        "conditions": len(selected),
        "decoding": "greedy",
        "property_reranking": False,
        "baseline_reused": True,
        "aggregate": {
            "strict_macro": mean([buckets[f"{count}p"]["strict_rate"] for count in range(2, 8)]),
            "valid_macro": mean([buckets[f"{count}p"]["valid_rate"] for count in range(2, 8)]),
        },
        "buckets": buckets,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "prompts.jsonl").open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    (args.output_dir / "baseline_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    (args.output_dir / "condition_ids.txt").write_text(
        "".join(str(row["condition_id"]) + "\n" for row in selected)
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

