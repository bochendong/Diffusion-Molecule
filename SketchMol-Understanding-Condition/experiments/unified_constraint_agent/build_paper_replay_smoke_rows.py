#!/usr/bin/env python3
"""Deterministically sample small 2p-7p and Table1 paper-replay subsets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--denovo-eval-csv", required=True, type=Path)
    parser.add_argument("--denovo-candidate-csv", required=True, type=Path)
    parser.add_argument("--table1-input-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--denovo-per-bucket", type=int, default=10)
    parser.add_argument("--table1-per-task", type=int, default=10)
    parser.add_argument("--candidate-budget", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1713)
    return parser.parse_args(argv)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def stable_key(seed: int, value: str) -> bytes:
    return hashlib.sha256(f"{seed}:{value}".encode()).digest()


def row_id(row: Mapping[str, object], index: int) -> str:
    for key in ("condition_id", "sample_id", "example_id", "pair_hash"):
        value = str(row.get(key, "") or "").strip()
        if value:
            return value
    return f"row_{index:08d}"


def property_count(row: Mapping[str, object]) -> int:
    value = str(row.get("property_count", "") or "").strip()
    if value:
        return int(float(value))
    return len([item for item in str(row.get("condition_properties", "") or "").split(",") if item.strip()])


def table1_task_key(row: Mapping[str, object]) -> str:
    raw = str(row.get("instruction_tasks", "") or "").strip()
    try:
        tasks = json.loads(raw) if raw else []
    except json.JSONDecodeError:
        tasks = []
    pairs = []
    for item in tasks if isinstance(tasks, list) else []:
        if isinstance(item, Mapping):
            pairs.append((str(item.get("property", "")), str(item.get("direction", ""))))
    return "+".join(f"{prop}:{direction}" for prop, direction in sorted(pairs))


def sample_by_group(
    rows: Sequence[Mapping[str, str]],
    *,
    group_key,
    per_group: int,
    seed: int,
) -> list[dict[str, str]]:
    grouped: dict[object, list[tuple[str, dict[str, str]]]] = defaultdict(list)
    for index, raw in enumerate(rows):
        row = dict(raw)
        identity = row_id(row, index)
        grouped[group_key(row)].append((identity, row))
    output = []
    for group, items in sorted(grouped.items(), key=lambda item: str(item[0])):
        ranked = sorted(items, key=lambda item: stable_key(seed, item[0]))
        if len(ranked) < int(per_group):
            raise ValueError(f"Group {group!r} has only {len(ranked)} rows")
        output.extend(row for _identity, row in ranked[: int(per_group)])
    return output


def candidate_index(row: Mapping[str, object], fallback: int) -> int:
    for key in ("direct_candidate_index", "candidate_index", "candidate_rank"):
        value = str(row.get(key, "") or "").strip()
        if value:
            return int(float(value))
    return fallback


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if int(args.candidate_budget) != 20:
        raise ValueError("Paper replay smoke fixes exact n=20")
    all_denovo_rows = [
        row for row in read_csv(args.denovo_eval_csv) if 2 <= property_count(row) <= 7
    ]
    denovo_rows = sample_by_group(
        all_denovo_rows,
        group_key=property_count,
        per_group=int(args.denovo_per_bucket),
        seed=int(args.seed),
    )
    denovo_ids = {row_id(row, index) for index, row in enumerate(denovo_rows)}
    candidate_groups: dict[str, list[tuple[int, dict[str, str]]]] = defaultdict(list)
    for index, row in enumerate(read_csv(args.denovo_candidate_csv)):
        identity = row_id(row, index)
        if identity in denovo_ids:
            candidate_groups[identity].append((candidate_index(row, index), row))
    denovo_candidates = []
    counts: dict[str, int] = {}
    for identity in sorted(denovo_ids):
        frozen = [
            row
            for _rank, row in sorted(candidate_groups.get(identity, []), key=lambda item: item[0])[
                : int(args.candidate_budget)
            ]
        ]
        counts[identity] = len(frozen)
        denovo_candidates.extend(frozen)
    bad_counts = {key: counts.get(key, 0) for key in denovo_ids if counts.get(key, 0) != 20}
    if bad_counts:
        raise ValueError(f"De novo exact n=20 contract failed: {bad_counts}")
    all_table1_rows = [row for row in read_csv(args.table1_input_csv) if table1_task_key(row)]
    table1_rows = sample_by_group(
        all_table1_rows,
        group_key=table1_task_key,
        per_group=int(args.table1_per_task),
        seed=int(args.seed) + 1,
    )
    if len({table1_task_key(row) for row in table1_rows}) != 10:
        raise ValueError("Table1 smoke does not cover all ten tasks")
    write_csv(args.output_dir / "denovo_eval.csv", denovo_rows)
    write_csv(args.output_dir / "denovo_candidates_n20.csv", denovo_candidates)
    write_csv(args.output_dir / "table1_rows.csv", table1_rows)
    manifest = {
        "protocol": "paper_replay_smoke_v1",
        "seed": int(args.seed),
        "candidate_budget": 20,
        "evaluation_target_access": False,
        "common_llm_adapter_in_denovo_execution_graph": False,
        "denovo_candidate_prefix": "first_20_by_original_candidate_index",
        "denovo_conditions": len(denovo_rows),
        "denovo_candidate_rows": len(denovo_candidates),
        "denovo_bucket_counts": {
            str(count): sum(property_count(row) == count for row in denovo_rows)
            for count in range(2, 8)
        },
        "table1_conditions": len(table1_rows),
        "table1_task_counts": {
            key: sum(table1_task_key(row) == key for row in table1_rows)
            for key in sorted({table1_task_key(row) for row in table1_rows})
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
