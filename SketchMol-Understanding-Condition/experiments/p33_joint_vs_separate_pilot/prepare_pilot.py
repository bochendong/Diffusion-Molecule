#!/usr/bin/env python3
"""Freeze the matched P33 train subsets and small de-novo gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence


EDIT_TASKS = (
    "DRD2:decrease+MW:decrease+SA:decrease",
    "GSK3B:increase",
    "HBA:decrease+LogP:increase",
    "HBA:decrease+MW:decrease",
    "HBA:decrease+SA:decrease",
    "HBA:increase+MW:increase+QED:decrease",
    "MW:increase",
    "QED:increase+SA:decrease",
    "RB:decrease",
    "SA:decrease",
)


def read_jsonl(paths: Sequence[Path]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in paths:
        rows.extend(
            json.loads(line) for line in path.read_text().splitlines() if line.strip()
        )
    return rows


def stable_key(row: Mapping[str, object], seed: int) -> str:
    identity = row.get("example_id", row.get("condition_id", row.get("sample_id", "")))
    return hashlib.sha256(f"{seed}:{identity}".encode()).hexdigest()


def property_count(row: Mapping[str, object]) -> int:
    program = row.get("condition_program")
    if isinstance(program, list):
        return len(program)
    for message in list(row.get("messages", [])):
        if str(message.get("role")) == "user":
            payload = json.loads(str(message.get("content", "{}")))
            conditions = payload.get("conditions", []) if isinstance(payload, dict) else []
            return len(conditions) if isinstance(conditions, list) else 0
    return 0


def select_train(rows: Sequence[dict[str, object]], seed: int):
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        mode = str(row.get("task_mode", ""))
        if mode == "de_novo" and 2 <= property_count(row) <= 7:
            grouped[f"de_novo:{property_count(row)}p"].append(row)
        elif mode == "edit" and str(row.get("task_key", "")) in EDIT_TASKS:
            grouped[f"edit:{row['task_key']}"].append(row)

    de_keys = tuple(f"de_novo:{count}p" for count in range(2, 8))
    edit_keys = tuple(f"edit:{task}" for task in EDIT_TASKS)
    quotas = {**{key: 500 for key in de_keys}, **{key: 300 for key in edit_keys}}
    selected: dict[str, list[dict[str, object]]] = {}
    for key, quota in quotas.items():
        values = sorted(grouped[key], key=lambda row: stable_key(row, seed))
        if len(values) < quota:
            raise ValueError(f"P33 bucket {key} has {len(values)} rows; needs {quota}")
        selected[key] = values[:quota]

    de_novo = [selected[key][index] for index in range(500) for key in de_keys]
    editing = [selected[key][index] for index in range(300) for key in edit_keys]
    joint: list[dict[str, object]] = []
    for index in range(max(len(de_novo), len(editing))):
        if index < len(de_novo):
            joint.append(de_novo[index])
        if index < len(editing):
            joint.append(editing[index])
    return de_novo, editing, joint, quotas


def select_denovo_gate(rows: Sequence[dict[str, object]], seed: int):
    grouped: dict[int, list[dict[str, object]]] = defaultdict(list)
    seen: set[str] = set()
    for row in rows:
        identity = str(row.get("condition_id", row.get("sample_id", "")))
        count = property_count(row)
        if identity and identity not in seen and 2 <= count <= 7:
            grouped[count].append(row)
            seen.add(identity)
    selected: list[dict[str, object]] = []
    for count in range(2, 8):
        values = sorted(grouped[count], key=lambda row: stable_key(row, seed))
        if len(values) < 20:
            raise ValueError(f"P33 gate has only {len(values)} rows for {count}p")
        selected.extend(values[:20])
    return selected


def write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-source", required=True, type=Path)
    parser.add_argument("--denovo-prompts", action="append", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=33001)
    args = parser.parse_args(argv)

    train_rows = read_jsonl([args.train_source])
    de_novo, editing, joint, quotas = select_train(train_rows, args.seed)
    gate = select_denovo_gate(read_jsonl(args.denovo_prompts), args.seed + 1)
    write_jsonl(args.output_dir / "train.denovo.jsonl", de_novo)
    write_jsonl(args.output_dir / "train.edit.jsonl", editing)
    write_jsonl(args.output_dir / "train.joint.jsonl", joint)
    write_jsonl(args.output_dir / "gate.denovo.jsonl", gate)
    summary = {
        "protocol": "p33_clean_joint_vs_separate_pilot_v1",
        "source": str(args.train_source),
        "seed": args.seed,
        "rows": {"joint": len(joint), "denovo": len(de_novo), "edit": len(editing)},
        "bucket_quotas": quotas,
        "denovo_gate_rows": len(gate),
        "matched_examples": True,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    (args.output_dir / "DATA_COMPLETE").touch()
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
