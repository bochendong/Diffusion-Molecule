#!/usr/bin/env python3
"""Prepare leakage-audited, task-capped rows for one joint Unified checkpoint."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping, Sequence


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-train-csv", required=True, type=Path)
    parser.add_argument("--source-eval-csv", required=True, type=Path)
    parser.add_argument("--train-output-csv", required=True, type=Path)
    parser.add_argument("--eval-output-csv", required=True, type=Path)
    parser.add_argument("--manifest-json", required=True, type=Path)
    parser.add_argument("--denovo-train-per-count", type=int, default=2000)
    parser.add_argument("--denovo-eval-per-count", type=int, default=1000)
    parser.add_argument("--edit-train-per-task", type=int, default=500)
    parser.add_argument("--edit-eval-per-task", type=int, default=100)
    parser.add_argument("--overlap-policy", choices=("fail", "drop_train"), default="drop_train")
    parser.add_argument("--seed", type=int, default=41)
    return parser.parse_args(argv)


def read_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = [
            {str(key): "" if value is None else str(value) for key, value in row.items()}
            for row in reader
        ]
        return rows, list(reader.fieldnames or [])


def task_mode(row: Mapping[str, str]) -> str:
    raw = str(row.get("task_mode", "") or row.get("unified_task_mode", "")).strip().lower()
    normalized = raw.replace("-", "_").replace(" ", "_")
    if normalized in {"de_novo", "denovo", "generation", "generate"}:
        return "de_novo"
    if normalized in {"edit", "source_edit", "conditional_edit", "edit_generation"}:
        return "edit"
    return "edit" if str(row.get("source_smiles", "")).strip() else "de_novo"


def include_joint_row(row: Mapping[str, str]) -> bool:
    mode = task_mode(row)
    benchmark = str(row.get("benchmark_task", "") or "").strip().lower()
    sample_id = str(row.get("sample_id", "") or row.get("condition_id", "")).strip().lower()
    if mode == "de_novo":
        if "ood" in benchmark or "ood" in sample_id:
            return False
        return "2p7p" in benchmark or sample_id.startswith("denovo_")
    if benchmark and ("external" in benchmark or "ood" in benchmark):
        return False
    return not benchmark or "table1" in benchmark or "moledit" in benchmark


def parse_instruction_tasks(row: Mapping[str, str]) -> list[tuple[str, str]]:
    raw = str(row.get("instruction_tasks", "") or "").strip()
    if raw:
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = None
        if isinstance(value, list):
            specs = []
            for item in value:
                if isinstance(item, dict):
                    prop = str(item.get("property", "") or item.get("name", "") or "").strip()
                    direction = str(item.get("direction", "") or item.get("operation", "") or "").strip().lower()
                    if prop:
                        specs.append((prop, direction))
            if specs:
                return sorted(specs)
    props = [part.strip() for part in str(row.get("condition_properties", "") or "").replace(";", ",").split(",") if part.strip()]
    specs = []
    for prop in props:
        direction = str(row.get(f"{prop}_direction", "") or row.get(f"{prop.lower()}_direction", "")).strip().lower()
        specs.append((prop, direction))
    return sorted(specs)


def group_key(row: Mapping[str, str]) -> str:
    mode = task_mode(row)
    if mode == "de_novo":
        raw = str(row.get("property_count", "") or "0").strip()
        try:
            count = int(float(raw))
        except ValueError:
            count = len([part for part in str(row.get("condition_properties", "")).split(",") if part.strip()])
        return f"de_novo:{count}p"
    specs = parse_instruction_tasks(row)
    rendered = "+".join(f"{prop}:{direction or 'none'}" for prop, direction in specs)
    return f"edit:{rendered or 'unknown'}"


def cap_groups(rows: Sequence[dict[str, str]], *, per_group: int, seed: int) -> list[dict[str, str]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[group_key(row)].append(row)
    selected = []
    for offset, key in enumerate(sorted(grouped)):
        values = list(grouped[key])
        random.Random(seed + offset).shuffle(values)
        selected.extend(values[:per_group] if per_group > 0 else values)
    return selected


def overlap_keys(row: Mapping[str, str]) -> set[str]:
    mode = task_mode(row)
    target = str(row.get("target_smiles", "") or "").strip()
    if mode == "de_novo":
        molecule_id = str(row.get("molecule_id", "") or row.get("mol_id", "")).strip()
        keys = {f"denovo_target:{target}"} if target else set()
        if molecule_id:
            keys.add(f"denovo_molecule:{molecule_id}")
        return keys
    source = str(row.get("source_smiles", "") or "").strip()
    example_id = str(row.get("example_id", "") or "").strip()
    keys = {f"edit_pair:{source}|{target}|{group_key(row)}"} if source or target else set()
    if example_id:
        keys.add(f"edit_example:{example_id}")
    return keys


def remove_train_eval_overlap(
    train_rows: Sequence[dict[str, str]],
    eval_rows: Sequence[dict[str, str]],
    *,
    policy: str,
) -> tuple[list[dict[str, str]], list[dict[str, object]]]:
    eval_keys = set().union(*(overlap_keys(row) for row in eval_rows)) if eval_rows else set()
    kept = []
    dropped = []
    for row in train_rows:
        collisions = sorted(overlap_keys(row) & eval_keys)
        if not collisions:
            kept.append(row)
            continue
        dropped.append({"sample_id": row.get("sample_id", ""), "group": group_key(row), "keys": collisions})
    if dropped and policy == "fail":
        raise ValueError(f"Detected {len(dropped)} train rows overlapping eval rows; see overlap audit.")
    return kept, dropped


def write_rows(path: Path, rows: Sequence[Mapping[str, str]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def counts(rows: Sequence[Mapping[str, str]]) -> dict[str, object]:
    return {
        "rows": len(rows),
        "task_modes": dict(sorted(Counter(task_mode(row) for row in rows).items())),
        "groups": dict(sorted(Counter(group_key(row) for row in rows).items())),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    source_train, train_fields = read_rows(args.source_train_csv)
    source_eval, eval_fields = read_rows(args.source_eval_csv)
    train_rows = [row for row in source_train if include_joint_row(row)]
    eval_rows = [row for row in source_eval if include_joint_row(row)]

    train_denovo = cap_groups(
        [row for row in train_rows if task_mode(row) == "de_novo"],
        per_group=int(args.denovo_train_per_count),
        seed=int(args.seed),
    )
    train_edit = cap_groups(
        [row for row in train_rows if task_mode(row) == "edit"],
        per_group=int(args.edit_train_per_task),
        seed=int(args.seed) + 100,
    )
    eval_denovo = cap_groups(
        [row for row in eval_rows if task_mode(row) == "de_novo"],
        per_group=int(args.denovo_eval_per_count),
        seed=int(args.seed) + 200,
    )
    eval_edit = cap_groups(
        [row for row in eval_rows if task_mode(row) == "edit"],
        per_group=int(args.edit_eval_per_task),
        seed=int(args.seed) + 300,
    )
    joint_eval = eval_denovo + eval_edit
    joint_train, dropped = remove_train_eval_overlap(
        train_denovo + train_edit,
        joint_eval,
        policy=str(args.overlap_policy),
    )

    fields = list(dict.fromkeys(train_fields + eval_fields))
    write_rows(args.train_output_csv, joint_train, fields)
    write_rows(args.eval_output_csv, joint_eval, fields)
    manifest = {
        "protocol": "unified_joint_v2",
        "seed": int(args.seed),
        "overlap_policy": str(args.overlap_policy),
        "source_train_csv": str(args.source_train_csv),
        "source_train_sha256": sha256(args.source_train_csv),
        "source_eval_csv": str(args.source_eval_csv),
        "source_eval_sha256": sha256(args.source_eval_csv),
        "train_output_csv": str(args.train_output_csv),
        "train_output_sha256": sha256(args.train_output_csv),
        "eval_output_csv": str(args.eval_output_csv),
        "eval_output_sha256": sha256(args.eval_output_csv),
        "train": counts(joint_train),
        "eval": counts(joint_eval),
        "dropped_train_eval_overlap_rows": len(dropped),
        "overlap_examples": dropped[:50],
    }
    args.manifest_json.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_json.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
