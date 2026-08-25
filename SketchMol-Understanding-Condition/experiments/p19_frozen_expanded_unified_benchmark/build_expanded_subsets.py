#!/usr/bin/env python3
"""Freeze leak-audited expanded subsets while retaining every P17 pilot row."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import p17_protocol as protocol


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def row_id(row: Mapping[str, object]) -> str:
    return str(row.get("condition_id") or row.get("sample_id") or row.get("pair_id") or "")


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha_smiles(value: object) -> str:
    canonical = protocol.canonical_smiles(value)
    return hashlib.sha256(canonical.encode()).hexdigest() if canonical else ""


def stable_rank(row: Mapping[str, object], seed: int) -> str:
    return hashlib.sha256(f"{seed}:{row_id(row)}".encode()).hexdigest()


def prompt_record(row: Mapping[str, object]) -> dict[str, object]:
    messages, source, mode = protocol.build_prompt(row)
    record = {
        "condition_id": row_id(row),
        "sample_id": str(row.get("sample_id") or row.get("condition_id") or ""),
        "task_mode": mode,
        "source_smiles": source,
        "messages": messages,
        "condition_hash": protocol.condition_hash(row),
        "condition_family_hash": protocol.condition_family_hash(row),
    }
    serialized = json.dumps(record, sort_keys=True)
    for forbidden in ("target_smiles", "policy_target_smiles", "target_scaffold", "oracle"):
        if forbidden in serialized:
            raise AssertionError(f"benchmark inference prompt leaked {forbidden}")
    return record


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")


def choose_with_mandatory(
    pool: Sequence[dict[str, str]], mandatory: Sequence[dict[str, str]], count: int, seed: int
) -> list[dict[str, str]]:
    by_id = {row_id(row): row for row in pool}
    mandatory_ids = [row_id(row) for row in mandatory]
    if len(set(mandatory_ids)) != len(mandatory_ids):
        raise ValueError("duplicate mandatory row ids")
    missing = sorted(set(mandatory_ids) - set(by_id))
    if missing:
        raise ValueError(f"mandatory P17 rows absent after leakage audit: {missing[:5]}")
    ranked = sorted(pool, key=lambda row: stable_rank(row, seed))
    chosen_ids = set(mandatory_ids)
    selected = [by_id[key] for key in mandatory_ids]
    selected.extend(row for row in ranked if row_id(row) not in chosen_ids and len(selected) < count)
    if len(selected) != count:
        raise ValueError(f"only {len(selected)} eligible rows for requested {count}")
    return sorted(selected, key=lambda row: stable_rank(row, seed))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table1-csv", required=True, type=Path)
    parser.add_argument("--denovo-csv", required=True, type=Path)
    parser.add_argument("--p16-train-jsonl", required=True, type=Path)
    parser.add_argument("--p17-train-jsonl", required=True, type=Path)
    parser.add_argument("--p17-table-reference", required=True, type=Path)
    parser.add_argument("--p17-denovo-reference", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=1717)
    args = parser.parse_args(argv)

    train_rows = [*read_jsonl(args.p16_train_jsonl), *read_jsonl(args.p17_train_jsonl)]
    train_sources = {str(row.get("source_hash", "")) for row in train_rows if row.get("source_hash")}
    train_targets = {
        str(row.get("target_hash") or sha_smiles(row.get("target_smiles", "")))
        for row in train_rows
        if row.get("target_hash") or row.get("target_smiles")
    }
    excluded = Counter()

    table_pool = []
    for row in read_csv(args.table1_csv):
        if sha_smiles(row.get("source_smiles")) in train_sources:
            excluded["table1_source"] += 1
            continue
        if sha_smiles(row.get("target_smiles")) in train_targets:
            excluded["table1_target"] += 1
            continue
        try:
            prompt_record(row)
        except (ValueError, AssertionError):
            excluded["table1_unusable_prompt"] += 1
            continue
        table_pool.append(row)
    table_by_task: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in table_pool:
        table_by_task[str(row.get("moledit_task_key", ""))].append(row)
    mandatory_table: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(args.p17_table_reference):
        mandatory_table[str(row.get("moledit_task_key", ""))].append(row)
    if len(table_by_task) != 10 or set(table_by_task) != set(mandatory_table):
        raise ValueError("expected exactly the same ten Table1 task strata as P17")
    table_rows = []
    for task in sorted(table_by_task):
        table_rows.extend(choose_with_mandatory(table_by_task[task], mandatory_table[task], 10, args.seed))

    denovo_pool = []
    for row in read_csv(args.denovo_csv):
        if sha_smiles(row.get("target_smiles")) in train_targets:
            excluded["denovo_target"] += 1
            continue
        try:
            prompt_record(row)
        except (ValueError, AssertionError):
            excluded["denovo_unusable_prompt"] += 1
            continue
        denovo_pool.append(row)
    denovo_by_count: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in denovo_pool:
        denovo_by_count[int(float(row.get("property_count") or 0))].append(row)
    mandatory_denovo: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(args.p17_denovo_reference):
        mandatory_denovo[int(float(row.get("property_count") or 0))].append(row)
    denovo_rows = []
    for count in (6, 7):
        denovo_rows.extend(
            choose_with_mandatory(denovo_by_count[count], mandatory_denovo[count], 20, args.seed + count)
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    table_ref = args.output_dir / "table1_expanded.reference.csv"
    denovo_ref = args.output_dir / "denovo_expanded.reference.csv"
    table_prompts_path = args.output_dir / "table1_expanded.prompts.jsonl"
    denovo_prompts_path = args.output_dir / "denovo_expanded.prompts.jsonl"
    write_csv(table_ref, table_rows)
    write_csv(denovo_ref, denovo_rows)
    write_jsonl(table_prompts_path, [prompt_record(row) for row in table_rows])
    write_jsonl(denovo_prompts_path, [prompt_record(row) for row in denovo_rows])

    source_overlap = sum(sha_smiles(row.get("source_smiles")) in train_sources for row in table_rows)
    target_overlap = sum(
        sha_smiles(row.get("target_smiles")) in train_targets for row in [*table_rows, *denovo_rows]
    )
    table_mandatory_ids = {row_id(row) for rows in mandatory_table.values() for row in rows}
    denovo_mandatory_ids = {row_id(row) for rows in mandatory_denovo.values() for row in rows}
    if source_overlap or target_overlap:
        raise AssertionError(f"training leakage: source={source_overlap}, target={target_overlap}")
    if not table_mandatory_ids <= {row_id(row) for row in table_rows}:
        raise AssertionError("not all P17 Table1 rows retained")
    if not denovo_mandatory_ids <= {row_id(row) for row in denovo_rows}:
        raise AssertionError("not all P17 de-novo rows retained")
    manifest = {
        "protocol": "p19_frozen_expanded_subsets_v1",
        "status_label": "expanded paired pilot estimate; not full benchmarks",
        "seed": args.seed,
        "table1_rows": len(table_rows),
        "table1_strata": {task: 10 for task in sorted(table_by_task)},
        "denovo_rows": len(denovo_rows),
        "denovo_strata": {"6p": 20, "7p": 20},
        "p17_original_rows_retained": {"table1": len(table_mandatory_ids), "denovo": len(denovo_mandatory_ids)},
        "training_overlap": {"source": source_overlap, "target": target_overlap},
        "excluded_pool_rows": dict(excluded),
        "locked_sha256": {
            "table1_reference": sha_file(table_ref),
            "denovo_reference": sha_file(denovo_ref),
            "table1_prompts": sha_file(table_prompts_path),
            "denovo_prompts": sha_file(denovo_prompts_path),
        },
        "raw_candidate_budgets": [1, 4, 8],
        "raw_generation_count": 8,
        "inference_prompt_target_fields": False,
        "static_candidate_pool": False,
        "property_reranking": False,
        "selection": "candidate 0 greedy; candidates 1-7 sampled once; K uses raw prefix",
        "created_before_expanded_generation": True,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (args.output_dir / "LOCKED.sha256").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in [
            ("table1_expanded.reference.csv", manifest["locked_sha256"]["table1_reference"]),
            ("denovo_expanded.reference.csv", manifest["locked_sha256"]["denovo_reference"]),
            ("table1_expanded.prompts.jsonl", manifest["locked_sha256"]["table1_prompts"]),
            ("denovo_expanded.prompts.jsonl", manifest["locked_sha256"]["denovo_prompts"]),
        ])
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
