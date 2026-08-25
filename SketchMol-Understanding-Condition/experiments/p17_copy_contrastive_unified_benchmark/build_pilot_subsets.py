#!/usr/bin/env python3
"""Freeze deterministic leak-audited 20-row Table1 and de-novo pilot subsets."""

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


def sha_smiles(value: object) -> str:
    canonical = protocol.canonical_smiles(value)
    return hashlib.sha256(canonical.encode()).hexdigest() if canonical else ""


def rank(row: Mapping[str, object], seed: int) -> str:
    key = str(row.get("condition_id") or row.get("sample_id") or row.get("pair_id") or "")
    return hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")


def prompt_record(row: Mapping[str, object]) -> dict[str, object]:
    messages, source, mode = protocol.build_prompt(row)
    record = {
        "condition_id": str(row.get("condition_id") or row.get("sample_id") or row.get("pair_id") or ""),
        "sample_id": str(row.get("sample_id") or row.get("condition_id") or ""),
        "task_mode": mode, "source_smiles": source, "messages": messages,
        "condition_hash": protocol.condition_hash(row),
        "condition_family_hash": protocol.condition_family_hash(row),
    }
    serialized = json.dumps(record, sort_keys=True)
    for forbidden in ("target_smiles", "policy_target_smiles", "target_scaffold", "oracle"):
        if forbidden in serialized:
            raise AssertionError(f"benchmark inference prompt leaked {forbidden}")
    return record


def stored_family_hash(row: Mapping[str, object]) -> str:
    direct = str(row.get("condition_family_hash", ""))
    if direct:
        return direct
    try:
        payload = json.loads(row["messages"][1]["content"])
        family = sorted(
            (str(item["property"]), item["goal"] if isinstance(item["goal"], str) else "around")
            for item in payload["conditions"]
        )
        return hashlib.sha256(json.dumps(family, separators=(",", ":")).encode()).hexdigest()
    except (KeyError, TypeError, ValueError, IndexError):
        return ""


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table1-csv", required=True, type=Path)
    parser.add_argument("--denovo-csv", required=True, type=Path)
    parser.add_argument("--p16-train-jsonl", required=True, type=Path)
    parser.add_argument("--p17-train-jsonl", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=1717)
    args = parser.parse_args(argv)

    train_rows = [*read_jsonl(args.p16_train_jsonl), *read_jsonl(args.p17_train_jsonl)]
    train_sources = {str(row.get("source_hash", "")) for row in train_rows if row.get("source_hash")}
    train_targets = {
        str(row.get("target_hash") or sha_smiles(row.get("target_smiles", "")))
        for row in train_rows if row.get("target_hash") or row.get("target_smiles")
    }
    train_conditions = {str(row.get("condition_hash", "")) for row in train_rows}
    train_families = {value for row in train_rows if (value := stored_family_hash(row))}

    table_pool = []
    leak_counts = Counter()
    for row in read_csv(args.table1_csv):
        source_hash, target_hash = sha_smiles(row.get("source_smiles")), sha_smiles(row.get("target_smiles"))
        if source_hash in train_sources:
            leak_counts["table1_source"] += 1
            continue
        if target_hash in train_targets:
            leak_counts["table1_target"] += 1
            continue
        try:
            prompt_record(row)
        except (ValueError, AssertionError):
            leak_counts["table1_unusable_prompt"] += 1
            continue
        table_pool.append(row)
    by_task = defaultdict(list)
    for row in table_pool:
        by_task[str(row.get("moledit_task_key", ""))].append(row)
    if len(by_task) != 10:
        raise ValueError(f"expected 10 Table1 strata after audit, got {len(by_task)}")
    table_rows = []
    for task in sorted(by_task):
        chosen = sorted(by_task[task], key=lambda row: rank(row, args.seed))[:2]
        if len(chosen) != 2:
            raise ValueError(f"Table1 task {task} has only {len(chosen)} leak-free rows")
        table_rows.extend(chosen)

    denovo_pool = []
    for row in read_csv(args.denovo_csv):
        target_hash = sha_smiles(row.get("target_smiles"))
        if target_hash in train_targets:
            leak_counts["denovo_target"] += 1
            continue
        try:
            prompt_record(row)
        except (ValueError, AssertionError):
            leak_counts["denovo_unusable_prompt"] += 1
            continue
        denovo_pool.append(row)
    by_count = defaultdict(list)
    for row in denovo_pool:
        by_count[int(float(row.get("property_count") or 0))].append(row)
    denovo_rows = []
    for count in (6, 7):
        chosen = sorted(by_count[count], key=lambda row: rank(row, args.seed + count))[:10]
        if len(chosen) != 10:
            raise ValueError(f"de-novo {count}p has only {len(chosen)} leak-free rows")
        denovo_rows.extend(chosen)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "table1_pilot.reference.csv", table_rows)
    write_csv(args.output_dir / "denovo_pilot.reference.csv", denovo_rows)
    table_prompts = [prompt_record(row) for row in table_rows]
    denovo_prompts = [prompt_record(row) for row in denovo_rows]
    write_jsonl(args.output_dir / "table1_pilot.prompts.jsonl", table_prompts)
    write_jsonl(args.output_dir / "denovo_pilot.prompts.jsonl", denovo_prompts)
    benchmark_conditions = {row["condition_hash"] for row in [*table_prompts, *denovo_prompts]}
    benchmark_families = {row["condition_family_hash"] for row in [*table_prompts, *denovo_prompts]}
    manifest = {
        "protocol": "p17_frozen_pilot_subsets_v1", "seed": args.seed,
        "status_label": "pilot estimate; not full Table1 or full de-novo benchmark",
        "locked_sources": {"table1": str(args.table1_csv), "denovo": str(args.denovo_csv)},
        "table1_rows": len(table_rows),
        "table1_strata": {task: 2 for task in sorted(by_task)},
        "denovo_rows": len(denovo_rows), "denovo_strata": {"6p": 10, "7p": 10},
        "raw_candidate_budgets": [1, 4, 8], "raw_generation_count": 8,
        "training_overlap": {
            "source": 0, "target": 0,
            "condition_identity_count": len(benchmark_conditions & train_conditions),
            "condition_family_count": len(benchmark_families & train_families),
        },
        "excluded_pool_rows": dict(leak_counts),
        "inference_prompt_target_fields": False,
        "static_candidate_pool": False, "property_reranking": False,
        "selection": "candidate 0 greedy, candidates 1-7 sampled once; budgets use prefix only",
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
