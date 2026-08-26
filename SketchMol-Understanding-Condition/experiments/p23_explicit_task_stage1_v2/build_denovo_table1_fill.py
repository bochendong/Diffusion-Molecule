#!/usr/bin/env python3
"""Freeze the missing 100-condition 5p slice for the paper de-novo table."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import p23_protocol as protocol


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def row_id(row: dict[str, str]) -> str:
    return str(row.get("condition_id") or row.get("sample_id") or "")


def target_hash(row: dict[str, object]) -> str:
    existing = str(row.get("target_hash") or "").strip()
    if existing:
        return existing
    canonical = protocol.canonical_smiles(row.get("target_smiles") or "")
    return protocol.smiles_hash(canonical)


def stable_rank(row: dict[str, str], seed: int) -> str:
    return hashlib.sha256(f"{seed}:{row_id(row)}".encode()).hexdigest()


def prompt_record(row: dict[str, str]) -> dict[str, object]:
    messages, source, mode = protocol.build_prompt(row)
    if mode != "de_novo" or source != "<EMPTY>":
        raise ValueError(f"expected target-blind de_novo prompt for {row_id(row)}")
    program = protocol.condition_program(row, mode)
    record = {
        "condition_id": row_id(row),
        "sample_id": str(row.get("sample_id") or row_id(row)),
        "task_mode": mode,
        "source_smiles": source,
        "messages": messages,
        "condition_hash": protocol.condition_hash_from_program(program),
        "task_key": protocol.task_key(program),
    }
    serialized = json.dumps(record, sort_keys=True)
    for forbidden in ("target_smiles", "policy_target_smiles", "target_scaffold", "oracle"):
        if forbidden in serialized:
            raise AssertionError(f"5p prompt leaked {forbidden}")
    return record


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-csv", required=True, type=Path)
    parser.add_argument("--train-jsonl", action="append", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--conditions", type=int, default=100)
    args = parser.parse_args()

    raw = read_csv(args.eval_csv)
    distribution = Counter(int(float(row["property_count"])) for row in raw)
    if len(raw) != 6000 or any(distribution[count] != 1000 for count in range(2, 8)):
        raise ValueError(f"expected official 6000-row 2p-7p distribution, got {distribution}")
    train_targets = {
        digest
        for path in args.train_jsonl
        for row in read_jsonl(path)
        if (digest := target_hash(row))
    }
    eligible = []
    excluded = Counter()
    for row in raw:
        if int(float(row["property_count"])) != 5:
            continue
        if target_hash(row) in train_targets:
            excluded["training_target_overlap"] += 1
            continue
        try:
            prompt_record(row)
        except (ValueError, AssertionError):
            excluded["unusable_prompt"] += 1
            continue
        eligible.append(row)
    selected = sorted(eligible, key=lambda row: stable_rank(row, args.seed))[: args.conditions]
    if len(selected) != args.conditions or len({row_id(row) for row in selected}) != args.conditions:
        raise ValueError(f"could not freeze {args.conditions} unique 5p rows")
    overlap = sum(target_hash(row) in train_targets for row in selected)
    if overlap:
        raise AssertionError(f"selected training-target overlap={overlap}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    reference = args.output_dir / "denovo_5p.reference.csv"
    prompts = args.output_dir / "denovo_5p.prompts.jsonl"
    with reference.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selected[0]), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(selected)
    with prompts.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(prompt_record(row), sort_keys=True) + "\n")
    manifest = {
        "protocol": "p23_paper_table1_denovo_missing_cells_v1",
        "seed": args.seed,
        "source_eval_csv": str(args.eval_csv),
        "source_eval_sha256": sha_file(args.eval_csv),
        "source_distribution": {f"{count}p": distribution[count] for count in range(2, 8)},
        "selected_distribution": {"5p": len(selected)},
        "training_files": [str(path) for path in args.train_jsonl],
        "training_target_overlap": overlap,
        "excluded": dict(excluded),
        "generation_target_access": False,
        "candidate_budget": 40,
        "selection": "property-aware best-of-40 finalizer",
        "locked_sha256": {"reference": sha_file(reference), "prompts": sha_file(prompts)},
    }
    (args.output_dir / "denovo_5p.manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
