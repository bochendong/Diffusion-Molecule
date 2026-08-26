#!/usr/bin/env python3
"""Freeze a target-blind, 500-output-per-task P23 MolEdit Table 1 evaluation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import p23_protocol as protocol
from build_stage1_data import select_oracle_aligned_paper_edits


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--heldout-csv", required=True, type=Path)
    parser.add_argument("--train-jsonl", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--rows-per-task", type=int, default=500)
    parser.add_argument("--candidate-pool-per-task", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=23500)
    args = parser.parse_args()

    selected, alignment, _pairs = select_oracle_aligned_paper_edits(
        args.heldout_csv,
        args.rows_per_task,
        args.candidate_pool_per_task,
        set(),
        set(),
        0.65,
        args.seed,
    )
    expected = args.rows_per_task * 10
    if len(selected) != expected:
        raise AssertionError(f"selected {len(selected)} rows, expected {expected}")

    train_sources: set[str] = set()
    train_targets: set[str] = set()
    with args.train_jsonl.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("source_hash"):
                train_sources.add(str(row["source_hash"]))
            if row.get("target_hash"):
                train_targets.add(str(row["target_hash"]))
    source_overlap = sum(str(row["source_hash"]) in train_sources for row in selected)
    target_overlap = sum(str(row["target_hash"]) in train_targets for row in selected)
    if source_overlap or target_overlap:
        raise AssertionError(f"training overlap: source={source_overlap}, target={target_overlap}")

    references: list[dict[str, object]] = []
    prompts: list[dict[str, object]] = []
    task_counts: Counter[str] = Counter()
    for row in selected:
        condition_id = str(row["example_id"])
        program = list(row["condition_program"])
        task_key = str(row["task_key"])
        tasks = [
            {"property": str(item["property"]), "direction": str(item["goal"])}
            for item in program
        ]
        references.append(
            {
                "condition_id": condition_id,
                "sample_id": condition_id,
                "source_smiles": row["source_smiles"],
                "target_smiles": row["target_smiles"],
                "source_target_tanimoto": row["source_tanimoto"],
                "moledit_task_key": task_key,
                "instruction_tasks": json.dumps(tasks, separators=(",", ":")),
                "instruction_task_properties": "|".join(item["property"] for item in tasks),
                "instruction_task_directions": json.dumps(
                    {item["property"]: item["direction"] for item in tasks},
                    separators=(",", ":"),
                    sort_keys=True,
                ),
            }
        )
        messages = list(row["messages"][:2])
        serialized = json.dumps(messages, sort_keys=True)
        for forbidden in ("target_smiles", "target_canonical_smiles", "oracle"):
            if forbidden in serialized:
                raise AssertionError(f"prompt leaked {forbidden}")
        prompts.append(
            {
                "condition_id": condition_id,
                "sample_id": condition_id,
                "task_mode": "edit",
                "source_smiles": row["source_smiles"],
                "messages": messages,
                "condition_hash": protocol.condition_hash_from_program(program),
                "task_key": task_key,
            }
        )
        task_counts[task_key] += 1
    if set(task_counts.values()) != {args.rows_per_task} or len(task_counts) != 10:
        raise AssertionError(f"unbalanced tasks: {dict(task_counts)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    reference_path = args.output_dir / "table1_500.reference.csv"
    with reference_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(references[0]))
        writer.writeheader()
        writer.writerows(references)
    prompt_path = args.output_dir / "table1_500.prompts.jsonl"
    with prompt_path.open("w", encoding="utf-8") as handle:
        for row in prompts:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    manifest = {
        "protocol": "p23_moledit_table1_sampled_once_500_per_task_v1",
        "seed": args.seed,
        "heldout_csv": str(args.heldout_csv),
        "tasks": dict(sorted(task_counts.items())),
        "total_outputs": len(prompts),
        "outputs_per_task": args.rows_per_task,
        "outputs_per_source": 1,
        "aggregation": "output_level_candidate_pool",
        "property_reranking": False,
        "any_at_k": False,
        "greedy_candidate": False,
        "prompt_target_access": False,
        "training_overlap": {"source": source_overlap, "target": target_overlap},
        "alignment": alignment,
        "reference_sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
        "prompts_sha256": hashlib.sha256(prompt_path.read_bytes()).hexdigest(),
    }
    (args.output_dir / "table1_500.manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: manifest[k] for k in ("protocol", "tasks", "total_outputs", "training_overlap")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
