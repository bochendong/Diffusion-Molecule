#!/usr/bin/env python3
"""Freeze a target-disjoint, arity-stratified de novo overlap gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
P23_DIR = SCRIPT_DIR.parent / "p23_explicit_task_stage1_v2"
if str(P23_DIR) not in sys.path:
    sys.path.insert(0, str(P23_DIR))
import p23_protocol as protocol  # noqa: E402


SHARED_PROPERTIES = frozenset({"MW", "LogP", "QED", "HBA", "RB"})


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_key(identity: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{identity}".encode()).hexdigest()


def active_properties(row: Mapping[str, object]) -> frozenset[str]:
    declared = str(row.get("condition_properties", "")).strip()
    if declared:
        return frozenset(item.strip() for item in declared.split(",") if item.strip())
    return frozenset(
        prop
        for prop in protocol.PROPERTIES
        if str(row.get(f"{prop}_active", "")).lower() in {"1", "true", "yes"}
    )


def overlap_group(properties: frozenset[str]) -> str:
    if properties <= SHARED_PROPERTIES:
        return "shared_only"
    unknown = properties - SHARED_PROPERTIES - {"TPSA", "HBD"}
    if unknown:
        raise ValueError(f"unexpected de novo properties: {sorted(unknown)}")
    return "contains_denovo_only"


def target_hash(row: Mapping[str, object]) -> str:
    existing = str(row.get("target_hash", "")).strip()
    if existing:
        return existing
    canonical = protocol.canonical_smiles(row.get("target_smiles", ""))
    return protocol.smiles_hash(canonical)


def prompt_record(row: Mapping[str, object], group: str) -> dict[str, object]:
    messages, source, mode = protocol.build_prompt(row)
    if mode != "de_novo" or source != "<EMPTY>":
        raise ValueError(f"expected de novo row: {row.get('condition_id')}")
    program = protocol.condition_program(row, mode)
    result = {
        "condition_id": str(row.get("condition_id") or row.get("sample_id")),
        "sample_id": str(row.get("sample_id") or row.get("condition_id")),
        "task_mode": mode,
        "source_smiles": source,
        "messages": messages,
        "condition_hash": protocol.condition_hash_from_program(program),
        "task_key": protocol.task_key(program),
        "overlap_group": group,
        "property_count": len(program),
    }
    serialized = json.dumps(result, sort_keys=True)
    for forbidden in ("target_smiles", "policy_target_smiles", "oracle"):
        if forbidden in serialized:
            raise AssertionError(f"prompt leaked {forbidden}")
    return result


def select_rows(
    rows: Sequence[dict[str, str]],
    *,
    excluded_hashes: set[str],
    excluded_ids: set[str],
    seed: int,
    per_2p4p_cell: int,
    per_5p_cell: int,
) -> tuple[list[dict[str, str]], dict[str, object]]:
    grouped: dict[tuple[int, str], list[dict[str, str]]] = defaultdict(list)
    excluded = Counter()
    for row in rows:
        arity = int(float(row["property_count"]))
        if arity not in {2, 3, 4, 5}:
            continue
        identity = str(row.get("condition_id") or row.get("sample_id") or "")
        digest = target_hash(row)
        if identity in excluded_ids:
            excluded["prior_gate_condition"] += 1
            continue
        if digest in excluded_hashes:
            excluded["training_target"] += 1
            continue
        group = overlap_group(active_properties(row))
        grouped[(arity, group)].append(row)

    selected: list[dict[str, str]] = []
    available: dict[str, int] = {}
    selected_counts: dict[str, int] = {}
    for arity in (2, 3, 4, 5):
        quota = per_2p4p_cell if arity <= 4 else per_5p_cell
        for group in ("shared_only", "contains_denovo_only"):
            key = (arity, group)
            values = sorted(
                grouped[key],
                key=lambda row: stable_key(
                    str(row.get("condition_id") or row.get("sample_id")), seed
                ),
            )
            label = f"{arity}p:{group}"
            available[label] = len(values)
            if len(values) < quota:
                raise ValueError(f"{label} has {len(values)} eligible rows; needs {quota}")
            chosen = values[:quota]
            selected.extend(chosen)
            selected_counts[label] = len(chosen)
    return selected, {
        "eligible_counts": available,
        "selected_counts": selected_counts,
        "excluded": dict(excluded),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-csv", required=True, type=Path)
    parser.add_argument("--training-jsonl", action="append", required=True, type=Path)
    parser.add_argument("--prior-gate", action="append", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=37101)
    parser.add_argument("--per-2p4p-cell", type=int, default=100)
    parser.add_argument("--per-5p-cell", type=int, default=40)
    args = parser.parse_args(argv)

    training_hashes = {
        digest
        for path in args.training_jsonl
        for row in read_jsonl(path)
        if (digest := target_hash(row))
    }
    prior_ids = {
        str(row["condition_id"])
        for path in args.prior_gate
        for row in read_jsonl(path)
    }
    with args.eval_csv.open(newline="", encoding="utf-8") as handle:
        source_rows = [dict(row) for row in csv.DictReader(handle)]
    selected, audit = select_rows(
        source_rows,
        excluded_hashes=training_hashes,
        excluded_ids=prior_ids,
        seed=args.seed,
        per_2p4p_cell=args.per_2p4p_cell,
        per_5p_cell=args.per_5p_cell,
    )
    if len({target_hash(row) for row in selected}) != len(selected):
        raise ValueError("selected gate contains duplicate target molecules")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    gate_path = args.output_dir / "gate.denovo_overlap.jsonl"
    reference_path = args.output_dir / "gate.denovo_overlap.reference.csv"
    prompts = [
        prompt_record(row, overlap_group(active_properties(row))) for row in selected
    ]
    gate_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in prompts)
    )
    with reference_path.open("w", newline="", encoding="utf-8") as handle:
        fields = list(selected[0])
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(selected)
    manifest = {
        "protocol": "p37_denovo_overlap_expanded_raw1_v1",
        "source_eval_csv": str(args.eval_csv),
        "source_eval_sha256": sha_file(args.eval_csv),
        "selection_seed": args.seed,
        "training_files": [str(path) for path in args.training_jsonl],
        "training_target_hash_count": len(training_hashes),
        "prior_gates": [str(path) for path in args.prior_gate],
        "prior_gate_condition_count": len(prior_ids),
        "gate_rows": len(prompts),
        "per_2p4p_cell": args.per_2p4p_cell,
        "per_5p_cell": args.per_5p_cell,
        "shared_properties": sorted(SHARED_PROPERTIES),
        "denovo_only_properties": ["HBD", "TPSA"],
        "generation_target_access": False,
        "property_reranking": False,
        "audit": audit,
        "locked_sha256": {
            "gate": sha_file(gate_path),
            "reference": sha_file(reference_path),
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    (args.output_dir / "DATA_COMPLETE").touch()
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
