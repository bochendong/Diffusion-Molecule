#!/usr/bin/env python3
"""Fail closed unless the corrected P24 gate is valid and avoids exact copying."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from rdkit import Chem


def canonical(smiles: str) -> str:
    molecule = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(molecule, canonical=True) if molecule is not None else ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-summary", required=True, type=Path)
    parser.add_argument("--prompts-jsonl", required=True, type=Path)
    parser.add_argument("--candidates-csv", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--minimum-validity", type=float, default=0.80)
    parser.add_argument("--minimum-bucket-validity", type=float, default=0.60)
    parser.add_argument("--maximum-edit-copy-rate", type=float, default=0.20)
    args = parser.parse_args()

    training = json.loads(args.training_summary.read_text())
    exact_contract = (
        training.get("max_steps") == 500
        and training.get("gradient_accumulation") == 26
        and training.get("effective_examples") == 13000
        and training.get("adapter_nonfinite_parameters") == 0
    )
    prompts = {
        row["condition_id"]: row
        for row in (json.loads(line) for line in args.prompts_jsonl.read_text().splitlines() if line.strip())
    }
    totals: dict[str, int] = defaultdict(int)
    valid: dict[str, int] = defaultdict(int)
    edit_total = edit_valid = edit_copies = 0
    with args.candidates_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            prompt = prompts[row["condition_id"]]
            bucket = f"{prompt['task_mode']}:{prompt['property_count']}p"
            totals[bucket] += 1
            generated = canonical(row.get("generated_smiles", ""))
            if generated:
                valid[bucket] += 1
            if prompt["task_mode"] == "edit":
                edit_total += 1
                if generated:
                    edit_valid += 1
                    if generated == canonical(str(prompt["source_smiles"])):
                        edit_copies += 1
    total = sum(totals.values())
    total_valid = sum(valid.values())
    bucket_validity = {key: valid[key] / totals[key] for key in sorted(totals)}
    validity = total_valid / total
    edit_copy_rate = edit_copies / edit_valid if edit_valid else 1.0
    expected_candidates = len(prompts) * 8
    expected_buckets = {
        *{f"de_novo:{count}p" for count in range(2, 8)},
        *{f"edit:{count}p" for count in range(1, 8)},
    }
    checks = {
        "exact_training_contract": exact_contract,
        "complete_candidate_matrix": total == expected_candidates and set(totals) == expected_buckets,
        "overall_validity": validity >= args.minimum_validity,
        "per_bucket_validity": min(bucket_validity.values()) >= args.minimum_bucket_validity,
        "edit_copy_rate": edit_copy_rate <= args.maximum_edit_copy_rate,
    }
    result = {
        "protocol": "p24_gate_validity_noncopy_v1",
        "training": {
            key: training.get(key)
            for key in ("max_steps", "gradient_accumulation", "effective_examples", "adapter_nonfinite_parameters")
        },
        "candidate_rows": total,
        "validity": validity,
        "bucket_validity": bucket_validity,
        "edit_valid_candidates": edit_valid,
        "edit_exact_copies": edit_copies,
        "edit_copy_rate": edit_copy_rate,
        "thresholds": {
            "minimum_validity": args.minimum_validity,
            "minimum_bucket_validity": args.minimum_bucket_validity,
            "maximum_edit_copy_rate": args.maximum_edit_copy_rate,
        },
        "checks": checks,
        "passed": all(checks.values()),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit("P24 corrected gate failed validity/non-copy contract")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
