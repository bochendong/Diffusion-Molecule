#!/usr/bin/env python3
"""Build frozen P18 train-only negative pairs from the exact P17 training rows."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

LOCKED_TRAIN_SHA256 = "6d92e05a8a5c5ae8351fa4d2887942fc10d2cf4dfd5e81512784d8bb0d95a6dd"


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def response(mode: str, smiles: str) -> str:
    plan = "BUILD" if mode == "de_novo" else "MODIFY"
    return json.dumps({"plan": plan, "smiles": smiles}, separators=(",", ":"))


def donor_for(row: Mapping[str, object], pool: Sequence[Mapping[str, object]]) -> Mapping[str, object]:
    start = int(hashlib.sha256(str(row["example_id"]).encode()).hexdigest(), 16) % len(pool)
    for offset in range(len(pool)):
        donor = pool[(start + offset) % len(pool)]
        if (donor["condition_hash"] != row["condition_hash"]
                and donor["target_hash"] != row["target_hash"]
                and str(donor["target_smiles"]) != str(row.get("source_smiles", ""))):
            return donor
    raise ValueError(f"no condition-mismatch donor for {row['example_id']}")


def instance(
    row: Mapping[str, object], negative_type: str, rejected: str,
    margin: float, weight: float, ce_weight: float,
) -> dict[str, object]:
    return {
        "pair_id": f"{row['example_id']}:{negative_type}",
        "example_id": row["example_id"],
        "task_mode": row["task_mode"],
        "condition_hash": row["condition_hash"],
        "source_hash": row["source_hash"],
        "target_hash": row["target_hash"],
        "messages": row["messages"],
        "negative_type": negative_type,
        "rejected_assistant": rejected,
        "margin": margin,
        "negative_weight": weight,
        "chosen_ce_weight": ce_weight,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p17-train", required=True, type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args(argv)

    actual_hash = sha256(args.p17_train)
    if actual_hash != LOCKED_TRAIN_SHA256:
        raise SystemExit(f"locked P17 train hash mismatch: {actual_hash}")
    rows = read_jsonl(args.p17_train)
    by_mode = {
        mode: sorted((row for row in rows if row["task_mode"] == mode), key=lambda row: str(row["example_id"]))
        for mode in ("de_novo", "edit")
    }
    if {mode: len(items) for mode, items in by_mode.items()} != {"de_novo": 160, "edit": 160}:
        raise SystemExit("P18 requires the exact balanced P17 160+160 training set")

    output: list[dict[str, object]] = []
    donors: list[dict[str, str]] = []
    for row in rows:
        mode = str(row["task_mode"])
        target = str(row["target_smiles"])
        negatives = []
        if mode == "edit":
            negatives.append(("source_copy", str(row["rejected_assistant"]), 0.10, 0.12))
        negatives.append(("invalid_corruption", response(mode, target + "("), 0.15, 0.20))
        donor = donor_for(row, by_mode[mode])
        negatives.append(("condition_mismatch", response(mode, str(donor["target_smiles"])), 0.10, 0.10))
        donors.append({"example_id": str(row["example_id"]), "donor_example_id": str(donor["example_id"])})
        ce_weight = 1.0 / len(negatives)
        output.extend(instance(row, kind, text, margin, weight, ce_weight) for kind, text, margin, weight in negatives)

    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.error")
    invalid_rows = [item for item in output if item["negative_type"] == "invalid_corruption"]
    invalid_json_parse_failures = 0
    invalid_rdkit_valid_count = 0
    for item in invalid_rows:
        try:
            payload = json.loads(str(item["rejected_assistant"]))
            if set(payload) != {"plan", "smiles"}:
                invalid_json_parse_failures += 1
                continue
            invalid_rdkit_valid_count += Chem.MolFromSmiles(str(payload["smiles"])) is not None
        except (TypeError, ValueError):
            invalid_json_parse_failures += 1
    if invalid_json_parse_failures or invalid_rdkit_valid_count:
        raise SystemExit(
            f"invalid-negative audit failed: json={invalid_json_parse_failures} rdkit_valid={invalid_rdkit_valid_count}"
        )

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as handle:
        for row in output:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    counts = Counter(str(row["negative_type"]) for row in output)
    manifest = {
        "protocol": "p18_validity_aware_multinegative_data_v1",
        "p17_train": str(args.p17_train),
        "p17_train_sha256": actual_hash,
        "unique_train_rows": len(rows),
        "unique_rows_by_mode": {mode: len(items) for mode, items in by_mode.items()},
        "logical_pair_instances": len(output),
        "negative_instances": dict(sorted(counts.items())),
        "invalid_negative_strict_json_count": len(invalid_rows) - invalid_json_parse_failures,
        "invalid_negative_rdkit_valid_count": invalid_rdkit_valid_count,
        "condition_mismatch_same_target_count": sum(
            item["negative_type"] == "condition_mismatch"
            and json.loads(str(item["rejected_assistant"]))["smiles"] == json.loads(str(item["messages"][-1]["content"]))["smiles"]
            for item in output
        ),
        "condition_mismatch_source_copy_count": sum(
            item["negative_type"] == "condition_mismatch"
            and item["task_mode"] == "edit"
            and json.loads(str(item["rejected_assistant"]))["smiles"]
                == next(row["source_smiles"] for row in rows if row["example_id"] == item["example_id"])
            for item in output
        ),
        "chosen_ce_total_weight_by_mode": {
            mode: sum(float(item["chosen_ce_weight"]) for item in output if item["task_mode"] == mode)
            for mode in by_mode
        },
        "negative_target_source": "P17 train rows only",
        "development_target_access": False,
        "benchmark_target_access": False,
        "mode_router": False,
        "donor_assignment_sha256": hashlib.sha256(json.dumps(donors, sort_keys=True).encode()).hexdigest(),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
