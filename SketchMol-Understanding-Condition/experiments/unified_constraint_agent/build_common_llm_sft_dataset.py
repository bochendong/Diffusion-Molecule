#!/usr/bin/env python3
"""Build a contamination-safe chat SFT set for the common constraint LLM."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

from molecular_constraint_ir import build_constraint_ir, truthy


SYSTEM_PROMPT = (
    "You are a unified molecular constraint agent. Read the constraint IR and return exactly one JSON object. "
    "For de_novo use action_type=smiles. For edit use action_type=graph_edit_dsl and emit one executable action. "
    "Do not add prose or markdown."
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--denovo-train-csv", required=True, type=Path)
    parser.add_argument("--table1-action-train-csv", required=True, type=Path)
    parser.add_argument("--mumo-action-train-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-denovo-rows", type=int, default=2048)
    parser.add_argument("--table1-repeat", type=int, default=3)
    parser.add_argument("--mumo-repeat", type=int, default=3)
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=1701)
    return parser.parse_args(argv)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def condition_id(row: Mapping[str, object]) -> str:
    for key in ("condition_id", "sample_id", "example_id", "variant_id", "pair_id"):
        value = str(row.get(key, "") or "").strip()
        if value:
            return value
    return hashlib.sha256(json.dumps(dict(row), sort_keys=True).encode()).hexdigest()[:20]


def response_record(row: Mapping[str, object], *, origin: str) -> dict[str, object] | None:
    ir = build_constraint_ir(row)
    if origin == "denovo":
        target = str(row.get("target_smiles", "") or "").strip()
        if ir.task_mode != "de_novo" or not target:
            return None
        action = {"action_type": "smiles", "value": target}
    else:
        raw_action = str(row.get("policy_target_action_json", "") or "").strip()
        if ir.task_mode != "edit" or not raw_action:
            return None
        try:
            action_value = json.loads(raw_action)
        except json.JSONDecodeError:
            return None
        action = {"action_type": "graph_edit_dsl", "value": action_value}
    user_payload = {
        "constraint_ir": ir.to_dict(),
        "response_schema": {"action_type": ir.action_space, "value": "one action"},
    }
    return {
        "example_id": f"{origin}:{condition_id(row)}",
        "origin": origin,
        "data_role": "train_only",
        "task_mode": ir.task_mode,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(user_payload, sort_keys=True, separators=(",", ":"))},
            {"role": "assistant", "content": json.dumps(action, sort_keys=True, separators=(",", ":"))},
        ],
    }


def stable_validation(example_id: str, *, seed: int, fraction: float) -> bool:
    digest = hashlib.sha256(f"{seed}:{example_id}".encode()).digest()
    bucket = int.from_bytes(digest[:8], "big") / float(2**64)
    return bucket < fraction


def sample_denovo(rows: Sequence[Mapping[str, object]], *, limit: int, seed: int) -> list[Mapping[str, object]]:
    eligible = [row for row in rows if not str(row.get("source_smiles", "") or "").strip()]
    random.Random(seed).shuffle(eligible)
    return eligible[: max(0, limit)]


def strict_table1(row: Mapping[str, object]) -> bool:
    return truthy(row.get("policy_target_strict_success"))


def paired_mumo(row: Mapping[str, object]) -> bool:
    return truthy(row.get("policy_target_paired_teacher"))


def repeated(records: Sequence[dict[str, object]], count: int) -> list[dict[str, object]]:
    output = []
    for repeat_index in range(max(1, count)):
        for record in records:
            clone = dict(record)
            clone["repeat_index"] = repeat_index
            output.append(clone)
    return output


def write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not 0.0 < args.validation_fraction < 0.5:
        raise ValueError("--validation-fraction must be between 0 and 0.5")
    denovo_rows = sample_denovo(
        read_rows(args.denovo_train_csv),
        limit=args.max_denovo_rows,
        seed=args.seed,
    )
    table1_rows = [row for row in read_rows(args.table1_action_train_csv) if strict_table1(row)]
    mumo_rows = [row for row in read_rows(args.mumo_action_train_csv) if paired_mumo(row)]

    unique_records = []
    for origin, rows in (("denovo", denovo_rows), ("table1", table1_rows), ("mumo", mumo_rows)):
        for row in rows:
            record = response_record(row, origin=origin)
            if record is not None:
                unique_records.append(record)
    if not unique_records:
        raise SystemExit("No common-LLM SFT records were built")

    by_id = {str(record["example_id"]): record for record in unique_records}
    unique_records = list(by_id.values())
    validation_unique = [
        record
        for record in unique_records
        if stable_validation(str(record["example_id"]), seed=args.seed, fraction=args.validation_fraction)
    ]
    train_unique = [record for record in unique_records if record not in validation_unique]

    repeats = {"denovo": 1, "table1": args.table1_repeat, "mumo": args.mumo_repeat}
    train = []
    for origin in ("denovo", "table1", "mumo"):
        train.extend(repeated([row for row in train_unique if row["origin"] == origin], repeats[origin]))
    random.Random(args.seed).shuffle(train)
    validation = sorted(validation_unique, key=lambda row: str(row["example_id"]))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "train.jsonl", train)
    write_jsonl(args.output_dir / "validation.jsonl", validation)
    manifest = {
        "protocol": "unified_constraint_common_llm_sft_v1",
        "data_role": "train_only",
        "inputs": {
            "denovo_train_csv": str(args.denovo_train_csv),
            "table1_action_train_csv": str(args.table1_action_train_csv),
            "mumo_action_train_csv": str(args.mumo_action_train_csv),
        },
        "unique_rows": len(unique_records),
        "train_rows_after_repeat": len(train),
        "validation_rows": len(validation),
        "unique_by_origin": dict(sorted(Counter(row["origin"] for row in unique_records).items())),
        "train_by_origin": dict(sorted(Counter(row["origin"] for row in train).items())),
        "repeat_factors": repeats,
        "seed": args.seed,
        "validation_fraction": args.validation_fraction,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
