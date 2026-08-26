#!/usr/bin/env python3
"""Validate the two release modes and write one frozen dataset manifest/card."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from collections import Counter
from pathlib import Path


REQUIRED = {
    "dataset", "release_version", "example_id", "task_mode", "source_smiles",
    "target_smiles", "target_hash", "condition_program", "condition_hash",
    "task_key", "property_count", "messages", "provenance",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_mode(root: Path, mode: str, expected: int) -> dict[str, object]:
    rows = 0
    properties: Counter[str] = Counter()
    examples: set[str] = set()
    targets: set[str] = set()
    pairs: set[str] = set()
    shards = []
    for path in sorted((root / mode).glob("*.jsonl")):
        index = path.with_suffix(".idx")
        if not index.is_file():
            raise FileNotFoundError(index)
        offsets = index.stat().st_size // 8
        first_offset = None
        last_offset = None
        with index.open("rb") as handle:
            if offsets:
                first_offset = struct.unpack("<Q", handle.read(8))[0]
                handle.seek((offsets - 1) * 8)
                last_offset = struct.unpack("<Q", handle.read(8))[0]
        shard_rows = 0
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                missing = REQUIRED.difference(row)
                if missing:
                    raise ValueError(f"{path}: missing {sorted(missing)}")
                if row["task_mode"] != mode:
                    raise ValueError(f"{path}: wrong task mode {row['task_mode']}")
                messages = row["messages"]
                if [item.get("role") for item in messages] != ["system", "user", "assistant"]:
                    raise ValueError(f"{path}: invalid messages")
                if len(row["condition_program"]) != int(row["property_count"]):
                    raise ValueError(f"{path}: property count mismatch")
                example_id = str(row["example_id"])
                if example_id in examples:
                    raise ValueError(f"duplicate example id: {example_id}")
                examples.add(example_id)
                targets.add(str(row["target_hash"]))
                if mode == "edit":
                    pair = str(row.get("pair_hash", ""))
                    if not pair or pair in pairs:
                        raise ValueError(f"missing or duplicate pair hash: {pair}")
                    pairs.add(pair)
                properties[f"{row['property_count']}p"] += 1
                shard_rows += 1
        if shard_rows != offsets:
            raise ValueError(f"{path}: JSONL rows={shard_rows}, index rows={offsets}")
        rows += shard_rows
        shards.append({
            "jsonl": str(path.relative_to(root)), "index": str(index.relative_to(root)),
            "rows": shard_rows, "first_offset": first_offset, "last_offset": last_offset,
            "jsonl_sha256": sha256_file(path), "index_sha256": sha256_file(index),
        })
    if rows != expected:
        raise ValueError(f"{mode}: expected {expected} rows, found {rows}")
    return {
        "rows": rows, "unique_example_ids": len(examples),
        "unique_target_molecules": len(targets), "unique_source_target_pairs": len(pairs),
        "property_count_distribution": dict(sorted(properties.items())), "shards": shards,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", required=True, type=Path)
    parser.add_argument("--de-novo-rows", type=int, default=2_000_000)
    parser.add_argument("--edit-rows", type=int, default=569_919)
    parser.add_argument("--output-manifest", required=True, type=Path)
    args = parser.parse_args()
    result = {
        "dataset": "MolProgramInstruct-Balanced", "release_version": "1.0",
        "protocol": "molprogram_instruct_4m_release_v1",
        "total_instruction_examples": args.de_novo_rows + args.edit_rows,
        "count_semantics": "instruction examples; unique structures and pairs reported separately",
        "modes": {
            "de_novo": validate_mode(args.release_root, "de_novo", args.de_novo_rows),
            "edit": validate_mode(args.release_root, "edit", args.edit_rows),
        },
    }
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (args.output_manifest.parent / "RELEASE_COMPLETE").write_text(
        sha256_file(args.output_manifest) + "  " + args.output_manifest.name + "\n"
    )
    print(json.dumps({"dataset": result["dataset"], "rows": result["total_instruction_examples"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
