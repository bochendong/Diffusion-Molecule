#!/usr/bin/env python3
"""Build leakage-isolated mixed and matched single-mode P16 SFT sets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path
from typing import Mapping, Sequence

import p16_protocol as protocol


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")


def row_id(row: Mapping[str, object]) -> str:
    for key in ("condition_id", "sample_id", "pair_id", "example_id"):
        value = str(row.get(key, "") or "").strip()
        if value:
            return value
    return hashlib.sha256(json.dumps(dict(row), sort_keys=True).encode()).hexdigest()[:20]


def convert(row: Mapping[str, object], declared_mode: str) -> dict[str, object]:
    messages, source, mode = protocol.build_prompt(row)
    if mode != declared_mode:
        raise ValueError(f"mode mismatch: {declared_mode} != {mode}")
    target = str(row.get("target_smiles", "") or row.get("policy_target_smiles", "") or "").strip()
    target = protocol.canonical_smiles(target)
    if not target or (mode == "edit" and target == source):
        raise ValueError("missing/identity target")
    assistant = protocol.response(target, mode)
    return {
        "example_id": f"p16:{mode}:{row_id(row)}",
        "task_mode": mode,
        "condition_hash": protocol.condition_hash(row),
        "source_hash": protocol.source_hash(source),
        "source_smiles": source,
        "target_smiles": target,
        "messages": [*messages, {"role": "assistant", "content": assistant}],
    }


def clean(rows: Sequence[Mapping[str, object]], mode: str) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    seen: set[tuple[str, str, str]] = set()
    for raw in rows:
        try:
            item = convert(raw, mode)
        except (ValueError, AssertionError):
            continue
        key = (str(item["condition_hash"]), str(item["source_hash"]), str(item["target_smiles"]))
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result


def isolated_split(rows: Sequence[dict[str, object]], seed: int, train_limit: int, dev_limit: int) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    ranked = sorted(
        rows,
        key=lambda row: protocol.split_score(str(row["condition_hash"]), str(row["source_hash"]), seed),
    )
    dev: list[dict[str, object]] = []
    dev_conditions: set[str] = set()
    dev_sources: set[str] = set()
    for row in ranked:
        cond = str(row["condition_hash"])
        source = str(row["source_hash"])
        if cond in dev_conditions or (source and source in dev_sources):
            continue
        dev.append(row)
        dev_conditions.add(cond)
        if source:
            dev_sources.add(source)
        if len(dev) >= dev_limit:
            break
    train = [
        row for row in ranked
        if str(row["condition_hash"]) not in dev_conditions
        and (not str(row["source_hash"]) or str(row["source_hash"]) not in dev_sources)
    ][:train_limit]
    if len(train) < train_limit or len(dev) < dev_limit:
        raise ValueError(f"insufficient isolated rows: train={len(train)} dev={len(dev)}")
    return train, dev


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--denovo-csv", required=True, type=Path)
    parser.add_argument("--edit-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--train-per-mode", type=int, default=128)
    parser.add_argument("--dev-per-mode", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1616)
    args = parser.parse_args(argv)

    denovo_train, denovo_dev = isolated_split(clean(read_rows(args.denovo_csv), "de_novo"), args.seed, args.train_per_mode, args.dev_per_mode)
    edit_train, edit_dev = isolated_split(clean(read_rows(args.edit_csv), "edit"), args.seed + 1, args.train_per_mode, args.dev_per_mode)
    mixed_train: list[dict[str, object]] = []
    for left, right in zip(denovo_train, edit_train):
        mixed_train.extend((left, right))
    random.Random(args.seed).shuffle(mixed_train)
    dev = [*denovo_dev, *edit_dev]

    write_jsonl(args.output_dir / "train.mixed.jsonl", mixed_train)
    write_jsonl(args.output_dir / "train.denovo.jsonl", denovo_train)
    write_jsonl(args.output_dir / "train.edit.jsonl", edit_train)
    write_jsonl(args.output_dir / "dev.jsonl", dev)
    train_conditions = {str(row["condition_hash"]) for row in mixed_train}
    dev_conditions = {str(row["condition_hash"]) for row in dev}
    train_sources = {str(row["source_hash"]) for row in mixed_train if row["source_hash"]}
    dev_sources = {str(row["source_hash"]) for row in dev if row["source_hash"]}
    manifest = {
        "protocol": protocol.PROTOCOL,
        "seed": args.seed,
        "train_rows": {"mixed": len(mixed_train), "de_novo": len(denovo_train), "edit": len(edit_train)},
        "dev_rows": {"total": len(dev), "de_novo": len(denovo_dev), "edit": len(edit_dev)},
        "condition_hash_overlap": len(train_conditions & dev_conditions),
        "source_hash_overlap": len(train_sources & dev_sources),
        "prompt_target_fields": False,
        "training_label_contains_target": True,
        "same_schema": True,
        "same_tokenizer": True,
        "mode_router": False,
        "mode_expression": "source=<EMPTY> versus canonical source SMILES",
        "response_schema": '{"plan":"BUILD|MODIFY","smiles":"CANONICAL_SMILES"}',
        "static_candidate_pool": False,
        "property_reranking": False,
        "official_test_access": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    if manifest["condition_hash_overlap"] or manifest["source_hash_overlap"]:
        raise SystemExit("P16 leakage isolation failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
