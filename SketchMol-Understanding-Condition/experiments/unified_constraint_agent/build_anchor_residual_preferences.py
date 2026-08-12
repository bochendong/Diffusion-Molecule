#!/usr/bin/env python3
"""Build balanced train-only Table1/MuMO preferences for an anchored residual policy."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verifier-preference-dir", required=True, type=Path)
    parser.add_argument("--mumo-residual-preference-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-table1-pairs", type=int, default=438)
    parser.add_argument("--max-mumo-graph-pairs", type=int, default=96)
    parser.add_argument("--max-mumo-residual-pairs", type=int, default=438)
    parser.add_argument("--seed", type=int, default=1714)
    return parser.parse_args(argv)


def read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(path)
    return dict(value)


def stable_key(row: Mapping[str, object], seed: int) -> bytes:
    identity = str(row.get("pair_id") or row.get("example_id") or json.dumps(row, sort_keys=True))
    return hashlib.sha256(f"{seed}:{identity}".encode()).digest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prompt_target_hidden(row: Mapping[str, object]) -> bool:
    prompt = json.dumps(row.get("prompt_messages", []), sort_keys=True).lower()
    return "target_smiles" not in prompt and "evaluation_oracle" not in prompt


def select(
    rows: Sequence[Mapping[str, object]],
    *,
    family: str,
    limit: int,
    seed: int,
) -> list[dict[str, object]]:
    ranked = sorted(rows, key=lambda row: stable_key(row, seed))[: max(0, int(limit))]
    output = []
    for row in ranked:
        item = dict(row)
        item["preference_family"] = family
        item["coverage_rank"] = 5
        if not prompt_target_hidden(item):
            raise ValueError(f"Target/oracle content leaked into {family} prompt")
        output.append(item)
    return output


def split_families(
    verifier_rows: Sequence[Mapping[str, object]],
    residual_rows: Sequence[Mapping[str, object]],
    *,
    args: argparse.Namespace,
    validation: bool,
) -> list[dict[str, object]]:
    offset = 100 if validation else 0
    table1 = [row for row in verifier_rows if str(row.get("origin")) == "table1"]
    mumo_graph = [row for row in verifier_rows if str(row.get("origin")) == "mumo"]
    limits = (
        len(table1) if validation else int(args.max_table1_pairs),
        len(mumo_graph) if validation else int(args.max_mumo_graph_pairs),
        len(residual_rows) if validation else int(args.max_mumo_residual_pairs),
    )
    output = [
        *select(table1, family="table1_graph_edit", limit=limits[0], seed=int(args.seed) + offset),
        *select(mumo_graph, family="mumo_graph_edit", limit=limits[1], seed=int(args.seed) + offset + 1),
        *select(
            residual_rows,
            family="mumo_rank_candidate",
            limit=limits[2],
            seed=int(args.seed) + offset + 2,
        ),
    ]
    return sorted(output, key=lambda row: stable_key(row, int(args.seed) + offset + 3))


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    verifier_manifest = load_json(args.verifier_preference_dir / "manifest.json")
    residual_manifest = load_json(args.mumo_residual_preference_dir / "manifest.json")
    if verifier_manifest.get("data_role") != "train_only":
        raise ValueError("Verifier preferences are not train-only")
    if residual_manifest.get("prompt_target_access") is not False:
        raise ValueError("MuMO residual preference prompt contract drifted")
    if int(residual_manifest.get("source_group_overlap", -1)) != 0:
        raise ValueError("MuMO residual train/validation source groups overlap")

    train = split_families(
        read_jsonl(args.verifier_preference_dir / "train.jsonl"),
        read_jsonl(args.mumo_residual_preference_dir / "train.jsonl"),
        args=args,
        validation=False,
    )
    validation = split_families(
        read_jsonl(args.verifier_preference_dir / "validation.jsonl"),
        read_jsonl(args.mumo_residual_preference_dir / "validation.jsonl"),
        args=args,
        validation=True,
    )
    train_ids = {str(row.get("pair_id")) for row in train}
    validation_ids = {str(row.get("pair_id")) for row in validation}
    overlap = train_ids & validation_ids
    if overlap:
        raise ValueError(f"Combined preference train/validation overlap: {sorted(overlap)[:5]}")
    if not train or not validation:
        raise ValueError("Combined preference split is empty")
    write_jsonl(args.output_dir / "train.jsonl", train)
    write_jsonl(args.output_dir / "validation.jsonl", validation)
    manifest = {
        "protocol": "unified_anchor_residual_topk_preference_v1",
        "data_role": "train_only",
        "evaluation_target_access": False,
        "evaluation_oracle_access": False,
        "official_test_content_access": False,
        "prompt_target_access": False,
        "coverage_rank": 5,
        "seed": int(args.seed),
        "train_pairs": len(train),
        "validation_pairs": len(validation),
        "train_family_counts": dict(sorted(Counter(row["preference_family"] for row in train).items())),
        "validation_family_counts": dict(
            sorted(Counter(row["preference_family"] for row in validation).items())
        ),
        "pair_id_overlap": len(overlap),
        "mumo_source_group_overlap": int(residual_manifest["source_group_overlap"]),
        "source_group_overlap": int(residual_manifest["source_group_overlap"]),
        "source_sha256": {
            "verifier_train": sha256(args.verifier_preference_dir / "train.jsonl"),
            "verifier_validation": sha256(args.verifier_preference_dir / "validation.jsonl"),
            "mumo_residual_train": sha256(args.mumo_residual_preference_dir / "train.jsonl"),
            "mumo_residual_validation": sha256(
                args.mumo_residual_preference_dir / "validation.jsonl"
            ),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
