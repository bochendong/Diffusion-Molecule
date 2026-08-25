#!/usr/bin/env python3
"""Freeze a leak-audited paired 2p/3p/4p pilot from the official 6000-row eval CSV."""

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
P17_DIR = SCRIPT_DIR.parent / "p17_copy_contrastive_unified_benchmark"
sys.path.insert(0, str(P17_DIR))
import p17_protocol as protocol  # noqa: E402


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha_smiles(value: object) -> str:
    canonical = protocol.canonical_smiles(value)
    return hashlib.sha256(canonical.encode()).hexdigest() if canonical else ""


def row_id(row: Mapping[str, object]) -> str:
    return str(row.get("condition_id") or row.get("sample_id") or "")


def stable_rank(row: Mapping[str, object], seed: int) -> str:
    return hashlib.sha256(f"{seed}:{row_id(row)}".encode()).hexdigest()


def prompt_record(row: Mapping[str, object]) -> dict[str, object]:
    messages, source, mode = protocol.build_prompt(row)
    result = {
        "condition_id": row_id(row),
        "sample_id": str(row.get("sample_id") or row_id(row)),
        "task_mode": mode,
        "source_smiles": source,
        "messages": messages,
        "condition_hash": protocol.condition_hash(row),
        "condition_family_hash": protocol.condition_family_hash(row),
    }
    serialized = json.dumps(result, sort_keys=True)
    for forbidden in ("target_smiles", "policy_target_smiles", "target_scaffold", "oracle"):
        if forbidden in serialized:
            raise AssertionError(f"benchmark prompt leaked {forbidden}")
    return result


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-csv", required=True, type=Path)
    parser.add_argument("--p16-train-jsonl", required=True, type=Path)
    parser.add_argument("--p17-train-jsonl", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=2020)
    parser.add_argument("--per-count", type=int, default=100)
    args = parser.parse_args(argv)

    raw = read_csv(args.eval_csv)
    full_counts = Counter(int(float(row["property_count"])) for row in raw)
    if len(raw) != 6000 or any(full_counts[count] != 1000 for count in range(2, 8)):
        raise ValueError(f"expected official 6000-row 2p-7p eval distribution, got {full_counts}")
    train_rows = [*read_jsonl(args.p16_train_jsonl), *read_jsonl(args.p17_train_jsonl)]
    train_targets = {
        str(row.get("target_hash") or sha_smiles(row.get("target_smiles", "")))
        for row in train_rows if row.get("target_hash") or row.get("target_smiles")
    }
    eligible: dict[int, list[dict[str, str]]] = defaultdict(list)
    excluded = Counter()
    for row in raw:
        count = int(float(row["property_count"]))
        if count not in (2, 3, 4):
            continue
        if sha_smiles(row.get("target_smiles")) in train_targets:
            excluded[f"{count}p_train_target"] += 1
            continue
        try:
            prompt_record(row)
        except (ValueError, AssertionError):
            excluded[f"{count}p_unusable_prompt"] += 1
            continue
        eligible[count].append(row)
    selected: list[dict[str, str]] = []
    for count in (2, 3, 4):
        ranked = sorted(eligible[count], key=lambda row: stable_rank(row, args.seed + count))
        if len(ranked) < args.per_count:
            raise ValueError(f"only {len(ranked)} eligible {count}p rows")
        selected.extend(ranked[: args.per_count])
    if len({row_id(row) for row in selected}) != len(selected):
        raise AssertionError("duplicate condition ids")
    overlap = sum(sha_smiles(row.get("target_smiles")) in train_targets for row in selected)
    if overlap:
        raise AssertionError(f"selected train-target overlap={overlap}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    reference = args.output_dir / "denovo_2p4p.reference.csv"
    prompts = args.output_dir / "denovo_2p4p.prompts.jsonl"
    write_csv(reference, selected)
    write_jsonl(prompts, [prompt_record(row) for row in selected])
    manifest = {
        "protocol": "p20_frozen_denovo_2p4p_paired_pilot_v1",
        "status_label": "paired pilot estimate; not full 1000-condition-per-stratum benchmark",
        "seed": args.seed,
        "source_eval_csv": str(args.eval_csv),
        "source_eval_sha256": sha_file(args.eval_csv),
        "source_eval_rows": len(raw),
        "source_distribution": {f"{count}p": full_counts[count] for count in range(2, 8)},
        "selected_rows": len(selected),
        "selected_distribution": {f"{count}p": args.per_count for count in (2, 3, 4)},
        "excluded": dict(excluded),
        "training_target_overlap": overlap,
        "inference_prompt_target_fields": False,
        "raw_generation_count": 8,
        "raw_candidate_budgets": [1, 4, 8],
        "property_reranking": False,
        "locked_sha256": {"reference": sha_file(reference), "prompts": sha_file(prompts)},
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (args.output_dir / "LOCKED.sha256").write_text(
        f"{manifest['locked_sha256']['reference']}  {reference.name}\n"
        f"{manifest['locked_sha256']['prompts']}  {prompts.name}\n"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
