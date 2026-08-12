#!/usr/bin/env python3
"""Read MuMO train.json once and freeze balanced, leakage-audited shards."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
PROJECT_SCRIPTS = PROJECT_DIR / "scripts"
for path in (SCRIPT_DIR, PROJECT_DIR, PROJECT_SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import export_external_multiproperty_benchmark_rows as export  # noqa: E402
import mumo_parallel_protocol as protocol  # noqa: E402
from sketchmol_understanding_condition.chem import canonical_smiles  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-json", required=True, type=Path)
    parser.add_argument("--audit-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--test-json-digest-only", type=Path, default=None)
    parser.add_argument("--rows-per-task", type=int, default=5500)
    parser.add_argument("--min-rows-per-task", type=int, default=100)
    parser.add_argument("--dev-fraction", type=float, default=0.10)
    parser.add_argument("--shard-count", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1711)
    return parser.parse_args(argv)


def mumo_specs() -> list[export.ExternalTaskSpec]:
    return [spec for spec in export.TASK_SPECS if spec.suite == "mumo"]


def alias_index() -> dict[str, export.ExternalTaskSpec]:
    output: dict[str, export.ExternalTaskSpec] = {}
    for spec in mumo_specs():
        for alias in protocol.task_aliases(spec.task_id, spec.task_key):
            output[alias] = spec
    return output


def row_spec(row: Mapping[str, object], aliases: Mapping[str, export.ExternalTaskSpec]) -> export.ExternalTaskSpec | None:
    raw = protocol.first_value(
        row,
        ("task", "task_id", "task_key", "external_task_id", "external_task_key", "property_combination"),
    )
    token = protocol.normalize_task_token(raw)
    if token in aliases:
        return aliases[token]
    properties = export.parse_properties_payload(row.get("properties"))
    property_token = protocol.normalize_task_token("+".join(sorted(str(key).lower() for key in properties)))
    return aliases.get(property_token)


def audit_forbidden_sources(path: Path) -> tuple[set[str], dict[str, int]]:
    forbidden: set[str] = set()
    raw_rows = 0
    invalid = 0
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            raw_rows += 1
            source = canonical_smiles(str(row.get("source_smiles", "") or ""))
            if not source:
                invalid += 1
                continue
            forbidden.add(source)
    return forbidden, {"rows": raw_rows, "canonical_sources": len(forbidden), "invalid_sources": invalid}


def git_commit(repo_dir: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_dir,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if int(args.rows_per_task) < 100:
        raise ValueError("rows_per_task must be at least 100")
    if not 1 <= int(args.min_rows_per_task) <= int(args.rows_per_task):
        raise ValueError("min_rows_per_task must be positive and no larger than rows_per_task")
    if not 0.0 < float(args.dev_fraction) < 0.5:
        raise ValueError("dev_fraction must be between 0 and 0.5")
    if int(args.shard_count) < 1:
        raise ValueError("shard_count must be positive")

    aliases = alias_index()
    forbidden_sources, audit_summary = audit_forbidden_sources(args.audit_csv)
    reservoirs: dict[str, list[dict[str, object]]] = defaultdict(list)
    unique_counts: Counter[str] = Counter()
    raw_counts: Counter[str] = Counter()
    seen_pairs: dict[str, set[int]] = defaultdict(set)
    outcomes: Counter[str] = Counter()
    rngs = {task_id: random.Random(int(args.seed) + offset) for offset, task_id in enumerate(protocol.TASK_IDS)}

    for raw_index, value in enumerate(protocol.iter_json_array(args.train_json)):
        outcomes["raw_rows"] += 1
        if not isinstance(value, Mapping):
            outcomes["non_object_rows"] += 1
            continue
        row = dict(value)
        raw_split = str(row.get("split", "train") or "train").strip().lower()
        if raw_split not in {"train", "training"}:
            outcomes["non_train_rows"] += 1
            continue
        spec = row_spec(row, aliases)
        if spec is None:
            outcomes["unknown_task_rows"] += 1
            continue
        raw_counts[spec.task_id] += 1
        source = protocol.raw_source_smiles(row)
        target = protocol.raw_target_smiles(row)
        if not source or not target:
            outcomes["missing_smiles_rows"] += 1
            continue
        pair_text = protocol.canonical_pair_key(spec.task_id, source, target)
        pair_digest = int.from_bytes(hashlib.sha256(pair_text.encode()).digest()[:8], "big")
        if pair_digest in seen_pairs[spec.task_id]:
            outcomes["duplicate_raw_pairs"] += 1
            continue
        seen_pairs[spec.task_id].add(pair_digest)
        unique_counts[spec.task_id] += 1
        record = {
            **row,
            "_uca_raw_index": raw_index,
            "_uca_task_id": spec.task_id,
            "_uca_task_key": spec.task_key,
            "_uca_pair_digest": f"{pair_digest:016x}",
        }
        reservoir = reservoirs[spec.task_id]
        seen = int(unique_counts[spec.task_id])
        if len(reservoir) < int(args.rows_per_task):
            reservoir.append(record)
        else:
            replacement = rngs[spec.task_id].randrange(seen)
            if replacement < int(args.rows_per_task):
                reservoir[replacement] = record
        if (raw_index + 1) % 100000 == 0:
            print(f"[mumo-prepare] scanned={raw_index + 1}", flush=True)

    missing_tasks = [
        task
        for task in protocol.TASK_IDS
        if len(reservoirs.get(task, [])) < int(args.min_rows_per_task)
    ]
    if missing_tasks:
        raise ValueError(
            "Insufficient balanced rows: "
            + ", ".join(f"{task}={len(reservoirs.get(task, []))}" for task in missing_tasks)
        )

    selected: list[dict[str, object]] = []
    selected_counts: Counter[str] = Counter()
    partition_counts: Counter[str] = Counter()
    invalid_selected = 0
    forbidden_overlap = 0
    duplicate_canonical_groups = 0
    seen_task_source: set[tuple[str, str]] = set()
    label_coverage: Counter[str] = Counter()
    label_expected: Counter[str] = Counter()
    spec_by_id = {spec.task_id: spec for spec in mumo_specs()}
    for task_id in protocol.TASK_IDS:
        task_rows = list(reservoirs[task_id])
        rngs[task_id].shuffle(task_rows)
        for row in task_rows:
            source = canonical_smiles(protocol.raw_source_smiles(row))
            target = canonical_smiles(protocol.raw_target_smiles(row))
            if not source or not target:
                invalid_selected += 1
                continue
            if source in forbidden_sources:
                forbidden_overlap += 1
                continue
            group = (task_id, source)
            if group in seen_task_source:
                duplicate_canonical_groups += 1
                continue
            seen_task_source.add(group)
            partition = (
                "dev"
                if protocol.stable_fraction(f"{task_id}:{source}", seed=int(args.seed)) < float(args.dev_fraction)
                else "fit"
            )
            record = {
                **row,
                "source_smiles": source,
                "target_smiles": target,
                "_uca_source_group": f"{task_id}:{source}",
                "_uca_partition": partition,
            }
            selected.append(record)
            selected_counts[task_id] += 1
            partition_counts[f"{task_id}:{partition}"] += 1
            spec = spec_by_id[task_id]
            for prop in spec.properties:
                label_expected[prop] += 2
                label_coverage[prop] += int(export.read_property_value(row, prop, prefix="source") is not None)
                label_coverage[prop] += int(export.read_property_value(row, prop, prefix="target") is not None)

    underfilled = [
        task
        for task in protocol.TASK_IDS
        if selected_counts[task] < int(args.min_rows_per_task)
    ]
    if underfilled:
        raise ValueError(
            "Canonical filtering underfilled tasks: "
            + ", ".join(f"{task}={selected_counts[task]}" for task in underfilled)
        )

    shards: list[list[dict[str, object]]] = [[] for _ in range(int(args.shard_count))]
    for row in selected:
        shard = protocol.stable_shard(
            str(row["_uca_source_group"]),
            seed=int(args.seed),
            shard_count=int(args.shard_count),
        )
        row["_uca_shard"] = shard
        shards[shard].append(row)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for shard_index, rows in enumerate(shards):
        rows.sort(key=lambda row: (str(row["_uca_task_id"]), str(row["_uca_source_group"])))
        protocol.write_jsonl(args.output_dir / f"train_shard_{shard_index:03d}.jsonl", rows)

    test_digest = None
    if args.test_json_digest_only:
        test_digest = {
            "path": str(args.test_json_digest_only),
            "sha256": protocol.sha256_file(args.test_json_digest_only),
            "bytes": args.test_json_digest_only.stat().st_size,
            "content_parsed": False,
        }
    manifest = {
        "protocol": protocol.PROTOCOL_VERSION,
        "sampling_algorithm": "per_task_reservoir_then_canonical_source_group_v1",
        "evaluation_target_access": False,
        "evaluation_oracle_access": False,
        "official_test_content_access": False,
        "candidate_budget": 20,
        "seed": int(args.seed),
        "rows_per_task_requested": int(args.rows_per_task),
        "rows_per_task_role": "unique_pair_cap_not_required_quota",
        "min_rows_per_task": int(args.min_rows_per_task),
        "dev_fraction": float(args.dev_fraction),
        "shard_count": int(args.shard_count),
        "code_commit": git_commit(SCRIPT_DIR.parents[2]),
        "train_source": {
            "path": str(args.train_json),
            "sha256": protocol.sha256_file(args.train_json),
            "bytes": args.train_json.stat().st_size,
        },
        "test_source_digest_only": test_digest,
        "audit_forbidden": audit_summary,
        "raw_task_counts": dict(sorted(raw_counts.items())),
        "unique_pair_counts": dict(sorted(unique_counts.items())),
        "selected_task_counts": dict(sorted(selected_counts.items())),
        "partition_counts": dict(sorted(partition_counts.items())),
        "selected_rows": len(selected),
        "fit_rows": sum(value for key, value in partition_counts.items() if key.endswith(":fit")),
        "dev_rows": sum(value for key, value in partition_counts.items() if key.endswith(":dev")),
        "invalid_selected": invalid_selected,
        "audit_canonical_source_overlap_removed": forbidden_overlap,
        "selected_audit_canonical_source_overlap": 0,
        "duplicate_canonical_task_sources_removed": duplicate_canonical_groups,
        "shard_rows": [len(rows) for rows in shards],
        "raw_label_coverage": {
            prop: {
                "observed": int(label_coverage[prop]),
                "expected": int(label_expected[prop]),
                "rate": float(label_coverage[prop] / max(label_expected[prop], 1)),
            }
            for prop in sorted(label_expected)
        },
        "outcomes": dict(sorted(outcomes.items())),
    }
    protocol.write_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
