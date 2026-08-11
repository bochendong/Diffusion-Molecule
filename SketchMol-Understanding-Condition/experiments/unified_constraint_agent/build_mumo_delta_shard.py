#!/usr/bin/env python3
"""Extract mergeable train-only matched-pair delta statistics from one shard."""

from __future__ import annotations

import argparse
import json
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

import build_composed_retrieved_delta_candidates as composed  # noqa: E402
import build_retrieved_delta_edit_candidates as delta  # noqa: E402
import export_external_multiproperty_benchmark_rows as export  # noqa: E402
import mumo_parallel_protocol as protocol  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True, type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--manifest-json", required=True, type=Path)
    parser.add_argument("--min-core-heavy-atoms", type=int, default=5)
    parser.add_argument("--max-variable-heavy-atoms", type=int, default=30)
    return parser.parse_args(argv)


def spec_index() -> dict[str, export.ExternalTaskSpec]:
    return {spec.task_id: spec for spec in export.TASK_SPECS if spec.suite == "mumo"}


def normalized_row(raw: Mapping[str, object], spec: export.ExternalTaskSpec) -> dict[str, object]:
    row: dict[str, object] = {
        "condition_id": str(raw.get("_uca_source_group", "")),
        "source_smiles": str(raw.get("source_smiles", "") or ""),
        "target_smiles": str(raw.get("target_smiles", "") or ""),
        "external_task_id": spec.task_id,
        "external_task_key": spec.task_key,
        "external_task_properties": ",".join(spec.properties),
        "external_property_directions_json": json.dumps(spec.directions, sort_keys=True),
        "external_property_thresholds_json": json.dumps(dict(spec.thresholds), sort_keys=True),
    }
    for prop in spec.properties:
        source = export.read_property_value(raw, prop, prefix="source")
        target = export.read_property_value(raw, prop, prefix="target")
        if source is not None:
            row[f"external_source_{prop}"] = float(source)
        if target is not None:
            row[f"external_target_{prop}"] = float(target)
    return row


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    specs = spec_index()
    rows = protocol.read_jsonl(args.input_jsonl)
    counts: Counter[tuple[str, str, str]] = Counter()
    effect_sums: dict[tuple[str, str, str], dict[str, float]] = defaultdict(lambda: defaultdict(float))
    effect_counts: dict[tuple[str, str, str], Counter[str]] = defaultdict(Counter)
    first_condition: dict[tuple[str, str, str], str] = {}
    task_rows: Counter[str] = Counter()
    outcomes: Counter[str] = Counter()

    for raw in rows:
        if str(raw.get("_uca_partition", "")) != "fit":
            outcomes["dev_rows_skipped"] += 1
            continue
        task_id = str(raw.get("_uca_task_id", ""))
        spec = specs.get(task_id)
        if spec is None:
            outcomes["unknown_task_rows"] += 1
            continue
        row = normalized_row(raw, spec)
        task_rows[task_id] += 1
        source_by_core: dict[str, set[str]] = defaultdict(set)
        target_by_core: dict[str, set[str]] = defaultdict(set)
        for split in delta.fragment_splits(
            str(row["source_smiles"]),
            int(args.min_core_heavy_atoms),
            int(args.max_variable_heavy_atoms),
        ):
            source_by_core[split.core].add(split.variable)
        for split in delta.fragment_splits(
            str(row["target_smiles"]),
            int(args.min_core_heavy_atoms),
            int(args.max_variable_heavy_atoms),
        ):
            target_by_core[split.core].add(split.variable)
        effects = composed.observed_property_effects(row)
        added = False
        for core in sorted(set(source_by_core) & set(target_by_core)):
            for source_variable in sorted(source_by_core[core]):
                for target_variable in sorted(target_by_core[core]):
                    if source_variable == target_variable:
                        continue
                    key = (spec.task_key, source_variable, target_variable)
                    counts[key] += 1
                    first_condition.setdefault(key, str(raw.get("_uca_source_group", "")))
                    for prop, value in effects.items():
                        effect_sums[key][prop] += float(value)
                        effect_counts[key][prop] += 1
                    added = True
        outcomes["rows_with_transform"] += int(added)
        outcomes["rows_without_transform"] += int(not added)

    transform_rows = []
    for key in sorted(counts):
        task_key, source_variable, target_variable = key
        transform_rows.append(
            {
                "task_key": task_key,
                "source_variable": source_variable,
                "target_variable": target_variable,
                "frequency": int(counts[key]),
                "first_train_condition_id": first_condition[key],
                "effect_sums": dict(sorted(effect_sums[key].items())),
                "effect_counts": dict(sorted(effect_counts[key].items())),
            }
        )
    protocol.write_jsonl(args.output_jsonl, transform_rows)
    manifest = {
        "protocol": protocol.PROTOCOL_VERSION,
        "stage": "delta_shard",
        "data_role": "fit_train_only",
        "evaluation_target_access": False,
        "evaluation_oracle_access": False,
        "input": str(args.input_jsonl),
        "input_rows": len(rows),
        "fit_task_rows": dict(sorted(task_rows.items())),
        "unique_transform_partials": len(transform_rows),
        "transform_observations": int(sum(counts.values())),
        "outcomes": dict(sorted(outcomes.items())),
        "min_core_heavy_atoms": int(args.min_core_heavy_atoms),
        "max_variable_heavy_atoms": int(args.max_variable_heavy_atoms),
    }
    protocol.write_json(args.manifest_json, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
