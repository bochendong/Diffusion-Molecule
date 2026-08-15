#!/usr/bin/env python3
"""Audit train-only one-cut fragment attachment support for B24.

The audit reproduces the frozen B22 train/development split, then decomposes
only selected training source-target pairs with RDKit MMPA.  A pair is covered
when source and target expose different one-cut variables around an identical
core and joining the target variable back to that core reconstructs the target.
No development target is fragmented or used to build fragment support.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
UCA_DIR = PROJECT_DIR / "experiments" / "unified_constraint_agent"
for path in (SCRIPT_DIR, PROJECT_DIR, UCA_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import build_retrieved_delta_edit_candidates as fragments  # noqa: E402
import source_relative_delta_diffusion as delta  # noqa: E402


base = delta.base
belief = delta.belief

PROTOCOL = "train_only_fragment_attachment_coverage_gate_v24"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--validation-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-limit", type=int, default=1500)
    parser.add_argument("--validation-limit", type=int, default=20)
    parser.add_argument("--property-counts", default="2,3")
    parser.add_argument("--max-atoms", type=int, default=64)
    parser.add_argument("--fingerprint-bits", type=int, default=512)
    parser.add_argument("--condition-dim", type=int, default=64)
    parser.add_argument("--mcs-timeout", type=int, default=1)
    parser.add_argument("--min-common-fraction", type=float, default=0.45)
    parser.add_argument("--min-core-heavy-atoms", type=int, default=5)
    parser.add_argument("--max-variable-heavy-atoms", type=int, default=30)
    parser.add_argument("--validation-selection-seed", type=int, default=2719)
    parser.add_argument("--validation-exclusion-seed", type=int, default=1742)
    parser.add_argument("--train-selection-seed", type=int, default=1741)
    parser.add_argument("--gate-overall-coverage", type=float, default=0.30)
    parser.add_argument("--gate-three-property-coverage", type=float, default=0.30)
    parser.add_argument("--gate-growth-task-coverage", type=float, default=0.30)
    parser.add_argument("--gate-exact-reconstruction", type=float, default=0.95)
    parser.add_argument("--gate-unique-target-fragments", type=int, default=100)
    return parser.parse_args(argv)


def pair_fragment_support(pair: object, args: argparse.Namespace) -> dict[str, object]:
    source_splits = fragments.fragment_splits(
        pair.source_smiles,
        int(args.min_core_heavy_atoms),
        int(args.max_variable_heavy_atoms),
    )
    target_splits = fragments.fragment_splits(
        pair.target_smiles,
        int(args.min_core_heavy_atoms),
        int(args.max_variable_heavy_atoms),
    )
    source_by_core: dict[str, set[str]] = defaultdict(set)
    target_by_core: dict[str, set[str]] = defaultdict(set)
    for split in source_splits:
        source_by_core[split.core].add(split.variable)
    for split in target_splits:
        target_by_core[split.core].add(split.variable)
    target_canonical = fragments.canonical_smiles(pair.target_smiles)
    transforms: set[tuple[str, str, str]] = set()
    exact_transforms: set[tuple[str, str, str]] = set()
    for core in sorted(set(source_by_core) & set(target_by_core)):
        for source_variable in source_by_core[core]:
            for target_variable in target_by_core[core]:
                if source_variable == target_variable:
                    continue
                transform = (core, source_variable, target_variable)
                transforms.add(transform)
                joined = fragments.canonical_smiles(
                    fragments.join_fragments(core, target_variable)
                )
                if joined == target_canonical:
                    exact_transforms.add(transform)
    return {
        "covered": bool(transforms),
        "exact": bool(exact_transforms),
        "transform_count": len(transforms),
        "exact_transform_count": len(exact_transforms),
        "transforms": exact_transforms or transforms,
    }


def summarize_group(rows: Sequence[dict[str, object]]) -> dict[str, object]:
    count = len(rows)
    covered = sum(bool(row["covered"]) for row in rows)
    exact = sum(bool(row["exact"]) for row in rows)
    return {
        "pairs": count,
        "covered_pairs": covered,
        "coverage": covered / max(1, count),
        "exact_pairs": exact,
        "exact_reconstruction_rate": exact / max(1, covered),
        "mean_transforms_per_pair": sum(int(row["transform_count"]) for row in rows)
        / max(1, count),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    allowed_counts = base.parse_property_counts(str(args.property_counts))
    validation_rows = base.read_rows(args.validation_csv)
    historical_pairs, historical_counts = base.build_pairs(
        validation_rows,
        max_atoms=int(args.max_atoms),
        fingerprint_bits=int(args.fingerprint_bits),
        condition_dim=int(args.condition_dim),
        allowed_counts=allowed_counts,
        timeout=int(args.mcs_timeout),
        min_common_fraction=float(args.min_common_fraction),
        limit=int(args.validation_limit),
        seed=int(args.validation_exclusion_seed),
    )
    historical_sources = {pair.source_smiles for pair in historical_pairs}
    historical_keys = {
        (pair.source_smiles, pair.target_smiles) for pair in historical_pairs
    }
    development_pairs, development_counts = base.build_pairs(
        validation_rows,
        max_atoms=int(args.max_atoms),
        fingerprint_bits=int(args.fingerprint_bits),
        condition_dim=int(args.condition_dim),
        allowed_counts=allowed_counts,
        timeout=int(args.mcs_timeout),
        min_common_fraction=float(args.min_common_fraction),
        limit=int(args.validation_limit),
        seed=int(args.validation_selection_seed),
        forbidden_sources=historical_sources,
        forbidden_pairs=historical_keys,
    )
    development_sources = {pair.source_smiles for pair in development_pairs}
    development_keys = {
        (pair.source_smiles, pair.target_smiles) for pair in development_pairs
    }
    train_pairs, train_counts = base.build_pairs(
        base.read_rows(args.train_csv),
        max_atoms=int(args.max_atoms),
        fingerprint_bits=int(args.fingerprint_bits),
        condition_dim=int(args.condition_dim),
        allowed_counts=allowed_counts,
        timeout=int(args.mcs_timeout),
        min_common_fraction=float(args.min_common_fraction),
        limit=int(args.train_limit),
        seed=int(args.train_selection_seed),
        forbidden_sources=development_sources,
        forbidden_pairs=development_keys,
    )
    if len(train_pairs) < 32:
        raise ValueError(f"Need at least 32 selected train pairs, found {len(train_pairs)}")

    evidence_rows: list[dict[str, object]] = []
    transform_vocab: set[tuple[str, str]] = set()
    target_fragments: set[str] = set()
    source_fragments: set[str] = set()
    for pair in train_pairs:
        support = pair_fragment_support(pair, args)
        transforms = support.pop("transforms")
        for _core, source_variable, target_variable in transforms:
            transform_vocab.add((source_variable, target_variable))
            source_fragments.add(source_variable)
            target_fragments.add(target_variable)
        evidence_rows.append(
            {
                "property_count": int(pair.property_count),
                "task": str(pair.task),
                **support,
            }
        )

    overall = summarize_group(evidence_rows)
    by_property_count = {
        str(count): summarize_group(
            [row for row in evidence_rows if int(row["property_count"]) == int(count)]
        )
        for count in sorted(allowed_counts)
    }
    by_task = {
        task: summarize_group([row for row in evidence_rows if row["task"] == task])
        for task in sorted({str(row["task"]) for row in evidence_rows})
    }
    growth_rows = [
        row
        for row in evidence_rows
        if "HBA:+1" in str(row["task"])
        and "MW:+1" in str(row["task"])
        and "QED:-1" in str(row["task"])
    ]
    growth = summarize_group(growth_rows)
    checks = {
        "overall_coverage": {
            "value": overall["coverage"],
            "threshold": float(args.gate_overall_coverage),
        },
        "three_property_coverage": {
            "value": by_property_count.get("3", {}).get("coverage", 0.0),
            "threshold": float(args.gate_three_property_coverage),
        },
        "growth_task_coverage": {
            "value": growth["coverage"],
            "threshold": float(args.gate_growth_task_coverage),
        },
        "exact_reconstruction": {
            "value": overall["exact_reconstruction_rate"],
            "threshold": float(args.gate_exact_reconstruction),
        },
        "unique_target_fragments": {
            "value": len(target_fragments),
            "threshold": int(args.gate_unique_target_fragments),
        },
    }
    failures = [
        name
        for name, item in checks.items()
        if float(item["value"]) < float(item["threshold"])
    ]
    manifest = {
        "protocol": PROTOCOL,
        "train_csv": str(args.train_csv),
        "train_csv_sha256": belief.file_sha256(args.train_csv),
        "validation_csv": str(args.validation_csv),
        "validation_csv_sha256": belief.file_sha256(args.validation_csv),
        "selected_train_pairs": len(train_pairs),
        "selected_development_pairs": len(development_pairs),
        "train_filter_counts": train_counts,
        "development_filter_counts": development_counts,
        "historical_development_filter_counts": historical_counts,
        "train_development_source_overlap": len(
            {pair.source_smiles for pair in train_pairs} & development_sources
        ),
        "train_development_pair_overlap": len(
            {(pair.source_smiles, pair.target_smiles) for pair in train_pairs}
            & development_keys
        ),
        "development_target_fragment_access": False,
        "train_only_fragment_vocabulary": True,
        "min_core_heavy_atoms": int(args.min_core_heavy_atoms),
        "max_variable_heavy_atoms": int(args.max_variable_heavy_atoms),
        "train_selection_seed": int(args.train_selection_seed),
        "validation_selection_seed": int(args.validation_selection_seed),
        "validation_exclusion_seed": int(args.validation_exclusion_seed),
        "requested_accelerator_hours": 0,
    }
    summary = {
        "protocol": PROTOCOL,
        "manifest": manifest,
        "overall": overall,
        "by_property_count": by_property_count,
        "growth_task": growth,
        "by_task": by_task,
        "vocabulary": {
            "unique_transforms": len(transform_vocab),
            "unique_source_fragments": len(source_fragments),
            "unique_target_fragments": len(target_fragments),
            "transform_observations": sum(
                int(row["exact_transform_count"] or row["transform_count"])
                for row in evidence_rows
            ),
        },
        "gate": {"passed": not failures, "checks": checks, "failures": failures},
        "next_stage": (
            "train_latent_fragment_attachment_kernel"
            if not failures
            else "train_two_step_residual_local_delta_without_fragment_vocabulary"
        ),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
