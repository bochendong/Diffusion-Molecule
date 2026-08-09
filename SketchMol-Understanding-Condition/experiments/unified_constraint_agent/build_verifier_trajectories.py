#!/usr/bin/env python3
"""Build strict-first preferences and revision cases from audited candidate pools."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

from audit_candidate_pools import (
    RunSpec,
    generation_rank,
    generated_smiles,
    group_id,
    load_specs,
    property_distance,
    property_fraction,
    read_rows,
    source_similarity,
    strict_success,
    valid_candidate,
)
from molecular_constraint_ir import build_constraint_ir, truthy


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-json", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args(argv)


def preference_key(row: Mapping[str, object], index: int) -> tuple[float, ...]:
    distance = property_distance(row)
    similarity = source_similarity(row)
    return (
        float(strict_success(row)),
        property_fraction(row),
        -(distance if distance is not None else 1e6),
        similarity if similarity is not None else 0.0,
        -generation_rank(row, index)[0],
    )


def property_feedback(row: Mapping[str, object]) -> dict[str, object]:
    per_property = {}
    payload = str(row.get("external_property_success_json", "") or "").strip()
    if payload:
        try:
            parsed = json.loads(payload)
        except (ValueError, json.JSONDecodeError):
            parsed = {}
        if isinstance(parsed, Mapping):
            per_property = {str(key): truthy(value) for key, value in parsed.items()}
    unsatisfied = sorted(key for key, passed in per_property.items() if not passed)
    return {
        "valid": valid_candidate(row),
        "strict_success": strict_success(row),
        "property_success_fraction": property_fraction(row),
        "property_distance": property_distance(row),
        "source_similarity": source_similarity(row),
        "unsatisfied_properties": unsatisfied,
    }


def trajectory_row(
    spec: RunSpec,
    source_row: Mapping[str, object],
    chosen: Mapping[str, object] | None,
    rejected: Mapping[str, object],
    *,
    trajectory_type: str,
) -> dict[str, object]:
    ir = build_constraint_ir(source_row)
    return {
        "run": spec.name,
        "suite": spec.suite,
        "condition_id": ir.condition_id,
        "trajectory_type": trajectory_type,
        "constraint_ir_json": ir.to_json(),
        "task_mode": ir.task_mode,
        "action_space": ir.action_space,
        "instruction": ir.instruction,
        "source_smiles": ir.source_smiles,
        "chosen_smiles": generated_smiles(chosen or {}),
        "rejected_smiles": generated_smiles(rejected),
        "chosen_feedback_json": json.dumps(property_feedback(chosen or {}), sort_keys=True),
        "rejected_feedback_json": json.dumps(property_feedback(rejected), sort_keys=True),
        "candidate_budget": spec.budget,
    }


def build_run(spec: RunSpec) -> tuple[list[dict[str, object]], dict[str, int]]:
    rows = read_rows(spec.candidate_csv)
    grouped: dict[str, list[tuple[int, dict[str, str]]]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[group_id(row, index)].append((index, row))
    output = []
    strict_preferences = 0
    revision_cases = 0
    skipped_no_negative = 0
    for indexed in grouped.values():
        ordered_pairs = sorted(indexed, key=lambda pair: generation_rank(pair[1], pair[0]))[: spec.budget]
        candidates = [row for _, row in ordered_pairs]
        source_row = candidates[0]
        positives = [(index, row) for index, row in enumerate(candidates) if strict_success(row)]
        hard_negatives = [
            (index, row)
            for index, row in enumerate(candidates)
            if valid_candidate(row) and not strict_success(row)
        ]
        if positives and hard_negatives:
            chosen = max(positives, key=lambda pair: preference_key(pair[1], pair[0]))[1]
            rejected = max(hard_negatives, key=lambda pair: preference_key(pair[1], pair[0]))[1]
            output.append(
                trajectory_row(
                    spec,
                    source_row,
                    chosen,
                    rejected,
                    trajectory_type="strict_preference",
                )
            )
            strict_preferences += 1
        elif hard_negatives:
            rejected = max(hard_negatives, key=lambda pair: preference_key(pair[1], pair[0]))[1]
            output.append(
                trajectory_row(
                    spec,
                    source_row,
                    None,
                    rejected,
                    trajectory_type="revision_needed",
                )
            )
            revision_cases += 1
        else:
            skipped_no_negative += 1
    return output, {
        "groups": len(grouped),
        "strict_preferences": strict_preferences,
        "revision_cases": revision_cases,
        "skipped_no_negative": skipped_no_negative,
    }


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    summaries = {}
    for spec in load_specs(args.config_json):
        run_rows, summary = build_run(spec)
        rows.extend(run_rows)
        summaries[spec.name] = summary
    if not rows:
        raise SystemExit("No verifier trajectories could be built")
    write_csv(args.output_dir / "verifier_trajectories.csv", rows)
    with (args.output_dir / "verifier_trajectories.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    manifest = {
        "protocol": "unified_constraint_agent_verifier_trajectory_v1",
        "config_json": str(args.config_json),
        "rows": len(rows),
        "trajectory_type_counts": {
            key: sum(row["trajectory_type"] == key for row in rows)
            for key in ("strict_preference", "revision_needed")
        },
        "runs": summaries,
    }
    (args.output_dir / "verifier_trajectories.manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
