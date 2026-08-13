#!/usr/bin/env python3
"""Merge exact-20 direct-repair trajectory shards without ranking semantics."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-dir", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--manifest-json", required=True, type=Path)
    parser.add_argument("--shard-count", type=int, default=16)
    parser.add_argument("--expected-conditions", type=int, required=True)
    return parser.parse_args(argv)


def truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    rows: list[dict[str, str]] = []
    seen_conditions: set[str] = set()
    shard_manifests = []
    for shard_index in range(int(args.shard_count)):
        tag = f"{shard_index:03d}"
        shard_rows = read_csv(args.shard_dir / f"trajectories_{tag}.csv")
        manifest = json.loads(
            (args.shard_dir / f"manifest_{tag}.json").read_text(encoding="utf-8")
        )
        if (
            manifest.get("output_selection") != "none"
            or manifest.get("internal_molecular_candidate_pool") is not False
            or manifest.get("evaluation_target_access") is not False
            or manifest.get("evaluation_oracle_access") is not False
            or int(manifest.get("attempts_per_condition", 0)) != 20
        ):
            raise ValueError(f"Direct-repair shard {tag} contract violation")
        shard_conditions = {str(row["condition_id"]) for row in shard_rows}
        if seen_conditions & shard_conditions:
            raise ValueError(f"Duplicate conditions in direct-repair shard {tag}")
        if len(shard_rows) != 20 * len(shard_conditions):
            raise ValueError(f"Direct-repair shard {tag} is not exact n=20")
        seen_conditions.update(shard_conditions)
        rows.extend(shard_rows)
        shard_manifests.append(manifest)

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if "candidate_rank" in row or "candidate_selected" in row:
            raise ValueError("Rank/selection columns are forbidden in direct-repair output")
        grouped[str(row["condition_id"])].append(row)
    if len(grouped) != int(args.expected_conditions):
        raise ValueError(f"conditions={len(grouped)} expected={args.expected_conditions}")
    for condition_id, items in grouped.items():
        attempts = sorted(int(row["generation_attempt_index"]) for row in items)
        trajectories = {row["trajectory_id"] for row in items}
        if attempts != list(range(1, 21)) or len(trajectories) != 20:
            raise ValueError(f"{condition_id} direct trajectory contract failed")
        if not all(truthy(row.get("candidate_valid")) for row in items):
            raise ValueError(f"{condition_id} contains an invalid frozen attempt")
        if not all(truthy(row.get("candidate_source_similarity_pass")) for row in items):
            raise ValueError(f"{condition_id} violates source similarity floor")

    rows.sort(key=lambda row: (row["condition_id"], int(row["generation_attempt_index"])))
    write_csv(args.output_csv, rows)
    unique_counts = [len({row["generated_smiles"] for row in items}) for items in grouped.values()]
    repeated = sum(truthy(row.get("candidate_attempt_is_repeat")) for row in rows)
    noop = sum(truthy(row.get("candidate_is_noop")) for row in rows)
    trace_rows = sum(int(row.get("trajectory_step_count", 0)) > 0 for row in rows)
    transactional = all(
        item.get("train_verifier_transactional_acceptance") is True
        for item in shard_manifests
    )
    protocols = {str(item.get("protocol", "")) for item in shard_manifests}
    if len(protocols) != 1:
        raise ValueError("Direct-repair shard protocols differ")
    transaction_policies = {
        json.dumps(item.get("transaction_policy"), sort_keys=True)
        for item in shard_manifests
    }
    if len(transaction_policies) != 1:
        raise ValueError("Direct-repair shard transaction policies differ")
    manifest = {
        "protocol": protocols.pop(),
        "method": str(shard_manifests[0].get("method", "")),
        "data_role": "fit_tools_to_source_only_disjoint_dev",
        "evaluation_target_access": False,
        "evaluation_oracle_access": False,
        "official_test_content_access": False,
        "output_selection": "none",
        "output_rows_have_rank": False,
        "output_rows_have_selected_flag": False,
        "internal_molecular_candidate_pool": False,
        "candidate_budget": 20,
        "attempts_per_condition": 20,
        "conditions": len(grouped),
        "output_rows": len(rows),
        "attempted_candidates_total": len(rows),
        "unique_candidates_total": sum(unique_counts),
        "unique_valid_candidates_total": sum(unique_counts),
        "mean_unique_candidates_per_condition": sum(unique_counts) / max(len(unique_counts), 1),
        "min_unique_candidates_per_condition": min(unique_counts, default=0),
        "repeated_attempt_rows": repeated,
        "noop_attempt_rows": noop,
        "trajectory_rows_with_executed_edit": trace_rows,
        "trajectory_trace_rate": trace_rows / max(len(rows), 1),
        "mean_steps_per_attempt": sum(
            int(row.get("trajectory_step_count", 0)) for row in rows
        ) / max(len(rows), 1),
        "mean_proposals_per_attempt": sum(
            int(row.get("trajectory_proposal_count", row.get("trajectory_step_count", 0)))
            for row in rows
        )
        / max(len(rows), 1),
        "committed_edits_total": sum(
            int(item.get("committed_edits_total", 0)) for item in shard_manifests
        ),
        "transaction_rollbacks_total": sum(
            int(item.get("transaction_rollbacks_total", 0)) for item in shard_manifests
        ),
        "train_verifier_transactional_acceptance": transactional,
        "transaction_policy": (
            shard_manifests[0].get("transaction_policy") if transactional else None
        ),
        "train_verifier_observation_after_each_edit": all(
            item.get("train_verifier_observation_after_each_edit") is True
            for item in shard_manifests
        ),
        "shard_count": int(args.shard_count),
    }
    args.manifest_json.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_json.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
