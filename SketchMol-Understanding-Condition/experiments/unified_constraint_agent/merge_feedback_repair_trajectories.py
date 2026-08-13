#!/usr/bin/env python3
"""Merge and audit exact-20 feedback-repair trajectory shards."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping, Sequence


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-dir", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--manifest-json", required=True, type=Path)
    parser.add_argument("--controller-mode", choices=("llm", "deterministic"), required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--expected-conditions", type=int, default=200)
    parser.add_argument("--expected-ind", type=int, default=100)
    parser.add_argument("--expected-ood", type=int, default=100)
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
    manifests: list[dict[str, object]] = []
    seen_conditions: set[str] = set()
    for shard_index in range(int(args.shard_count)):
        tag = f"{shard_index:03d}"
        shard_rows = read_csv(args.shard_dir / f"trajectories_{tag}.csv")
        manifest = json.loads(
            (args.shard_dir / f"manifest_{tag}.json").read_text(encoding="utf-8")
        )
        if (
            manifest.get("protocol") != "common_llm_feedback_repair_signal_v14a"
            or manifest.get("controller_mode") != args.controller_mode
            or manifest.get("output_selection") != "none"
            or manifest.get("internal_molecular_candidate_pool") is not False
            or manifest.get("evaluation_target_access") is not False
            or manifest.get("evaluation_oracle_access") is not False
            or int(manifest.get("attempts_per_condition", 0)) != 20
            or int(manifest.get("shard_index", -1)) != shard_index
            or int(manifest.get("shard_count", 0)) != int(args.shard_count)
        ):
            raise ValueError(f"Feedback-repair shard {tag} contract violation")
        shard_conditions = {str(row["condition_id"]) for row in shard_rows}
        if seen_conditions & shard_conditions:
            raise ValueError(f"Duplicate conditions across feedback-repair shard {tag}")
        if len(shard_rows) != 20 * len(shard_conditions):
            raise ValueError(f"Feedback-repair shard {tag} is not exact n=20")
        seen_conditions.update(shard_conditions)
        rows.extend(shard_rows)
        manifests.append(manifest)

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if "candidate_rank" in row or "candidate_selected" in row:
            raise ValueError("Rank/selection columns are forbidden in feedback-repair output")
        grouped[str(row["condition_id"])].append(row)
    if len(grouped) != int(args.expected_conditions):
        raise ValueError(f"conditions={len(grouped)} expected={args.expected_conditions}")
    for condition_id, items in grouped.items():
        attempts = sorted(int(row["generation_attempt_index"]) for row in items)
        trajectories = {str(row["trajectory_id"]) for row in items}
        if attempts != list(range(1, 21)) or len(trajectories) != 20:
            raise ValueError(f"{condition_id} feedback trajectory contract failed")
        if not all(truthy(row.get("candidate_valid")) for row in items):
            raise ValueError(f"{condition_id} contains an invalid frozen attempt")
        if not all(truthy(row.get("candidate_source_similarity_pass")) for row in items):
            raise ValueError(f"{condition_id} violates source similarity floor")

    ind_conditions = sum(int(item.get("ind_conditions", 0)) for item in manifests)
    ood_conditions = sum(int(item.get("ood_conditions", 0)) for item in manifests)
    if ind_conditions != int(args.expected_ind) or ood_conditions != int(args.expected_ood):
        raise ValueError(
            f"split conditions IND/OOD={ind_conditions}/{ood_conditions} "
            f"expected={args.expected_ind}/{args.expected_ood}"
        )
    protocols = {str(item.get("protocol")) for item in manifests}
    seeds = {int(item.get("seed", -1)) for item in manifests}
    max_committed_edits = {int(item.get("max_committed_edits", -1)) for item in manifests}
    max_proposals = {int(item.get("max_proposals", -1)) for item in manifests}
    if len(protocols) != 1 or len(seeds) != 1 or len(max_committed_edits) != 1 or len(max_proposals) != 1:
        raise ValueError("Feedback-repair shard configuration drift")

    rows.sort(key=lambda row: (str(row["condition_id"]), int(row["generation_attempt_index"])))
    write_csv(args.output_csv, rows)
    unique_counts = [len({str(row["generated_smiles"]) for row in items}) for items in grouped.values()]
    action_counts: Counter[str] = Counter()
    for item in manifests:
        action_counts.update(
            {str(key): int(value) for key, value in dict(item.get("controller_action_counts", {})).items()}
        )
    decisions = sum(int(item.get("controller_decisions", 0)) for item in manifests)
    divergence_count = sum(
        float(item.get("controller_deterministic_divergence_rate", 0.0))
        * int(item.get("controller_decisions", 0))
        for item in manifests
    )
    manifest = {
        "protocol": protocols.pop(),
        "controller_mode": args.controller_mode,
        "data_role": "fit_tools_to_stable_disjoint_dev_signal",
        "evaluation_target_access": False,
        "evaluation_oracle_access": False,
        "official_test_content_access": False,
        "output_selection": "none",
        "internal_molecular_candidate_pool": False,
        "candidate_budget": 20,
        "attempts_per_condition": 20,
        "conditions_per_split": int(args.expected_ind),
        "ind_conditions": ind_conditions,
        "ood_conditions": ood_conditions,
        "conditions": len(grouped),
        "candidate_rows": len(rows),
        "max_committed_edits": max_committed_edits.pop(),
        "max_proposals": max_proposals.pop(),
        "committed_edits_total": sum(int(item.get("committed_edits_total", 0)) for item in manifests),
        "transaction_rollbacks_total": sum(
            int(item.get("transaction_rollbacks_total", 0)) for item in manifests
        ),
        "noop_attempt_rows": sum(int(item.get("noop_attempt_rows", 0)) for item in manifests),
        "mean_unique_candidates_per_condition": sum(unique_counts) / max(len(unique_counts), 1),
        "min_unique_candidates_per_condition": min(unique_counts, default=0),
        "controller_decisions": decisions,
        "controller_action_counts": dict(sorted(action_counts.items())),
        "controller_deterministic_divergence_rate": divergence_count / max(decisions, 1),
        "soft_uncertainty_feedback": all(item.get("soft_uncertainty_feedback") is True for item in manifests),
        "exact_qed_safety_critic": all(item.get("exact_qed_safety_critic") is True for item in manifests),
        "nonexact_verifier_is_hard_veto": not all(
            item.get("nonexact_verifier_is_hard_veto") is False for item in manifests
        ),
        "shard_count": int(args.shard_count),
        "seed": seeds.pop(),
    }
    args.manifest_json.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_json.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
