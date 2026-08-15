#!/usr/bin/env python3
"""Merge the eight preregistered B30 assay-support shards."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Sequence


PROTOCOL = "target_free_table1_assay_latent_action_support_v30"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--protocol-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    preregistration = json.loads(args.protocol_manifest.read_text(encoding="utf-8"))
    if preregistration.get("protocol") != PROTOCOL:
        raise ValueError("B30 merge preregistration drift")
    expected = int(preregistration["shards"])
    shards = []
    digests = set()
    condition_ids = set()
    for index in range(expected):
        path = args.shard_root / f"shard_{index:03d}" / "summary.json"
        if not path.is_file():
            raise ValueError(f"B30 shard is incomplete: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("protocol") != PROTOCOL:
            raise ValueError(f"B30 shard protocol drift: {index}")
        manifest = dict(payload.get("manifest", {}))
        if manifest.get("shard_index") != index:
            raise ValueError(f"B30 shard index drift: {index}")
        if manifest.get("generation_target_access") is not False:
            raise ValueError(f"B30 shard accessed a target: {index}")
        if manifest.get("molecular_candidate_ranking") is not False:
            raise ValueError(f"B30 shard performed molecular ranking: {index}")
        support = dict(payload["support"])
        condition_id = str(support["condition_id"])
        if condition_id in condition_ids:
            raise ValueError(f"B30 duplicate condition: {condition_id}")
        condition_ids.add(condition_id)
        digests.add(manifest.get("vocabulary_sha256"))
        shards.append(support)
    if len(digests) != 1:
        raise ValueError("B30 shard vocabulary drift")

    grouped: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    for row in shards:
        grouped[str(row["task"])].append(row)
    by_task = {}
    for task in preregistration["tasks"]:
        rows = grouped[task]
        if len(rows) != 4:
            raise ValueError(f"B30 task coverage drift: {task}={len(rows)}")
        by_task[task] = {
            "conditions": len(rows),
            "assay_support_conditions_t0_15": sum(
                bool(row["has_assay_support_t0_15"]) for row in rows
            ),
            "assay_support_rate_t0_15": sum(
                bool(row["has_assay_support_t0_15"]) for row in rows
            ) / len(rows),
            "full_support_conditions_t0_15": sum(
                bool(row["has_full_support_t0_15"]) for row in rows
            ),
            "full_support_rate_t0_15": sum(
                bool(row["has_full_support_t0_15"]) for row in rows
            ) / len(rows),
            "assay_improved_candidates_t0_15": sum(
                int(row["assay_improved_t0_15"]) for row in rows
            ),
            "full_success_candidates_t0_15": sum(
                int(row["full_property_success_t0_15"]) for row in rows
            ),
            "mean_assay_oracle_coverage": sum(
                float(row["assay_oracle_coverage"]) for row in rows
            ) / len(rows),
        }
    gates = dict(preregistration["gates"])
    checks = {}
    for task, values in by_task.items():
        checks[f"{task}:assay_support_rate_t0_15"] = {
            "value": values["assay_support_rate_t0_15"],
            "threshold": gates["minimum_task_assay_support_rate_t0_15"],
        }
        checks[f"{task}:full_support_rate_t0_15"] = {
            "value": values["full_support_rate_t0_15"],
            "threshold": gates["minimum_task_full_support_rate_t0_15"],
        }
        checks[f"{task}:oracle_coverage"] = {
            "value": values["mean_assay_oracle_coverage"],
            "threshold": gates["oracle_coverage"],
        }
    failures = [
        name
        for name, item in checks.items()
        if float(item["value"]) < float(item["threshold"])
    ]
    enough_assay_support = all(
        float(values["assay_support_rate_t0_15"])
        >= float(gates["minimum_task_assay_support_rate_t0_15"])
        for values in by_task.values()
    )
    decision = (
        "train_property_conditioned_joint_site_token_latent"
        if enough_assay_support
        else "expand_to_connected_region_latent_action_grammar"
    )
    summary = {
        "protocol": PROTOCOL,
        "manifest": {
            "shards": expected,
            "conditions": len(shards),
            "vocabulary_sha256": next(iter(digests)),
            "generation_target_access": False,
            "moledit_target_access": False,
            "diagnostic_only": True,
            "molecular_candidate_ranking": False,
            "selected_prediction_output": False,
        },
        "by_task": by_task,
        "conditions": shards,
        "gate": {"passed": not failures, "checks": checks, "failures": failures},
        "decision": decision,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
